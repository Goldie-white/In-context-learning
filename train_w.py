#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文学习训练脚本 - W预测版本
预测权重向量w而不是直接预测y，然后通过y=w^Tx计算y值
"""

import os
# 强制使用单GPU避免NCCL多GPU问题
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import pickle
from absl import app, flags, logging
from flax import jax_utils
from flax.training import train_state
from flax.training import common_utils
from flax.training import checkpoints
from jax import random
import jax
import jax.numpy as jnp
import optax
from tensorflow.io import gfile

# 导入项目模块
from incontext import utils
from incontext import sampler_lib
from incontext import transformer_lib_flax
from incontext import predictor_flax_w  # ⭐ 改动1：导入w预测器（在incontext文件夹内）

# 基础训练参数 (与原始项目model_trainer.py和main.py一致)
flags.DEFINE_integer("seed", default=0, help="随机种子")
flags.DEFINE_integer("batch_size", default=64, help="批次大小")
flags.DEFINE_integer("x_dim", default=20, help="输入维度")
flags.DEFINE_integer("num_exemplars", default=40, help="示例数量")
flags.DEFINE_integer("n_epochs", default=5000, help="训练轮数")  # 原始项目: 5001 epochs
flags.DEFINE_integer("n_iter_per_epoch", default=100, help="每轮迭代次数")  # 原始项目: 100 iters, 总共~500K步
flags.DEFINE_float("learning_rate", default=1e-4, help="学习率")  # 原始项目: 1e-4
flags.DEFINE_float("weight_decay", default=0, help="权重衰减")  # 原始项目: 0
flags.DEFINE_string("exp_folder", default="experiments/w_predictor", help="实验文件夹")  # ⭐ 改动2：独立文件夹避免混淆

# 数据分布参数
flags.DEFINE_string("x_distribution_str", default="normal*1.0+0.0", help="输入分布")
flags.DEFINE_string("w_distribution_str", default="normal*1.0+0.0", help="权重分布")
flags.DEFINE_float("noise_std", default=0.0, help="噪声标准差")
flags.DEFINE_float("prob0", default=1.0, help="任务1概率: y=w^Tx (标准线性回归)")
flags.DEFINE_float("prob1", default=0.0, help="任务2概率: y=w^Tsort(x) (排序线性回归)")
flags.DEFINE_float("prob2", default=0.0, help="任务3概率: y=(dim/sqrt(2))*w^Tsoftmax(x) (缩放softmax线性回归)")
flags.DEFINE_float("prob3", default=0.0, help="任务4概率: y=||x-w||^2 (平方距离)")
flags.DEFINE_float("task_mix_alpha", default=1.0, help="[已弃用，请使用prob0-prob3] 任务混合比例")
flags.DEFINE_float("task3_prob", default=0.0, help="[已弃用，请使用prob0-prob3] 任务3概率")

# Transformer模型参数 (在 transformer_lib_flax.py 中定义)

# 优化器参数
flags.DEFINE_string("lr_scheduler_type", default="cosine", help="学习率调度器类型")
flags.DEFINE_float("adam_b1", default=0.9, help="Adam b1")
flags.DEFINE_float("adam_b2", default=0.98, help="Adam b2")
flags.DEFINE_float("adam_eps", default=1e-9, help="Adam eps")

FLAGS = flags.FLAGS


def train_step(state, seq, task_ids, model, learning_rate_fn, dropout_rng=None):
    """执行单步训练"""
    dropout_rng = random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """训练损失函数"""
        output = model.apply({"params": params},
                           inputs=seq,
                           task_ids=task_ids,
                           train=True,
                           rngs={"dropout": dropout_rng})
        return output[0].mean(), output

    lr = learning_rate_fn(state.step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, extras), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(extras[0], "batch")
    y_errors = jax.lax.psum(extras[1][0], "batch").sum(axis=0)
    metrics = {"loss": loss, "lr": lr, "y_errors": y_errors}
    return new_state, metrics


def get_model(rng, args):
    """初始化模型和优化器"""
    rng, init_rng = random.split(rng)

    # 论文标准配置：L=16, H=512, M=4
    # 如果args中没有设置，使用论文标准值
    n_layers = getattr(args, 'n_layers', 16)
    hidden_size = getattr(args, 'hidden_size', 512)
    n_heads = getattr(args, 'n_heads', 4)
    
    logging.info(f"Transformer配置: L={n_layers}, H={hidden_size}, M={n_heads}")
    logging.info(f"⭐ 使用W预测器: 输出维度={args.x_dim}")

    # 创建Transformer配置
    config = transformer_lib_flax.TransformerConfig(
        num_heads=n_heads,
        num_layers=n_layers,
        hidden_size=hidden_size,
        loss_on_x_steps=args.loss_on_x_steps,
        norm_first=args.norm_first,
        disable_layer_norms=args.disable_layer_norms,
        final_layer_norm=args.final_layer_norm,
        kernel_init=transformer_lib_flax.nn_init_parser(args.kernel_init),
        bias_init=transformer_lib_flax.nn_init_parser(args.bias_init),
        linear_w_init=transformer_lib_flax.nn_init_parser(args.linear_w_init),
        linear_bias_init=transformer_lib_flax.nn_init_parser(args.linear_bias_init),
        posemb_init=transformer_lib_flax.nn_init_parser(args.posemb_init),
        max_len=(args.num_exemplars + 1) * 2,
        inner_dim=None,
        activation_fn=transformer_lib_flax.nn_activation_parser(args.activation_fn),
    )

    # ⭐ 改动3：使用CausalLM_W并传入x_dim
    model = predictor_flax_w.CausalLM_W(config=config, x_dim=args.x_dim)

    @jax.jit
    def initialize_variables(init_rng):
        init_batch = jnp.ones((1, config.max_len, args.x_dim + 1), jnp.float32)
        init_variables = model.init(init_rng, inputs=init_batch, train=False)
        return init_variables

    init_variables = initialize_variables(init_rng)

    # 创建学习率调度器
    if args.lr_scheduler_type == "cosine":
        scheduler = transformer_lib_flax.create_learning_rate_scheduler(
            base_learning_rate=args.learning_rate,
            num_warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
            num_training_steps=args.n_epochs * args.n_iter_per_epoch,
        )
    elif args.lr_scheduler_type == "warmup":
        scheduler = transformer_lib_flax.create_learning_rate_scheduler_v2(
            factors="constant * linear_warmup",
            base_learning_rate=args.learning_rate,
            warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
        )
    else:
        def scheduler(_):
            return args.learning_rate

    # 创建优化器
    opt = optax.adamw(
        scheduler,
        b1=args.adam_b1,
        b2=args.adam_b2,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # 创建训练状态
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_variables["params"],
        tx=opt
    )

    # 复制到多个设备
    state = jax_utils.replicate(state)

    # 创建并行训练步骤
    p_train_step = jax.pmap(
        lambda state, seq, task_ids, dropout_rng: train_step(state, seq, task_ids, model, scheduler, dropout_rng),
        axis_name="batch",
    )

    return model, state, p_train_step


def save_checkpoint(state, exp_folder):
    """保存检查点"""
    # unreplicate会自动处理，不需要额外的get_array
    state = jax_utils.unreplicate(state)
    ckpt_dir = os.path.abspath(os.path.join(exp_folder, "ckpt/"))
    gfile.makedirs(ckpt_dir)
    checkpoints.save_checkpoint(
        ckpt_dir,
        state,
        step=int(state.step),
        keep=3,
        overwrite=True,
    )
    logging.info(f"保存检查点到: {ckpt_dir}")


def train_model(args):
    """主训练函数"""
    # 设置随机种子
    utils.set_seed(args.seed)
    rng = random.PRNGKey(args.seed)
    rng, new_rng = random.split(rng)

    # 创建实验文件夹
    gfile.makedirs(args.exp_folder)
    
    # 保存配置
    with gfile.GFile(os.path.join(args.exp_folder, "config.json"), "w") as handle:
        json.dump(args.initial_dict, handle)

    logging.info("开始训练...")
    logging.info("="*70)
    logging.info("训练配置 (W预测版本):")
    logging.info(f"  模型: L=16, H=512, M=4")
    logging.info(f"  ⭐ 预测目标: w向量 (维度={args.x_dim}), 然后计算y=w^Tx")
    logging.info(f"  训练: {args.n_epochs * args.n_iter_per_epoch} iterations ({args.n_epochs} epochs × {args.n_iter_per_epoch} iters)")
    logging.info(f"  数据: {args.num_exemplars} (x,y)对, x_dim={args.x_dim}, batch={args.batch_size}")
    logging.info(f"  分布: p(w)=N(0,I), p(x)=N(0,I)")
    logging.info(f"  优化: lr={args.learning_rate}, scheduler={args.lr_scheduler_type}, Adam(β1={args.adam_b1}, β2={args.adam_b2})")
    logging.info(f"  Warmup: {args.n_epochs // 5} epochs (~{(args.n_epochs // 5) * args.n_iter_per_epoch} steps, 20% of training)")
    logging.info("="*70)

    # 初始化模型
    model, state, p_train_step = get_model(new_rng, args)

    # 创建数据采样器
    rng, new_rng = random.split(rng)
    # Parse task probabilities
    task_probs = [args.prob0, args.prob1, args.prob2, args.prob3]
    prob_sum = sum(task_probs)
    
    # Check if probabilities sum to 1.0
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"任务概率之和必须等于1.0，当前为 {prob_sum}。\n"
            f"当前设置: prob0={args.prob0}, prob1={args.prob1}, prob2={args.prob2}, prob3={args.prob3}\n"
            f"请调整参数使其和为1.0"
        )
    
    logging.info(f"📝 任务概率设置: [Task1={args.prob0}, Task2={args.prob1}, Task3={args.prob2}, Task4={args.prob3}]")
    sampler = sampler_lib.Sampler(
        args.num_exemplars,
        args.x_dim,
        args.hidden_size,
        x_distribution_fn=sampler_lib.str_to_distribution_fn(args.x_distribution_str),
        w_distribution_fn=sampler_lib.str_to_distribution_fn(args.w_distribution_str),
        noise_std=args.noise_std,
        task_probs=task_probs,
    )

    # 准备dropout随机数
    dropout_rngs = random.split(new_rng, jax.local_device_count())

    # 训练循环
    metrics_history = []
    for epoch in range(args.n_epochs):
        epoch_metrics = []
        
        for iteration in range(args.n_iter_per_epoch):
            # 采样数据
            seqs, coefficients, *_ = sampler.sample(n=args.batch_size)
            # 获取任务类型
            task_ids = sampler.get_last_task_ids()
            
            # 转换为JAX数组并分片
            seqs = jnp.array(seqs)
            coefficients = jnp.array(coefficients)
            task_ids = jnp.array(task_ids, dtype=jnp.int32)
            seqs = common_utils.shard(seqs)
            task_ids = common_utils.shard(task_ids)
            
            # 执行训练步骤
            state, metrics = p_train_step(state, seqs, task_ids, dropout_rng=dropout_rngs)
            
            # 收集指标
            metrics = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))
            epoch_metrics.append(metrics)
        
        # Epoch结束，计算平均指标
        epoch_metrics = common_utils.stack_forest(epoch_metrics)
        avg_loss = jnp.mean(epoch_metrics["loss"])
        avg_lr = epoch_metrics["lr"][-1]
        y_errors = jnp.mean(epoch_metrics["y_errors"], axis=0) / args.batch_size
        
        logging.info(f"Epoch {epoch+1}/{args.n_epochs} - "
                    f"Loss: {avg_loss:.6f}, "
                    f"LR: {avg_lr:.2e}")
        
        # 输出位置loss数组（简洁格式）
        if len(y_errors) <= 40:
            loss_str = "[" + ", ".join([f"{float(y_errors[i]):.4f}" for i in range(len(y_errors))]) + "]"
            logging.info(f"Position losses: {loss_str}")

        # 定期保存检查点
        if (epoch + 1) % 100 == 0:
            save_checkpoint(state, args.exp_folder)
        
        metrics_history.append(epoch_metrics)
    
    # 保存最终模型
    save_checkpoint(state, args.exp_folder)
    
    # 保存训练指标
    metrics_history = common_utils.stack_forest(metrics_history)
    metrics_history["y_errors"] = jnp.mean(metrics_history["y_errors"], axis=0) / args.batch_size
    
    with gfile.GFile(os.path.join(args.exp_folder, "metrics.pickle"), "wb") as handle:
        pickle.dump(metrics_history, handle)
    
    logging.info("训练完成！")
    logging.info(f"最终损失: {jnp.mean(metrics_history['loss'][-100:]):.6f}")
    
    return state, metrics_history


def main(_):
    """主函数"""
    args = utils.flags_to_args()
    train_model(args)


if __name__ == "__main__":
    app.run(main)

