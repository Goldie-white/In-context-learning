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
import numpy as np
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
from incontext import predictor_flax_w_loss_w  # ⭐ 改动1：导入w预测器（使用w loss版本）

# 基础训练参数 (与原始项目model_trainer.py和main.py一致)
flags.DEFINE_integer("seed", default=0, help="随机种子")
flags.DEFINE_integer("batch_size", default=64, help="批次大小")
flags.DEFINE_integer("x_dim", default=20, help="输入维度")
flags.DEFINE_integer("num_exemplars", default=None, help="示例数量（固定长度模式，设置此项则忽略min/max）")
flags.DEFINE_integer("min_num_exemplars", default=20, help="最小示例数量（可变长度模式）")
flags.DEFINE_integer("max_num_exemplars", default=50, help="最大示例数量（可变长度模式）")
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
flags.DEFINE_float("dropout_rate", default=0.0, help="Dropout rate (0.0-1.0), 建议值: 0.1-0.2")
flags.DEFINE_float("attention_dropout_rate", default=0.0, help="Attention dropout rate (0.0-1.0), 建议值: 0.1-0.2")

# 优化器参数
flags.DEFINE_string("lr_scheduler_type", default="cosine", help="学习率调度器类型")
flags.DEFINE_float("adam_b1", default=0.9, help="Adam b1")
flags.DEFINE_float("adam_b2", default=0.98, help="Adam b2")
flags.DEFINE_float("adam_eps", default=1e-9, help="Adam eps")

FLAGS = flags.FLAGS


def train_step(state, seq, task_ids, w_target, model, learning_rate_fn, dropout_rng=None):
    """执行单步训练"""
    dropout_rng = random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """训练损失函数 - 使用 w 的 MSE"""
        output = model.apply({"params": params},
                           inputs=seq,
                           task_ids=task_ids,
                           w_target=w_target,  # ⭐ 传入真实的w向量
                           train=True,
                           rngs={"dropout": dropout_rng})
        return output[0].mean(), output

    lr = learning_rate_fn(state.step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, extras), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(extras[0], "batch")
    # extras[1] = (y_errors, w_errors, y_pred, w_pred, seq_pred, seq_hiddens)
    y_errors = jax.lax.psum(extras[1][0], "batch").sum(axis=0)
    w_errors = jax.lax.psum(extras[1][1], "batch").sum(axis=0)
    metrics = {"loss": loss, "lr": lr, "y_errors": y_errors, "w_errors": w_errors}
    return new_state, metrics


def get_model(rng, args):
    """初始化模型和优化器"""
    rng, init_rng = random.split(rng)

    # 论文标准配置：L=16, H=512, M=4
    # 如果args中没有设置，使用论文标准值
    n_layers = getattr(args, 'n_layers', 16)
    hidden_size = getattr(args, 'hidden_size', 512)
    n_heads = getattr(args, 'n_heads', 4)
    
    # 确定最大长度：如果使用可变长度，使用max_num_exemplars；否则使用num_exemplars
    if args.num_exemplars is not None:
        max_num_exemplars = args.num_exemplars
    else:
        max_num_exemplars = args.max_num_exemplars
    
    logging.info(f"Transformer配置: L={n_layers}, H={hidden_size}, M={n_heads}")
    logging.info(f"⭐ 使用W预测器: 输出维度={args.x_dim}")
    logging.info(f"⭐ 最大序列长度: {max_num_exemplars} exemplars")

    # 创建Transformer配置
    config = transformer_lib_flax.TransformerConfig(
        num_heads=n_heads,
        num_layers=n_layers,
        hidden_size=hidden_size,
        dropout_rate=args.dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        loss_on_x_steps=args.loss_on_x_steps,
        norm_first=args.norm_first,
        disable_layer_norms=args.disable_layer_norms,
        final_layer_norm=args.final_layer_norm,
        kernel_init=transformer_lib_flax.nn_init_parser(args.kernel_init),
        bias_init=transformer_lib_flax.nn_init_parser(args.bias_init),
        linear_w_init=transformer_lib_flax.nn_init_parser(args.linear_w_init),
        linear_bias_init=transformer_lib_flax.nn_init_parser(args.linear_bias_init),
        posemb_init=transformer_lib_flax.nn_init_parser(args.posemb_init),
        max_len=(max_num_exemplars + 1) * 2,  # 使用最大长度
        inner_dim=None,
        activation_fn=transformer_lib_flax.nn_activation_parser(args.activation_fn),
    )

    # ⭐ 改动3：使用CausalLM_W并传入x_dim（使用w loss版本）
    model = predictor_flax_w_loss_w.CausalLM_W(config=config, x_dim=args.x_dim)

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
        lambda state, seq, task_ids, w_target, dropout_rng: train_step(state, seq, task_ids, w_target, model, scheduler, dropout_rng),
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

    # 确定训练模式：固定长度 vs 可变长度
    if args.num_exemplars is not None:
        # 固定长度模式
        use_variable_length = False
        train_num_exemplars = args.num_exemplars
        length_info = f"{args.num_exemplars} (固定)"
    else:
        # 可变长度模式
        use_variable_length = True
        train_num_exemplars = args.max_num_exemplars  # 初始化sampler用最大长度
        length_info = f"{args.min_num_exemplars}-{args.max_num_exemplars} (均匀随机)"

    logging.info("开始训练...")
    logging.info("="*70)
    logging.info("训练配置 (W预测版本 - 使用W MSE损失):")
    logging.info(f"  模型: L=16, H=512, M=4")
    logging.info(f"  ⭐ 预测目标: w向量 (维度={args.x_dim})")
    logging.info(f"  ⭐ 损失函数: MSE(w_pred, w_true) - 直接优化w的预测")
    logging.info(f"  训练: {args.n_epochs * args.n_iter_per_epoch} iterations ({args.n_epochs} epochs × {args.n_iter_per_epoch} iters)")
    logging.info(f"  数据: {length_info} (x,y)对, x_dim={args.x_dim}, batch={args.batch_size}")
    logging.info(f"  分布: p(w)=N(0,I), p(x)=N(0,I)")
    logging.info(f"  优化: lr={args.learning_rate}, scheduler={args.lr_scheduler_type}, Adam(β1={args.adam_b1}, β2={args.adam_b2})")
    logging.info(f"  Warmup: {args.n_epochs // 5} epochs (~{(args.n_epochs // 5) * args.n_iter_per_epoch} steps, 20% of training)")
    if use_variable_length:
        logging.info(f"  ⭐ 可变长度训练: 每次迭代从[{args.min_num_exemplars}, {args.max_num_exemplars}]均匀随机抽样")
    logging.info("="*70)

    # 初始化模型
    model, state, p_train_step = get_model(new_rng, args)

    # 检查并恢复 checkpoint
    checkpoint_dir = os.path.abspath(os.path.join(args.exp_folder, "ckpt"))  # 转换为绝对路径
    start_epoch = 0
    start_iteration = 0
    
    if gfile.exists(checkpoint_dir):
        try:
            # 先 unreplicate 以便恢复
            state_unreplicated = jax_utils.unreplicate(state)
            # 恢复最新的 checkpoint（使用绝对路径）
            restored_state = checkpoints.restore_checkpoint(checkpoint_dir, state_unreplicated)
            
            if restored_state is not None and hasattr(restored_state, 'step'):
                # 获取已训练的 step 数
                start_step = int(restored_state.step)
                start_epoch = start_step // args.n_iter_per_epoch
                start_iteration = start_step % args.n_iter_per_epoch
                
                # 重新 replicate 恢复的状态
                state = jax_utils.replicate(restored_state)
                
                logging.info("="*70)
                logging.info("✅ 从 checkpoint 恢复训练")
                logging.info(f"  已完成: {start_step} steps = {start_epoch} epochs + {start_iteration} iterations")
                logging.info(f"  继续训练: 从 epoch {start_epoch}, iteration {start_iteration} 开始")
                logging.info("="*70)
            else:
                logging.info("checkpoint 无效，从头开始训练")
        except Exception as e:
            logging.warning(f"恢复 checkpoint 失败: {e}")
            logging.info("从头开始训练")
    else:
        logging.info("未找到 checkpoint，从头开始训练")

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

    # 准备dropout随机数
    dropout_rngs = random.split(new_rng, jax.local_device_count())
    
    # 确定最大长度用于padding
    if args.num_exemplars is not None:
        max_len_for_padding = args.num_exemplars
    else:
        max_len_for_padding = args.max_num_exemplars

    # 训练循环
    metrics_history = []
    for epoch in range(start_epoch, args.n_epochs):
        epoch_metrics = []
        epoch_lengths = []  # 记录本epoch使用的所有长度
        
        # 如果是恢复训练，第一个epoch从start_iteration开始；否则从0开始
        start_iter = start_iteration if epoch == start_epoch else 0
        
        for iteration in range(start_iter, args.n_iter_per_epoch):
            # 如果是可变长度模式，每次迭代随机选择长度
            if use_variable_length:
                current_length = np.random.randint(args.min_num_exemplars, args.max_num_exemplars + 1)
                epoch_lengths.append(current_length)  # 记录长度
            else:
                current_length = args.num_exemplars
            
            # 创建当前长度的sampler
            sampler = sampler_lib.Sampler(
                current_length,
                args.x_dim,
                args.hidden_size,
                x_distribution_fn=sampler_lib.str_to_distribution_fn(args.x_distribution_str),
                w_distribution_fn=sampler_lib.str_to_distribution_fn(args.w_distribution_str),
                noise_std=args.noise_std,
                task_probs=task_probs,
            )
            
            # 采样数据
            seqs, coefficients, *_ = sampler.sample(n=args.batch_size)
            # 获取任务类型
            task_ids = sampler.get_last_task_ids()
            
            # 如果是可变长度，需要padding到最大长度
            if use_variable_length and current_length < max_len_for_padding:
                # seqs shape: (batch, current_length*2, x_dim+1)
                # 需要padding到 (batch, max_len_for_padding*2, x_dim+1)
                pad_length = (max_len_for_padding - current_length) * 2
                padding = np.zeros((seqs.shape[0], pad_length, seqs.shape[2]))
                seqs = np.concatenate([seqs, padding], axis=1)
            
            # 转换为JAX数组并分片
            seqs = jnp.array(seqs)
            coefficients = jnp.array(coefficients)  # 真实的w向量
            task_ids = jnp.array(task_ids, dtype=jnp.int32)
            seqs = common_utils.shard(seqs)
            coefficients = common_utils.shard(coefficients)  # ⭐ 分片w向量
            task_ids = common_utils.shard(task_ids)
            
            # 执行训练步骤
            state, metrics = p_train_step(state, seqs, task_ids, coefficients, dropout_rng=dropout_rngs)
            
            # 收集指标
            metrics = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))
            # 如果是可变长度，记录该iteration的实际长度，用于后续正确统计
            if use_variable_length:
                metrics["actual_length"] = current_length
            epoch_metrics.append(metrics)
        
        # Epoch结束，计算平均指标
        epoch_metrics = common_utils.stack_forest(epoch_metrics)
        avg_loss = jnp.mean(epoch_metrics["loss"])
        avg_lr = epoch_metrics["lr"][-1]
        
        # 获取最后一个iteration的指标（用于显示实际长度和对应的loss）
        # epoch_metrics经过stack_forest后，y_errors和w_errors的shape是(num_iterations, num_positions)
        last_y_errors = epoch_metrics["y_errors"][-1] / args.batch_size if epoch_metrics["y_errors"].shape[0] > 0 else jnp.array([])
        last_w_errors = epoch_metrics["w_errors"][-1] / args.batch_size if epoch_metrics["w_errors"].shape[0] > 0 else jnp.array([])
        
        # 获取最后一个iteration的实际长度
        if use_variable_length and len(epoch_lengths) > 0:
            last_length = epoch_lengths[-1]
            # 只输出到实际长度，不包含padding部分
            last_y_errors = last_y_errors[:last_length]
            last_w_errors = last_w_errors[:last_length]
            
            # 输出长度统计（整个epoch的统计）
            avg_length = np.mean(epoch_lengths)
            min_length = np.min(epoch_lengths)
            max_length = np.max(epoch_lengths)
            std_length = np.std(epoch_lengths)
            # 计算长度分布（最多显示前10个最常见的长度）
            length_counts = {}
            for length in epoch_lengths:
                length_counts[length] = length_counts.get(length, 0) + 1
            sorted_lengths = sorted(length_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            length_str = ", ".join([f"{length}({count})" for length, count in sorted_lengths])
            logging.info(f"Epoch {epoch+1}/{args.n_epochs} - "
                        f"Loss (W MSE): {avg_loss:.6f}, "
                        f"LR: {avg_lr:.2e}")
            logging.info(f"  📏 序列长度统计: 平均={avg_length:.1f}±{std_length:.1f}, 范围=[{min_length}, {max_length}], 主要分布: {length_str}")
            logging.info(f"  📏 最后批次: 序列长度={last_length}")
        else:
            last_length = args.num_exemplars if args.num_exemplars is not None else max_len_for_padding
        logging.info(f"Epoch {epoch+1}/{args.n_epochs} - "
                    f"Loss (W MSE): {avg_loss:.6f}, "
                    f"LR: {avg_lr:.2e}")
        
        # 输出最后一个iteration的位置loss数组（简洁格式）
        if len(last_w_errors) > 0:
            if len(last_w_errors) <= 100:
                w_loss_str = "[" + ", ".join([f"{float(last_w_errors[i]):.4f}" for i in range(len(last_w_errors))]) + "]"
                logging.info(f"Position W Loss (MSE, 长度={last_length}): {w_loss_str}")
            else:
                logging.info(f"Position W Loss (MSE, 长度={last_length}): (序列太长，共{len(last_w_errors)}个位置，仅显示前10个和后10个)")
                w_first = ", ".join([f"{float(last_w_errors[i]):.4f}" for i in range(10)])
                w_last = ", ".join([f"{float(last_w_errors[i]):.4f}" for i in range(len(last_w_errors)-10, len(last_w_errors))])
                logging.info(f"  前10: [{w_first}]")
                logging.info(f"  后10: [{w_last}]")
        
        if len(last_y_errors) > 0:
            if len(last_y_errors) <= 100:
                y_loss_str = "[" + ", ".join([f"{float(last_y_errors[i]):.4f}" for i in range(len(last_y_errors))]) + "]"
                logging.info(f"Position Y Loss (MSE, 长度={last_length}): {y_loss_str}")
            else:
                logging.info(f"Position Y Loss (MSE, 长度={last_length}): (序列太长，共{len(last_y_errors)}个位置，仅显示前10个和后10个)")
                y_first = ", ".join([f"{float(last_y_errors[i]):.4f}" for i in range(10)])
                y_last = ", ".join([f"{float(last_y_errors[i]):.4f}" for i in range(len(last_y_errors)-10, len(last_y_errors))])
                logging.info(f"  前10: [{y_first}]")
                logging.info(f"  后10: [{y_last}]")

        # 定期保存检查点
        if (epoch + 1) % 100 == 0:
            save_checkpoint(state, args.exp_folder)
        
        metrics_history.append(epoch_metrics)
    
    # 保存最终模型
    save_checkpoint(state, args.exp_folder)
    
    # 保存训练指标
    metrics_history = common_utils.stack_forest(metrics_history)
    metrics_history["y_errors"] = jnp.mean(metrics_history["y_errors"], axis=0) / args.batch_size
    metrics_history["w_errors"] = jnp.mean(metrics_history["w_errors"], axis=0) / args.batch_size
    
    with gfile.GFile(os.path.join(args.exp_folder, "metrics.pickle"), "wb") as handle:
        pickle.dump(metrics_history, handle)
    
    logging.info("训练完成！")
    logging.info(f"最终W MSE损失: {jnp.mean(metrics_history['loss'][-100:]):.6f}")
    logging.info(f"最终Y MSE损失: {jnp.mean(metrics_history['y_errors'][:, -1][-100:]):.6f}")
    
    return state, metrics_history


def main(_):
    """主函数"""
    args = utils.flags_to_args()
    train_model(args)


if __name__ == "__main__":
    app.run(main)

