#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Y预测器：分析各个位置的y预测loss
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import pickle
import numpy as np
from absl import app, flags, logging
from flax import jax_utils
from flax.training import checkpoints
from jax import random
import jax
import jax.numpy as jnp

from incontext import utils
from incontext import sampler_lib
from incontext import transformer_lib_flax
from incontext import predictor_flax

flags.DEFINE_string("checkpoint_dir", default="experiments/y_predictor/ckpt", help="检查点目录")
flags.DEFINE_integer("n_test_samples", default=500, help="测试样本数")
flags.DEFINE_integer("seed", default=42, help="测试随机种子")
flags.DEFINE_string("output_file", default="experiments/y_predictor/y_analysis.pkl", help="输出文件")
flags.DEFINE_string("test_x_distribution_str", default=None, help="测试时x分布（None则使用训练分布）")
flags.DEFINE_string("test_w_distribution_str", default=None, help="测试时w分布（None则使用训练分布）")
flags.DEFINE_float("test_prob0", default=None, help="测试时任务1概率（None则使用训练设置）")
flags.DEFINE_float("test_prob1", default=None, help="测试时任务2概率（None则使用训练设置）")
flags.DEFINE_float("test_prob2", default=None, help="测试时任务3概率（None则使用训练设置）")
flags.DEFINE_float("test_prob3", default=None, help="测试时任务4概率（None则使用训练设置）")
flags.DEFINE_float("test_task_mix_alpha", default=None, help="[已弃用] 测试时任务混合比例（None则使用训练设置）")
flags.DEFINE_float("test_task3_prob", default=None, help="[已弃用] 测试时任务3概率（None则使用训练设置）")

FLAGS = flags.FLAGS


def load_model_and_config(checkpoint_dir):
    """加载模型和配置"""
    # 转换为绝对路径
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(checkpoint_dir), "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    args = utils.dict_to_args(config_dict)
    
    # 创建Transformer配置
    config = transformer_lib_flax.TransformerConfig(
        num_heads=getattr(args, 'n_heads', 4),
        num_layers=getattr(args, 'n_layers', 16),
        hidden_size=getattr(args, 'hidden_size', 512),
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
    
    # 创建模型
    model = predictor_flax.CausalLM(config)
    
    # 初始化变量
    rng = random.PRNGKey(0)
    init_batch = jnp.ones((1, config.max_len, args.x_dim + 1), jnp.float32)
    init_variables = model.init(rng, inputs=init_batch, train=False)
    
    # 加载检查点
    restored = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
    params = restored['params']
    
    logging.info(f"✅ 模型加载成功: {checkpoint_dir}")
    logging.info(f"   配置: L={config.num_layers}, H={config.hidden_size}, M={config.num_heads}")
    logging.info(f"   数据: {args.num_exemplars}个样本对, x_dim={args.x_dim}")
    
    # 显示分布配置
    x_dist = getattr(args, 'x_distribution_str', 'N/A')
    w_dist = getattr(args, 'w_distribution_str', 'N/A')
    logging.info(f"   分布: p(x)={x_dist}, p(w)={w_dist}")
    
    return model, params, args


def test_y_predictor(args):
    """测试Y预测器"""
    # 设置随机种子
    utils.set_seed(args.seed)
    rng = random.PRNGKey(args.seed)
    
    # 加载模型
    model, params, train_args = load_model_and_config(args.checkpoint_dir)
    
    # 确定测试使用的分布（优先使用命令行参数，否则使用训练分布）
    test_x_dist_str = args.test_x_distribution_str if args.test_x_distribution_str else train_args.x_distribution_str
    test_w_dist_str = args.test_w_distribution_str if args.test_w_distribution_str else train_args.w_distribution_str
    
    # 确定任务概率（优先使用测试指定的值，否则使用训练设置）
    # Check if any test_prob is specified
    if any(p is not None for p in [args.test_prob0, args.test_prob1, args.test_prob2, args.test_prob3]):
        # Use test probabilities (default to training values for unspecified)
        test_prob0 = args.test_prob0 if args.test_prob0 is not None else getattr(train_args, 'prob0', 1.0)
        test_prob1 = args.test_prob1 if args.test_prob1 is not None else getattr(train_args, 'prob1', 0.0)
        test_prob2 = args.test_prob2 if args.test_prob2 is not None else getattr(train_args, 'prob2', 0.0)
        test_prob3 = args.test_prob3 if args.test_prob3 is not None else getattr(train_args, 'prob3', 0.0)
        
        task_probs = [test_prob0, test_prob1, test_prob2, test_prob3]
        prob_sum = sum(task_probs)
        
        # Validate probabilities sum to 1.0
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(
                f"测试任务概率之和必须等于1.0，当前为 {prob_sum}。\n"
                f"当前设置: test_prob0={test_prob0}, test_prob1={test_prob1}, "
                f"test_prob2={test_prob2}, test_prob3={test_prob3}\n"
                f"请调整参数使其和为1.0"
            )
        
        logging.info(f"📝 使用测试任务概率: [Task1={test_prob0}, Task2={test_prob1}, Task3={test_prob2}, Task4={test_prob3}]")
    else:
        # Use training probabilities
        train_prob0 = getattr(train_args, 'prob0', 1.0)
        train_prob1 = getattr(train_args, 'prob1', 0.0)
        train_prob2 = getattr(train_args, 'prob2', 0.0)
        train_prob3 = getattr(train_args, 'prob3', 0.0)
        task_probs = [train_prob0, train_prob1, train_prob2, train_prob3]
        logging.info(f"📝 使用训练时的任务概率: [Task1={train_prob0}, Task2={train_prob1}, Task3={train_prob2}, Task4={train_prob3}]")
    
    # Create sampler
    sampler = sampler_lib.Sampler(
        train_args.num_exemplars,
        train_args.x_dim,
        train_args.hidden_size,
        x_distribution_fn=sampler_lib.str_to_distribution_fn(test_x_dist_str),
        w_distribution_fn=sampler_lib.str_to_distribution_fn(test_w_dist_str),
        noise_std=train_args.noise_std,
        task_probs=task_probs,
    )
    
    logging.info(f"🧪 开始测试，生成 {args.n_test_samples} 个测试样本...")
    
    # 生成测试数据
    seqs, coefficients, xs, ys = sampler.sample(n=args.n_test_samples)
    seqs = jnp.array(seqs)
    ys_true = np.array(ys)  # (n_samples, num_exemplars, 1) - 保存真实y值
    
    # 前向传播
    logging.info("📊 计算各位置的y预测loss...")
    errors, (y_errors, y_pred, seq_pred, seq_hiddens) = model.apply(
        {"params": params},
        inputs=seqs,
        train=False,
        return_attention=False
    )
    
    # 转为numpy
    y_errors = np.array(y_errors)
    y_pred = np.array(y_pred)  # (n_samples, num_exemplars, 1)
    
    # 计算平均loss
    avg_y_loss = np.mean(y_errors, axis=0)  # (num_exemplars,)
    
    # 计算各位置y预测的均值和标准差
    y_mean_per_pos = np.mean(y_pred, axis=0)  # (num_exemplars, 1)
    y_std_per_pos = np.std(y_pred, axis=0)    # (num_exemplars, 1)
    
    # 计算各位置y真实值的均值和标准差
    y_true_mean_per_pos = np.mean(ys_true, axis=0)  # (num_exemplars, 1)
    y_true_std_per_pos = np.std(ys_true, axis=0)    # (num_exemplars, 1)
    
    # 输出结果
    logging.info("\n" + "="*70)
    logging.info("测试结果分析")
    logging.info(f"实验: {train_args.exp_folder if hasattr(train_args, 'exp_folder') else args.checkpoint_dir}")
    train_x_dist = getattr(train_args, 'x_distribution_str', 'N/A')
    train_w_dist = getattr(train_args, 'w_distribution_str', 'N/A')
    logging.info(f"训练分布: p(x)={train_x_dist}, p(w)={train_w_dist}")
    logging.info(f"测试分布: p(x)={test_x_dist_str}, p(w)={test_w_dist_str}")
    if test_x_dist_str != train_x_dist or test_w_dist_str != train_w_dist:
        logging.info("⚠️  注意: 测试分布与训练分布不同 (Out-of-Distribution 测试)")
    logging.info("="*70)
    
    # 输出各位置的loss（和训练时格式一致）
    loss_str = "[" + ", ".join([f"{float(avg_y_loss[i]):.4f}" for i in range(len(avg_y_loss))]) + "]"
    logging.info(f"\n各位置Y预测Loss: {loss_str}")
    
    # 统计信息
    logging.info("\n" + "="*70)
    logging.info("统计信息")
    logging.info("="*70)
    logging.info(f"平均Loss: {np.mean(avg_y_loss):.6f}")
    logging.info(f"最小Loss（位置{np.argmin(avg_y_loss)}）: {np.min(avg_y_loss):.6f}")
    logging.info(f"最大Loss（位置{np.argmax(avg_y_loss)}）: {np.max(avg_y_loss):.6f}")
    logging.info(f"第1个位置Loss: {avg_y_loss[0]:.6f}")
    logging.info(f"最后位置Loss: {avg_y_loss[-1]:.6f}")
    logging.info(f"Loss下降: {avg_y_loss[0] - avg_y_loss[-1]:.6f} ({(1 - avg_y_loss[-1]/avg_y_loss[0])*100:.1f}%)")
    
    # 保存结果（包含分布信息）
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    analysis = {
        'avg_y_loss': avg_y_loss,
        'y_errors': y_errors,  # 所有样本的loss
        'y_mean_per_pos': y_mean_per_pos,  # 各位置y预测的均值
        'y_std_per_pos': y_std_per_pos,    # 各位置y预测的标准差
        'y_true_mean_per_pos': y_true_mean_per_pos,  # 各位置y真实值的均值
        'y_true_std_per_pos': y_true_std_per_pos,    # 各位置y真实值的标准差
        'train_x_distribution_str': train_x_dist,
        'train_w_distribution_str': train_w_dist,
        'test_x_distribution_str': test_x_dist_str,
        'test_w_distribution_str': test_w_dist_str,
        'exp_folder': getattr(train_args, 'exp_folder', 'N/A'),
    }
    with open(args.output_file, 'wb') as f:
        pickle.dump(analysis, f)
    
    logging.info(f"\n✅ 分析结果已保存到: {args.output_file}")
    logging.info("="*70)
    
    return analysis


def main(_):
    """主函数"""
    args = utils.flags_to_args()
    test_y_predictor(args)


if __name__ == "__main__":
    app.run(main)

