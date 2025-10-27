#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试W预测器：分析各个位置的w预测值和loss
验证ICL的学习过程：从先验分布到真实w
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
from incontext import predictor_flax_w

flags.DEFINE_string("checkpoint_dir", default="experiments/w_predictor/ckpt", help="检查点目录")
flags.DEFINE_integer("n_test_samples", default=500, help="测试样本数")
flags.DEFINE_integer("seed", default=42, help="测试随机种子")
flags.DEFINE_string("output_file", default="experiments/w_predictor/w_analysis.pkl", help="输出文件")
flags.DEFINE_string("test_x_distribution_str", default=None, help="测试时x分布（None则使用训练分布）")
flags.DEFINE_string("test_w_distribution_str", default=None, help="测试时w分布（None则使用训练分布）")
flags.DEFINE_float("test_prob0", default=None, help="测试时任务1概率（None则使用训练设置）")
flags.DEFINE_float("test_prob1", default=None, help="测试时任务2概率（None则使用训练设置）")
flags.DEFINE_float("test_prob2", default=None, help="测试时任务3概率（None则使用训练设置）")
flags.DEFINE_float("test_prob3", default=None, help="测试时任务4概率（None则使用训练设置）")
flags.DEFINE_float("test_task_mix_alpha", default=None, help="[已弃用] 测试时任务混合比例（None则使用训练设置）")

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
    model = predictor_flax_w.CausalLM_W(config=config, x_dim=args.x_dim)
    
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


def extract_w_predictions(model, params, seqs, num_exemplars, task_ids=None):
    """
    提取各个位置的w预测值
    
    Args:
        model: 模型
        params: 模型参数
        seqs: 输入序列 (batch, seq_len, x_dim+1)
        num_exemplars: 样本对数量
        task_ids: 任务类型 (batch,), 可选
    
    Returns:
        w_preds: (batch, num_exemplars, x_dim) - 各位置预测的w
        y_preds: (batch, num_exemplars, 1) - 各位置预测的y
        y_errors: (batch, num_exemplars) - 各位置的loss
    """
    # 前向传播
    errors, (y_errors, y_pred, seq_pred, seq_hiddens) = model.apply(
        {"params": params},
        inputs=seqs,
        task_ids=task_ids,
        train=False,
        return_attention=False
    )
    
    # seq_pred shape: (batch, seq_len-1, x_dim)
    # 提取y位置的w预测 (偶数位置：0, 2, 4, ...)
    w_preds = seq_pred[:, jnp.arange(0, seq_pred.shape[1], 2), :]
    # w_preds shape: (batch, num_exemplars, x_dim)
    
    return w_preds, y_pred, y_errors


def analyze_w_predictions(w_preds, w_true, y_errors, x_dim):
    """
    分析w预测值的统计特性
    
    Args:
        w_preds: (n_samples, num_exemplars, x_dim) - 预测的w
        w_true: (n_samples, x_dim) - 真实的w
        y_errors: (n_samples, num_exemplars) - 各位置loss
        x_dim: x的维度
    
    Returns:
        分析结果字典
    """
    n_samples, num_exemplars, _ = w_preds.shape
    
    # 计算各位置w预测的均值向量和协方差矩阵
    w_mean_per_pos = np.mean(w_preds, axis=0)  # (num_exemplars, x_dim)
    w_cov_per_pos = []  # 列表，每个元素是 (x_dim, x_dim) 的协方差矩阵
    for pos in range(num_exemplars):
        # w_preds[:, pos, :] 是 (n_samples, x_dim)
        cov_matrix = np.cov(w_preds[:, pos, :], rowvar=False)  # (x_dim, x_dim)
        w_cov_per_pos.append(cov_matrix)
    w_cov_per_pos = np.array(w_cov_per_pos)  # (num_exemplars, x_dim, x_dim)
    
    # 计算各位置w预测与真实w的MSE
    w_true_expanded = w_true[:, None, :]  # (n_samples, 1, x_dim)
    w_mse_per_pos = np.mean((w_preds - w_true_expanded)**2, axis=(0, 2))  # (num_exemplars,)
    
    # 计算各位置w预测与真实w的余弦相似度
    w_norm = np.linalg.norm(w_preds, axis=2, keepdims=True)  # (n_samples, num_exemplars, 1)
    w_true_norm = np.linalg.norm(w_true, axis=1, keepdims=True)  # (n_samples, 1)
    w_normalized = w_preds / (w_norm + 1e-8)
    w_true_normalized = w_true / (w_true_norm + 1e-8)
    
    cosine_sim = np.sum(w_normalized * w_true_normalized[:, None, :], axis=2)  # (n_samples, num_exemplars)
    cosine_sim_mean = np.mean(cosine_sim, axis=0)  # (num_exemplars,)
    
    # 平均y loss
    avg_y_loss = np.mean(y_errors, axis=0)  # (num_exemplars,)
    
    return {
        'w_mean_per_pos': w_mean_per_pos,             # 各位置w的均值向量 (num_exemplars, x_dim)
        'w_cov_per_pos': w_cov_per_pos,               # 各位置w的协方差矩阵 (num_exemplars, x_dim, x_dim)
        'w_mse_per_pos': w_mse_per_pos,               # 各位置w预测的MSE
        'cosine_sim_mean': cosine_sim_mean,           # 各位置w与真实w的余弦相似度
        'avg_y_loss': avg_y_loss,                     # 各位置y的平均loss
        'w_preds': w_preds,                           # 所有w预测值（用于进一步分析）
        'w_true': w_true,                             # 真实w值
    }


def test_w_predictor(args):
    """测试W预测器"""
    # 设置随机种子
    utils.set_seed(args.seed)
    rng = random.PRNGKey(args.seed)
    
    # 加载模型
    model, params, train_args = load_model_and_config(args.checkpoint_dir)
    
    # 确定测试使用的分布（优先使用命令行参数，否则使用训练分布）
    test_x_dist_str = args.test_x_distribution_str if args.test_x_distribution_str else train_args.x_distribution_str
    test_w_dist_str = args.test_w_distribution_str if args.test_w_distribution_str else train_args.w_distribution_str
    
    # 确定任务概率（优先使用测试指定的值，否则使用训练设置）
    if any(p is not None for p in [args.test_prob0, args.test_prob1, args.test_prob2, args.test_prob3]):
        # 使用测试概率（未指定的默认为训练值）
        test_prob0 = args.test_prob0 if args.test_prob0 is not None else getattr(train_args, 'prob0', 1.0)
        test_prob1 = args.test_prob1 if args.test_prob1 is not None else getattr(train_args, 'prob1', 0.0)
        test_prob2 = args.test_prob2 if args.test_prob2 is not None else getattr(train_args, 'prob2', 0.0)
        test_prob3 = args.test_prob3 if args.test_prob3 is not None else getattr(train_args, 'prob3', 0.0)
        
        task_probs = [test_prob0, test_prob1, test_prob2, test_prob3]
        prob_sum = sum(task_probs)
        
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(
                f"测试任务概率之和必须等于1.0，当前为 {prob_sum}。\n"
                f"当前设置: test_prob0={test_prob0}, test_prob1={test_prob1}, "
                f"test_prob2={test_prob2}, test_prob3={test_prob3}\n"
                f"请调整参数使其和为1.0"
            )
        
        logging.info(f"📝 使用测试任务概率: [Task1={test_prob0}, Task2={test_prob1}, Task3={test_prob2}, Task4={test_prob3}]")
    else:
        # 使用训练概率
        train_prob0 = getattr(train_args, 'prob0', 1.0)
        train_prob1 = getattr(train_args, 'prob1', 0.0)
        train_prob2 = getattr(train_args, 'prob2', 0.0)
        train_prob3 = getattr(train_args, 'prob3', 0.0)
        task_probs = [train_prob0, train_prob1, train_prob2, train_prob3]
        logging.info(f"📝 使用训练时的任务概率: [Task1={train_prob0}, Task2={train_prob1}, Task3={train_prob2}, Task4={train_prob3}]")
    
    # 创建数据采样器
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
    # 获取任务类型
    task_ids = sampler.get_last_task_ids()
    
    seqs = jnp.array(seqs)
    coefficients = jnp.array(coefficients)  # 真实的w
    xs_true = np.array(xs)  # (n_samples, num_exemplars, x_dim) - 保存真实x值
    ys_true = np.array(ys)  # (n_samples, num_exemplars, 1) - 保存真实y值
    task_ids_np = np.array(task_ids) if task_ids is not None else None  # 保存task_ids
    task_ids = jnp.array(task_ids, dtype=jnp.int32) if task_ids is not None else None
    
    # 提取w预测
    logging.info("📊 提取各位置的w预测值...")
    w_preds, y_preds, y_errors = extract_w_predictions(
        model, params, seqs, train_args.num_exemplars, task_ids=task_ids
    )
    
    # 转为numpy
    w_preds = np.array(w_preds)
    y_preds = np.array(y_preds)  # (n_samples, num_exemplars, 1) - W预测器预测的y
    y_errors = np.array(y_errors)
    coefficients = np.array(coefficients)
    
    # 计算W预测器预测的y的均值和标准差（用于与Y预测器对比）
    y_pred_mean_per_pos = np.mean(y_preds, axis=0)  # (num_exemplars, 1)
    y_pred_std_per_pos = np.std(y_preds, axis=0)    # (num_exemplars, 1)
    
    # 计算y真实值的均值和标准差
    y_true_mean_per_pos = np.mean(ys_true, axis=0)  # (num_exemplars, 1)
    y_true_std_per_pos = np.std(ys_true, axis=0)    # (num_exemplars, 1)
    
    # 分析结果
    logging.info("🔍 分析w预测值的统计特性...")
    analysis = analyze_w_predictions(w_preds, coefficients, y_errors, train_args.x_dim)
    
    # 添加y预测值和真实值统计（用于与Y预测器对比）
    analysis['y_pred_mean_per_pos'] = y_pred_mean_per_pos
    analysis['y_pred_std_per_pos'] = y_pred_std_per_pos
    analysis['y_true_mean_per_pos'] = y_true_mean_per_pos
    analysis['y_true_std_per_pos'] = y_true_std_per_pos
    
    # 输出结果
    logging.info("\n" + "="*70)
    logging.info("测试结果分析")
    logging.info("="*70)
    
    # 输出各位置的loss（和训练时格式一致）
    avg_y_loss = analysis['avg_y_loss']
    loss_str = "[" + ", ".join([f"{float(avg_y_loss[i]):.4f}" for i in range(len(avg_y_loss))]) + "]"
    logging.info(f"\n各位置Y预测Loss: {loss_str}")
    
    # 输出各位置w预测的MSE
    w_mse = analysis['w_mse_per_pos']
    mse_str = "[" + ", ".join([f"{float(w_mse[i]):.4f}" for i in range(len(w_mse))]) + "]"
    logging.info(f"\n各位置W预测MSE:  {mse_str}")
    
    # 输出各位置w与真实w的余弦相似度
    cosine_sim = analysis['cosine_sim_mean']
    cosine_str = "[" + ", ".join([f"{float(cosine_sim[i]):.4f}" for i in range(len(cosine_sim))]) + "]"
    logging.info(f"\n各位置W余弦相似度: {cosine_str}")
    
    # 输出测试分布信息
    logging.info("\n" + "="*70)
    train_x_dist = getattr(train_args, 'x_distribution_str', 'N/A')
    train_w_dist = getattr(train_args, 'w_distribution_str', 'N/A')
    logging.info(f"训练分布: p(x)={train_x_dist}, p(w)={train_w_dist}")
    logging.info(f"测试分布: p(x)={test_x_dist_str}, p(w)={test_w_dist_str}")
    if test_x_dist_str != train_x_dist or test_w_dist_str != train_w_dist:
        logging.info("⚠️  注意: 测试分布与训练分布不同 (Out-of-Distribution 测试)")
    logging.info("="*70)
    
    # 保存结果（包含分布信息和原始数据）
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    analysis['train_x_distribution_str'] = train_x_dist
    analysis['train_w_distribution_str'] = train_w_dist
    analysis['test_x_distribution_str'] = test_x_dist_str
    analysis['test_w_distribution_str'] = test_w_dist_str
    analysis['exp_folder'] = getattr(train_args, 'exp_folder', 'N/A')
    # 保存原始数据用于详细分析
    analysis['xs_true'] = xs_true  # (n_samples, num_exemplars, x_dim)
    analysis['ys_true'] = ys_true  # (n_samples, num_exemplars, 1)
    analysis['task_ids'] = task_ids_np  # (n_samples,)
    with open(args.output_file, 'wb') as f:
        pickle.dump(analysis, f)
    
    logging.info(f"\n✅ 分析结果已保存到: {args.output_file}")
    logging.info("="*70)
    
    return analysis


def main(_):
    """主函数"""
    args = utils.flags_to_args()
    test_w_predictor(args)


if __name__ == "__main__":
    app.run(main)
