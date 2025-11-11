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
flags.DEFINE_integer("test_num_exemplars", default=None, help="测试时序列长度/示例对数量（None则使用训练设置）")
flags.DEFINE_float("test_task_mix_alpha", default=None, help="[已弃用] 测试时任务混合比例（None则使用训练设置）")
flags.DEFINE_float("test_task3_prob", default=None, help="[已弃用] 测试时任务3概率（None则使用训练设置）")
flags.DEFINE_bool("use_mc_dropout", default=False, help="是否使用 MC Dropout 进行不确定性估计")
flags.DEFINE_integer("n_mc_samples", default=50, help="MC Dropout 采样次数（仅在 use_mc_dropout=True 时有效）")

FLAGS = flags.FLAGS


def load_model_and_config(checkpoint_dir, test_max_len=None):
    """加载模型和配置
    
    Args:
        checkpoint_dir: 检查点目录
        test_max_len: 测试时的最大序列长度（如果指定，会扩展位置编码以支持更长序列）
    """
    # 转换为绝对路径
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(checkpoint_dir), "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    args = utils.dict_to_args(config_dict)
    
    # 创建Transformer配置
    # 处理可变长度训练的情况
    if args.num_exemplars is not None:
        train_max_len = (args.num_exemplars + 1) * 2
    else:
        # 可变长度模式：使用 max_num_exemplars
        train_max_len = (args.max_num_exemplars + 1) * 2
    
    # 如果测试序列长度超过训练时的max_len，使用测试时的max_len
    actual_max_len = test_max_len if (test_max_len is not None and test_max_len > train_max_len) else train_max_len
    
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
        max_len=actual_max_len,  # 使用实际需要的最大长度
        inner_dim=None,
        activation_fn=transformer_lib_flax.nn_activation_parser(args.activation_fn),
        dropout_rate=getattr(args, 'dropout_rate', 0.0),  # ⭐ 使用训练时的dropout_rate
        attention_dropout_rate=getattr(args, 'attention_dropout_rate', 0.0),  # ⭐ 使用训练时的attention_dropout_rate
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
    
    # 如果测试序列长度超过训练时的max_len，需要扩展位置编码
    if test_max_len is not None and test_max_len > train_max_len:
        # 获取原始位置编码（找到正确的键名）
        # 参数结构可能是 params['Transformer_0']['PositionEmbeddings_0']['pos_embedding']
        if 'Transformer_0' in params:
            original_pos_emb = params['Transformer_0']['PositionEmbeddings_0']['pos_embedding']
            transformer_key = 'Transformer_0'
        elif 'transformer' in params:
            original_pos_emb = params['transformer']['PositionEmbeddings_0']['pos_embedding']
            transformer_key = 'transformer'
        else:
            # 尝试找到包含PositionEmbeddings的键
            transformer_key = None
            for key in params.keys():
                if isinstance(params[key], dict) and 'PositionEmbeddings_0' in params[key]:
                    original_pos_emb = params[key]['PositionEmbeddings_0']['pos_embedding']
                    transformer_key = key
                    break
            if transformer_key is None:
                raise KeyError(f"Cannot find position embeddings in params. Available keys: {list(params.keys())}")
        
        # 检查位置编码初始化方式
        posemb_init_fn = transformer_lib_flax.nn_init_parser(args.posemb_init)
        
        # 生成扩展的位置编码
        hidden_size = config.hidden_size
        extended_shape = (1, test_max_len, hidden_size)
        
        # 如果是正弦位置编码，可以动态生成
        if 'sinusoidal' in str(args.posemb_init).lower() or 'sin' in str(args.posemb_init).lower():
            # 使用正弦位置编码生成器
            sinusoidal_init = transformer_lib_flax.sinusoidal_init(max_len=test_max_len)
            rng_pos = random.PRNGKey(0)
            extended_pos_emb = sinusoidal_init(rng_pos, extended_shape, jnp.float32)
        else:
            # 对于可学习的位置编码，扩展原始编码并初始化新部分
            # 先复制原始编码
            extended_pos_emb = jnp.zeros(extended_shape, dtype=jnp.float32)
            extended_pos_emb = extended_pos_emb.at[:, :train_max_len, :].set(original_pos_emb)
            
            # 对于超出部分，使用原始初始化器生成
            new_pos_emb = posemb_init_fn(random.PRNGKey(1), (1, test_max_len - train_max_len, hidden_size), jnp.float32)
            extended_pos_emb = extended_pos_emb.at[:, train_max_len:, :].set(new_pos_emb)
        
        # 更新参数
        params[transformer_key]['PositionEmbeddings_0']['pos_embedding'] = extended_pos_emb
        
        logging.info(f"🔧 扩展位置编码: {train_max_len} -> {test_max_len}")
    
    logging.info(f"✅ 模型加载成功: {checkpoint_dir}")
    logging.info(f"   配置: L={config.num_layers}, H={config.hidden_size}, M={config.num_heads}")
    logging.info(f"   数据: {args.num_exemplars}个样本对, x_dim={args.x_dim}")
    if test_max_len is not None and test_max_len > train_max_len:
        logging.info(f"   最大序列长度: {train_max_len} (训练) -> {test_max_len} (测试)")
    
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
    
    # 先加载配置以获取训练时的序列长度
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    config_path = os.path.join(os.path.dirname(checkpoint_dir), "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    train_args = utils.dict_to_args(config_dict)
    
    # 确定测试序列长度（优先使用命令行参数，否则使用训练设置）
    test_num_exemplars = args.test_num_exemplars if args.test_num_exemplars is not None else train_args.num_exemplars
    test_max_len = (test_num_exemplars + 1) * 2
    
    if test_num_exemplars != train_args.num_exemplars:
        logging.info(f"📏 使用测试序列长度: {test_num_exemplars} (训练时: {train_args.num_exemplars})")
    else:
        logging.info(f"📏 使用训练序列长度: {test_num_exemplars}")
    
    # 加载模型（如果需要扩展位置编码，会在这里处理）
    model, params, train_args = load_model_and_config(args.checkpoint_dir, test_max_len=test_max_len)
    
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
        test_num_exemplars,  # 使用测试序列长度
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
    ys_true = np.array(ys)  # (n_samples, test_num_exemplars, 1) - 保存真实y值
    
    # 前向传播
    if args.use_mc_dropout:
        # 检查训练时是否启用了 dropout
        train_dropout = getattr(train_args, 'dropout_rate', 0.0)
        train_attention_dropout = getattr(train_args, 'attention_dropout_rate', 0.0)
        if train_dropout == 0.0:
            logging.warning("⚠️  警告: 训练时 dropout_rate=0.0，MC Dropout 可能无效！")
            logging.warning("     建议重新训练模型并设置 dropout_rate > 0 (推荐 0.1)")
        else:
            logging.info(f"📊 MC Dropout 配置:")
            logging.info(f"   训练时 dropout_rate: {train_dropout}")
            logging.info(f"   训练时 attention_dropout_rate: {train_attention_dropout}")
            logging.info(f"   测试时使用相同的 dropout rate (MC Dropout 要求)")
        
        logging.info(f"📊 使用 MC Dropout 计算各位置的y预测loss (采样 {args.n_mc_samples} 次)...")
        rng, dropout_rng = random.split(rng)
        
        # MC Dropout: 多次采样
        y_preds_all = []
        y_errors_all = []
        
        for i in range(args.n_mc_samples):
            if (i + 1) % 10 == 0 or i == 0:
                logging.info(f"  采样进度: {i+1}/{args.n_mc_samples}")
            dropout_rng, sub_rng = random.split(dropout_rng)
            errors, (y_errors, y_pred, seq_pred, seq_hiddens) = model.apply(
                {"params": params},
                inputs=seqs,
                train=True,  # ⭐ MC Dropout: 测试时也启用 dropout
                rngs={"dropout": sub_rng},
                return_attention=False
            )
            y_preds_all.append(y_pred)
            y_errors_all.append(y_errors)
        
        # 堆叠所有采样结果
        y_preds_all = jnp.stack(y_preds_all, axis=0)  # (n_mc, n_samples, test_num_exemplars, 1)
        y_errors_all = jnp.stack(y_errors_all, axis=0)  # (n_mc, n_samples, test_num_exemplars)
        
        # 计算均值和标准差（跨MC采样）
        y_pred_mean = jnp.mean(y_preds_all, axis=0)  # (n_samples, test_num_exemplars, 1)
        y_pred_std = jnp.std(y_preds_all, axis=0)    # (n_samples, test_num_exemplars, 1)
        y_errors_mean = jnp.mean(y_errors_all, axis=0)  # (n_samples, test_num_exemplars)
        
        # 转为numpy
        y_errors = np.array(y_errors_mean)
        y_pred = np.array(y_pred_mean)  # (n_samples, test_num_exemplars, 1)
        y_pred_std_mc = np.array(y_pred_std)  # MC Dropout 不确定性
        logging.info(f"✓ MC Dropout 采样完成")
    else:
        logging.info("📊 计算各位置的y预测loss...")
        errors, (y_errors, y_pred, seq_pred, seq_hiddens) = model.apply(
            {"params": params},
            inputs=seqs,
            train=False,
            return_attention=False
        )
        
        # 转为numpy
        y_errors = np.array(y_errors)
        y_pred = np.array(y_pred)  # (n_samples, test_num_exemplars, 1)
        y_pred_std_mc = None  # 标准测试模式没有不确定性估计
    
    # 计算平均loss
    avg_y_loss = np.mean(y_errors, axis=0)  # (test_num_exemplars,)
    
    # 计算各位置y预测的均值和标准差
    y_mean_per_pos = np.mean(y_pred, axis=0)  # (test_num_exemplars, 1)
    y_std_per_pos = np.std(y_pred, axis=0) if y_pred_std_mc is None else np.mean(y_pred_std_mc, axis=0)  # (test_num_exemplars, 1)
    
    # 计算各位置y真实值的均值和标准差
    y_true_mean_per_pos = np.mean(ys_true, axis=0)  # (test_num_exemplars, 1)
    y_true_std_per_pos = np.std(ys_true, axis=0)    # (test_num_exemplars, 1)
    
    # 输出结果
    logging.info("\n" + "="*70)
    logging.info("测试结果分析")
    if args.use_mc_dropout:
        logging.info(f"模式: MC Dropout (采样 {args.n_mc_samples} 次)")
    else:
        logging.info("模式: 标准测试")
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
    
    if args.use_mc_dropout and y_pred_std_mc is not None:
        # MC Dropout 不确定性统计
        uncertainty_per_pos = np.mean(y_pred_std_mc, axis=0).squeeze()  # (test_num_exemplars,)
        logging.info("\n" + "-"*70)
        logging.info("MC Dropout 不确定性分析")
        logging.info("-"*70)
        logging.info(f"平均不确定性: {np.mean(uncertainty_per_pos):.6f}")
        logging.info(f"第1个位置不确定性: {uncertainty_per_pos[0]:.6f}")
        logging.info(f"最后位置不确定性: {uncertainty_per_pos[-1]:.6f}")
        logging.info(f"不确定性下降: {uncertainty_per_pos[0] - uncertainty_per_pos[-1]:.6f} ({(1 - uncertainty_per_pos[-1]/uncertainty_per_pos[0])*100:.1f}%)")
        uncertainty_str = "[" + ", ".join([f"{float(uncertainty_per_pos[i]):.4f}" for i in range(len(uncertainty_per_pos))]) + "]"
        logging.info(f"各位置不确定性: {uncertainty_str}")
    
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
    
    if args.use_mc_dropout:
        analysis['use_mc_dropout'] = True
        analysis['n_mc_samples'] = args.n_mc_samples
        if y_pred_std_mc is not None:
            analysis['y_pred_std_mc'] = y_pred_std_mc  # MC Dropout 不确定性
            analysis['uncertainty_per_pos'] = np.mean(y_pred_std_mc, axis=0).squeeze()
    else:
        analysis['use_mc_dropout'] = False
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

