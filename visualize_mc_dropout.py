#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 MC Dropout 结果
展示预测不确定性随序列位置的变化
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from absl import app, flags

flags.DEFINE_string("input_file", default="experiments/w_predictor/mc_dropout_analysis.pkl", help="MC Dropout分析结果文件（单个文件）")
flags.DEFINE_string("input_files", default="", help="MC Dropout分析结果文件（多个文件，用逗号分隔）")
flags.DEFINE_string("output_dir", default="experiments/w_predictor/mc_dropout_plots", help="输出图片目录")

FLAGS = flags.FLAGS


def plot_mc_dropout_results(analyses_list, output_dir, input_files_list=None):
    """绘制 MC Dropout 结果（支持多个分析文件合并绘制）
    
    Args:
        analyses_list: 分析数据列表，每个元素是一个字典
        output_dir: 输出目录
        input_files_list: 输入文件路径列表（可选，用于检测模型类型）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果只有一个分析，转换为列表
    if not isinstance(analyses_list, list):
        analyses_list = [analyses_list]
    
    # 检测模型类型（从第一个分析的 exp_folder 或输入文件路径）
    first_analysis = analyses_list[0]
    model_type = "W Predictor"
    exp_folder = first_analysis.get('exp_folder', '')
    if input_files_list and len(input_files_list) > 0:
        input_file = input_files_list[0]
        if 'W_pred_loss_W' in input_file:
            model_type = "W Predictor (Loss_W)"
        elif 'W_pred' in input_file:
            model_type = "W Predictor"
        elif 'Y_pred' in input_file:
            model_type = "Y Predictor"
    elif 'W_pred_loss_W' in exp_folder:
        model_type = "W Predictor (Loss_W)"
    elif 'W_pred' in exp_folder:
        model_type = "W Predictor"
    elif 'Y_pred' in exp_folder:
        model_type = "Y Predictor"
    
    # 检测是 W 预测器还是 Y 预测器
    is_w_predictor = 'w_mse_per_pos' in first_analysis
    is_y_predictor = 'avg_y_loss' in first_analysis or 'y_loss_per_pos' in first_analysis
    
    # 从多个分析文件中提取数据
    w_pred_data = []  # W_pred 和 W_pred_loss_W 的数据
    y_pred_data = []  # Y_pred 的数据
    all_y_data = []   # 所有预测器的 Y prediction 数据
    
    for i, analysis in enumerate(analyses_list):
        input_file = input_files_list[i] if input_files_list and i < len(input_files_list) else None
        
        # 判断预测器类型
        pred_type = None
        if input_file:
            if 'W_pred_loss_W' in input_file:
                pred_type = 'W_pred_loss_W'
            elif 'W_pred' in input_file:
                pred_type = 'W_pred'
            elif 'Y_pred' in input_file:
                pred_type = 'Y_pred'
        
        # 提取数据
        if 'w_mse_per_pos' in analysis:
            # W 预测器数据
            w_mse = analysis['w_mse_per_pos']
            w_uncertainty = analysis.get('w_uncertainty_per_pos', None)
            y_loss = analysis.get('avg_y_loss', analysis.get('y_loss_per_pos', None))
            y_uncertainty = analysis.get('y_uncertainty_per_pos', None)
            cosine_sim = analysis.get('cosine_sim_mean', None)
            w_preds = analysis.get('w_preds', None)  # (n_samples, num_exemplars, x_dim)
            w_true = analysis.get('w_true', None)  # (n_samples, x_dim)
            
            w_pred_data.append({
                'type': pred_type or 'W_pred',
                'w_mse': w_mse,
                'w_uncertainty': w_uncertainty,
                'y_loss': y_loss,
                'y_uncertainty': y_uncertainty,
                'cosine_sim': cosine_sim,
                'w_preds': w_preds,
                'w_true': w_true
            })
            
            if y_loss is not None:
                all_y_data.append({
                    'type': pred_type or 'W_pred',
                    'y_loss': y_loss,
                    'y_uncertainty': y_uncertainty
                })
        else:
            # Y 预测器数据
            y_loss = analysis.get('avg_y_loss', analysis.get('y_loss_per_pos', None))
            y_uncertainty = analysis.get('uncertainty_per_pos', None)
            
            y_pred_data.append({
                'type': pred_type or 'Y_pred',
                'y_loss': y_loss,
                'y_uncertainty': y_uncertainty
            })
            
            if y_loss is not None:
                all_y_data.append({
                    'type': pred_type or 'Y_pred',
                    'y_loss': y_loss,
                    'y_uncertainty': y_uncertainty
                })
    
    # 确定 positions（使用第一个有数据的分析）
    positions = None
    for analysis in analyses_list:
        y_loss = analysis.get('avg_y_loss', analysis.get('y_loss_per_pos', None))
        if y_loss is not None:
            positions = np.arange(1, len(y_loss) + 1)
            break
    
    if positions is None:
        print("⚠️  警告：无法找到 Y loss 数据")
        return
    
    # ========================================================================
    # 图1：MSE 和不确定性随位置变化（合并绘制，3x2布局）
    # ========================================================================
    
    # 判断是否有 W prediction 数据
    has_w_pred = len(w_pred_data) > 0
    
    if has_w_pred:
        # 有 W prediction 数据：6个子图（3行2列）
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # 颜色和标签定义
        colors_w = {'W_pred': 'b-', 'W_pred_loss_W': 'orange'}
        labels_w = {'W_pred': 'W Predictor', 'W_pred_loss_W': 'W Predictor (Loss_W)'}
        colors_y = {'Y_pred': 'r-', 'W_pred': 'b--', 'W_pred_loss_W': 'orange'}
        labels_y = {'Y_pred': 'Y Predictor', 'W_pred': 'W Predictor', 'W_pred_loss_W': 'W Predictor (Loss_W)'}
        
        # 第一行：W MSE 和 Y Loss
        # 子图1 (0,0)：W MSE（合并 W_pred 和 W_pred_loss_W）
        for w_data in w_pred_data:
            if w_data['w_mse'] is not None:
                pred_type = w_data['type']
                color = colors_w.get(pred_type, 'b-')
                label = labels_w.get(pred_type, pred_type)
                axes[0, 0].plot(positions, w_data['w_mse'], color, linewidth=2, label=label)
        axes[0, 0].set_xlabel('Position', fontsize=11)
        axes[0, 0].set_ylabel('W MSE', fontsize=11)
        axes[0, 0].set_title('(a) W Prediction Error', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 子图2 (0,1)：Y Loss（合并所有预测器：Y_pred, W_pred, W_pred_loss_W）
        for y_data in all_y_data:
            if y_data['y_loss'] is not None:
                pred_type = y_data['type']
                color = colors_y.get(pred_type, 'r-')
                label = labels_y.get(pred_type, pred_type)
                axes[0, 1].plot(positions, y_data['y_loss'], color, linewidth=2, label=label)
        axes[0, 1].set_xlabel('Position', fontsize=11)
        axes[0, 1].set_ylabel('Y Loss (MSE)', fontsize=11)
        axes[0, 1].set_title('(b) Y Prediction Error', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 第二行：Cosine Similarity 和 W Norm
        # 子图3 (1,0)：Cosine Similarity（合并 W_pred 和 W_pred_loss_W）
        has_cosine = False
        for w_data in w_pred_data:
            if w_data['cosine_sim'] is not None:
                has_cosine = True
                pred_type = w_data['type']
                color = colors_w.get(pred_type, 'b-')
                label = labels_w.get(pred_type, pred_type)
                axes[1, 0].plot(positions, w_data['cosine_sim'], color, linewidth=2, label=label)
        if has_cosine:
            axes[1, 0].set_xlabel('Position', fontsize=11)
            axes[1, 0].set_ylabel('Cosine Similarity', fontsize=11)
            axes[1, 0].set_title('(c) W Cosine Similarity (w_pred · w_true)', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim([-0.1, 1.1])
        else:
            axes[1, 0].text(0.5, 0.5, 'No Cosine Similarity Data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].axis('off')
        
        # 子图4 (1,1)：W Norm（合并 W_pred 和 W_pred_loss_W）
        has_norm = False
        # 先绘制 w_true 的范数（作为参考线，所有位置相同）
        w_true_norm_mean = None
        for w_data in w_pred_data:
            if w_data['w_true'] is not None:
                w_true_norms = np.linalg.norm(w_data['w_true'], axis=1)  # (n_samples,)
                w_true_norm_mean = np.mean(w_true_norms)
                break
        
        if w_true_norm_mean is not None:
            axes[1, 1].axhline(y=w_true_norm_mean, label='Ground Truth', color='green',
                              linestyle=':', linewidth=2.5, alpha=0.8)
            has_norm = True
        
        # 绘制各预测器的 w_pred 范数
        for w_data in w_pred_data:
            if w_data['w_preds'] is not None:
                has_norm = True
                # w_preds: (n_samples, num_exemplars, x_dim)
                norms = np.linalg.norm(w_data['w_preds'], axis=2)  # (n_samples, num_exemplars)
                w_norm_mean = np.mean(norms, axis=0)  # (num_exemplars,)
                pred_type = w_data['type']
                color = colors_w.get(pred_type, 'b-')
                label = labels_w.get(pred_type, pred_type)
                axes[1, 1].plot(positions, w_norm_mean, color, linewidth=2, label=label)
        
        if has_norm:
            axes[1, 1].set_xlabel('Position', fontsize=11)
            axes[1, 1].set_ylabel('||w|| (L2 norm, mean)', fontsize=11)
            axes[1, 1].set_title('(d) W-Predictor Norm Comparison', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No W Norm Data', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].axis('off')
        
        # 第三行：W Uncertainty 和 Y Uncertainty
        # 子图5 (2,0)：W Uncertainty（合并 W_pred 和 W_pred_loss_W）
        has_w_uncertainty = False
        for w_data in w_pred_data:
            if w_data['w_uncertainty'] is not None:
                has_w_uncertainty = True
                pred_type = w_data['type']
                color = colors_w.get(pred_type, 'g-')
                label = labels_w.get(pred_type, pred_type)
                axes[2, 0].plot(positions, w_data['w_uncertainty'], color, linewidth=2, label=label)
        if has_w_uncertainty:
            axes[2, 0].set_xlabel('Position', fontsize=11)
            axes[2, 0].set_ylabel('W Uncertainty', fontsize=11)
            axes[2, 0].set_title('(e) W Prediction Uncertainty (MC Dropout)', fontsize=12, fontweight='bold')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()
        else:
            axes[2, 0].text(0.5, 0.5, 'No W Uncertainty Data', ha='center', va='center', 
                           transform=axes[2, 0].transAxes, fontsize=14)
            axes[2, 0].axis('off')
        
        # 子图6 (2,1)：Y Uncertainty（合并所有预测器）
        has_y_uncertainty = False
        for y_data in all_y_data:
            if y_data['y_uncertainty'] is not None:
                has_y_uncertainty = True
                pred_type = y_data['type']
                color = colors_y.get(pred_type, 'm-')
                label = labels_y.get(pred_type, pred_type)
                axes[2, 1].plot(positions, y_data['y_uncertainty'], color, linewidth=2, label=label)
        if has_y_uncertainty:
            axes[2, 1].set_xlabel('Position', fontsize=11)
            axes[2, 1].set_ylabel('Y Uncertainty', fontsize=11)
            axes[2, 1].set_title('(f) Y Prediction Uncertainty (MC Dropout)', fontsize=12, fontweight='bold')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()
        else:
            axes[2, 1].text(0.5, 0.5, 'No Y Uncertainty Data', ha='center', va='center',
                           transform=axes[2, 1].transAxes, fontsize=14)
            axes[2, 1].axis('off')
        
        n_mc = first_analysis.get('n_mc_samples', 'N/A')
        plt.suptitle(f'MC Dropout Analysis - Combined Predictors (n_mc={n_mc})', fontsize=14, fontweight='bold', y=0.995)
    
    else:
        # 只有 Y 预测器：2个子图（合并所有预测器的 Y prediction）
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Y Loss（合并所有预测器）
        colors_y = {'Y_pred': 'r-', 'W_pred': 'b--', 'W_pred_loss_W': 'orange'}
        labels_y = {'Y_pred': 'Y Predictor', 'W_pred': 'W Predictor', 'W_pred_loss_W': 'W Predictor (Loss_W)'}
        for y_data in all_y_data:
            if y_data['y_loss'] is not None:
                pred_type = y_data['type']
                color = colors_y.get(pred_type, 'r-')
                label = labels_y.get(pred_type, pred_type)
                axes[0].plot(positions, y_data['y_loss'], color, linewidth=2, label=label)
        axes[0].set_xlabel('Position', fontsize=11)
        axes[0].set_ylabel('Y Loss (MSE)', fontsize=11)
        axes[0].set_title('Y Prediction Error', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Y Uncertainty（合并所有预测器）
        has_y_uncertainty = False
        for y_data in all_y_data:
            if y_data['y_uncertainty'] is not None:
                has_y_uncertainty = True
                pred_type = y_data['type']
                color = colors_y.get(pred_type, 'm-')
                label = labels_y.get(pred_type, pred_type)
                axes[1].plot(positions, y_data['y_uncertainty'], color, linewidth=2, label=label)
        if has_y_uncertainty:
            axes[1].set_xlabel('Position', fontsize=11)
            axes[1].set_ylabel('Y Uncertainty', fontsize=11)
            axes[1].set_title('Y Prediction Uncertainty (MC Dropout)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No Uncertainty Data', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=14)
            axes[1].axis('off')
        
        n_mc = first_analysis.get('n_mc_samples', 'N/A')
        plt.suptitle(f'MC Dropout Analysis - Combined Predictors (n_mc={n_mc})', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mc_dropout_overview.png'), dpi=150, bbox_inches='tight')
    print(f"✓ 保存图片: mc_dropout_overview.png")
    plt.close()
    
    # ========================================================================
    # 打印统计总结
    # ========================================================================
    
    # 确定测试样本数（使用第一个分析）
    n_test_samples = first_analysis.get('n_test_samples', 0)
    if n_test_samples == 0:
        # 尝试从其他字段推断
        w_preds = first_analysis.get('w_preds', None)
        ys_true = first_analysis.get('ys_true', None)
        if w_preds is not None:
            n_test_samples = len(w_preds)
        elif ys_true is not None:
            n_test_samples = len(ys_true)
    
    print("\n" + "="*70)
    print("MC Dropout 统计总结:")
    print("="*70)
    print(f"MC 采样次数: {first_analysis.get('n_mc_samples', 'N/A')}")
    print(f"测试样本数: {n_test_samples}")
    print(f"合并的预测器数量: {len(analyses_list)}")
    
    # 获取 dropout rate
    dropout_rate = first_analysis.get('dropout_rate', None)
    attention_dropout_rate = first_analysis.get('attention_dropout_rate', None)
    if dropout_rate is None:
        # 尝试从训练配置中获取
        dropout_rate = first_analysis.get('train_dropout_rate', 'N/A')
        attention_dropout_rate = first_analysis.get('train_attention_dropout_rate', 'N/A')
    
    print(f"Dropout rate: {dropout_rate}")
    print(f"Attention dropout rate: {attention_dropout_rate}")
    print("-"*70)
    
    # 打印各预测器的最后位置统计
    for w_data in w_pred_data:
        if w_data['w_mse'] is not None:
            print(f"{w_data['type']} - 最后位置的 W MSE: {w_data['w_mse'][-1]:.6f}")
            if w_data['w_uncertainty'] is not None:
                print(f"{w_data['type']} - 最后位置的 W 不确定性: {w_data['w_uncertainty'][-1]:.6f}")
    
    for y_data in all_y_data:
        if y_data['y_loss'] is not None:
            print(f"{y_data['type']} - 最后位置的 Y Loss (MSE): {y_data['y_loss'][-1]:.6f}")
        if y_data['y_uncertainty'] is not None:
            print(f"{y_data['type']} - 最后位置的 Y 不确定性: {y_data['y_uncertainty'][-1]:.6f}")
    
    print("-"*70)
    
    # 打印不确定性减少统计
    for w_data in w_pred_data:
        if w_data['w_uncertainty'] is not None and len(w_data['w_uncertainty']) > 1:
            reduction = (1 - w_data['w_uncertainty'][-1] / w_data['w_uncertainty'][0]) * 100
            print(f"{w_data['type']} - W 不确定性减少: {w_data['w_uncertainty'][0]:.6f} → {w_data['w_uncertainty'][-1]:.6f} ({reduction:.1f}%)")
    
    for y_data in all_y_data:
        if y_data['y_uncertainty'] is not None and len(y_data['y_uncertainty']) > 1:
            reduction = (1 - y_data['y_uncertainty'][-1] / y_data['y_uncertainty'][0]) * 100
            print(f"{y_data['type']} - Y 不确定性减少: {y_data['y_uncertainty'][0]:.6f} → {y_data['y_uncertainty'][-1]:.6f} ({reduction:.1f}%)")
    
    print("="*70)


def main(_):
    """主函数"""
    # 确定输入文件列表
    input_files = []
    if FLAGS.input_files:
        # 使用多个文件（逗号分隔）
        input_files = [f.strip() for f in FLAGS.input_files.split(',')]
    elif FLAGS.input_file:
        # 使用单个文件
        input_files = [FLAGS.input_file]
    else:
        print("❌ 错误：必须指定 --input_file 或 --input_files")
        return
    
    # 加载所有分析文件
    analyses_list = []
    input_files_list = []
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"⚠️  警告：找不到输入文件: {input_file}，跳过")
            continue
        
        with open(input_file, 'rb') as f:
            analysis = pickle.load(f)
        
        # 检查是否是 MC Dropout 结果
        if not analysis.get('use_mc_dropout', False):
            print(f"⚠️  警告：{input_file} 不是 MC Dropout 测试结果，跳过")
            continue
        
        analyses_list.append(analysis)
        input_files_list.append(input_file)
        print(f"✓ 加载数据: {input_file}")
    
    if len(analyses_list) == 0:
        print("❌ 错误：没有有效的 MC Dropout 分析文件")
        return
    
    plot_mc_dropout_results(analyses_list, FLAGS.output_dir, input_files_list=input_files_list)
    print(f"\n✅ 所有图片已保存到: {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)
