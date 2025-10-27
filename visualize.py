#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化测试结果
自动扫描 test_results/ 目录，为每个训练配置生成可视化图表
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


def extract_prob_from_dirname(dirname):
    """从目录名提取概率配置
    
    Args:
        dirname: 目录名，如 'prob_0.25_0.25_0.25_0.25'
    
    Returns:
        tuple: (p0, p1, p2, p3) or None
    """
    match = re.match(r'prob_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)', dirname)
    if match:
        return tuple(float(x) for x in match.groups())
    return None


def get_task_label(task_name):
    """获取任务标签"""
    task_labels = {
        'T1': 'Task 1: y=w^T·x',
        'T2': 'Task 2: y=w^T·sort(x)',
        'T3': 'Task 3: y=(d/√2)·w^T·softmax(x)',
        'T4': 'Task 4: y=||x-w||²'
    }
    return task_labels.get(task_name, task_name)


def get_training_label(probs):
    """根据训练概率生成标签"""
    p0, p1, p2, p3 = probs
    
    # 判断是否为纯任务
    if p0 > 0.99:
        return 'Trained on Pure T1'
    elif p1 > 0.99:
        return 'Trained on Pure T2'
    elif p2 > 0.99:
        return 'Trained on Pure T3'
    elif p3 > 0.99:
        return 'Trained on Pure T4'
    else:
        # 混合任务
        return f'Trained on Mix (T1:{p0:.2f}, T2:{p1:.2f}, T3:{p2:.2f}, T4:{p3:.2f})'


def load_test_results(test_dir):
    """加载测试结果
    
    Args:
        test_dir: 测试结果目录，如 'test_results/Y_pred/prob_1.0_0.0_0.0_0.0'
    
    Returns:
        dict: {task_name: data_dict}
    """
    results = {}
    test_path = Path(test_dir)
    
    if not test_path.exists():
        return results
    
    # 扫描所有 test_on_*.pkl 文件
    for pkl_file in test_path.glob('test_on_*.pkl'):
        task_name = pkl_file.stem.replace('test_on_', '')  # 提取 T1, T2, T3, T4
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                results[task_name] = data
        except Exception as e:
            print(f"Warning: Failed to load {pkl_file}: {e}")
    
    return results


def visualize_y_pred(test_dir, output_dir):
    """可视化 Y 预测器的测试结果
    
    Args:
        test_dir: Y预测器测试结果目录
        output_dir: 输出图片目录
    """
    results = load_test_results(test_dir)
    
    if not results:
        print(f"No results found in {test_dir}")
        return
    
    # 提取训练配置
    dirname = Path(test_dir).name
    train_probs = extract_prob_from_dirname(dirname)
    if train_probs is None:
        print(f"Cannot parse training probs from {dirname}")
        return
    
    train_label = get_training_label(train_probs)
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 颜色映射
    colors = {'T1': '#1f77b4', 'T2': '#ff7f0e', 'T3': '#2ca02c', 'T4': '#d62728'}
    markers = {'T1': 'o', 'T2': 's', 'T3': '^', 'T4': 'D'}
    
    # 画每个测试任务的曲线
    for task_name in sorted(results.keys()):
        data = results[task_name]
        avg_y_loss = data['avg_y_loss']
        positions = np.arange(1, len(avg_y_loss) + 1)
        
        ax.plot(positions, avg_y_loss,
                label=get_task_label(task_name),
                color=colors.get(task_name, 'gray'),
                marker=markers.get(task_name, 'o'),
                markersize=4,
                linewidth=2,
                alpha=0.8)
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Y Prediction Loss (MSE)', fontsize=12)
    ax.set_title(f'Y-Predictor: Loss vs Position\n{train_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dirname}_y_pred_loss.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Y-Pred visualization saved: {output_file}")


def visualize_w_pred(test_dir, output_dir):
    """可视化 W 预测器的测试结果
    
    Args:
        test_dir: W预测器测试结果目录
        output_dir: 输出图片目录
    """
    results = load_test_results(test_dir)
    
    if not results:
        print(f"No results found in {test_dir}")
        return
    
    # 提取训练配置
    dirname = Path(test_dir).name
    train_probs = extract_prob_from_dirname(dirname)
    if train_probs is None:
        print(f"Cannot parse training probs from {dirname}")
        return
    
    train_label = get_training_label(train_probs)
    
    # 尝试加载对应的Y-predictor结果（用于第4个子图）
    y_test_dir = test_dir.replace('/W_pred/', '/Y_pred/')
    y_results = load_test_results(y_test_dir) if Path(y_test_dir).exists() else {}
    
    # 创建2x2四宫格布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 颜色映射
    colors = {'T1': '#1f77b4', 'T2': '#ff7f0e', 'T3': '#2ca02c', 'T4': '#d62728'}
    markers = {'T1': 'o', 'T2': 's', 'T3': '^', 'T4': 'D'}
    
    # 子图1 (左上): Y Loss
    ax1 = axes[0, 0]
    for task_name in sorted(results.keys()):
        data = results[task_name]
        avg_y_loss = data['avg_y_loss']
        positions = np.arange(1, len(avg_y_loss) + 1)
        
        ax1.plot(positions, avg_y_loss,
                 label=get_task_label(task_name),
                 color=colors.get(task_name, 'gray'),
                 marker=markers.get(task_name, 'o'),
                 markersize=4,
                 linewidth=2,
                 alpha=0.8)
    
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('Y Loss (computed from w)', fontsize=11)
    ax1.set_title('(a) Y Prediction Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 子图2 (右上): W MSE
    ax2 = axes[0, 1]
    for task_name in sorted(results.keys()):
        data = results[task_name]
        w_mse = data['w_mse_per_pos']
        positions = np.arange(1, len(w_mse) + 1)
        
        ax2.plot(positions, w_mse,
                 label=get_task_label(task_name),
                 color=colors.get(task_name, 'gray'),
                 marker=markers.get(task_name, 'o'),
                 markersize=4,
                 linewidth=2,
                 alpha=0.8)
    
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('W MSE (||w_pred - w_true||²)', fontsize=11)
    ax2.set_title('(b) W Prediction MSE', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    # 子图3 (左下): Cosine Similarity
    ax3 = axes[1, 0]
    for task_name in sorted(results.keys()):
        data = results[task_name]
        cosine_sim = data['cosine_sim_mean']
        positions = np.arange(1, len(cosine_sim) + 1)
        
        ax3.plot(positions, cosine_sim,
                 label=get_task_label(task_name),
                 color=colors.get(task_name, 'gray'),
                 marker=markers.get(task_name, 'o'),
                 markersize=4,
                 linewidth=2,
                 alpha=0.8)
    
    ax3.set_xlabel('Position', fontsize=11)
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title('(c) W Cosine Similarity (w_pred · w_true)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim([-0.1, 1.1])
    
    # 子图4 (右下): Y-Predictor vs W-Predictor 的预测值对比
    ax4 = axes[1, 1]
    if y_results:
        # Task颜色: T1红, T2橙, T3绿, T4蓝
        # W-Pred颜色: T1蓝, T2绿, T3橙, T4红（交叉对比）
        y_colors = {'T1': '#d62728', 'T2': '#ff7f0e', 'T3': '#2ca02c', 'T4': '#1f77b4'}  # 红橙绿蓝
        w_colors = {'T1': '#1f77b4', 'T2': '#2ca02c', 'T3': '#ff7f0e', 'T4': '#d62728'}  # 蓝绿橙红
        
        # Y-Pred和W-Pred用不同marker
        y_marker = 'o'  # 圆圈
        w_marker = 's'  # 方块
        
        for task_name in sorted(results.keys()):
            if task_name not in y_results:
                continue
            
            w_data = results[task_name]
            y_data = y_results[task_name]
            
            # 获取y预测值
            y_pred_from_y = y_data.get('y_mean_per_pos')  # Y-predictor预测的y
            y_pred_from_w = w_data.get('y_pred_mean_per_pos')  # W-predictor预测的y
            
            if y_pred_from_y is not None and y_pred_from_w is not None:
                y_pred_from_y = y_pred_from_y.flatten()
                y_pred_from_w = y_pred_from_w.flatten()
                positions = np.arange(1, len(y_pred_from_y) + 1)
                
                task_label = get_task_label(task_name)
                
                # Y-predictor曲线（圆圈marker）
                ax4.plot(positions, y_pred_from_y,
                         label=f'{task_label} (Y-Pred)',
                         color=y_colors.get(task_name, 'red'),
                         marker=y_marker,
                         markersize=4,
                         linewidth=2.5,
                         linestyle='-',
                         alpha=0.9)
                
                # W-predictor曲线（方块marker）
                ax4.plot(positions, y_pred_from_w,
                         label=f'{task_label} (W-Pred)',
                         color=w_colors.get(task_name, 'blue'),
                         marker=w_marker,
                         markersize=4,
                         linewidth=2.5,
                         linestyle='-',
                         alpha=0.9)
        
        ax4.set_xlabel('Position', fontsize=11)
        ax4.set_ylabel('Predicted Y Value (mean)', fontsize=11)
        ax4.set_title('(d) Y-Predictor vs W-Predictor Predictions', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8, loc='best', ncol=1)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=0)
    else:
        ax4.text(0.5, 0.5, 'Y-Predictor results not available\nfor comparison',
                 ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('(d) Y-Predictor vs W-Predictor Predictions', fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    # 总标题
    fig.suptitle(f'W-Predictor: Analysis\n{train_label}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dirname}_w_pred_analysis.png')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ W-Pred visualization saved: {output_file}")


def compare_y_vs_w_predictor(y_test_dir, w_test_dir, output_dir):
    """对比Y预测器和W预测器在同一测试任务上的表现
    
    Args:
        y_test_dir: Y预测器测试结果目录
        w_test_dir: W预测器测试结果目录
        output_dir: 输出图片目录
    """
    # 加载两个预测器的结果
    y_results = load_test_results(y_test_dir)
    w_results = load_test_results(w_test_dir)
    
    if not y_results or not w_results:
        print(f"  ⚠️  Missing results for comparison")
        return
    
    # 提取训练配置
    dirname = Path(y_test_dir).name
    train_probs = extract_prob_from_dirname(dirname)
    if train_probs is None:
        return
    
    train_label = get_training_label(train_probs)
    
    # 找出共同的测试任务
    common_tasks = set(y_results.keys()) & set(w_results.keys())
    if not common_tasks:
        print(f"  ⚠️  No common test tasks")
        return
    
    # 为每个测试任务创建对比图
    for task_name in sorted(common_tasks):
        y_data = y_results[task_name]
        w_data = w_results[task_name]
        
        # 创建图表 - 主图显示y值，子图显示误差
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 提取三种数据
        y_true_mean = y_data.get('y_true_mean_per_pos', None)  # 真实y值
        y_pred_mean = y_data.get('y_mean_per_pos', None)  # Y预测器预测的y
        w_y_pred_mean = w_data.get('y_pred_mean_per_pos', None)  # W预测器预测的y
        
        # 如果没有数据，跳过这个对比图
        if y_true_mean is None or y_pred_mean is None or w_y_pred_mean is None:
            print(f"    ⚠️  Missing data for {task_name}, skipping comparison")
            plt.close(fig)
            continue
        
        y_true_mean = y_true_mean.flatten()
        y_pred_mean = y_pred_mean.flatten()
        w_y_pred_mean = w_y_pred_mean.flatten()
        positions = np.arange(1, len(y_true_mean) + 1)
        
        # 计算误差
        y_pred_error = np.abs(y_pred_mean - y_true_mean)
        w_pred_error = np.abs(w_y_pred_mean - y_true_mean)
        
        # === 上半部分：Y值曲线 ===
        # 1. 真实y值
        ax1.plot(positions, y_true_mean,
                label='Ground Truth',
                color='green',
                marker='x',
                markersize=6,
                linewidth=3,
                linestyle='-',
                alpha=0.8,
                zorder=1)
        
        # 2. Y预测器预测值
        ax1.plot(positions, y_pred_mean,
                label='Y-Predictor',
                color='#1f77b4',
                marker='o',
                markersize=5,
                linewidth=2.5,
                linestyle='-',
                alpha=0.9,
                zorder=3)
        
        # 3. W预测器预测值
        ax1.plot(positions, w_y_pred_mean,
                label='W-Predictor',
                color='#ff7f0e',
                marker='s',
                markersize=5,
                linewidth=2.5,
                linestyle='--',
                alpha=0.9,
                zorder=2)
        
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Y Value (Mean across samples)', fontsize=12)
        ax1.set_title(f'Y-Predictor vs W-Predictor: Predictions and Errors\n{train_label} | Test on {get_task_label(task_name)}',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # 添加最终位置的统计信息
        y_true_final = y_true_mean[-1]
        y_pred_final = y_pred_mean[-1]
        w_pred_final = w_y_pred_mean[-1]
        y_pred_error_final = y_pred_error[-1]
        w_pred_error_final = w_pred_error[-1]
        
        textstr = (f'Final Position (#{len(positions)}):\n'
                  f'Ground Truth: {y_true_final:.4f}\n'
                  f'Y-Predictor: {y_pred_final:.4f}  (error: {y_pred_error_final:.4f})\n'
                  f'W-Predictor: {w_pred_final:.4f}  (error: {w_pred_error_final:.4f})')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # === 下半部分：误差曲线 ===
        ax2.plot(positions, y_pred_error,
                label='Y-Predictor Error',
                color='#1f77b4',
                marker='o',
                markersize=4,
                linewidth=2.5,
                linestyle='-',
                alpha=0.9)
        
        ax2.plot(positions, w_pred_error,
                label='W-Predictor Error',
                color='#ff7f0e',
                marker='s',
                markersize=4,
                linewidth=2.5,
                linestyle='--',
                alpha=0.9)
        
        # 添加零线
        ax2.axhline(y=0, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Zero Error')
        
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Absolute Error |y_pred - y_true|', fontsize=12)
        ax2.legend(fontsize=11, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        
        # 添加误差统计信息
        y_pred_error_mean = np.mean(y_pred_error)
        w_pred_error_mean = np.mean(w_pred_error)
        y_pred_error_max = np.max(y_pred_error)
        w_pred_error_max = np.max(w_pred_error)
        
        error_textstr = (f'Error Statistics:\n'
                        f'Y-Pred: mean={y_pred_error_mean:.4f}, max={y_pred_error_max:.4f}\n'
                        f'W-Pred: mean={w_pred_error_mean:.4f}, max={w_pred_error_max:.4f}')
        error_props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax2.text(0.98, 0.98, error_textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=error_props)
        
        # 保存图片
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{dirname}_compare_{task_name}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Comparison saved: {output_file}")


def main():
    """主函数：扫描所有测试结果并生成可视化"""
    base_dir = Path('test_results')
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist!")
        return
    
    print("="*70)
    print("Starting visualization for all test results...")
    print("="*70)
    
    # 可视化 Y 预测器结果
    y_pred_dir = base_dir / 'Y_pred'
    if y_pred_dir.exists():
        print("\n📊 Processing Y-Predictor results...")
        for prob_dir in sorted(y_pred_dir.iterdir()):
            if prob_dir.is_dir() and prob_dir.name.startswith('prob_'):
                print(f"\n  Processing: {prob_dir.name}")
                visualize_y_pred(str(prob_dir), 'visualization_results/Y_pred')
    
    # 可视化 W 预测器结果
    w_pred_dir = base_dir / 'W_pred'
    if w_pred_dir.exists():
        print("\n📊 Processing W-Predictor results...")
        for prob_dir in sorted(w_pred_dir.iterdir()):
            if prob_dir.is_dir() and prob_dir.name.startswith('prob_'):
                print(f"\n  Processing: {prob_dir.name}")
                visualize_w_pred(str(prob_dir), 'visualization_results/W_pred')
    
    # 对比 Y 预测器 vs W 预测器
    if y_pred_dir.exists() and w_pred_dir.exists():
        print("\n📊 Comparing Y-Predictor vs W-Predictor...")
        for prob_dir_name in sorted([d.name for d in y_pred_dir.iterdir() if d.is_dir() and d.name.startswith('prob_')]):
            y_dir = y_pred_dir / prob_dir_name
            w_dir = w_pred_dir / prob_dir_name
            
            if y_dir.exists() and w_dir.exists():
                print(f"\n  Comparing: {prob_dir_name}")
                compare_y_vs_w_predictor(str(y_dir), str(w_dir), 'visualization_results/Comparison')
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print("="*70)
    print("Results saved in:")
    print("  - visualization_results/Y_pred/")
    print("  - visualization_results/W_pred/")
    print("  - visualization_results/Comparison/")
    print("="*70)


if __name__ == '__main__':
    main()

