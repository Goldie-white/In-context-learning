#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化测试结果
递归扫描指定目录（默认 test_results/），为每个训练配置生成可视化图表

使用方法:
    python visualize.py                    # 扫描默认的 test_results/ 目录
    python visualize.py --input_dir /path/to/results  # 扫描指定目录
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse
import sys

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
        test_dir: 测试结果目录，如 'test_results/xdim_5/Y_pred/prob_1.0_0.0_0.0_0.0'
    
    Returns:
        dict: {task_name: data_dict}
    
    Note:
        实际文件是 y_analysis.pkl 或 w_analysis.pkl，需要从数据中或文件名推断测试任务
    """
    results = {}
    test_path = Path(test_dir)
    
    if not test_path.exists():
        return results
    
    # 尝试加载 y_analysis.pkl 或 w_analysis.pkl（包括 MC Dropout 版本）
    pkl_files = (list(test_path.glob('y_analysis.pkl')) + 
                 list(test_path.glob('w_analysis.pkl')) +
                 list(test_path.glob('y_mc_analysis.pkl')) +
                 list(test_path.glob('w_mc_analysis.pkl')))
    
    if not pkl_files:
        # 兼容旧格式：扫描所有 test_on_*.pkl 文件
        pkl_files = list(test_path.glob('test_on_*.pkl'))
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
                # 从文件名或数据中推断任务名称
                if 'test_on_' in pkl_file.stem:
                    # 旧格式：test_on_T1.pkl
                    task_name = pkl_file.stem.replace('test_on_', '')
                else:
                    # 新格式：从目录名或数据中推断测试任务
                    # 目录名格式：prob_0.0_0.0_0.0_1.0 表示测试时使用的任务概率
                    dirname = test_path.name
                    prob_match = re.match(r'prob_([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)', dirname)
                    if prob_match:
                        p0, p1, p2, p3 = tuple(float(x) for x in prob_match.groups())
                        # 根据概率确定任务（假设只有一个非零概率）
                        if p0 > 0.99:
                            task_name = 'T1'
                        elif p1 > 0.99:
                            task_name = 'T2'
                        elif p2 > 0.99:
                            task_name = 'T3'
                        elif p3 > 0.99:
                            task_name = 'T4'
                        else:
                            # 混合任务，使用第一个非零概率
                            if p0 > 0:
                                task_name = 'T1'
                            elif p1 > 0:
                                task_name = 'T2'
                            elif p2 > 0:
                                task_name = 'T3'
                            else:
                                task_name = 'T4'
                    else:
                        # 无法推断，使用默认名称
                        task_name = 'T1'
                
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
    
    # 从数据中推断序列长度信息
    # 测试序列长度：从数据数组长度推断（如 avg_y_loss 的长度）
    first_data = list(results.values())[0] if results else {}
    test_num_exemplars = None
    if first_data:
        # 从 avg_y_loss 的长度推断测试序列长度
        if 'avg_y_loss' in first_data:
            test_num_exemplars = len(first_data['avg_y_loss'])
        elif 'y_mean_per_pos' in first_data:
            test_num_exemplars = len(first_data['y_mean_per_pos'])
    
    # 训练序列长度：尝试从保存的数据中获取，如果没有则尝试从路径推断
    train_num_exemplars = first_data.get('train_num_exemplars', None)
    if train_num_exemplars is None:
        # 尝试从 exp_folder 路径推断（如果路径包含序列长度信息）
        exp_folder = first_data.get('exp_folder', '')
        if exp_folder and 'num_exemplars' in str(exp_folder):
            # 这里可以添加更复杂的路径解析逻辑
            pass
    
    # 构建序列长度信息字符串
    seq_info = ""
    if test_num_exemplars is not None:
        if train_num_exemplars is not None:
            if train_num_exemplars == test_num_exemplars:
                seq_info = f" (Train & Test: {train_num_exemplars} exemplars)"
            else:
                seq_info = f" (Train: {train_num_exemplars}, Test: {test_num_exemplars} exemplars)"
        else:
            seq_info = f" (Test: {test_num_exemplars} exemplars)"
    
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
    ax.set_title(f'Y-Predictor: Loss vs Position\n{train_label}{seq_info}', fontsize=14, fontweight='bold')
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


def visualize_w_pred(test_dir, output_dir, other_w_test_dir=None):
    """可视化 W 预测器的测试结果（可同时对比两个版本）
    
    Args:
        test_dir: W预测器测试结果目录（W_pred 或 W_pred_loss_W）
        output_dir: 输出图片目录
        other_w_test_dir: 另一个W预测器版本的结果目录（用于对比），可为None
    """
    results = load_test_results(test_dir)
    
    if not results:
        print(f"No results found in {test_dir}")
        return
    
    # 尝试加载另一个W预测器版本的结果（如果提供）
    other_results = {}
    if other_w_test_dir:
        if Path(other_w_test_dir).exists():
            other_results = load_test_results(other_w_test_dir)
            if other_results:
                print(f"    ✅ Loaded {len(other_results)} tasks from {other_w_test_dir}")
            else:
                print(f"    ⚠️  No results loaded from {other_w_test_dir}")
        else:
            print(f"    ⚠️  Path does not exist: {other_w_test_dir}")
    
    # 确定当前版本名称
    test_path = Path(test_dir)
    is_loss_w = 'W_pred_loss_W' in str(test_dir)
    current_version = 'W_pred_loss_W' if is_loss_w else 'W_pred'
    other_version = 'W_pred' if is_loss_w else 'W_pred_loss_W'
    
    # 提取训练配置
    dirname = Path(test_dir).name
    train_probs = extract_prob_from_dirname(dirname)
    if train_probs is None:
        print(f"Cannot parse training probs from {dirname}")
        return
    
    train_label = get_training_label(train_probs)
    
    # 从数据中推断序列长度信息
    first_data = list(results.values())[0] if results else {}
    test_num_exemplars = None
    if first_data:
        # 从数据数组长度推断测试序列长度
        if 'avg_y_loss' in first_data:
            test_num_exemplars = len(first_data['avg_y_loss'])
        elif 'w_mse_per_pos' in first_data:
            test_num_exemplars = len(first_data['w_mse_per_pos'])
    
    train_num_exemplars = first_data.get('train_num_exemplars', None)
    
    # 构建序列长度信息字符串
    seq_info = ""
    if test_num_exemplars is not None:
        if train_num_exemplars is not None:
            if train_num_exemplars == test_num_exemplars:
                seq_info = f" (Train & Test: {train_num_exemplars} exemplars)"
            else:
                seq_info = f" (Train: {train_num_exemplars}, Test: {test_num_exemplars} exemplars)"
        else:
            seq_info = f" (Test: {test_num_exemplars} exemplars)"
    
    # 尝试加载对应的Y-predictor结果（用于第4个子图）
    # 从 test_dir 中找到对应的 Y_pred 目录
    # 如果 test_dir 是 xdim_*/W_pred/prob_...，则找 xdim_*/Y_pred/prob_...
    if 'W_pred' in str(test_dir):
        y_test_dir = str(test_path.parent.parent / 'Y_pred' / test_path.name)
    else:
        y_test_dir = test_dir.replace('/W_pred/', '/Y_pred/').replace('/W_pred_loss_W/', '/Y_pred/')
    y_results = load_test_results(y_test_dir) if Path(y_test_dir).exists() else {}
    
    # 创建2x2四宫格布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 颜色映射
    colors = {'T1': '#1f77b4', 'T2': '#ff7f0e', 'T3': '#2ca02c', 'T4': '#d62728'}
    # 另一个版本使用不同的颜色（用于区分，确保明显可见）
    colors_other = {'T1': '#9467bd', 'T2': '#ffbb78', 'T3': '#98df8a', 'T4': '#ff9896'}
    markers = {'T1': 'o', 'T2': 's', 'T3': '^', 'T4': 'D'}
    
    # 子图1 (左上): Y Loss
    ax1 = axes[0, 0]
    # 获取所有任务（包括两个版本的所有任务）
    all_tasks = set(results.keys())
    if other_results:
        all_tasks.update(other_results.keys())
    
    for task_name in sorted(all_tasks):
        # 绘制当前版本的数据
        if task_name in results:
            data = results[task_name]
            avg_y_loss = data.get('avg_y_loss')
            if avg_y_loss is not None:
                positions = np.arange(1, len(avg_y_loss) + 1)
                ax1.plot(positions, avg_y_loss,
                         label=f'{get_task_label(task_name)} ({current_version})',
                         color=colors.get(task_name, 'gray'),
                         marker=markers.get(task_name, 'o'),
                         markersize=4,
                         linewidth=2.5,
                         alpha=0.9,
                         zorder=3)
        
        # 绘制另一个版本的数据
        if task_name in other_results:
            other_data = other_results[task_name]
            other_avg_y_loss = other_data.get('avg_y_loss')
            if other_avg_y_loss is not None:
                other_positions = np.arange(1, len(other_avg_y_loss) + 1)
                ax1.plot(other_positions, other_avg_y_loss,
                         label=f'{get_task_label(task_name)} ({other_version})',
                         color=colors_other.get(task_name, '#666666'),
                         marker='^',
                         markersize=5,
                         linewidth=2.5,
                         linestyle='--',
                         alpha=0.9,
                         zorder=4)
        
        # 绘制 Y predictor 的 loss 曲线
        if task_name in y_results:
            y_data = y_results[task_name]
            y_avg_y_loss = y_data.get('avg_y_loss')
            if y_avg_y_loss is not None:
                y_positions = np.arange(1, len(y_avg_y_loss) + 1)
                ax1.plot(y_positions, y_avg_y_loss,
                         label=f'{get_task_label(task_name)} (Y-Predictor)',
                         color=colors.get(task_name, 'gray'),
                         marker='*',
                         markersize=6,
                         linewidth=2.5,
                         linestyle=':',
                         alpha=0.9,
                         zorder=5)
    
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('Y Loss', fontsize=11)
    ax1.set_title('(a) Y Prediction Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 子图2 (右上): W MSE
    ax2 = axes[0, 1]
    # 获取所有任务（包括两个版本的所有任务）
    all_tasks = set(results.keys())
    if other_results:
        all_tasks.update(other_results.keys())
    
    for task_name in sorted(all_tasks):
        # 绘制当前版本的数据
        if task_name in results:
            data = results[task_name]
            w_mse = data.get('w_mse_per_pos')
            if w_mse is not None:
                positions = np.arange(1, len(w_mse) + 1)
                ax2.plot(positions, w_mse,
                         label=f'{get_task_label(task_name)} ({current_version})',
                         color=colors.get(task_name, 'gray'),
                         marker=markers.get(task_name, 'o'),
                         markersize=4,
                         linewidth=2.5,
                         alpha=0.9,
                         zorder=3)
        
        # 绘制另一个版本的数据
        if task_name in other_results:
            other_data = other_results[task_name]
            other_w_mse = other_data.get('w_mse_per_pos')
            if other_w_mse is not None:
                other_positions = np.arange(1, len(other_w_mse) + 1)
                ax2.plot(other_positions, other_w_mse,
                         label=f'{get_task_label(task_name)} ({other_version})',
                         color=colors_other.get(task_name, '#666666'),
                         marker='^',
                         markersize=5,
                         linewidth=2.5,
                         linestyle='--',
                         alpha=0.9,
                         zorder=4)
    
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('W MSE (||w_pred - w_true||²)', fontsize=11)
    ax2.set_title('(b) W Prediction MSE', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    # 子图3 (左下): Cosine Similarity
    ax3 = axes[1, 0]
    # 获取所有任务（包括两个版本的所有任务）
    all_tasks = set(results.keys())
    if other_results:
        all_tasks.update(other_results.keys())
    
    for task_name in sorted(all_tasks):
        # 绘制当前版本的数据
        if task_name in results:
            data = results[task_name]
            cosine_sim = data.get('cosine_sim_mean')
            if cosine_sim is not None:
                positions = np.arange(1, len(cosine_sim) + 1)
                ax3.plot(positions, cosine_sim,
                         label=f'{get_task_label(task_name)} ({current_version})',
                         color=colors.get(task_name, 'gray'),
                         marker=markers.get(task_name, 'o'),
                         markersize=4,
                         linewidth=2.5,
                         alpha=0.9,
                         zorder=3)
        
        # 绘制另一个版本的数据
        if task_name in other_results:
            other_data = other_results[task_name]
            other_cosine_sim = other_data.get('cosine_sim_mean')
            if other_cosine_sim is not None:
                other_positions = np.arange(1, len(other_cosine_sim) + 1)
                ax3.plot(other_positions, other_cosine_sim,
                         label=f'{get_task_label(task_name)} ({other_version})',
                         color=colors_other.get(task_name, '#666666'),
                         marker='^',
                         markersize=5,
                         linewidth=2.5,
                         linestyle='--',
                         alpha=0.9,
                         zorder=4)
    
    ax3.set_xlabel('Position', fontsize=11)
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title('(c) W Cosine Similarity (w_pred · w_true)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim([-0.1, 1.1])
    
    # 子图4 (右下): W-Predictor 范数曲线对比
    ax4 = axes[1, 1]
    # 获取所有任务（包括两个版本的所有任务）
    all_tasks = set(results.keys())
    if other_results:
        all_tasks.update(other_results.keys())
    
    for task_name in sorted(all_tasks):
        # 先绘制w_true的范数（作为参考线，所有位置相同）
        if task_name in results:
            data = results[task_name]
            w_true = data.get('w_true')  # shape: (n_samples, x_dim)
            w_preds = data.get('w_preds')  # shape: (n_samples, num_exemplars, x_dim)
            
            if w_true is not None and w_preds is not None:
                # 计算w_true的平均范数（对所有样本求平均）
                w_true_norms = np.linalg.norm(w_true, axis=1)  # (n_samples,)
                w_true_norm_mean = np.mean(w_true_norms)
                
                # 获取位置数量
                num_positions = w_preds.shape[1]
                positions = np.arange(1, num_positions + 1)
                
                # 绘制w_true范数（水平线）
                ax4.axhline(y=w_true_norm_mean,
                           label=f'{get_task_label(task_name)} (Ground Truth)',
                           color='green',
                           linestyle=':',
                           linewidth=2.5,
                           alpha=0.8,
                           zorder=1)
        
        # 绘制当前版本的w_pred范数
        if task_name in results:
            data = results[task_name]
            w_preds = data.get('w_preds')  # shape: (n_samples, num_exemplars, x_dim)
            if w_preds is not None:
                # 计算每个位置的平均L2范数
                # w_preds: (n_samples, num_exemplars, x_dim)
                # norms: (n_samples, num_exemplars)
                norms = np.linalg.norm(w_preds, axis=2)
                # 对所有样本求平均: (num_exemplars,)
                w_norm_mean = np.mean(norms, axis=0)
                positions = np.arange(1, len(w_norm_mean) + 1)
                
                ax4.plot(positions, w_norm_mean,
                         label=f'{get_task_label(task_name)} ({current_version})',
                         color=colors.get(task_name, 'gray'),
                         marker=markers.get(task_name, 'o'),
                         markersize=4,
                         linewidth=2.5,
                         alpha=0.9,
                         zorder=3)
        
        # 绘制另一个版本的w_pred范数
        if task_name in other_results:
            other_data = other_results[task_name]
            other_w_preds = other_data.get('w_preds')
            if other_w_preds is not None:
                # 计算每个位置的平均L2范数
                other_norms = np.linalg.norm(other_w_preds, axis=2)
                other_w_norm_mean = np.mean(other_norms, axis=0)
                other_positions = np.arange(1, len(other_w_norm_mean) + 1)
                
                ax4.plot(other_positions, other_w_norm_mean,
                         label=f'{get_task_label(task_name)} ({other_version})',
                         color=colors_other.get(task_name, '#666666'),
                         marker='^',
                         markersize=5,
                         linewidth=2.5,
                         linestyle='--',
                         alpha=0.9,
                         zorder=4)
    
    ax4.set_xlabel('Position', fontsize=11)
    ax4.set_ylabel('||w|| (L2 norm, mean)', fontsize=11)
    ax4.set_title('(d) W-Predictor Norm Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    
    # 总标题
    title = f'W-Predictor: Analysis ({current_version}'
    if other_results:
        title += f' vs {other_version}'
    title += f')\n{train_label}{seq_info}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dirname}_w_pred_analysis.png')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ W-Pred visualization saved: {output_file}")


def compare_y_vs_w_predictor(y_test_dir, w_test_dir, w_loss_w_test_dir, output_dir):
    """对比Y预测器和W预测器（两个版本）在同一测试任务上的表现
    
    Args:
        y_test_dir: Y预测器测试结果目录
        w_test_dir: W预测器测试结果目录（W_pred）
        w_loss_w_test_dir: W预测器Loss_W测试结果目录（W_pred_loss_W），可为None
        output_dir: 输出图片目录
    """
    # 加载三个预测器的结果
    y_results = load_test_results(y_test_dir)
    w_results = load_test_results(w_test_dir)
    w_loss_w_results = load_test_results(w_loss_w_test_dir) if w_loss_w_test_dir and Path(w_loss_w_test_dir).exists() else {}
    
    if not y_results:
        print(f"  ⚠️  Missing Y-Predictor results")
        return
    
    if not w_results and not w_loss_w_results:
        print(f"  ⚠️  Missing W-Predictor results")
        return
    
    # 提取训练配置
    dirname = Path(y_test_dir).name
    train_probs = extract_prob_from_dirname(dirname)
    if train_probs is None:
        return
    
    train_label = get_training_label(train_probs)
    
    # 从数据中推断序列长度信息
    first_y_data = list(y_results.values())[0] if y_results else {}
    test_num_exemplars = None
    if first_y_data:
        if 'avg_y_loss' in first_y_data:
            test_num_exemplars = len(first_y_data['avg_y_loss'])
        elif 'y_mean_per_pos' in first_y_data:
            test_num_exemplars = len(first_y_data['y_mean_per_pos'])
    
    train_num_exemplars = first_y_data.get('train_num_exemplars', None)
    
    # 构建序列长度信息字符串
    seq_info = ""
    if test_num_exemplars is not None:
        if train_num_exemplars is not None:
            if train_num_exemplars == test_num_exemplars:
                seq_info = f" | Train & Test: {train_num_exemplars} exemplars"
            else:
                seq_info = f" | Train: {train_num_exemplars}, Test: {test_num_exemplars} exemplars"
        else:
            seq_info = f" | Test: {test_num_exemplars} exemplars"
    
    # 找出共同的测试任务
    common_tasks = set(y_results.keys())
    if w_results:
        common_tasks = common_tasks & set(w_results.keys())
    if w_loss_w_results:
        common_tasks = common_tasks & set(w_loss_w_results.keys())
    
    if not common_tasks:
        print(f"  ⚠️  No common test tasks")
        return
    
    # 为每个测试任务创建对比图
    for task_name in sorted(common_tasks):
        y_data = y_results[task_name]
        w_data = w_results.get(task_name, {}) if w_results else {}
        w_loss_w_data = w_loss_w_results.get(task_name, {}) if w_loss_w_results else {}
        
        # 创建图表 - 主图显示y值，子图显示误差
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 提取数据
        y_true_mean = y_data.get('y_true_mean_per_pos', None)  # 真实y值
        y_pred_mean = y_data.get('y_mean_per_pos', None)  # Y预测器预测的y
        w_y_pred_mean = w_data.get('y_pred_mean_per_pos', None) if w_data else None  # W预测器预测的y
        w_loss_w_y_pred_mean = w_loss_w_data.get('y_pred_mean_per_pos', None) if w_loss_w_data else None  # W预测器Loss_W预测的y
        
        # 至少需要Y预测器的数据
        if y_true_mean is None or y_pred_mean is None:
            print(f"    ⚠️  Missing Y-Predictor data for {task_name}, skipping comparison")
            plt.close(fig)
            continue
        
        y_true_mean = y_true_mean.flatten()
        y_pred_mean = y_pred_mean.flatten()
        positions = np.arange(1, len(y_true_mean) + 1)
        
        # 计算误差（原始值，不用绝对值）
        y_pred_error = y_pred_mean - y_true_mean
        
        # 处理W预测器的数据
        w_pred_error = None
        if w_y_pred_mean is not None:
            w_y_pred_mean = w_y_pred_mean.flatten()
            w_pred_error = w_y_pred_mean - y_true_mean
        
        w_loss_w_pred_error = None
        if w_loss_w_y_pred_mean is not None:
            w_loss_w_y_pred_mean = w_loss_w_y_pred_mean.flatten()
            w_loss_w_pred_error = w_loss_w_y_pred_mean - y_true_mean
        
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
                zorder=4)
        
        # 3. W预测器预测值（W_pred）
        if w_y_pred_mean is not None:
            ax1.plot(positions, w_y_pred_mean,
                    label='W-Predictor (W_pred)',
                    color='#ff7f0e',
                    marker='s',
                    markersize=5,
                    linewidth=2.5,
                    linestyle='--',
                    alpha=0.9,
                    zorder=3)
        
        # 4. W预测器Loss_W预测值（W_pred_loss_W）
        if w_loss_w_y_pred_mean is not None:
            ax1.plot(positions, w_loss_w_y_pred_mean,
                    label='W-Predictor Loss_W (W_pred_loss_W)',
                    color='#d62728',
                    marker='^',
                    markersize=5,
                    linewidth=2.5,
                    linestyle='-.',
                    alpha=0.9,
                    zorder=2)
        
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Y Value (Mean across samples)', fontsize=12)
        ax1.set_title(f'Y-Predictor vs W-Predictors: Predictions and Errors\n{train_label} | Test on {get_task_label(task_name)}{seq_info}',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # 添加最终位置的统计信息
        y_true_final = y_true_mean[-1]
        y_pred_final = y_pred_mean[-1]
        y_pred_error_final = y_pred_error[-1]
        
        textstr_lines = [f'Final Position (#{len(positions)}):',
                        f'Ground Truth: {y_true_final:.4f}',
                        f'Y-Predictor: {y_pred_final:.4f}  (error: {y_pred_error_final:.4f})']
        
        if w_y_pred_mean is not None:
            w_pred_final = w_y_pred_mean[-1]
            w_pred_error_final = w_pred_error[-1]
            textstr_lines.append(f'W-Pred: {w_pred_final:.4f}  (error: {w_pred_error_final:.4f})')
        
        if w_loss_w_y_pred_mean is not None:
            w_loss_w_pred_final = w_loss_w_y_pred_mean[-1]
            w_loss_w_pred_error_final = w_loss_w_pred_error[-1]
            textstr_lines.append(f'W-Pred Loss_W: {w_loss_w_pred_final:.4f}  (error: {w_loss_w_pred_error_final:.4f})')
        
        textstr = '\n'.join(textstr_lines)
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
        
        if w_pred_error is not None:
            ax2.plot(positions, w_pred_error,
                    label='W-Predictor Error (W_pred)',
                    color='#ff7f0e',
                    marker='s',
                    markersize=4,
                    linewidth=2.5,
                    linestyle='--',
                    alpha=0.9)
        
        if w_loss_w_pred_error is not None:
            ax2.plot(positions, w_loss_w_pred_error,
                    label='W-Predictor Loss_W Error',
                    color='#d62728',
                    marker='^',
                    markersize=4,
                    linewidth=2.5,
                    linestyle='-.',
                    alpha=0.9)
        
        # 添加零线
        ax2.axhline(y=0, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Zero Error')
        
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Error (y_pred - y_true)', fontsize=12)
        ax2.legend(fontsize=11, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        # 移除 bottom=0 限制，允许显示负值
        
        # 误差统计信息已移除（避免遮挡图表）
        
        # 保存图片
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{dirname}_compare_{task_name}.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Comparison saved: {output_file}")


def process_experiment_dir(exp_dir, output_base_name=None, base_dir=None):
    """处理一个实验目录，生成可视化
    
    Args:
        exp_dir: 实验目录路径（Path对象），应包含 Y_pred, W_pred, W_pred_loss_W 子目录
        output_base_name: 输出目录的基础名称，如果为None则使用exp_dir的相对路径
        base_dir: 基础目录（Path对象），用于计算相对路径，如果为None则使用exp_dir的父目录
    """
    if output_base_name is None:
        # 生成输出目录名：将路径中的 / 替换为 _
        if base_dir is not None:
            try:
                rel_path = exp_dir.relative_to(base_dir)
                output_base_name = str(rel_path).replace('/', '_')
            except ValueError:
                # 如果无法计算相对路径，使用目录名
                output_base_name = exp_dir.name
        else:
            # 如果没有指定base_dir，尝试从exp_dir推断
            # 如果exp_dir包含test_results，去掉它
            exp_str = str(exp_dir)
            if 'test_results' in exp_str:
                parts = exp_str.split('test_results')
                if len(parts) > 1:
                    rel_path = parts[1].lstrip('/')
                    output_base_name = rel_path.replace('/', '_') if rel_path else exp_dir.name
                else:
                    output_base_name = exp_dir.name
            else:
                output_base_name = exp_dir.name
    
    print(f"\n📁 Processing directory: {exp_dir}")
    
    # 可视化 Y 预测器结果
    y_pred_dir = exp_dir / 'Y_pred'
    if y_pred_dir.exists():
        print("\n  📊 Processing Y-Predictor results...")
        for prob_dir in sorted(y_pred_dir.iterdir()):
            if prob_dir.is_dir() and prob_dir.name.startswith('prob_'):
                print(f"    Processing: {prob_dir.name}")
                output_dir = f'visualization_results/{output_base_name}/Y_pred'
                visualize_y_pred(str(prob_dir), output_dir)
    
    # 可视化 W 预测器结果（包括 W_pred 和 W_pred_loss_W，同时对比两个版本）
    w_pred_dir = exp_dir / 'W_pred'
    w_pred_loss_w_dir = exp_dir / 'W_pred_loss_W'
    
    # 如果两个版本都存在，合并生成一份图；如果只有一个，单独生成
    if w_pred_dir.exists() and w_pred_loss_w_dir.exists():
        # 两个版本都存在，合并生成一份图
        print(f"\n  📊 Processing W-Predictor results (both versions)...")
        # 获取所有共同的 prob 目录
        w_pred_probs = {d.name for d in w_pred_dir.iterdir() if d.is_dir() and d.name.startswith('prob_')}
        w_loss_w_probs = {d.name for d in w_pred_loss_w_dir.iterdir() if d.is_dir() and d.name.startswith('prob_')}
        common_probs = sorted(w_pred_probs & w_loss_w_probs)
        
        for prob_dir_name in common_probs:
            print(f"    Processing: {prob_dir_name}")
            w_pred_prob_dir = w_pred_dir / prob_dir_name
            w_loss_w_prob_dir = w_pred_loss_w_dir / prob_dir_name
            output_dir = f'visualization_results/{output_base_name}/W_pred'
            # 生成合并图（以 W_pred 为主，对比 W_pred_loss_W）
            visualize_w_pred(str(w_pred_prob_dir), output_dir, str(w_loss_w_prob_dir))
    else:
        # 只有一个版本存在，单独生成
        if w_pred_dir.exists():
            print(f"\n  📊 Processing W_pred results...")
            for prob_dir in sorted(w_pred_dir.iterdir()):
                if prob_dir.is_dir() and prob_dir.name.startswith('prob_'):
                    print(f"    Processing: {prob_dir.name}")
                    output_dir = f'visualization_results/{output_base_name}/W_pred'
                    visualize_w_pred(str(prob_dir), output_dir, None)
        
        if w_pred_loss_w_dir.exists():
            print(f"\n  📊 Processing W_pred_loss_W results...")
            for prob_dir in sorted(w_pred_loss_w_dir.iterdir()):
                if prob_dir.is_dir() and prob_dir.name.startswith('prob_'):
                    print(f"    Processing: {prob_dir.name}")
                    output_dir = f'visualization_results/{output_base_name}/W_pred_loss_W'
                    visualize_w_pred(str(prob_dir), output_dir, None)
    
    # 对比 Y 预测器 vs W 预测器（两个版本）
    y_pred_dir = exp_dir / 'Y_pred'
    w_pred_dir = exp_dir / 'W_pred'
    w_pred_loss_w_dir = exp_dir / 'W_pred_loss_W'
    
    if y_pred_dir.exists() and (w_pred_dir.exists() or w_pred_loss_w_dir.exists()):
        print("\n  📊 Comparing Y-Predictor vs W-Predictors...")
        for prob_dir_name in sorted([d.name for d in y_pred_dir.iterdir() if d.is_dir() and d.name.startswith('prob_')]):
            y_dir = y_pred_dir / prob_dir_name
            w_dir = w_pred_dir / prob_dir_name if w_pred_dir.exists() else None
            w_loss_w_dir = w_pred_loss_w_dir / prob_dir_name if w_pred_loss_w_dir.exists() else None
            
            if y_dir.exists() and (w_dir is not None or w_loss_w_dir is not None):
                print(f"    Comparing: {prob_dir_name}")
                output_dir = f'visualization_results/{output_base_name}/Comparison'
                compare_y_vs_w_predictor(
                    str(y_dir), 
                    str(w_dir) if w_dir and w_dir.exists() else None,
                    str(w_loss_w_dir) if w_loss_w_dir and w_loss_w_dir.exists() else None,
                    output_dir
                )


def find_experiment_dirs(base_dir, current_path=None, max_depth=10):
    """递归查找所有包含 Y_pred/W_pred/W_pred_loss_W 的目录
    
    Args:
        base_dir: 基础目录（Path对象）
        current_path: 当前扫描路径（Path对象），用于递归
        max_depth: 最大递归深度，防止无限递归
    
    Returns:
        list: [(exp_dir, output_name), ...] 元组列表
    """
    if current_path is None:
        current_path = base_dir
    
    if max_depth <= 0:
        return []
    
    exp_dirs = []
    
    # 检查当前目录是否包含 Y_pred/W_pred/W_pred_loss_W
    has_predictors = any((current_path / pred_type).exists() 
                        for pred_type in ['Y_pred', 'W_pred', 'W_pred_loss_W'])
    
    if has_predictors:
        # 找到实验目录，生成输出名称
        try:
            rel_path = current_path.relative_to(base_dir)
            # 将路径转换为输出名称：用 _ 替换 /
            output_name = str(rel_path).replace('/', '_')
            exp_dirs.append((current_path, output_name))
        except ValueError:
            # 如果无法计算相对路径，使用目录名
            output_name = current_path.name
            exp_dirs.append((current_path, output_name))
    else:
        # 继续递归扫描子目录
        try:
            for subdir in sorted(current_path.iterdir()):
                if not subdir.is_dir():
                    continue
                
                # 跳过某些不需要扫描的目录（可选）
                if subdir.name in ['.git', '__pycache__', 'ckpt']:
                    continue
                
                # 递归查找
                exp_dirs.extend(find_experiment_dirs(base_dir, subdir, max_depth - 1))
        except (PermissionError, OSError):
            # 忽略无法访问的目录
            pass
    
    return exp_dirs


def main():
    """主函数：递归扫描所有测试结果并生成可视化
    
    完全自动扫描指定目录下的所有目录结构，不限制：
    - 目录名前缀
    - 嵌套层级
    - 目录结构
    - 输入目录路径（可通过命令行参数指定）
    
    支持的任意目录结构示例：
    - {input_dir}/{exp_name}/Y_pred/prob_*/
    - {input_dir}/LEN_VARIANCE/num_*/Y_pred/prob_*/
    - {input_dir}/NOISE/noise_std_*/Y_pred/prob_*/
    - {input_dir}/mc/xdim_*/Y_pred/prob_*/
    - {input_dir}/任意/嵌套/层级/Y_pred/prob_*/
    """
    parser = argparse.ArgumentParser(
        description='可视化测试结果，递归扫描指定目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize.py                              # 扫描默认的 test_results/ 目录
  python visualize.py --input_dir test_results     # 扫描 test_results/ 目录
  python visualize.py --input_dir /path/to/results  # 扫描任意指定目录
        """
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='test_results',
        help='输入目录路径（默认: test_results）'
    )
    
    args = parser.parse_args()
    base_dir = Path(args.input_dir)
    
    if not base_dir.exists():
        print(f"Error: 目录不存在: {base_dir}")
        print(f"请检查路径是否正确，或使用 --input_dir 指定其他目录")
        sys.exit(1)
    
    if not base_dir.is_dir():
        print(f"Error: {base_dir} 不是一个目录")
        sys.exit(1)
    
    print("="*70)
    print("Starting visualization for all test results...")
    print(f"Recursively scanning directory: {base_dir}")
    print("="*70)
    
    # 递归查找所有包含预测器目录的实验目录
    exp_dirs = find_experiment_dirs(base_dir)
    
    if not exp_dirs:
        print("Warning: 未找到包含 Y_pred/W_pred/W_pred_loss_W 的实验目录")
        print("期望的目录结构:")
        print(f"  {base_dir}/.../Y_pred/prob_*/")
        print(f"  {base_dir}/.../W_pred/prob_*/")
        print(f"  {base_dir}/.../W_pred_loss_W/prob_*/")
        return
    
    print(f"\n找到 {len(exp_dirs)} 个实验目录需要处理:")
    for exp_dir, output_name in exp_dirs:
        print(f"  - {exp_dir} -> {output_name}")
    print()
    
    # 处理所有找到的实验目录
    for exp_dir, output_name in exp_dirs:
        process_experiment_dir(exp_dir, output_name, base_dir)
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print("="*70)
    print("Results saved in:")
    print("  - visualization_results/{exp_name}/Y_pred/")
    print("  - visualization_results/{exp_name}/W_pred/")
    print("  - visualization_results/{exp_name}/W_pred_loss_W/")
    print("  - visualization_results/{exp_name}/Comparison/")
    print("")
    print(f"Note: 目录名称自动从 {base_dir} 的目录结构生成")
    print("="*70)


if __name__ == '__main__':
    main()

