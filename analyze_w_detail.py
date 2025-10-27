#!/usr/bin/env python3
"""
详细分析W预测器的预测结果：
- 显示预测的w和真实的w的具体数值
- 使用相同的任务函数计算两个w对应的y
- 验证为什么w差距很大但y很接近
"""

import pickle
import numpy as np
import argparse
import os


def apply_task_function(task_id, w, x):
    """根据任务类型计算y = T_i(w, x)
    
    Args:
        task_id: 任务类型 (0, 1, 2, 3)
        w: shape (x_dim,)
        x: shape (x_dim,)
    
    Returns:
        y: scalar
    """
    if task_id == 0:
        # Task 1: y = w^T x
        return np.dot(w, x)
    elif task_id == 1:
        # Task 2: y = w^T sort(x)
        x_sorted = np.sort(x)
        return np.dot(w, x_sorted)
    elif task_id == 2:
        # Task 3: y = (dim/sqrt(2)) * w^T softmax(x)
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        x_softmax = exp_x / np.sum(exp_x)
        dim = len(x)
        scale_factor = dim / np.sqrt(2.0)
        return scale_factor * np.dot(w, x_softmax)
    elif task_id == 3:
        # Task 4: y = ||x - w||^2
        diff = x - w
        return np.sum(diff ** 2)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def analyze_sample(sample_idx, w_pred_seq, w_true, x_seq, y_true_seq, task_id, positions_to_check):
    """分析单个样本
    
    Args:
        sample_idx: 样本索引
        w_pred_seq: shape (num_exemplars, x_dim) - 各位置的w预测
        w_true: shape (x_dim,) - 真实的w
        x_seq: shape (num_exemplars, x_dim) - 各位置的x
        y_true_seq: shape (num_exemplars,) - 各位置的真实y
        task_id: 任务类型
        positions_to_check: 要检查的位置列表
    """
    print("\n" + "="*80)
    print(f"样本 #{sample_idx + 1} | Task Type: T{task_id + 1}")
    print("="*80)
    
    x_dim = len(w_true)
    
    print(f"\n真实的w (ground truth):")
    print(f"  {w_true}")
    print(f"  ||w_true|| = {np.linalg.norm(w_true):.6f}")
    
    for pos_idx in positions_to_check:
        if pos_idx >= len(w_pred_seq):
            continue
            
        w_pred = w_pred_seq[pos_idx]
        x = x_seq[pos_idx]
        y_true = y_true_seq[pos_idx]
        
        print(f"\n{'─'*80}")
        print(f"位置 {pos_idx + 1}:")
        print(f"{'─'*80}")
        
        # 显示预测的w
        print(f"\n  预测的w:")
        print(f"    {w_pred}")
        print(f"    ||w_pred|| = {np.linalg.norm(w_pred):.6f}")
        
        # 计算w的差异
        w_diff = w_pred - w_true
        w_mse = np.mean(w_diff ** 2)
        w_cosine = np.dot(w_pred, w_true) / (np.linalg.norm(w_pred) * np.linalg.norm(w_true) + 1e-8)
        
        print(f"\n  w的差异:")
        print(f"    w_pred - w_true = {w_diff}")
        print(f"    ||w_pred - w_true|| = {np.linalg.norm(w_diff):.6f}")
        print(f"    MSE(w) = {w_mse:.6f}")
        print(f"    Cosine Similarity = {w_cosine:.6f}")
        
        # 显示x
        print(f"\n  输入x:")
        print(f"    {x}")
        print(f"    ||x|| = {np.linalg.norm(x):.6f}")
        
        # 使用两个不同的w计算y
        y_from_w_pred = apply_task_function(task_id, w_pred, x)
        y_from_w_true = apply_task_function(task_id, w_true, x)
        
        print(f"\n  使用任务函数 T{task_id + 1} 计算y:")
        print(f"    y(w_true, x)  = {y_from_w_true:.6f}  <- 真实y值")
        print(f"    y(w_pred, x)  = {y_from_w_pred:.6f}  <- W预测器预测的y")
        print(f"    y_true (from data) = {y_true:.6f}")
        print(f"    |y(w_pred) - y(w_true)| = {abs(y_from_w_pred - y_from_w_true):.6f}")
        print(f"    相对误差 = {abs(y_from_w_pred - y_from_w_true) / (abs(y_from_w_true) + 1e-8) * 100:.2f}%")
        
        # 分析为什么y接近
        if task_id == 0:  # 线性回归
            print(f"\n  分析 (Task 1: y = w^T x):")
            print(f"    w_true^T x = {np.dot(w_true, x):.6f}")
            print(f"    w_pred^T x = {np.dot(w_pred, x):.6f}")
            print(f"    (w_pred - w_true)^T x = {np.dot(w_diff, x):.6f}")
        elif task_id == 1:  # 排序线性回归
            x_sorted = np.sort(x)
            print(f"\n  分析 (Task 2: y = w^T sort(x)):")
            print(f"    x_sorted = {x_sorted}")
            print(f"    w_true^T x_sorted = {np.dot(w_true, x_sorted):.6f}")
            print(f"    w_pred^T x_sorted = {np.dot(w_pred, x_sorted):.6f}")
        elif task_id == 2:  # Softmax线性回归
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            x_softmax = exp_x / np.sum(exp_x)
            dim = len(x)
            scale_factor = dim / np.sqrt(2.0)
            print(f"\n  分析 (Task 3: y = (dim/sqrt(2)) * w^T softmax(x)):")
            print(f"    x_softmax = {x_softmax}")
            print(f"    w_true^T softmax(x) = {np.dot(w_true, x_softmax):.6f}")
            print(f"    w_pred^T softmax(x) = {np.dot(w_pred, x_softmax):.6f}")
            print(f"    scale_factor = {scale_factor:.6f}")
        elif task_id == 3:  # 平方距离
            print(f"\n  分析 (Task 4: y = ||x - w||^2):")
            print(f"    ||x - w_true||^2 = {np.sum((x - w_true)**2):.6f}")
            print(f"    ||x - w_pred||^2 = {np.sum((x - w_pred)**2):.6f}")


def main():
    parser = argparse.ArgumentParser(description='详细分析W预测器的预测结果')
    parser.add_argument('--test_result', type=str, required=True,
                        help='W预测器测试结果文件 (*.pkl)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='要分析的样本数量')
    parser.add_argument('--positions', type=int, nargs='+', default=[9, 19, 29],
                        help='要检查的位置 (从0开始)')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='指定要分析的样本索引 (默认随机选择)')
    
    args = parser.parse_args()
    
    # 加载测试结果
    print(f"加载测试结果: {args.test_result}")
    with open(args.test_result, 'rb') as f:
        data = pickle.load(f)
    
    w_preds = data['w_preds']  # (n_samples, num_exemplars, x_dim)
    w_true = data['w_true']    # (n_samples, x_dim)
    
    # 检查是否有x和y数据
    if 'xs_true' not in data or 'ys_true' not in data or 'task_ids' not in data:
        print("\n❌ 错误: 测试结果中没有保存 xs_true, ys_true 或 task_ids")
        print("    需要重新运行测试（使用更新后的 test_w.py）")
        return
    
    xs_true = data['xs_true']  # (n_samples, num_exemplars, x_dim)
    ys_true = data['ys_true']  # (n_samples, num_exemplars, 1)
    task_ids = data['task_ids']  # (n_samples,)
    
    n_samples, num_exemplars, x_dim = w_preds.shape
    
    print(f"\n✅ 数据加载成功")
    print(f"  样本数: {n_samples}")
    print(f"  位置数: {num_exemplars}")
    print(f"  x维度: {x_dim}")
    
    # 选择要分析的样本
    if args.sample_indices is not None:
        sample_indices = [idx for idx in args.sample_indices[:args.num_samples] if idx < n_samples]
    else:
        sample_indices = np.random.choice(n_samples, size=min(args.num_samples, n_samples), replace=False)
    
    print(f"\n将分析样本: {sample_indices}")
    print(f"检查位置: {args.positions}")
    
    # 分析每个样本
    for sample_idx in sample_indices:
        w_pred_seq = w_preds[sample_idx]  # (num_exemplars, x_dim)
        w_true_val = w_true[sample_idx]   # (x_dim,)
        x_seq = xs_true[sample_idx]       # (num_exemplars, x_dim)
        y_true_seq = ys_true[sample_idx].flatten()  # (num_exemplars,)
        task_id = task_ids[sample_idx]
        
        analyze_sample(sample_idx, w_pred_seq, w_true_val, x_seq, y_true_seq, task_id, args.positions)


if __name__ == '__main__':
    main()

