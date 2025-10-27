#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据采样器的逻辑
验证：
1. 任务类型在同一序列内是否一致
2. w在同一序列内是否共享
3. x在不同位置是否i.i.d.
4. y的计算是否符合任务定义
"""

import sys
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/datasets/project/zhengjunkan/simulation')

from incontext import sampler_lib


def test_task_1_standard():
    """测试Task 1: y = w^T x"""
    print("\n" + "="*70)
    print("测试 Task 1: y = w^T x (标准线性回归)")
    print("="*70)
    
    sampler = sampler_lib.Sampler(
        length=5,  # 5个(x,y)对
        dim=3,
        hidden_size=10,
        task_probs=[1.0, 0.0, 0.0, 0.0],  # 纯Task 1
        noise_std=0.0
    )
    
    # 生成2个样本
    seqs, coefficients, xs, ys = sampler.sample(n=2)
    
    print(f"\n生成了 {len(coefficients)} 个序列")
    
    for i in range(len(coefficients)):
        print(f"\n序列 {i+1}:")
        w = coefficients[i]
        print(f"  w = {w}")
        
        for j in range(5):
            x = xs[i][j]  # xs shape: (n_samples, length, dim)
            y = ys[i][j][0]  # ys shape: (n_samples, length, 1)
            y_expected = np.dot(w, x)
            
            print(f"  位置 {j+1}: x={x}, y={y:.6f}, 期望y={y_expected:.6f}, 误差={abs(y-y_expected):.10f}")
        
        # 验证
        errors = [abs(ys[i][j][0] - np.dot(w, xs[i][j])) for j in range(5)]
        max_error = max(errors)
        if max_error < 1e-6:
            print(f"  ✅ Task 1 验证通过 (最大误差: {max_error:.10e})")
        else:
            print(f"  ❌ Task 1 验证失败 (最大误差: {max_error})")


def test_task_2_sorted():
    """测试Task 2: y = w^T sort(x)"""
    print("\n" + "="*70)
    print("测试 Task 2: y = w^T sort(x) (排序线性回归)")
    print("="*70)
    
    sampler = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[0.0, 1.0, 0.0, 0.0],  # 纯Task 2
        noise_std=0.0
    )
    
    seqs, coefficients, xs, ys = sampler.sample(n=2)
    
    print(f"\n生成了 {len(coefficients)} 个序列")
    
    for i in range(len(coefficients)):
        print(f"\n序列 {i+1}:")
        w = coefficients[i]
        print(f"  w = {w}")
        
        for j in range(5):
            x = xs[i][j]
            x_sorted = np.sort(x)
            y = ys[i][j][0]
            y_expected = np.dot(w, x_sorted)
            
            print(f"  位置 {j+1}: x={x}, x_sorted={x_sorted}, y={y:.6f}, 期望y={y_expected:.6f}")
        
        # 验证
        errors = [abs(ys[i][j][0] - np.dot(w, np.sort(xs[i][j]))) for j in range(5)]
        max_error = max(errors)
        if max_error < 1e-6:
            print(f"  ✅ Task 2 验证通过 (最大误差: {max_error:.10e})")
        else:
            print(f"  ❌ Task 2 验证失败 (最大误差: {max_error})")


def test_task_3_softmax():
    """测试Task 3: y = (dim / sqrt(2)) * w^T softmax(x)"""
    print("\n" + "="*70)
    print("测试 Task 3: y = (dim / sqrt(2)) * w^T softmax(x) (缩放softmax线性回归)")
    print("="*70)
    
    sampler = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[0.0, 0.0, 1.0, 0.0],  # 纯Task 3
        noise_std=0.0
    )
    
    seqs, coefficients, xs, ys = sampler.sample(n=2)
    
    print(f"\n生成了 {len(coefficients)} 个序列")
    
    for i in range(len(coefficients)):
        print(f"\n序列 {i+1}:")
        w = coefficients[i]
        dim = len(w)
        scale_factor = dim / np.sqrt(2.0)
        print(f"  w = {w}, dim={dim}, scale_factor={scale_factor:.6f}")
        
        for j in range(5):
            x = xs[i][j]
            # 计算softmax
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            x_softmax = exp_x / np.sum(exp_x)
            y = ys[i][j][0]
            y_expected = scale_factor * np.dot(w, x_softmax)
            
            print(f"  位置 {j+1}: x={x}, softmax(x)={x_softmax}, y={y:.6f}, 期望y={y_expected:.6f}")
        
        # 验证
        def compute_softmax(x):
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x)
        
        errors = [abs(ys[i][j][0] - scale_factor * np.dot(w, compute_softmax(xs[i][j]))) for j in range(5)]
        max_error = max(errors)
        if max_error < 1e-6:
            print(f"  ✅ Task 3 验证通过 (最大误差: {max_error:.10e})")
        else:
            print(f"  ❌ Task 3 验证失败 (最大误差: {max_error})")


def test_task_4_distance():
    """测试Task 4: y = ||x - w||^2"""
    print("\n" + "="*70)
    print("测试 Task 4: y = ||x - w||^2 (平方距离)")
    print("="*70)
    
    sampler = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[0.0, 0.0, 0.0, 1.0],  # 纯Task 4
        noise_std=0.0
    )
    
    seqs, coefficients, xs, ys = sampler.sample(n=2)
    
    print(f"\n生成了 {len(coefficients)} 个序列")
    
    for i in range(len(coefficients)):
        print(f"\n序列 {i+1}:")
        w = coefficients[i]
        print(f"  w = {w}")
        
        for j in range(5):
            x = xs[i][j]
            diff = x - w
            y = ys[i][j][0]
            y_expected = np.sum(diff ** 2)
            
            print(f"  位置 {j+1}: x={x}, y={y:.6f}, 期望y={y_expected:.6f}")
        
        # 验证
        errors = [abs(ys[i][j][0] - np.sum((xs[i][j] - w)**2)) for j in range(5)]
        max_error = max(errors)
        if max_error < 1e-6:
            print(f"  ✅ Task 4 验证通过 (最大误差: {max_error:.10e})")
        else:
            print(f"  ❌ Task 4 验证失败 (最大误差: {max_error})")


def test_mixed_tasks():
    """测试混合任务：验证每个序列使用同一个任务"""
    print("\n" + "="*70)
    print("测试混合任务 (prob=[0.3, 0.3, 0.2, 0.2])")
    print("验证：同一序列内任务类型一致，不同序列可能使用不同任务")
    print("="*70)
    
    np.random.seed(42)  # 设置随机种子以便复现
    
    sampler = sampler_lib.Sampler(
        length=3,  # 3个(x,y)对
        dim=4,
        hidden_size=10,
        task_probs=[0.3, 0.3, 0.2, 0.2],  # 混合任务
        noise_std=0.0
    )
    
    # 生成10个样本，看看任务分布
    seqs, coefficients, xs, ys = sampler.sample(n=10)
    
    print(f"\n生成了 {len(coefficients)} 个序列，每个序列 {xs.shape[1]} 个(x,y)对")
    
    def identify_task(w, x0, y0):
        """根据y的值判断使用的是哪个任务"""
        
        # 计算4种任务的期望y值
        dim = len(w)
        scale_factor = dim / np.sqrt(2.0)
        
        y_task1 = np.dot(w, x0)
        y_task2 = np.dot(w, np.sort(x0))
        x_softmax = sampler_lib.apply_softmax(x0.reshape(1, -1))[0]
        y_task3 = scale_factor * np.dot(w, x_softmax)
        y_task4 = np.sum((x0 - w)**2)
        
        errors = [
            abs(y0 - y_task1),
            abs(y0 - y_task2),
            abs(y0 - y_task3),
            abs(y0 - y_task4)
        ]
        
        task_id = np.argmin(errors)
        return task_id, errors[task_id]
    
    task_counts = [0, 0, 0, 0]
    
    for i in range(10):
        w = coefficients[i]
        
        # 识别任务类型（用第一个位置）
        task_id, error = identify_task(w, xs[i][0], ys[i][0][0])
        task_counts[task_id] += 1
        
        # 验证序列内所有位置使用同一个任务
        all_same_task = True
        for j in range(3):
            task_j, _ = identify_task(w, xs[i][j], ys[i][j][0])
            if task_j != task_id:
                all_same_task = False
                break
        
        task_name = ['Task1(w^Tx)', 'Task2(w^Tsort(x))', 'Task3(w^Tsoftmax(x))', 'Task4(||x-w||^2)'][task_id]
        status = "✅" if all_same_task else "❌"
        print(f"  序列 {i+1}: {task_name} {status} (误差: {error:.10e})")
    
    print(f"\n任务分布统计:")
    print(f"  Task 1: {task_counts[0]}/10 ({task_counts[0]/10*100:.1f}%)")
    print(f"  Task 2: {task_counts[1]}/10 ({task_counts[1]/10*100:.1f}%)")
    print(f"  Task 3: {task_counts[2]}/10 ({task_counts[2]/10*100:.1f}%)")
    print(f"  Task 4: {task_counts[3]}/10 ({task_counts[3]/10*100:.1f}%)")


def test_w_consistency():
    """测试w在同一序列内的一致性"""
    print("\n" + "="*70)
    print("测试 w 在同一序列内的一致性")
    print("="*70)
    
    sampler = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[1.0, 0.0, 0.0, 0.0],
        noise_std=0.0
    )
    
    seqs, coefficients, xs, ys = sampler.sample(n=3)
    
    print(f"\n生成了 {len(coefficients)} 个序列")
    for i in range(len(coefficients)):
        w = coefficients[i]
        print(f"\n序列 {i+1}: w = {w}")
        print(f"  验证：所有(x,y)对应该使用同一个w")
        
        # 对于Task 1，反推w（如果x线性无关）
        # 这里只是验证y确实是用同一个w计算的
        for j in range(5):
            y_computed = np.dot(w, xs[i][j])
            y_actual = ys[i][j][0]
            print(f"    位置 {j+1}: y_computed={y_computed:.6f}, y_actual={y_actual:.6f}, 一致: {abs(y_computed-y_actual)<1e-6}")


def test_x_independence():
    """测试x在不同位置的独立性（i.i.d.）"""
    print("\n" + "="*70)
    print("测试 x 在不同位置的独立性 (i.i.d.)")
    print("="*70)
    
    sampler = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[1.0, 0.0, 0.0, 0.0],
        noise_std=0.0
    )
    
    seqs, coefficients, xs, ys = sampler.sample(n=2)
    
    print(f"\n生成了 {len(coefficients)} 个序列，每个序列 {xs.shape[1]} 个x")
    
    for i in range(len(coefficients)):
        print(f"\n序列 {i+1}:")
        print(f"  检查：不同位置的x应该不同")
        
        for j in range(5):
            print(f"    位置 {j+1}: x = {xs[i][j]}")
        
        # 检查是否有完全相同的x
        all_different = True
        for j1 in range(5):
            for j2 in range(j1+1, 5):
                if np.allclose(xs[i][j1], xs[i][j2]):
                    all_different = False
                    print(f"  ⚠️ 位置{j1+1}和位置{j2+1}的x相同（这在随机采样下极不可能）")
        
        if all_different:
            print(f"  ✅ 所有位置的x都不同")


def test_noise():
    """测试噪声的独立性"""
    print("\n" + "="*70)
    print("测试噪声的独立性 (noise_std=0.5)")
    print("="*70)
    
    np.random.seed(123)
    
    sampler_no_noise = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[1.0, 0.0, 0.0, 0.0],
        noise_std=0.0
    )
    
    sampler_with_noise = sampler_lib.Sampler(
        length=5,
        dim=3,
        hidden_size=10,
        task_probs=[1.0, 0.0, 0.0, 0.0],
        noise_std=0.5
    )
    
    # 生成相同的样本（使用相同的w）
    w = np.array([1.0, 2.0, 3.0])
    
    _, _, xs_no_noise, ys_no_noise = sampler_no_noise.sample(n=1, coefficients=w)
    _, _, xs_with_noise, ys_with_noise = sampler_with_noise.sample(n=1, coefficients=w)
    
    print(f"\n无噪声 vs 有噪声 (noise_std=0.5):")
    print(f"使用固定的 w = {w}")
    
    # 因为x是随机的，我们不能直接比较，所以我们只验证有噪声时y的值会偏离理论值
    print(f"\n统计信息：")
    
    # 重新生成多次来统计噪声
    noise_samples = []
    for _ in range(100):
        _, _, xs_temp, ys_temp = sampler_with_noise.sample(n=1, coefficients=w)
        for j in range(5):
            y_theoretical = np.dot(w, xs_temp[0][j])
            y_actual = ys_temp[0][j][0]
            noise = y_actual - y_theoretical
            noise_samples.append(noise)
    
    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)
    
    print(f"  噪声的均值: {noise_mean:.6f} (应该接近0)")
    print(f"  噪声的标准差: {noise_std:.6f} (应该接近0.5)")
    
    if abs(noise_mean) < 0.1 and abs(noise_std - 0.5) < 0.1:
        print(f"  ✅ 噪声统计特性正确")
    else:
        print(f"  ❌ 噪声统计特性异常")


def test_probability_validation():
    """测试概率验证逻辑"""
    print("\n" + "="*70)
    print("测试概率和验证逻辑")
    print("="*70)
    
    # 测试1：概率和为1（应该成功）
    try:
        sampler = sampler_lib.Sampler(
            length=3,
            dim=3,
            hidden_size=10,
            task_probs=[0.25, 0.25, 0.25, 0.25]
        )
        print("\n✅ 测试1通过: [0.25, 0.25, 0.25, 0.25] 和为1.0")
    except ValueError as e:
        print(f"\n❌ 测试1失败: {e}")
    
    # 测试2：概率和不为1（应该失败）
    try:
        sampler = sampler_lib.Sampler(
            length=3,
            dim=3,
            hidden_size=10,
            task_probs=[0.3, 0.3, 0.3, 0.3]  # 和为1.2
        )
        print("\n❌ 测试2失败: [0.3, 0.3, 0.3, 0.3] 应该抛出错误但没有")
    except ValueError as e:
        print(f"\n✅ 测试2通过: 正确检测到概率和不为1")
        print(f"   错误信息: {e}")
    
    # 测试3：概率数量不对（应该失败）
    try:
        sampler = sampler_lib.Sampler(
            length=3,
            dim=3,
            hidden_size=10,
            task_probs=[0.5, 0.5]  # 只有2个
        )
        print("\n❌ 测试3失败: [0.5, 0.5] 应该抛出错误但没有")
    except ValueError as e:
        print(f"\n✅ 测试3通过: 正确检测到概率数量不对")
        print(f"   错误信息: {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("数据采样器测试")
    print("="*70)
    
    # 运行所有测试
    test_task_1_standard()
    test_task_2_sorted()
    test_task_3_softmax()
    test_task_4_distance()
    test_mixed_tasks()
    test_w_consistency()
    test_x_independence()
    test_noise()
    test_probability_validation()
    
    print("\n" + "="*70)
    print("所有测试完成！")
    print("="*70)

