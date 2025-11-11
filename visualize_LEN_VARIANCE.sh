#!/bin/bash

# ============================================================================
# LEN_VARIANCE 完整可视化脚本
# ============================================================================
# 此脚本用于可视化 LEN_VARIANCE 测试结果
# 
# 功能：
#   1. 生成标准测试可视化（与 visualize.sh 相同）
#   2. 生成 MC Dropout 特有的不确定性分析可视化
# 
# 使用方法：
#   bash visualize_VARIANCE.sh
# 
# 注意：
#   1. 需要先运行 run_LEN_VARIANCE.sh 生成测试结果
#   2. 确保输入文件路径正确
# ============================================================================

echo "=========================================="
echo "LEN_VARIANCE 完整可视化"
echo "=========================================="
echo ""
echo "步骤1: 标准可视化（使用 visualize.py）"
echo "步骤2: MC Dropout 不确定性可视化（使用 visualize_mc_dropout.py）"
echo ""

# ============================================================================
# 第一部分：标准可视化（使用 visualize.py）
# ============================================================================

echo "=========================================="
echo "第1部分: 标准可视化"
echo "=========================================="
echo ""
echo "使用 visualize.py 生成标准测试可视化图表..."
echo "（Loss曲线、对比分析等）"
echo ""

# 扫描所有 LEN_VARIANCE 测试结果目录
base_dir="test_results/LEN_VARIANCE"

if [ ! -d "$base_dir" ]; then
    echo "⚠️  未找到 LEN_VARIANCE 测试结果目录: $base_dir"
    echo "请先运行 run_LEN_VARIANCE.sh 生成测试结果"
    exit 1
fi

# 使用 test_results 作为 base_dir，这样输出路径会包含 LEN_VARIANCE 前缀
# 例如：visualization_results/LEN_VARIANCE_num_24_to_60/
python visualize.py --input_dir "test_results"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 标准可视化完成！"
    echo ""
    echo "标准可视化结果保存在："
    echo "  visualization_results/LEN_VARIANCE_num_*_to_*/Y_pred/"
    echo "  visualization_results/LEN_VARIANCE_num_*/Y_pred/"
    echo "  visualization_results/LEN_VARIANCE_num_*_to_*/W_pred/"
    echo "  visualization_results/LEN_VARIANCE_num_*/W_pred/"
    echo "  visualization_results/LEN_VARIANCE_num_*_to_*/W_pred_loss_W/"
    echo "  visualization_results/LEN_VARIANCE_num_*/W_pred_loss_W/"
    echo "  visualization_results/LEN_VARIANCE_num_*_to_*/Comparison/"
    echo "  visualization_results/LEN_VARIANCE_num_*/Comparison/"
    echo ""
else
    echo "⚠️  标准可视化失败，继续执行 MC Dropout 可视化..."
    echo ""
fi

# ============================================================================
# 第二部分：MC Dropout 不确定性可视化（使用 visualize_mc_dropout.py）
# ============================================================================

echo "=========================================="
echo "第2部分: MC Dropout 不确定性分析"
echo "=========================================="
echo ""
echo "使用 visualize_mc_dropout.py 生成不确定性分析图表..."
echo "（不确定性曲线、不确定性vs误差、MC采样分布等）"
echo ""

# 使用之前定义的 base_dir（避免重复检查）
# base_dir 已在第一部分定义

# 计数器
total_visualized=0

# 扫描所有长度配置目录（num_*_to_* 和 num_*）
for len_dir in "$base_dir"/num_*; do
    if [ ! -d "$len_dir" ]; then
        continue
    fi
    
    len_name=$(basename "$len_dir")
    echo "📁 处理目录: $len_name"
    echo ""
    
    # 扫描所有 prob_* 目录（从任意预测器类型获取，因为它们都有相同的 prob 配置）
    # 先找到第一个存在的预测器目录来获取 prob 列表
    first_pred_dir=""
    for pred_type in "Y_pred" "W_pred" "W_pred_loss_W"; do
        pred_dir="$len_dir/$pred_type"
        if [ -d "$pred_dir" ]; then
            first_pred_dir="$pred_dir"
            break
        fi
    done
    
    if [ -z "$first_pred_dir" ]; then
        echo "  ⚠️  未找到任何预测器目录"
        continue
    fi
    
    # 扫描所有 prob_* 目录
    for prob_dir in "$first_pred_dir"/prob_*; do
        if [ ! -d "$prob_dir" ]; then
            continue
        fi
        
        prob_name=$(basename "$prob_dir")
        
        # 收集所有预测器的输入文件
        input_files=()
        
        # Y_pred
        y_file="$len_dir/Y_pred/$prob_name/y_analysis.pkl"
        if [ -f "$y_file" ]; then
            input_files+=("$y_file")
        fi
        
        # W_pred
        w_file="$len_dir/W_pred/$prob_name/w_analysis.pkl"
        if [ -f "$w_file" ]; then
            input_files+=("$w_file")
        fi
        
        # W_pred_loss_W
        w_loss_w_file="$len_dir/W_pred_loss_W/$prob_name/w_analysis.pkl"
        if [ -f "$w_loss_w_file" ]; then
            input_files+=("$w_loss_w_file")
        fi
        
        # 检查是否有输入文件
        if [ ${#input_files[@]} -eq 0 ]; then
            echo "  ⚠️  跳过 $prob_name (没有找到任何分析文件)"
            continue
        fi
        
        # 输出目录：保存在对应实验目录的 mc/ 子目录下
        # 路径结构：visualization_results/LEN_VARIANCE_num_24_to_60/mc/prob_*/
        output_dir="visualization_results/LEN_VARIANCE_${len_name}/mc/${prob_name}"
        
        # 执行可视化（传递多个文件，用逗号分隔）
        input_files_str=$(IFS=','; echo "${input_files[*]}")
        echo "  📊 分析: $prob_name (合并 ${#input_files[@]} 个预测器)"
        python visualize_mc_dropout.py \
            --input_files "$input_files_str" \
            --output_dir "$output_dir"
        
        if [ $? -eq 0 ]; then
            echo "     ✅ 完成: $output_dir"
            ((total_visualized++))
        else
            echo "     ❌ 失败"
        fi
        echo ""
    done
done

echo ""
echo "----------------------------------------"
echo "✅ MC Dropout 不确定性分析完成"
echo "   共生成 $total_visualized 组可视化结果"
echo "----------------------------------------"
echo ""

# ============================================================================
# 总结
# ============================================================================

echo "=========================================="
echo "✅ LEN_VARIANCE 完整可视化完成！"
echo "=========================================="
echo ""
echo "生成的可视化结果："
echo ""
echo "【标准可视化】（使用 visualize.py）"
echo "  位置: visualization_results/LEN_VARIANCE_*/"
echo "  内容:"
echo "    - Y_pred/prob_*_y_pred_loss.png          : Y预测器Loss曲线"
echo "    - W_pred/prob_*_w_pred_analysis.png      : W预测器分析（W MSE + 余弦相似度）"
echo "    - W_pred_loss_W/prob_*_w_pred_analysis.png : W预测器Loss_W分析"
echo "    - Comparison/prob_*_compare_*.png        : 预测器对比"
echo ""
echo "【MC Dropout 不确定性分析】（额外的）"
echo "  位置: visualization_results/LEN_VARIANCE_*/mc/"
echo "  内容（每个 prob 配置1张图，合并所有预测器）:"
echo "    - mc/prob_*/mc_dropout_overview.png"
echo "      包含："
echo "        - W Prediction: W_pred 和 W_pred_loss_W 合并"
echo "        - Y Prediction: Y_pred, W_pred, W_pred_loss_W 合并"
echo ""
echo "详细说明："
echo "  - 标准可视化展示模型性能（Loss、MSE、余弦相似度等）"
echo "  - MC Dropout可视化展示预测不确定性，帮助理解模型的置信度"
echo ""
echo "扫描的目录结构："
echo "  test_results/LEN_VARIANCE/"
echo "    ├── num_*_to_*/  (可变长度模型)"
echo "    │   ├── Y_pred/prob_*/"
echo "    │   ├── W_pred/prob_*/"
echo "    │   └── W_pred_loss_W/prob_*/"
echo "    └── num_*/  (固定长度模型)"
echo "        ├── Y_pred/prob_*/"
echo "        ├── W_pred/prob_*/"
echo "        └── W_pred_loss_W/prob_*/"
echo ""

# ============================================================================
# 输出目录结构日志
# ============================================================================

echo "=========================================="
echo "📁 生成的目录结构"
echo "=========================================="
echo ""

if [ -d "visualization_results" ]; then
    echo "visualization_results/"
    echo ""
    
    # 显示每个实验目录的结构
    for dir in visualization_results/LEN_VARIANCE_*; do
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            echo "  $dir_name/"
            
            # 显示标准可视化子目录
            for subdir in "$dir"/*; do
                if [ -d "$subdir" ] && [ "$(basename "$subdir")" != "mc" ]; then
                    count=$(find "$subdir" -name "*.png" 2>/dev/null | wc -l)
                    echo "    ├── $(basename "$subdir")/  ($count 张图)  [标准可视化]"
                fi
            done
            
            # 显示 MC Dropout 可视化目录
            if [ -d "$dir/mc" ]; then
                echo "    └── mc/  [MC Dropout 不确定性分析]"
                prob_count=0
                for prob_dir in "$dir/mc"/prob_*; do
                    if [ -d "$prob_dir" ]; then
                        ((prob_count++))
                        count=$(find "$prob_dir" -name "*.png" 2>/dev/null | wc -l)
                        echo "        ├── $(basename "$prob_dir")/  ($count 张图)"
                    fi
                done
                if [ $prob_count -eq 0 ]; then
                    echo "        (无数据)"
                fi
            fi
            echo ""
        fi
    done
    
    # 统计总文件数
    total_png=$(find visualization_results -name "*.png" 2>/dev/null | wc -l)
    echo "总计: $total_png 张可视化图片"
    echo ""
else
    echo "⚠️  visualization_results/ 目录不存在"
    echo ""
fi

echo "=========================================="
echo ""

