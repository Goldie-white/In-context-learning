#!/bin/bash
# 可视化 test.sh 的所有测试结果

cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation
source venv/bin/activate

echo "=========================================="
echo "可视化测试结果"
echo "=========================================="
echo ""
echo "扫描 test_results/ 目录并生成可视化..."
echo ""

# 直接运行 visualize.py，它会自动扫描 test_results/ 目录
python visualize.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 所有可视化完成！"
    echo "=========================================="
    echo ""
    echo "可视化结果保存位置："
    echo "  visualization_results/Y_pred/prob_{P0}_{P1}_{P2}_{P3}/"
    echo "  visualization_results/W_pred/prob_{P0}_{P1}_{P2}_{P3}/"
    echo "  visualization_results/Comparison/prob_{P0}_{P1}_{P2}_{P3}_compare_T#.png"
    echo ""
    echo "示例："
    echo "  - visualization_results/Y_pred/prob_1.0_0.0_0.0_0.0/loss_vs_position.png"
    echo "  - visualization_results/W_pred/prob_1.0_0.0_0.0_0.0/y_loss_vs_position.png"
    echo "  - visualization_results/Comparison/prob_1.0_0.0_0.0_0.0_compare_T1.png"
else
    echo ""
    echo "❌ 可视化失败！请检查错误信息。"
    exit 1
fi
