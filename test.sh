#!/bin/bash

# --- 1. 环境和基础设置 ---
BASE_DIR="/root/autodl-tmp/datasets/project/zhengjunkan/simulation"
cd "$BASE_DIR" || { echo "Error: Directory not found"; exit 1; }
source venv/bin/activate
echo "Starting automated batch testing for Y-Pred and W-Pred..."
echo "=================================================="

# --- 2. 测试参数 ---
N_TEST_SAMPLES=1000  # 固定测试样本数

# --- 3. 实验配置数组 (与run.sh一致) ---
declare -a PROB_CONFIGS=(
    "1.0,0.0,0.0,0.0"     # Pure T1
    "0.0,1.0,0.0,0.0"     # Pure T2
    "0.0,0.0,1.0,0.0"     # Pure T3
    "0.0,0.0,0.0,1.0"     # Pure T4
    "0.25,0.25,0.25,0.25" # All Mix
    "0.4,0.2,0.2,0.2"     # All Mix
    "0.2,0.4,0.2,0.2"     # All Mix
)

# --- 4. 辅助函数：判断是否为纯任务 ---
is_pure_task() {
    local p0=$1 p1=$2 p2=$3 p3=$4
    # 检查是否只有一个概率接近1.0，其他接近0.0
    awk -v p0="$p0" -v p1="$p1" -v p2="$p2" -v p3="$p3" 'BEGIN {
        count = 0
        if (p0 > 0.99) count++
        if (p1 > 0.99) count++
        if (p2 > 0.99) count++
        if (p3 > 0.99) count++
        exit (count == 1) ? 0 : 1
    }'
}

# --- 5. 测试任务配置 (4种纯任务) ---
declare -a ALL_TEST_TASKS=(
    "1.0,0.0,0.0,0.0:T1"  # 纯任务1
    "0.0,1.0,0.0,0.0:T2"  # 纯任务2
    "0.0,0.0,1.0,0.0:T3"  # 纯任务3
    "0.0,0.0,0.0,1.0:T4"  # 纯任务4
)

# --- 6. 循环测试 ---
for PROB_STR in "${PROB_CONFIGS[@]}"; do
    # 解析训练配置
    IFS=',' read -r P0 P1 P2 P3 <<< "$PROB_STR"
    
    echo ""
    echo "=========================================================="
    echo "Testing models trained on: prob=($P0, $P1, $P2, $P3)"
    echo "=========================================================="
    
    # 确定测试任务列表
    if is_pure_task "$P0" "$P1" "$P2" "$P3"; then
        # 纯任务：只测试对应的任务
        CURRENT_PROB="${P0},${P1},${P2},${P3}"
        if [ "$P0" == "1.0" ]; then
            TEST_TASKS=("1.0,0.0,0.0,0.0:T1")
            echo "📌 Pure Task 1 detected → Testing only on T1"
        elif [ "$P1" == "1.0" ]; then
            TEST_TASKS=("0.0,1.0,0.0,0.0:T2")
            echo "📌 Pure Task 2 detected → Testing only on T2"
        elif [ "$P2" == "1.0" ]; then
            TEST_TASKS=("0.0,0.0,1.0,0.0:T3")
            echo "📌 Pure Task 3 detected → Testing only on T3"
        elif [ "$P3" == "1.0" ]; then
            TEST_TASKS=("0.0,0.0,0.0,1.0:T4")
            echo "📌 Pure Task 4 detected → Testing only on T4"
        fi
    else
        # 混合任务：测试所有4种纯任务
        TEST_TASKS=("${ALL_TEST_TASKS[@]}")
        echo "📌 Mixed Tasks detected → Testing on all 4 pure tasks (T1, T2, T3, T4)"
    fi
    
    # =======================================================
    # 测试 Y 预测器 (test.py)
    # =======================================================
    TRAIN_DIR_Y="experiments/Y_pred/prob_${P0}_${P1}_${P2}_${P3}"
    TEST_OUTPUT_DIR_Y="test_results/Y_pred/prob_${P0}_${P1}_${P2}_${P3}"
    CHECKPOINT_DIR_Y="${TRAIN_DIR_Y}/ckpt"
    
    # 检查checkpoint是否存在
    if [ ! -d "$CHECKPOINT_DIR_Y" ]; then
        echo "⚠️  Y-PRED checkpoint not found: $CHECKPOINT_DIR_Y (skipping)"
    else
        echo "--- Testing Y-PRED model: $TRAIN_DIR_Y ---"
        mkdir -p "$TEST_OUTPUT_DIR_Y"
        
        # 对每种测试任务进行测试
        for TEST_TASK_STR in "${TEST_TASKS[@]}"; do
            # 解析测试任务配置
            IFS=':' read -r TASK_PROB TASK_NAME <<< "$TEST_TASK_STR"
            IFS=',' read -r TP0 TP1 TP2 TP3 <<< "$TASK_PROB"
            
            OUTPUT_FILE="${TEST_OUTPUT_DIR_Y}/test_on_${TASK_NAME}.pkl"
            
            echo "  Testing on $TASK_NAME (test_prob=$TP0,$TP1,$TP2,$TP3)..."
            
            # 使用固定种子确保Y和W测试使用相同数据
            TEST_SEED=42
            
            python test.py \
                --checkpoint_dir "$CHECKPOINT_DIR_Y" \
                --output_file "$OUTPUT_FILE" \
                --test_prob0 "$TP0" \
                --test_prob1 "$TP1" \
                --test_prob2 "$TP2" \
                --test_prob3 "$TP3" \
                --n_test_samples "$N_TEST_SAMPLES" \
                --seed "$TEST_SEED" \
                > "${TEST_OUTPUT_DIR_Y}/test_on_${TASK_NAME}.log" 2>&1
            
            if [ $? -ne 0 ]; then
                echo "    ❌ Failed (see ${TEST_OUTPUT_DIR_Y}/test_on_${TASK_NAME}.log)"
            else
                echo "    ✅ Success → $OUTPUT_FILE"
            fi
        done
    fi
    
    # =======================================================
    # 测试 W 预测器 (test_w.py)
    # =======================================================
    TRAIN_DIR_W="experiments/W_pred/prob_${P0}_${P1}_${P2}_${P3}"
    TEST_OUTPUT_DIR_W="test_results/W_pred/prob_${P0}_${P1}_${P2}_${P3}"
    CHECKPOINT_DIR_W="${TRAIN_DIR_W}/ckpt"
    
    # 检查checkpoint是否存在
    if [ ! -d "$CHECKPOINT_DIR_W" ]; then
        echo "⚠️  W-PRED checkpoint not found: $CHECKPOINT_DIR_W (skipping)"
    else
        echo "--- Testing W-PRED model: $TRAIN_DIR_W ---"
        mkdir -p "$TEST_OUTPUT_DIR_W"
        
        # 对每种测试任务进行测试
        for TEST_TASK_STR in "${TEST_TASKS[@]}"; do
            # 解析测试任务配置
            IFS=':' read -r TASK_PROB TASK_NAME <<< "$TEST_TASK_STR"
            IFS=',' read -r TP0 TP1 TP2 TP3 <<< "$TASK_PROB"
            
            OUTPUT_FILE="${TEST_OUTPUT_DIR_W}/test_on_${TASK_NAME}.pkl"
            
            echo "  Testing on $TASK_NAME (test_prob=$TP0,$TP1,$TP2,$TP3)..."
            
            # 使用固定种子确保Y和W测试使用相同数据
            TEST_SEED=42
            
            python test_w.py \
                --checkpoint_dir "$CHECKPOINT_DIR_W" \
                --output_file "$OUTPUT_FILE" \
                --test_prob0 "$TP0" \
                --test_prob1 "$TP1" \
                --test_prob2 "$TP2" \
                --test_prob3 "$TP3" \
                --n_test_samples "$N_TEST_SAMPLES" \
                --seed "$TEST_SEED" \
                > "${TEST_OUTPUT_DIR_W}/test_on_${TASK_NAME}.log" 2>&1
            
            if [ $? -ne 0 ]; then
                echo "    ❌ Failed (see ${TEST_OUTPUT_DIR_W}/test_on_${TASK_NAME}.log)"
            else
                echo "    ✅ Success → $OUTPUT_FILE"
            fi
        done
    fi
    
    echo "--------------------------------------------------"
done

echo ""
echo "=========================================================="
echo "Automated batch testing complete!"
echo "=========================================================="
echo "Results saved in:"
echo "  - test_results/Y_pred/prob_*/"
echo "  - test_results/W_pred/prob_*/"
echo ""
