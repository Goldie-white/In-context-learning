#!/bin/bash

# --- 1. 环境和基础设置 ---
BASE_DIR="/root/autodl-tmp/datasets/project/zhengjunkan/simulation"
cd "$BASE_DIR" || { echo "Error: Directory not found"; exit 1; }
source venv/bin/activate
echo "Starting automated batch training for Y-Pred (Task 1) and W-Pred (Task 2)..."
echo "=================================================="

# --- 2. 基础参数 (保持不变) ---
NUM_EXEMPLARS=30
X_DIM=10
BATCH_SIZE=64
N_LAYERS=8
N_HEADS=4
HIDDEN_SIZE=256
BASE_EPOCHS=1000

# --- 3. 实验配置数组 (保持不变) ---
declare -a PROB_CONFIGS=(
    "0.2,0.2,0.2,0.4"     # All Mix
)

# --- 4. 辅助函数：检查浮点数是否大于阈值 (不依赖bc) ---
is_nonzero() {
    awk -v val="$1" 'BEGIN { exit (val > 0.0001) ? 0 : 1 }'
}

# --- 5. 循环执行训练 ---
for PROB_STR in "${PROB_CONFIGS[@]}"; do
    # 解析概率值
    IFS=',' read -r P0 P1 P2 P3 <<< "$PROB_STR"
    
    # 统计非零参数个数
    NON_ZERO_COUNT=0
    is_nonzero "$P0" && NON_ZERO_COUNT=$((NON_ZERO_COUNT + 1))
    is_nonzero "$P1" && NON_ZERO_COUNT=$((NON_ZERO_COUNT + 1))
    is_nonzero "$P2" && NON_ZERO_COUNT=$((NON_ZERO_COUNT + 1))
    is_nonzero "$P3" && NON_ZERO_COUNT=$((NON_ZERO_COUNT + 1))
    
    # 验证至少有一个非零概率
    if [ "$NON_ZERO_COUNT" -eq 0 ]; then
        echo "❌ Error: All probabilities are zero for config ($P0, $P1, $P2, $P3)"
        continue
    fi
    
    # 计算 n_epochs
    N_EPOCHS=$((BASE_EPOCHS * NON_ZERO_COUNT))
    
    
    # =======================================================
    # 任务 A: Y 预测 (实验 1) - 使用 train.py
    # =======================================================
    PYTHON_SCRIPT="train.py"
    # 文件夹名使用 Y_pred 标识
    EXP_FOLDER_Y="experiments/Y_pred/prob_${P0}_${P1}_${P2}_${P3}"
    
    echo "--- Running Experiment A (Y-PRED): ${EXP_FOLDER_Y} ---"
    echo "PROBS: ($P0, $P1, $P2, $P3), NON_ZERO: $NON_ZERO_COUNT, EPOCHS: $N_EPOCHS"
    
    python "$PYTHON_SCRIPT" \
        --num_exemplars "$NUM_EXEMPLARS" \
        --x_dim "$X_DIM" \
        --n_epochs "$N_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --n_layers "$N_LAYERS" \
        --n_heads "$N_HEADS" \
        --hidden_size "$HIDDEN_SIZE" \
        --exp_folder "$EXP_FOLDER_Y" \
        --prob0 "$P0" \
        --prob1 "$P1" \
        --prob2 "$P2" \
        --prob3 "$P3"
    
    if [ $? -ne 0 ]; then
        echo "❌ Y-PRED Training failed for config ($P0, $P1, $P2, $P3)"
    else
        echo "✅ Y-PRED Training completed successfully"
    fi
    
    # =======================================================
    # 任务 B: W 预测 (实验 2) - 使用 train_w.py
    # =======================================================
    PYTHON_SCRIPT_W="train_w.py"
    # 文件夹名使用 W_pred 标识
    EXP_FOLDER_W="experiments/W_pred/prob_${P0}_${P1}_${P2}_${P3}"
    
    echo "--- Running Experiment B (W-PRED): ${EXP_FOLDER_W} ---"
    echo "PROBS: ($P0, $P1, $P2, P3), NON_ZERO: $NON_ZERO_COUNT, EPOCHS: $N_EPOCHS"
    
    python "$PYTHON_SCRIPT_W" \
        --num_exemplars "$NUM_EXEMPLARS" \
        --x_dim "$X_DIM" \
        --n_epochs "$N_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --n_layers "$N_LAYERS" \
        --n_heads "$N_HEADS" \
        --hidden_size "$HIDDEN_SIZE" \
        --exp_folder "$EXP_FOLDER_W" \
        --prob0 "$P0" \
        --prob1 "$P1" \
        --prob2 "$P2" \
        --prob3 "$P3"
    
    if [ $? -ne 0 ]; then
        echo "❌ W-PRED Training failed for config ($P0, $P1, $P2, $P3)"
    else
        echo "✅ W-PRED Training completed successfully"
    fi
    
    echo "--------------------------------------------------"
done

echo "Automated batch training for all 16 experiments complete!"

python train.py \
  --num_exemplars 40 --x_dim 10 --n_epochs 3000 --batch_size 64 \
  --n_layers 16 --n_heads 4 --hidden_size 512 \
  --exp_folder "experiments/aug_Y_pred/prob_0.0_0.0_0.0_1.0" \
  --prob0 0.0 --prob1 0.0 --prob2 0.0 --prob3 1.0
  
python train_w.py \
  --num_exemplars 40 --x_dim 10 --n_epochs 3000 --batch_size 64 \
  --n_layers 16 --n_heads 4 --hidden_size 512 \
  --exp_folder "experiments/aug_W_pred/prob_0.0_0.0_0.0_1.0" \
  --prob0 0.0 --prob1 0.0 --prob2 0.0 --prob3 1.0