#!/bin/bash

# ============================================================================
# ## LEN_VARIANCE: 序列长度的变异性对泛化性的影响
# ============================================================================

# ============================================================================
# GPU显存限制（最多24GB，避免占用过多资源）
# ============================================================================
GPU_TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MEM_LIMIT=24576  # 24GB (in MB)
GPU_MEM_FRACTION=$(python3 -c "print(min(0.90, $GPU_MEM_LIMIT / $GPU_TOTAL_MEM))")
export XLA_PYTHON_CLIENT_MEM_FRACTION=$GPU_MEM_FRACTION
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# JAX性能优化
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"  # 移除不兼容的 triton_softmax_fusion 标志
export JAX_ENABLE_X64=false  # 使用float32加快速度
export TF_CPP_MIN_LOG_LEVEL=2  # 减少日志输出

# 动态并行配置：根据GPU显存自动调整
# 检测当前GPU可用显存
GPU_FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "当前GPU可用显存: ${GPU_FREE_MEM}MB"

# 根据可用显存动态设置并行数
if [ "$GPU_FREE_MEM" -ge 24576 ]; then
    MAX_PARALLEL_JOBS=3  # 显存充足：3个任务并行（每个约8GB）
    echo "✅ 显存充足(>24GB)，启用3任务并行模式"
elif [ "$GPU_FREE_MEM" -ge 16384 ]; then
    MAX_PARALLEL_JOBS=2  # 显存中等：2个任务并行（每个约8GB）
    echo "⚠️  显存中等(16-24GB)，启用2任务并行模式"
else
    MAX_PARALLEL_JOBS=1  # 显存不足：串行执行
    echo "❌ 显存不足(<16GB)，使用串行模式"
fi
echo ""

# 跳过已训练模型配置
SKIP_EXISTING=true  # true: 跳过已训练的模型; false: 重新训练所有模型

# 计算每个任务的显存分配
PER_JOB_MEM_LIMIT=$((GPU_MEM_LIMIT / MAX_PARALLEL_JOBS))
PER_JOB_MEM_FRACTION=$(python3 -c "print(min(0.90, $PER_JOB_MEM_LIMIT / $GPU_TOTAL_MEM))")
PER_JOB_MEM_GB=$(python3 -c "print(f'{$GPU_TOTAL_MEM * $PER_JOB_MEM_FRACTION / 1024:.1f}')")

echo "======================================================================"
echo "GPU显存配置:"
echo "  - 总显存: ${GPU_TOTAL_MEM} MB (~$((GPU_TOTAL_MEM/1024)) GB)"
echo "  - 显存上限: ${GPU_MEM_LIMIT} MB (24 GB)"
echo "  - 并行任务数: ${MAX_PARALLEL_JOBS}"
echo "  - 每个任务显存: ${PER_JOB_MEM_LIMIT} MB (~${PER_JOB_MEM_GB} GB)"
echo "  - 总显存使用: $(python3 -c "print(f'{${PER_JOB_MEM_GB} * ${MAX_PARALLEL_JOBS}:.1f}')") GB"
echo "======================================================================"
echo ""

# ============================================================================
# 配置参数
# ============================================================================

# 基础配置
X_DIM=6
MIN_NUM_EXEMPLARS=24
MAX_NUM_EXEMPLARS=60
NUM_EXEMPLARS=60
N_EPOCHS=2000
BATCH_SIZE=64
N_LAYERS=12
N_HEADS=8
HIDDEN_SIZE=512
NOISE_STD=0.2
DROPOUT_RATE=0.1
ATTENTION_DROPOUT_RATE=0.08
SEED_TRAIN=0
SEED_TEST=123
N_TEST_SAMPLES=200
TEST_NUM_EXEMPLARS=150
N_MC_SAMPLES=100

# 任务概率配置（训练和测试使用相同的概率）
# 格式: "prob0_prob1_prob2_prob3"
# 可以添加多个配置，用空格分隔
TRAIN_PROBS=(
    "1.0_0.0_0.0_0.0"
    "0.0_1.0_0.0_0.0"
    "0.0_0.0_1.0_0.0"
    "0.0_0.0_0.0_1.0"
)

# 预测器类型配置
# 格式: "预测器类型:训练脚本:测试脚本:输出文件名前缀"
PREDICTOR_CONFIGS=(
    "W_pred_loss_W:train_w_loss_w.py:test_w.py:w_analysis.pkl"
    "W_pred:train_w.py:test_w.py:w_analysis.pkl"
    "Y_pred:train.py:test.py:y_analysis.pkl"
)

# ============================================================================
# 辅助函数：解析概率字符串
# ============================================================================

parse_probs() {
    local prob_str=$1
    IFS='_' read -ra PROB_ARRAY <<< "$prob_str"
    PROB0=${PROB_ARRAY[0]}
    PROB1=${PROB_ARRAY[1]}
    PROB2=${PROB_ARRAY[2]}
    PROB3=${PROB_ARRAY[3]}
}

# 并行任务管理
running_jobs=0

# 等待直到有空闲槽位
wait_for_slot() {
    while [ $running_jobs -ge $MAX_PARALLEL_JOBS ]; do
        wait -n  # 等待任意一个后台任务完成
        running_jobs=$((running_jobs - 1))
    done
}

# 获取预测器类型对应的日志前缀
get_pred_prefix() {
    local pred_type=$1
    case "$pred_type" in
        "Y_pred")
            echo "[Y-Pred]"
            ;;
        "W_pred")
            echo "[W-Pred]"
            ;;
        "W_pred_loss_W")
            echo "[W-Pred-by-W-Loss]"
            ;;
        *)
            echo "[$pred_type]"
            ;;
    esac
}

# 检查模型是否已训练完成（基于checkpoint的epoch数）
check_if_trained() {
    local exp_folder=$1
    local checkpoint_dir="${exp_folder}/ckpt"
    
    # 检查checkpoint目录是否存在
    if [ ! -d "$checkpoint_dir" ]; then
        return 1  # checkpoint目录不存在
    fi
    
    # 获取所有checkpoint目录，提取step数
    local max_step=0
    for ckpt in "${checkpoint_dir}"/checkpoint_*; do
        if [ -d "$ckpt" ]; then
            # 从checkpoint_200000中提取200000
            local step=$(basename "$ckpt" | sed 's/checkpoint_//')
            if [[ "$step" =~ ^[0-9]+$ ]]; then
                if [ "$step" -gt "$max_step" ]; then
                    max_step=$step
                fi
            fi
        fi
    done
    
    # 计算目标总步数：N_EPOCHS * n_iter_per_epoch (默认100)
    local n_iter_per_epoch=100
    local target_steps=$((N_EPOCHS * n_iter_per_epoch))
    
    # 如果最大step达到或超过目标步数，认为训练完成
    if [ "$max_step" -ge "$target_steps" ]; then
        return 0  # 已完整训练
    fi
    
    return 1  # 未训练或训练未完成
}

# 启动后台训练任务（每个任务使用分配的显存份额，带日志前缀）
start_background_train() {
    local prefix=$1
    shift  # 移除第一个参数，剩下的是实际命令
    
    wait_for_slot
    # 计算每个任务应该使用的显存（24GB / 并行数）
    PER_JOB_MEM_LIMIT=$((GPU_MEM_LIMIT / MAX_PARALLEL_JOBS))
    PER_JOB_MEM_FRACTION=$(python3 -c "print(min(0.90, $PER_JOB_MEM_LIMIT / $GPU_TOTAL_MEM))")
    # 为每个任务单独设置显存限制，并为输出添加前缀
    (XLA_PYTHON_CLIENT_MEM_FRACTION=$PER_JOB_MEM_FRACTION "$@" 2>&1 | sed "s/^/$prefix /") &
    running_jobs=$((running_jobs + 1))
}

# ============================================================================
# 训练阶段：启用 Dropout
# ============================================================================

echo "=========================================="
echo "开始训练（启用 Dropout）"
echo "=========================================="
echo ""

# 遍历所有概率配置
for prob_str in "${TRAIN_PROBS[@]}"; do
    parse_probs "$prob_str"
    prob_dir="prob_${prob_str}"
    
    echo "----------------------------------------"
    echo "训练配置: $prob_dir"
    echo "  概率: T1=$PROB0, T2=$PROB1, T3=$PROB2, T4=$PROB3"
    echo "----------------------------------------"
    echo ""
    
    # ========================================================================
    # 滚动并行训练：可变长度 + 固定长度（任务完成即启动下一个）
    # ========================================================================
    echo "  🚀 开始滚动并行训练"
    echo "     - 可变长度（${MIN_NUM_EXEMPLARS}-${MAX_NUM_EXEMPLARS}）: Y_pred, W_pred, W_pred_loss_W"
    echo "     - 固定长度（${NUM_EXEMPLARS}）: Y_pred, W_pred, W_pred_loss_W"
    echo "     - 最多 ${MAX_PARALLEL_JOBS} 个任务同时运行"
    echo ""
    
    # 可变长度训练：遍历所有预测器类型
    for pred_config in "${PREDICTOR_CONFIGS[@]}"; do
        IFS=':' read -ra CONFIG <<< "$pred_config"
        pred_type=${CONFIG[0]}
        train_script=${CONFIG[1]}
        test_script=${CONFIG[2]}
        output_prefix=${CONFIG[3]}
        
        # 获取日志前缀
        prefix=$(get_pred_prefix "$pred_type")
        
        # 构建实验文件夹路径
        exp_folder="experiments/LEN_VARIANCE/num_${MIN_NUM_EXEMPLARS}_to_${MAX_NUM_EXEMPLARS}/${pred_type}/${prob_dir}"
        
        # 检查是否需要跳过已训练的模型
        if [ "$SKIP_EXISTING" = true ] && check_if_trained "$exp_folder"; then
            echo "    $prefix ⏭️  跳过可变长度训练 (模型已存在)"
            echo "        路径: $exp_folder"
        else
            echo "    $prefix 启动可变长度训练 [滚动并行]"
            echo "        路径: $exp_folder"
            
            # 使用后台任务滚动并行训练（wait_for_slot会自动等待空闲槽位）
            start_background_train "$prefix" python "$train_script" \
                --seed $SEED_TRAIN \
                --x_dim $X_DIM \
                --n_epochs $N_EPOCHS \
                --batch_size $BATCH_SIZE \
                --min_num_exemplars $MIN_NUM_EXEMPLARS \
                --max_num_exemplars $MAX_NUM_EXEMPLARS \
                --n_layers $N_LAYERS \
                --n_heads $N_HEADS \
                --hidden_size $HIDDEN_SIZE \
                --exp_folder "$exp_folder" \
                --prob0 $PROB0 \
                --prob1 $PROB1 \
                --prob2 $PROB2 \
                --prob3 $PROB3 \
                --noise_std $NOISE_STD \
                --dropout_rate $DROPOUT_RATE \
                --attention_dropout_rate $ATTENTION_DROPOUT_RATE
        fi
        echo ""
    done
    
    # 固定长度训练：遍历所有预测器类型
    for pred_config in "${PREDICTOR_CONFIGS[@]}"; do
        IFS=':' read -ra CONFIG <<< "$pred_config"
        pred_type=${CONFIG[0]}
        train_script=${CONFIG[1]}
        test_script=${CONFIG[2]}
        output_prefix=${CONFIG[3]}
        
        # 获取日志前缀
        prefix=$(get_pred_prefix "$pred_type")
        
        # 构建实验文件夹路径
        exp_folder="experiments/LEN_VARIANCE/num_${NUM_EXEMPLARS}/${pred_type}/${prob_dir}"
        
        # 检查是否需要跳过已训练的模型
        if [ "$SKIP_EXISTING" = true ] && check_if_trained "$exp_folder"; then
            echo "    $prefix ⏭️  跳过固定长度训练 (模型已存在)"
            echo "        路径: $exp_folder"
        else
            echo "    $prefix 启动固定长度训练 [滚动并行]"
            echo "        路径: $exp_folder"
            
            # 使用后台任务滚动并行训练（wait_for_slot会自动等待空闲槽位）
            start_background_train "$prefix" python "$train_script" \
                --seed $SEED_TRAIN \
                --x_dim $X_DIM \
                --n_epochs $N_EPOCHS \
                --batch_size $BATCH_SIZE \
                --num_exemplars $NUM_EXEMPLARS \
                --n_layers $N_LAYERS \
                --n_heads $N_HEADS \
                --hidden_size $HIDDEN_SIZE \
                --exp_folder "$exp_folder" \
                --prob0 $PROB0 \
                --prob1 $PROB1 \
                --prob2 $PROB2 \
                --prob3 $PROB3 \
                --noise_std $NOISE_STD \
                --dropout_rate $DROPOUT_RATE \
                --attention_dropout_rate $ATTENTION_DROPOUT_RATE
        fi
        echo ""
    done
    
    # 等待当前概率配置的所有训练任务完成
    echo "  ⏳ 等待当前概率配置 ($prob_dir) 的所有训练任务完成..."
    wait
    running_jobs=0  # 重置计数
    echo "  ✅ 训练完成"
    echo ""
done

echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""

# ============================================================================
# 测试阶段：MC Dropout 不确定性估计
# ============================================================================

echo "=========================================="
echo "开始 MC Dropout 测试"
echo "=========================================="
echo ""

# 遍历所有概率配置
for prob_str in "${TRAIN_PROBS[@]}"; do
    parse_probs "$prob_str"
    prob_dir="prob_${prob_str}"
    
    echo "----------------------------------------"
    echo "测试配置: $prob_dir"
    echo "  概率: T1=$PROB0, T2=$PROB1, T3=$PROB2, T4=$PROB3"
    echo "----------------------------------------"
    echo ""
    
    # ========================================================================
    # 滚动并行测试：可变长度 + 固定长度（任务完成即启动下一个）
    # ========================================================================
    echo "  🚀 开始滚动并行测试"
    echo "     - 可变长度（${MIN_NUM_EXEMPLARS}-${MAX_NUM_EXEMPLARS}）: Y_pred, W_pred, W_pred_loss_W"
    echo "     - 固定长度（${NUM_EXEMPLARS}）: Y_pred, W_pred, W_pred_loss_W"
    echo "     - 最多 ${MAX_PARALLEL_JOBS} 个任务同时运行"
    echo ""
    
    # 可变长度测试：遍历所有预测器类型
    for pred_config in "${PREDICTOR_CONFIGS[@]}"; do
        IFS=':' read -ra CONFIG <<< "$pred_config"
        pred_type=${CONFIG[0]}
        train_script=${CONFIG[1]}
        test_script=${CONFIG[2]}
        output_prefix=${CONFIG[3]}
        
        # 获取日志前缀
        prefix=$(get_pred_prefix "$pred_type")
        
        # 构建路径
        base_path="num_${MIN_NUM_EXEMPLARS}_to_${MAX_NUM_EXEMPLARS}/${pred_type}/${prob_dir}"
        checkpoint_dir="experiments/LEN_VARIANCE/${base_path}/ckpt"
        output_file="test_results/LEN_VARIANCE/${base_path}/${output_prefix}"
        
        echo "    $prefix 启动可变长度测试 [滚动并行]"
        echo "        检查点: $checkpoint_dir"
        echo "        输出: $output_file"
        
        # 使用后台任务滚动并行测试（wait_for_slot会自动等待空闲槽位）
        start_background_train "$prefix" python "$test_script" \
            --checkpoint_dir "$checkpoint_dir" \
            --output_file "$output_file" \
            --test_prob0 $PROB0 \
            --test_prob1 $PROB1 \
            --test_prob2 $PROB2 \
            --test_prob3 $PROB3 \
            --n_test_samples $N_TEST_SAMPLES \
            --test_num_exemplars $TEST_NUM_EXEMPLARS \
            --seed $SEED_TEST \
            --use_mc_dropout \
            --n_mc_samples $N_MC_SAMPLES
        echo ""
    done
    
    # 固定长度测试：遍历所有预测器类型
    for pred_config in "${PREDICTOR_CONFIGS[@]}"; do
        IFS=':' read -ra CONFIG <<< "$pred_config"
        pred_type=${CONFIG[0]}
        train_script=${CONFIG[1]}
        test_script=${CONFIG[2]}
        output_prefix=${CONFIG[3]}
        
        # 获取日志前缀
        prefix=$(get_pred_prefix "$pred_type")
        
        # 构建路径
        base_path="num_${NUM_EXEMPLARS}/${pred_type}/${prob_dir}"
        checkpoint_dir="experiments/LEN_VARIANCE/${base_path}/ckpt"
        output_file="test_results/LEN_VARIANCE/${base_path}/${output_prefix}"
        
        echo "    $prefix 启动固定长度测试 [滚动并行]"
        echo "        检查点: $checkpoint_dir"
        echo "        输出: $output_file"
        
        # 使用后台任务滚动并行测试（wait_for_slot会自动等待空闲槽位）
        start_background_train "$prefix" python "$test_script" \
            --checkpoint_dir "$checkpoint_dir" \
            --output_file "$output_file" \
            --test_prob0 $PROB0 \
            --test_prob1 $PROB1 \
            --test_prob2 $PROB2 \
            --test_prob3 $PROB3 \
            --n_test_samples $N_TEST_SAMPLES \
            --test_num_exemplars $TEST_NUM_EXEMPLARS \
            --seed $SEED_TEST \
            --use_mc_dropout \
            --n_mc_samples $N_MC_SAMPLES
        echo ""
    done
    
    # 等待当前概率配置的所有测试任务完成
    echo "  ⏳ 等待当前概率配置 ($prob_dir) 的所有测试任务完成..."
    wait
    running_jobs=0  # 重置计数
    echo "  ✅ 测试完成"
    echo ""
done

echo "=========================================="
echo "MC Dropout 测试完成！"
echo "=========================================="
echo ""
echo "结果已保存到 test_results/LEN_VARIANCE/ 目录"
echo ""
echo "生成的文件结构："
echo "  test_results/LEN_VARIANCE/"
echo "    ├── num_${MIN_NUM_EXEMPLARS}_to_${MAX_NUM_EXEMPLARS}/  (可变长度模型)"
echo "    └── num_${NUM_EXEMPLARS}/  (固定长度模型)"
echo ""
for pred_config in "${PREDICTOR_CONFIGS[@]}"; do
    IFS=':' read -ra CONFIG <<< "$pred_config"
    pred_type=${CONFIG[0]}
    output_prefix=${CONFIG[3]}
    echo "    ${pred_type}/prob_*/${output_prefix}"
done