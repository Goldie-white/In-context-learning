#!/bin/bash
# 批量训练所有实验

# ============================================================================
# exp3: x~N(1,4), w~N(0,1)
# ============================================================================
cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation && \
source venv/bin/activate && \
python train.py --num_exemplars 20 --x_dim 10 --n_epochs 500 --batch_size 64 \
  --n_layers 8 --n_heads 4 --hidden_size 256 \
  --x_distribution_str "normal*2.0+1.0" --w_distribution_str "normal*1.0+0.0" \
  --exp_folder "experiments/exp3_x_N14_w_N01/y_predictor"

cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation && \
source venv/bin/activate && \
python train_w.py --num_exemplars 20 --x_dim 10 --n_epochs 500 --batch_size 64 \
  --n_layers 8 --n_heads 4 --hidden_size 256 \
  --x_distribution_str "normal*2.0+1.0" --w_distribution_str "normal*1.0+0.0" \
  --exp_folder "experiments/exp3_x_N14_w_N01/w_predictor"

# ============================================================================
# exp4: x~N(1,4), w~N(1,4)
# ============================================================================
cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation && \
source venv/bin/activate && \
python train.py --num_exemplars 20 --x_dim 10 --n_epochs 500 --batch_size 64 \
  --n_layers 8 --n_heads 4 --hidden_size 256 \
  --x_distribution_str "normal*2.0+1.0" --w_distribution_str "normal*2.0+1.0" \
  --exp_folder "experiments/exp4_x_N14_w_N14/y_predictor"

cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation && \
source venv/bin/activate && \
python train_w.py --num_exemplars 20 --x_dim 10 --n_epochs 500 --batch_size 64 \
  --n_layers 8 --n_heads 4 --hidden_size 256 \
  --x_distribution_str "normal*2.0+1.0" --w_distribution_str "normal*2.0+1.0" \
  --exp_folder "experiments/exp4_x_N14_w_N14/w_predictor"

# ============================================================================
# exp5: x~N(0,1), w 为固定值（由种子42生成）
# ============================================================================

# 训练 Y predictor
cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation && \
source venv/bin/activate && \
python train.py --num_exemplars 20 --x_dim 10 --n_epochs 500 --batch_size 64 \
  --n_layers 8 --n_heads 4 --hidden_size 256 \
  --x_distribution_str "normal*1.0+0.0" --w_distribution_str "fixed*42+0.0" \
  --exp_folder "experiments/exp5_x_N01_w_fixed/y_predictor"

# 训练 W predictor
cd /root/autodl-tmp/datasets/project/zhengjunkan/simulation && \
source venv/bin/activate && \
python train_w.py --num_exemplars 20 --x_dim 10 --n_epochs 500 --batch_size 64 \
  --n_layers 8 --n_heads 4 --hidden_size 256 \
  --x_distribution_str "normal*1.0+0.0" --w_distribution_str "fixed*42+0.0" \
  --exp_folder "experiments/exp5_x_N01_w_fixed/w_predictor"