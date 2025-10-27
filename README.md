# In-Context Learning for Linear Regression Variants

This project implements Transformer models for in-context learning on various linear regression tasks, comparing Y-predictor (direct prediction) and W-predictor (coefficient prediction) approaches.

## Tasks

The model is trained on four types of linear regression tasks:

1. **Task 1 (T1)**: Standard Linear Regression  
   `y = w^T x`

2. **Task 2 (T2)**: Sorted Linear Regression  
   `y = w^T sort(x)`

3. **Task 3 (T3)**: Scaled Softmax Regression  
   `y = (dim/√2) × w^T softmax(x)`

4. **Task 4 (T4)**: Squared Distance Regression  
   `y = -||x - w||^2`

## Model Architectures

### Y-Predictor
- Directly predicts the output `y` from input sequences
- Architecture: Transformer with causal masking
- Output: Single scalar prediction per position

### W-Predictor
- Predicts the coefficient vector `w`, then computes `y` based on task type
- Architecture: Transformer with task-adaptive computation layer
- Output: Coefficient vector (dimension = x_dim)

## Project Structure

```
simulation/
├── incontext/              # Core library code
│   ├── sampler_lib.py     # Data generation for all tasks
│   ├── predictor_flax.py  # Y-predictor model (Flax/JAX)
│   └── predictor_flax_w.py # W-predictor model (Flax/JAX)
├── train.py               # Training script for Y-predictor
├── train_w.py             # Training script for W-predictor
├── test.py                # Testing script for Y-predictor
├── test_w.py              # Testing script for W-predictor
├── visualize.py           # Visualization of test results
├── run.sh                 # Batch training script
├── test.sh                # Batch testing script
└── visualize.sh           # Batch visualization script
```

## Usage

### Training

Train on a single task (e.g., Task 1):
```bash
python train.py --num_exemplars 40 --x_dim 10 --n_epochs 3000 --batch_size 64 \
  --n_layers 16 --n_heads 4 --hidden_size 512 \
  --exp_folder "experiments/Y_pred/prob_1.0_0.0_0.0_0.0" \
  --prob0 1.0 --prob1 0.0 --prob2 0.0 --prob3 0.0
```

Train on mixed tasks:
```bash
python train_w.py --num_exemplars 40 --x_dim 10 --n_epochs 3000 --batch_size 64 \
  --n_layers 16 --n_heads 4 --hidden_size 512 \
  --exp_folder "experiments/W_pred/prob_0.25_0.25_0.25_0.25" \
  --prob0 0.25 --prob1 0.25 --prob2 0.25 --prob3 0.25
```

Batch training:
```bash
bash run.sh
```

### Testing

Test a trained model:
```bash
python test.py --checkpoint_dir "experiments/Y_pred/prob_1.0_0.0_0.0_0.0" \
  --test_prob0 1.0 --test_prob1 0.0 --test_prob2 0.0 --test_prob3 0.0 \
  --n_samples 1000 --seed 42
```

Batch testing:
```bash
bash test.sh
```

### Visualization

Generate plots from test results:
```bash
python visualize.py
```

Or run batch visualization:
```bash
bash visualize.sh
```

## Requirements

- Python 3.8+
- JAX with CUDA support
- Flax
- NumPy
- Matplotlib
- See `requirements.txt` for full dependencies

## Key Features

- **Task-Adaptive W-Predictor**: Automatically adjusts computation based on task type
- **Consistent Data Generation**: Fixed seed for reproducible testing
- **Comprehensive Visualization**: Loss curves, MSE, cosine similarity, and predictor comparisons
- **Mixed Task Training**: Train on multiple task distributions simultaneously

## Results Structure

```
test_results/
├── Y_pred/
│   └── prob_X_X_X_X/
│       └── test_on_TX.pkl
└── W_pred/
    └── prob_X_X_X_X/
        └── test_on_TX.pkl
```

## Notes

- Models trained on pure tasks are tested only on their corresponding task
- Models trained on mixed tasks are tested on all four pure tasks
- GPU is required for training (CPU inference is possible but slow)
- Large model checkpoints are excluded from git (see `.gitignore`)

## License

[Add your license here]

## Citation

[Add citation information if applicable]

