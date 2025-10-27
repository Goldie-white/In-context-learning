# Environment Requirements

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (required for training)
  - Tested on: Tesla V100, A100, RTX series
  - CUDA Compute Capability: 6.0+ recommended
- **RAM**: 16GB+ recommended
- **Disk**: 10GB+ for code and model checkpoints (venv excluded)

### Software
- **OS**: Linux (tested on Ubuntu/CentOS)
  - Windows/macOS: May work but not tested
- **Python**: 3.12.x (tested on 3.12.5)
  - Python 3.8-3.11 may work but not guaranteed
- **CUDA**: 12.x (CUDA 12.1+ recommended)
  - Driver version: 535.xx or newer

---

## Installation Methods

### Method 1: Full Reproducibility (Recommended for Production)

Use exact package versions from development environment:

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install exact versions
pip install -r requirements_full.txt
```

**Pros**: 
- Exact same environment as development
- Guaranteed reproducibility

**Cons**: 
- Requires Python 3.12.x
- Some packages are platform-specific (Linux)

---

### Method 2: Flexible Installation (Cross-Platform)

Use minimal requirements with latest compatible versions:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

**Pros**: 
- Works with Python 3.8-3.12
- More flexible across platforms

**Cons**: 
- Package versions may differ
- Potential compatibility issues

---

## Package Details

### Core Dependencies (from requirements_full.txt)
- `jax==0.8.0` + `jaxlib==0.8.0` - Deep learning framework
- `flax==0.12.0` - Neural network library
- `numpy==2.3.4` - Numerical computing
- `matplotlib==3.10.7` - Plotting
- `tensorflow==2.20.0` - For some utilities
- `optax==0.2.6` - Optimization library

### CUDA Dependencies (15 packages, ~4GB)
- `nvidia-cublas-cu12==12.9.1.4`
- `nvidia-cudnn-cu12==9.14.0.64`
- `nvidia-cuda-runtime-cu12==12.9.79`
- ... and 12 more nvidia-* packages

**Note**: These are automatically installed with `jax[cuda12]` and are Linux-only.

---

## Verification

After installation, verify the environment:

```bash
python -c "import jax; print('JAX version:', jax.__version__)"
python -c "import jax; print('Available devices:', jax.devices())"
```

Expected output (with GPU):
```
JAX version: 0.8.0
Available devices: [cuda(id=0)]
```

If no GPU:
```
Available devices: [cpu(id=0)]
```

---

## Troubleshooting

### Issue: "No CUDA-capable device is detected"
**Solution**: 
1. Check GPU availability: `nvidia-smi`
2. Ensure CUDA drivers are installed
3. Verify JAX can see GPU: `python -c "import jax; print(jax.devices())"`

### Issue: "ModuleNotFoundError: No module named 'jax'"
**Solution**: 
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall: `pip install -r requirements_full.txt`

### Issue: Package version conflicts
**Solution**: 
- Use `requirements_full.txt` for exact versions
- Or use Python 3.12.x to match development environment

---

## CPU-Only Installation (Not Recommended for Training)

If you only need inference or testing without GPU:

```bash
pip install jax[cpu] flax optax numpy matplotlib
```

**Warning**: Training large Transformer models on CPU is extremely slow and may cause OOM.

---

## Development Notes

- Virtual environment size: ~7GB (with CUDA packages)
- Some packages duplicate in `lib/` and `lib64/` (NVIDIA's packaging behavior)
- Training requires GPU; testing can run on CPU (slow)

