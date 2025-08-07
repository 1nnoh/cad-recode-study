# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Setup environment
./setup.sh
source .venv/bin/activate

# Run demo
jupyter notebook demo.ipynb

# Install in development mode
uv pip install -e .
```

## Key Commands

### Environment Setup
- `./setup.sh` - Full environment setup with CUDA detection
- `source .venv/bin/activate` - Activate virtual environment
- `uv pip install -e .` - Install package in development mode

### Development Commands
- `jupyter notebook demo.ipynb` - Run the main demo
- `pytest` - Run tests (if available)
- `black .` - Format code
- `isort .` - Sort imports

## Architecture Overview

### Core System
CAD-Recode is a transformer-based system that generates CadQuery Python code from 3D point clouds using a Qwen2-1.5B backbone with custom point cloud encoding.

### Main Components
- **CADRecode**: Main model class extending `Qwen2ForCausalLM` in demo.ipynb
- **FourierPointEncoder**: Point cloud encoding layer using Fourier features
- **Point cloud processing**: 256 points → 51 features (3 coords + 48 Fourier)

### Data Flow
```
Mesh → Sample 256 points → Normalize [-1,1] → Fourier encoding → Qwen2 → CadQuery code
```

## Key Files

- `demo.ipynb` - Primary inference demo with complete pipeline
- `setup.sh` - Environment setup script
- `pyproject.toml` - Project dependencies and configuration
- `data/` - Training dataset (~1M samples) and validation data
- `checkpoint/` - Empty directory for model storage

## Dependencies

### Core ML Stack
- PyTorch 2.5.1 with CUDA 12.4 support
- Transformers 4.47.1 (Hugging Face)
- PyTorch3D (may need manual installation)
- Flash Attention 2 (CUDA only)

### 3D Processing
- Open3D, Trimesh, CadQuery for 3D operations
- scikit-image for image processing
- SciPy for spatial data structures

## Usage Pattern

### Model Loading
```python
from transformers import AutoTokenizer
from demo import CADRecode

model = CADRecode.from_pretrained('filapro/cad-recode-v1.5')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B')
```

### Dataset Format
- Training: Python files with CadQuery expressions
- Input coordinates: Integers from -100 to +100
- Validation: ~1K samples in data/ directory

## Important Notes

- **PyTorch3D compatibility**: May require manual installation due to wheel availability issues
- **Memory management**: CadQuery has memory leaks in long-running processes
- **Code safety**: Generated code needs sandboxed execution environment
- **Precision limit**: Integer coordinates only (-100 to +100 range)

## Development Setup

The project uses uv for package management. If PyTorch3D fails to install, skip it initially and install core dependencies:

```bash
uv pip install torch transformers numpy scipy matplotlib jupyter
uv pip install open3d trimesh cadquery
```