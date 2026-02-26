# SEEM-OCTA: Geometry-Informed Interactive Refinement for OCTA Vessel Segmentation
## 🔥 Overview

<p align="center">
  <img src="assets/Picture1.png" width="800"/>
</p>
---
SEEM-OCTA is an interactive segmentation framework for retinal OCTA vessel extraction...

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **SEEM-OCTA with GIIR (Geometry-Informed Interactive Refinement)** for interactive vessel segmentation in Optical Coherence Tomography Angiography (OCTA) images.

## 🎯 Key Features

- **Parameter-Efficient Fine-Tuning**: Uses LoRA adaptation with only ~1.2M trainable parameters (1.25% of total)
- **Interactive Refinement**: Achieves 0.91+ Dice score with just 3-4 clicks on average
- **3-Channel Input**: Utilizes FULL, ILM_OPL, and OPL_BM projections for robust segmentation
- **Two Training Strategies**:
  - **GIIR**: Geometry-Informed deterministic click selection using distance transforms
  - **Random**: SEEM baseline with random click selection


## 📁 Project Structure

```
seem-octa/
├── README.md
├── requirements.txt
├── setup.py
├── assets/
│   ├── Picture1.png
├── configs/
│   └── seem/
│       └── focall_unicl_lang_demo.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lora.py              # LoRA implementation
│   │   ├── losses.py            # Loss functions (Dice, Focal, Tversky)
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── model_utils.py       # Model loading utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # OCTADataset class
│   │   └── point_generators.py  # Point generation strategies
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py              # Base click strategy interface
│   │   ├── giir.py              # GIIR deterministic clicks
│   │   └── random_clicks.py     # SEEM random clicks
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Unified trainer class
│   │   └── config.py            # Training configuration
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py         # Evaluation and comparison
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py          # Visualization utilities
├── scripts/
│   ├── train.py                 # Unified training script
│   ├── evaluate.py              # Evaluation script
│   └── compare_strategies.py    # Academic comparison
├── modeling/                    # SEEM model files (from original repo)
└── utils/                       # SEEM utilities (from original repo)
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/seem-octa.git
cd seem-octa
```

### 2. Create conda environment
```bash
conda create -n seem-octa python=3.9 -y
conda activate seem-octa
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download SEEM pretrained weights
```bash
wget https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt
```

## 📊 Data Preparation

Organize your OCTA-500 dataset as follows:
```
octa/
├── OCTA_3mm/
│   ├── OCTA(FULL)/
│   ├── OCTA(ILM_OPL)/
│   └── OCTA(OPL_BM)/
├── OCTA_6mm/
│   ├── OCTA(FULL)/
│   ├── OCTA(ILM_OPL)/
│   └── OCTA(OPL_BM)/
└── Label/
    └── Label/
        └── GT_LargeVessel/
```

## 🏋️ Training

### Train with GIIR Strategy (Recommended)
```bash
python scripts/train.py --strategy giir --epochs 50 --batch-size 2
```

### Train with Random Strategy (Baseline)
```bash
python scripts/train.py --strategy random --epochs 50 --batch-size 2
```

### Training Options
```bash
python scripts/train.py --help
```

## 📈 Evaluation

### Run Academic Comparison
```bash
python scripts/compare_strategies.py \
    --weights checkpoints/best_model_merged.pth \
    --num-samples 20 \
    --num-seeds 5
```

### Evaluate Single Model
```bash
python scripts/evaluate.py \
    --weights checkpoints/best_model.pth \
    --strategy giir
```

## 📋 Results

### OCTA-500 3mm Dataset

| Method | Dice | IoU | clDice | Avg Clicks | Params (M) |
|--------|------|-----|--------|------------|------------|
| SEEM Baseline | 0.8756 | 0.7821 | 0.8412 | 5.58 | 31.2 |
| **GIIR (Ours)** | **0.9109** | **0.8389** | **0.8847** | **3.35** | **1.25** |

### Key Findings

1. **40% Click Reduction**: GIIR reduces required clicks from 5.58 to 3.35
2. **Parameter Efficiency**: Only 1.25% trainable parameters via LoRA
3. **Basin of Attraction Effect**: Geometry-informed training improves performance even with random inference

## 🔬 Method Overview

### GIIR (Geometry-Informed Interactive Refinement)

1. **Deterministic Click Selection**: Uses distance transform to find optimal click locations
2. **Oracle Mask Selection**: Selects best mask based on GT overlap during training
3. **Local Refinement**: Combines global structure with click-guided local corrections

### Training Strategy

```
For each sample:
    1. Get initial prediction (0 clicks)
    2. For click in 1..10:
        a. Find largest error region (FN or FP)
        b. Select click point using distance transform (GIIR) or random (baseline)
        c. Forward pass with accumulated clicks
        d. Compute loss and backpropagate
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{jaafar2025seemocta,
  title={SEEM-OCTA: Parameter-Efficient Interactive Vessel Segmentation with Geometry-Informed Refinement},
  author={Jaafar, et al.},
  journal={arXiv preprint},
  year={2025}
}
```

## 🙏 Acknowledgments

- [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) for the base model
- [OCTA-500](https://ieee-dataport.org/open-access/octa-500) dataset
- Iran University of Science and Technology

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
