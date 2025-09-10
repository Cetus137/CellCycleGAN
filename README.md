# CycleGAN for Fluorescent Image Generation

This project implements a CycleGAN model for generating fluorescent microscopy images from synthetic masks. The model learns bidirectional mappings between synthetic segmentation masks and real fluorescent images.

## Overview

CycleGAN (Cycle-Consistent Adversarial Networks) enables unpaired image-to-image translation. In this implementation:
- **Domain A**: Single-channel synthetic masks/segmentation images
- **Domain B**: Single-channel real fluorescent microscopy images

The model learns to:
1. Generate realistic fluorescent images from synthetic masks (A → B)
2. Generate masks from fluorescent images (B → A)
3. Maintain cycle consistency (A → B → A and B → A → B)

## Features

- PyTorch implementation of CycleGAN
- **Single-channel image support** (grayscale masks ↔ grayscale fluorescent)
- Support for fluorescent microscopy image generation
- Automatic grayscale conversion for input images
- Customizable network architectures
- Training and inference scripts
- Data preprocessing utilities
- Tensorboard logging
- Model checkpointing

## Requirements

- **Python**: 3.8 - 3.13 (tested with Python 3.13.7)
- **CUDA**: Optional but recommended for GPU acceleration
- **Memory**: At least 8GB RAM (16GB+ recommended for training)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cycleGAN.git
cd cycleGAN

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: This implementation has been tested with Python 3.13.7 and PyTorch 2.8.0. All dependencies install successfully.

## Data Structure

```
data/
├── trainA/          # Synthetic masks for training
├── trainB/          # Fluorescent images for training
├── testA/           # Synthetic masks for testing
└── testB/           # Fluorescent images for testing
```

## Usage
```

## Model Architecture

- **Generator**: ResNet-based architecture with skip connections
- **Discriminator**: PatchGAN discriminator
- **Loss Functions**: 
  - Adversarial loss
  - Cycle consistency loss
  - Identity loss (optional)

## Configuration

Modify `config.py` to adjust:
- Network architectures
- Training hyperparameters
- Data preprocessing settings
- Loss function weights

## Results

The model generates realistic fluorescent images that maintain structural consistency with input masks while adding realistic fluorescent characteristics.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
