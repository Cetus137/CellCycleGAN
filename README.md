# CycleGAN for Fluorescent Image Generation

This project implements a CycleGAN model for generating fluorescent microscopy images from synthetic masks. The model learns bidirectional mappings between synthetic segmentation masks and real fluorescent images.

## Overview

CycleGAN (Cycle-Consistent Adversarial Networks) enables unpaired image-to-image translation. In this implementation:
- **Domain A**: Synthetic masks/segmentation images
- **Domain B**: Real fluorescent microscopy images

The model learns to:
1. Generate realistic fluorescent images from synthetic masks (A → B)
2. Generate masks from fluorescent images (B → A)
3. Maintain cycle consistency (A → B → A and B → A → B)

## Features

- PyTorch implementation of CycleGAN
- Support for fluorescent image generation
- Customizable network architectures
- Training and inference scripts
- Data preprocessing utilities
- Tensorboard logging
- Model checkpointing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cycleGAN.git
cd cycleGAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Structure

```
data/
├── trainA/          # Synthetic masks for training
├── trainB/          # Fluorescent images for training
├── testA/           # Synthetic masks for testing
└── testB/           # Fluorescent images for testing
```

## Usage

### Training
```bash
python train.py --dataroot ./data --name fluorescent_cyclegan --model cycle_gan
```

### Testing/Inference
```bash
python test.py --dataroot ./data --name fluorescent_cyclegan --model cycle_gan --phase test
```

### Generate from synthetic masks
```bash
python generate.py --input_path ./data/testA --output_path ./results --model_path ./checkpoints
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

## Citation

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
