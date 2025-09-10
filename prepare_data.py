"""
Data preparation script for CycleGAN fluorescent image generation.
Use this script to organize your existing mask and fluorescent image data.
"""

import os
import shutil
import argparse
from pathlib import Path


def prepare_data(mask_dir, fluorescent_dir, output_dir, train_split=0.8):
    """
    Organize mask and fluorescent images for CycleGAN training.
    
    Args:
        mask_dir (str): Directory containing mask images
        fluorescent_dir (str): Directory containing fluorescent images
        output_dir (str): Output directory (should be ./data)
        train_split (float): Fraction of data to use for training
    """
    
    # Create output directories
    train_A_dir = os.path.join(output_dir, 'trainA')
    train_B_dir = os.path.join(output_dir, 'trainB')
    test_A_dir = os.path.join(output_dir, 'testA')
    test_B_dir = os.path.join(output_dir, 'testB')
    
    for dir_path in [train_A_dir, train_B_dir, test_A_dir, test_B_dir]:
        os.makedirs(dir_path, exist_ok=True)
        # Remove .gitkeep files if they exist
        gitkeep_path = os.path.join(dir_path, '.gitkeep')
        if os.path.exists(gitkeep_path):
            os.remove(gitkeep_path)
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(Path(mask_dir).glob(ext))
        mask_files.extend(Path(mask_dir).glob(ext.upper()))
    
    fluorescent_files = []
    for ext in image_extensions:
        fluorescent_files.extend(Path(fluorescent_dir).glob(ext))
        fluorescent_files.extend(Path(fluorescent_dir).glob(ext.upper()))
    
    print(f"Found {len(mask_files)} mask images")
    print(f"Found {len(fluorescent_files)} fluorescent images")
    
    # Sort files for consistent ordering
    mask_files.sort()
    fluorescent_files.sort()
    
    # Split data
    num_train_masks = int(len(mask_files) * train_split)
    num_train_fluorescent = int(len(fluorescent_files) * train_split)
    
    # Copy mask files
    print("Copying mask images...")
    for i, mask_file in enumerate(mask_files):
        if i < num_train_masks:
            dst_dir = train_A_dir
        else:
            dst_dir = test_A_dir
        
        dst_path = os.path.join(dst_dir, mask_file.name)
        shutil.copy2(str(mask_file), dst_path)
    
    # Copy fluorescent files
    print("Copying fluorescent images...")
    for i, fluor_file in enumerate(fluorescent_files):
        if i < num_train_fluorescent:
            dst_dir = train_B_dir
        else:
            dst_dir = test_B_dir
        
        dst_path = os.path.join(dst_dir, fluor_file.name)
        shutil.copy2(str(fluor_file), dst_path)
    
    print(f"\nData preparation complete!")
    print(f"Training masks: {num_train_masks} images in {train_A_dir}")
    print(f"Training fluorescent: {num_train_fluorescent} images in {train_B_dir}")
    print(f"Test masks: {len(mask_files) - num_train_masks} images in {test_A_dir}")
    print(f"Test fluorescent: {len(fluorescent_files) - num_train_fluorescent} images in {test_B_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for CycleGAN training')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing mask images')
    parser.add_argument('--fluorescent_dir', type=str, required=True,
                        help='Directory containing fluorescent images')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training (default: 0.8)')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.mask_dir):
        raise ValueError(f"Mask directory does not exist: {args.mask_dir}")
    if not os.path.exists(args.fluorescent_dir):
        raise ValueError(f"Fluorescent directory does not exist: {args.fluorescent_dir}")
    
    prepare_data(args.mask_dir, args.fluorescent_dir, args.output_dir, args.train_split)


if __name__ == '__main__':
    main()
