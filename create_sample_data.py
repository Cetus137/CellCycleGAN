#!/usr/bin/env python3
"""
Create sample synthetic masks and fluorescent images for CycleGAN training
"""

import os
import numpy as np
from PIL import Image
import argparse
from utils import create_synthetic_masks


def create_sample_fluorescent_images(output_dir, num_images=100, image_size=256):
    """Create sample fluorescent-like images for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} sample fluorescent images...")
    
    for i in range(num_images):
        # Create a fluorescent-like image with random patterns
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add background noise
        background = np.random.normal(20, 10, (image_size, image_size))
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        # Add fluorescent patterns
        num_patterns = np.random.randint(5, 15)
        
        for _ in range(num_patterns):
            # Random bright spots with fluorescent-like colors
            center_x = np.random.randint(20, image_size-20)
            center_y = np.random.randint(20, image_size-20)
            
            # Create gradient pattern
            y, x = np.ogrid[:image_size, :image_size]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Random size and intensity
            max_radius = np.random.randint(10, 40)
            intensity = np.random.randint(100, 255)
            
            # Create fluorescent-like pattern
            pattern = np.exp(-dist**2 / (2 * (max_radius/3)**2)) * intensity
            pattern = np.clip(pattern, 0, 255).astype(np.uint8)
            
            # Choose fluorescent-like color (green, red, or blue dominant)
            color_choice = np.random.choice(['green', 'red', 'blue'])
            if color_choice == 'green':
                image[:, :, 1] = np.maximum(image[:, :, 1], pattern)
                image[:, :, 0] = np.maximum(image[:, :, 0], pattern // 3)
            elif color_choice == 'red':
                image[:, :, 0] = np.maximum(image[:, :, 0], pattern)
                image[:, :, 1] = np.maximum(image[:, :, 1], pattern // 4)
            else:  # blue
                image[:, :, 2] = np.maximum(image[:, :, 2], pattern)
                image[:, :, 1] = np.maximum(image[:, :, 1], pattern // 3)
        
        # Add background to all channels
        for c in range(3):
            image[:, :, c] = np.maximum(image[:, :, c], background)
        
        # Save image
        Image.fromarray(image).save(f'{output_dir}/fluorescent_{i:04d}.png')
    
    print(f"Sample fluorescent images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create sample data for CycleGAN training')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for sample data')
    parser.add_argument('--num_train', type=int, default=100,
                        help='Number of training images per domain')
    parser.add_argument('--num_test', type=int, default=20,
                        help='Number of test images per domain')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of generated images')
    
    args = parser.parse_args()
    
    # Create directories
    train_A_dir = os.path.join(args.output_dir, 'trainA')
    train_B_dir = os.path.join(args.output_dir, 'trainB')
    test_A_dir = os.path.join(args.output_dir, 'testA')
    test_B_dir = os.path.join(args.output_dir, 'testB')
    
    # Create training data
    print("Creating training data...")
    create_synthetic_masks(train_A_dir, args.num_train, args.image_size)
    create_sample_fluorescent_images(train_B_dir, args.num_train, args.image_size)
    
    # Create test data
    print("Creating test data...")
    create_synthetic_masks(test_A_dir, args.num_test, args.image_size)
    create_sample_fluorescent_images(test_B_dir, args.num_test, args.image_size)
    
    print(f"Sample data creation completed!")
    print(f"Training: {args.num_train} images per domain")
    print(f"Testing: {args.num_test} images per domain")
    print(f"Data saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
