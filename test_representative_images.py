#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive TIFF representative image saving functionality
"""

import torch
import os
from models.cycle_gan_model import CycleGANModel
from data.dataset import create_dataloader
from config import Config
from utils import save_representative_images_as_tiff

def test_representative_images():
    """Test the representative image saving functionality"""
    
    # Initialize configuration
    config = Config(
        dataroot='./data',
        name='test_representative',
        batch_size=1,
        image_size=256
    )
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and len(config.gpu_ids) > 0 else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loader
    try:
        train_loader = create_dataloader(
            config.dataroot, 
            phase='train',
            batch_size=config.batch_size,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            image_size=config.image_size
        )
        print(f'✓ Data loader created successfully with {len(train_loader.dataset)} images')
    except Exception as e:
        print(f'✗ Error creating data loader: {e}')
        return
    
    # Initialize model
    try:
        model = CycleGANModel(config)
        print('✓ CycleGAN model initialized successfully')
    except Exception as e:
        print(f'✗ Error initializing model: {e}')
        return
    
    # Test representative image saving
    try:
        # Create test output directory
        test_dir = './test_representative_output'
        os.makedirs(test_dir, exist_ok=True)
        
        # Get a sample batch
        sample_data = next(iter(train_loader))
        print(f'✓ Sample data loaded: A shape={sample_data["A"].shape}, B shape={sample_data["B"].shape}')
        
        # Set model to evaluation mode and process
        model.eval()
        with torch.no_grad():
            model.set_input(sample_data)
            model.forward()
            visuals = model.get_current_visuals()
            
            print(f'✓ Model forward pass completed')
            print(f'Available visuals: {list(visuals.keys())}')
            
            # Save representative images
            save_representative_images_as_tiff(visuals, test_dir, epoch=0, prefix="test")
            
            print(f'✓ Representative images saved to: {test_dir}')
            
            # List saved files
            saved_files = [f for f in os.listdir(test_dir) if f.endswith(('.tif', '.png'))]
            print(f'Saved files ({len(saved_files)}):')
            for f in sorted(saved_files):
                filepath = os.path.join(test_dir, f)
                size = os.path.getsize(filepath)
                print(f'  - {f} ({size:,} bytes)')
                
    except Exception as e:
        print(f'✗ Error during representative image test: {e}')
        import traceback
        traceback.print_exc()
        return
    
    print('\n✅ Representative image saving test completed successfully!')
    print(f'Check the images in: {os.path.abspath(test_dir)}')


if __name__ == '__main__':
    test_representative_images()
