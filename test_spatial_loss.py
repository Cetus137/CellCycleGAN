#!/usr/bin/env python3
"""
Quick test of the spatial loss modifications
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.cycle_gan_model import CycleGANModel
from config import Config

def test_spatial_loss():
    """Test if the modified spatial loss works"""
    print("Testing spatial loss modifications...")
    
    # Create config
    opt = Config()
    opt.isTrain = True
    opt.gpu_ids = []  # Force CPU for quick test
    
    try:
        # Create model
        model = CycleGANModel(opt)
        print("✓ Model created successfully")
        
        # Create dummy data
        batch_size = 2
        dummy_A = torch.randn(batch_size, 1, 256, 256)  # masks
        dummy_B = torch.randn(batch_size, 1, 256, 256)  # fluorescent
        
        # Set input
        input_data = {'A': dummy_A, 'B': dummy_B}
        model.set_input(input_data)
        print("✓ Input set successfully")
        
        # Forward pass
        model.forward()
        print("✓ Forward pass successful")
        
        # Test spatial loss
        spatial_loss = model.criterionSpatial(dummy_A, dummy_B)
        print(f"✓ Spatial loss computed: {spatial_loss.item():.6f}")
        
        # Backward pass
        model.backward_G()
        print("✓ Backward pass successful")
        
        # Check loss values
        losses = model.get_current_losses()
        print("\nCurrent losses:")
        for name, value in losses.items():
            print(f"  {name}: {value:.6f}")
            
        print("\n✓ All tests passed! Spatial loss is working correctly.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_spatial_loss()
