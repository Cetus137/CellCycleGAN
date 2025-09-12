import torch
import os
import argparse
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm

from models import define_G
from data.dataset import get_transforms
from config import Config
from utils import tensor_to_image


def generate_from_masks(input_path, output_path, model_path, direction='AtoB'):
    """Generate fluorescent images from synthetic masks or vice versa"""
    
    # Initialize configuration
    config = Config(isTrain=False)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and len(config.gpu_ids) > 0 else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load generator
    if direction == 'AtoB':
        # Masks to fluorescent images
        generator = define_G(
            config.input_nc, config.output_nc, config.ngf,
            config.netG, config.norm, not config.no_dropout,
            config.init_type, config.init_gain, config.gpu_ids
        )
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint['netG_A_state_dict'])
        print("Loaded generator A->B (masks to fluorescent)")
    else:
        # Fluorescent images to masks
        generator = define_G(
            config.output_nc, config.input_nc, config.ngf,
            config.netG, config.norm, not config.no_dropout,
            config.init_type, config.init_gain, config.gpu_ids
        )
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint['netG_B_state_dict'])
        print("Loaded generator B->A (fluorescent to masks)")
    
    generator.eval()
    
    # Get transforms
    transform = get_transforms(config.image_size, is_train=False)
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_path, ext)))
        image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
    
    print(f'Found {len(image_files)} images to process')
    
    # Generate images
    with torch.no_grad():
        for img_path in tqdm(image_files, desc='Generating'):
            # Load and preprocess image
            image = np.array(Image.open(img_path).convert('L'))
            
            # Apply transforms
            transformed = transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Generate output
            output_tensor = generator(input_tensor)
            
            # Convert to image
            output_image = tensor_to_image(output_tensor[0])
            
            # Save result
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_filename = f'{name}_generated{ext}'
            
            # Handle grayscale images
            if len(output_image.shape) == 2:  # Grayscale
                Image.fromarray(output_image, mode='L').save(
                    os.path.join(output_path, output_filename)
                )
            else:  # RGB
                Image.fromarray(output_image).save(
                    os.path.join(output_path, output_filename)
                )
    
    print(f'Generation completed! Results saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate images using trained CycleGAN')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input images')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save generated images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--direction', type=str, default='AtoB', choices=['AtoB', 'BtoA'],
                        help='Generation direction: AtoB (masks->fluorescent) or BtoA (fluorescent->masks)')
    
    args = parser.parse_args()
    
    generate_from_masks(
        args.input_path,
        args.output_path,
        args.model_path,
        args.direction
    )


if __name__ == '__main__':
    main()
