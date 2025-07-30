#!/usr/bin/env python3
"""
Inference script for Mila Logo Segmentation model.
Usage: python infer.py <image_dir> <output_dir>

This script loads the best trained model and generates segmentation masks
for all images in the input directory.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet
import argparse
from tqdm import tqdm
import cv2

def load_model(checkpoint_path="checkpoints/best_model.pth"):
    """
    Load the trained model from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Load model
    model = UNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f" Best IoU: {checkpoint.get('best_iou', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(" Loaded model state (legacy format)")
    
    model.to(device)
    model.eval()
    
    return model, device

def get_transform():
    """
    Get the same transform used during training.
    """
    return A.Compose([
        A.Resize(340, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])

def process_image(image_path, model, device, transform):
    """
    Process a single image and return the segmentation mask.
    """
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))
    original_size = image.shape[:2]  # (height, width)
    
    # Apply transform
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output)
    
    # Convert to numpy and resize to original size
    mask = prediction.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Binarize mask
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    return binary_mask, mask

def save_results(original_image, binary_mask, probability_mask, output_path, filename):
    """
    Save the binary mask with the same filename as the input image.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save binary mask with same filename as input
    mask_path = os.path.join(output_path, f"{filename}.png")
    Image.fromarray(binary_mask).save(mask_path)
    
    return mask_path

def main():
    parser = argparse.ArgumentParser(description='Inference script for Mila Logo Segmentation')
    parser.add_argument('image_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory to save output masks')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', 
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binarization (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.image_dir):
        print(f"Input directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(args.image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {args.image_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    try:
        # Load model
        model, device = load_model(args.checkpoint)
        
        # Get transform
        transform = get_transform()
        
        # Process images
        print(f"\nProcessing images...")
        processed_count = 0
        
        for filename in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(args.image_dir, filename)
                
                # Load original image
                original_image = np.array(Image.open(image_path).convert("RGB"))
                
                # Process image
                binary_mask, probability_mask = process_image(image_path, model, device, transform)
                
                # Save results
                base_name = os.path.splitext(filename)[0]
                mask_path = save_results(
                    original_image, binary_mask, probability_mask, args.output_dir, base_name
                )
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        print(f"\nSuccessfully processed {processed_count}/{len(image_files)} images")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"   - Binary segmentation masks saved with same filenames as input images")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()