#!/usr/bin/env python3
"""
Test script to verify data loading and model setup
"""

import torch
from dataset import MilaLogoDataset
from model import UNet
from utils import DiceLoss, BCEDiceLoss, iou_score
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_setup():
    print("=== Testing Data Loading and Model Setup ===")
    
    # Test dataset
    image_dir = 'train/img'
    mask_dir = 'train/mask'
    all_files = sorted(os.listdir(image_dir))[:5]  # Test with first 5 files
    
    print(f"Found {len(all_files)} files for testing")
    
    # Create simple transform for testing
    test_transform = A.Compose([
        A.Resize(340, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    dataset = MilaLogoDataset(image_dir, mask_dir, all_files, transform=test_transform)
    print(f"Dataset length: {len(dataset)}")
    
    # Test single sample
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, range: [{img.min():.4f}, {img.max():.4f}]")
    print(f"Mask shape: {mask.shape}, range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"Mask unique values: {torch.unique(mask)}")
    
    # Test model
    model = UNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_img = img.unsqueeze(0)  # Add batch dimension
    batch_mask = mask.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(batch_img)
        print(f"Model output shape: {output.shape}")
        print(f"Model output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test loss functions
    dice_loss = DiceLoss()
    bce_dice_loss = BCEDiceLoss()
    
    dice_loss_val = dice_loss(output, batch_mask)
    bce_dice_loss_val = bce_dice_loss(output, batch_mask)
    
    print(f"Dice Loss: {dice_loss_val.item():.4f}")
    print(f"BCE Dice Loss: {bce_dice_loss_val.item():.4f}")
    
    # Test IoU function
    iou_val = iou_score(torch.sigmoid(output), batch_mask)
    print(f"IoU Score: {iou_val.item():.4f}")
    
    # Test edge cases
    print("\n=== Testing IoU Edge Cases ===")
    # Perfect prediction
    perfect_pred = batch_mask.clone()
    perfect_iou = iou_score(perfect_pred, batch_mask)
    print(f"Perfect prediction IoU: {perfect_iou.item():.4f}")
    
    # No overlap
    no_overlap = 1 - batch_mask
    no_overlap_iou = iou_score(no_overlap, batch_mask)
    print(f"No overlap IoU: {no_overlap_iou.item():.4f}")
    
    # Empty masks
    empty_mask = torch.zeros_like(batch_mask)
    empty_iou = iou_score(empty_mask, empty_mask)
    print(f"Empty masks IoU: {empty_iou:.4f}")
    
    print("=== Setup test completed ===")

def test_trained_model():
    print("\n=== Testing Trained Model (best_model.pth) ===")
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load model
    model = UNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best IoU: {checkpoint.get('best_iou', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state (legacy format)")
    
    model.to(device)
    model.eval()
    
    # Test dataset
    image_dir = 'test/test_set/img'
    mask_dir = 'test/test_set/mask'
    all_files = sorted(os.listdir(image_dir))[:10]  # Test with first 10 files
    
    test_transform = A.Compose([
        A.Resize(340, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    dataset = MilaLogoDataset(image_dir, mask_dir, all_files, transform=test_transform)
    
    # Test trained model performance
    total_iou = 0
    total_dice_loss = 0
    total_bce_dice_loss = 0
    
    dice_loss = DiceLoss()
    bce_dice_loss = BCEDiceLoss()
    
    print(f"\nTesting on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            img, mask = dataset[i]
            batch_img = img.unsqueeze(0).to(device)
            batch_mask = mask.unsqueeze(0).to(device)
            
            output = model(batch_img)
            
            # Calculate metrics
            iou_val = iou_score(torch.sigmoid(output), batch_mask)
            dice_loss_val = dice_loss(output, batch_mask)
            bce_dice_loss_val = bce_dice_loss(output, batch_mask)
            
            total_iou += iou_val.item()
            total_dice_loss += dice_loss_val.item()
            total_bce_dice_loss += bce_dice_loss_val.item()
            
            if i < 3:  # Show first 3 samples
                print(f"Sample {i+1}: IoU={iou_val.item():.4f}, Dice Loss={dice_loss_val.item():.4f}, BCE Dice Loss={bce_dice_loss_val.item():.4f}")
    
    # Average metrics
    avg_iou = total_iou / len(dataset)
    avg_dice_loss = total_dice_loss / len(dataset)
    avg_bce_dice_loss = total_bce_dice_loss / len(dataset)
    
    print(f"=== Trained Model Performance ===")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice Loss: {avg_dice_loss:.4f}")
    print(f"Average BCE Dice Loss: {avg_bce_dice_loss:.4f}")
    
    # Performance assessment
    if avg_iou > 0.8:
        print("Excellent performance!")
    elif avg_iou > 0.6:
        print("Good performance!")
    elif avg_iou > 0.4:
        print("Decent performance")
    else:
        print(" Performance needs improvement")

if __name__ == "__main__":
    test_setup()
    test_trained_model() 