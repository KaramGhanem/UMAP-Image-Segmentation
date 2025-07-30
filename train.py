# Trains U-Net from scratch on Mila logo segmentation.
# Includes training/validation/test split, augmentation, and LR scheduling.
# Citation: Loop structure based on PyTorch Transfer Learning tutorial:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MilaLogoDataset
from model import UNet
from utils import iou_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.multiprocessing import freeze_support
from utils import BCEDiceLoss, DiceLoss
import time
import csv

def main():
    # Set image/mask directory
    image_dir = 'train/img'
    mask_dir = 'train/mask'
    all_files = sorted(os.listdir(image_dir))

    # Split dataset into train, val, test: 70%, 15%, 15%  
    trainval_files, test_files = train_test_split(all_files, test_size=0.15, random_state=42) 
    train_files, val_files = train_test_split(trainval_files, test_size=0.1765, random_state=42) 

    # Albumentations transforms
    train_transform = A.Compose([
        A.Resize(340, 512),  # Resize both image and mask
        A.HorizontalFlip(p=0.5),  # Randomly flip image and mask horizontally
        A.RandomBrightnessContrast(p=0.2),  # Random brightness/contrast change
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),  # Random shift/scale/rotate
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0),  # Normalize image only (not mask)
        ToTensorV2(),  # Converts image to CHW tensor and mask to [H, W]
    ])

    val_transform = A.Compose([
        A.Resize(340, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Datasets and loaders
    train_dataset = MilaLogoDataset(image_dir, mask_dir, train_files, transform=train_transform)
    val_dataset = MilaLogoDataset(image_dir, mask_dir, val_files, transform=val_transform)
    test_dataset = MilaLogoDataset(image_dir, mask_dir, test_files, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)  

    # Load model and optimizer
    model = UNet()

    # Initialize mps gpu for training
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(x)
    else:
        print("MPS device not found.")
        device = torch.device("cpu")  

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    checkpoint_path = "checkpoints/best_model.pth"
    start_epoch = 0
    best_iou = 0
    
    # Restore checkpoint if available, otherwise start from scratch
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(torch.load(checkpoint_path)) #for checkpoints without optimizer/scheduler info and other metadata
            pass 
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_iou' in checkpoint:
            best_iou = checkpoint['best_iou']
        if 'metadata' in checkpoint:
            print("Metadata:", checkpoint.get('metadata', {}))
    else:
        print("No checkpoint found. Training from scratch.")
        

    use_bce_dice = True  # Set to True if you want to BCEDiceLoss
    criterion = BCEDiceLoss() if use_bce_dice else DiceLoss()

    # Training loop
    epochs = 20
    best_iou = 0

    # Tracking setup
    log_path = "logs/training_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create log file and write header if not exists
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "val_iou", "learning_rate", "duration_sec"])

    # Early stopping: stop training if validation IoU does not improve for 'early_stop_patience' consecutive epochs
    early_stop_patience = 5
    no_improve_epochs = 0

    # === Training Loop ===
    # - The following loop will train the model for a set number of epochs.
    # - It tracks the best validation IoU and implements early stopping if no improvement is seen.
    # - Training and validation metrics are logged to CSV for later analysis.
    # - Model checkpoints are saved for both the latest and best-performing models.
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        model.train()
        train_loss = 0

        for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Debug info (only every 10 batches for time efficiency)
            if i % 10 == 0:
                print(f"Prediction mean: {torch.sigmoid(outputs).mean().item():.4f}")
                print(f"Output shape: {outputs.shape}, Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"Mask shape: {masks.shape}, Mask range: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
                print(f"Mask unique values: {torch.unique(masks)}")
            
            loss = criterion(outputs, masks)
            if i % 10 == 0:
                print(f"Batch {i} loss: {loss.item():.4f}")
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()

        # === VALIDATION PHASE ===
        model.eval()  # Set model to evaluation mode (no gradient computation, batch norm uses running stats)
        val_iou = 0   
        
        with torch.no_grad(): 
            for imgs, masks in val_loader:
                imgs = imgs.to(device)    
                masks = masks.to(device)  
                outputs = model(imgs)     # Forward pass through model (no gradients computed)
                val_iou += iou_score(torch.sigmoid(outputs), masks)  # Calculate IoU for this batch and accumulate

        # Calculate average IoU across all validation batches
        val_iou /= len(val_loader)
        
        # Update learning rate based on validation performance
        # If val_iou doesn't improve for 'patience' epochs, reduce LR by 'factor'
        scheduler.step(val_iou)
        
        # Calculate average training loss across all batches
        train_loss /= len(train_loader)
        
        # Calculate epoch duration for monitoring
        duration = time.time() - start_time
        
        # Get current learning rate for logging
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, LR: {lr:.6f}")
        print(f"Epoch {epoch+1} completed in {duration:.2f}s")

        # === Log metrics to CSV ===
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1, train_loss, val_iou, lr, duration])

        # === Save latest model checkpoint ===
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'metadata': {
                'loss_function': 'BCEDiceLoss' if use_bce_dice else 'DiceLoss',
                'comment': 'Training on Mila logo segmentation dataset'
            }
        }, "checkpoints/last_model.pth")

        # === Save best model ===
        if val_iou > best_iou:
            best_iou = val_iou
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'metadata': {
                    'loss_function': 'BCEDiceLoss' if use_bce_dice else 'DiceLoss',
                    'comment': 'Training on Mila logo segmentation dataset'
                }
            }, "checkpoints/best_model.pth")
            print(f"Saved new best model at epoch {epoch+1} with IoU {val_iou:.4f}")
        else:
            # No improvement in validation IoU; increment patience counter and check for early stopping
            no_improve_epochs += 1
            print(f"No improvement. Patience counter: {no_improve_epochs}/{early_stop_patience}")
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ---- Final Test Evaluation ----  # 
    print("\n--- Final Test Evaluation ---")
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_iou = 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            test_iou += iou_score(torch.sigmoid(outputs), masks)

    test_iou /= len(test_loader)
    print(f"Final Test IoU: {test_iou:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()