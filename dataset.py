# Custom dataset class for Mila logo segmentation project.
# Citation: Based on PyTorch Dataset patterns - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MilaLogoDataset(Dataset):
    """
    Dataset class for Mila Logo Segmentation.
    Loads an image and its corresponding mask given a filename list.
    """
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.bmp')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']    # Already a tensor [3, H, W]
            mask = augmented['mask']      # Already a tensor [H, W]

        # Ensure mask is properly binarized and normalized
        mask = mask.float()
        
        # Convert mask from [0, 255] to [0, 1] and binarize
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Binarize mask: any pixel > 0.5 is considered foreground
        mask = (mask > 0.5).float()

        return image, mask

