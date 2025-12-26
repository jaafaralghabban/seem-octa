"""
OCTA Dataset for Vessel Segmentation

This module provides the OCTADataset class for loading OCTA-500 data
with support for 3-channel input (FULL, ILM_OPL, OPL_BM projections).
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .point_generators import generate_hybrid_points, generate_random_points


class OCTADataset(Dataset):
    """
    OCTA-500 Dataset for vessel segmentation.
    
    Loads 3-channel OCTA images (FULL, ILM_OPL, OPL_BM projections)
    with corresponding vessel segmentation masks.
    
    Args:
        base_dir: Root directory containing OCTA data
        image_ids: List of image filenames to include
        image_size: Target image size (H, W)
        augment: Whether to apply data augmentation
        max_points: Maximum number of initial points to generate
        dataset_type: "OCTA_3mm" or "OCTA_6mm"
        point_generator: "hybrid" or "random"
    """
    
    def __init__(
        self,
        base_dir: str,
        image_ids: List[str],
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        max_points: int = 10,
        dataset_type: str = "OCTA_3mm",
        point_generator: str = "hybrid"
    ):
        self.base_dir = base_dir
        self.image_ids = image_ids
        self.image_size = image_size
        self.max_points = max_points
        self.point_generator = point_generator
        
        # 3-Channel projection paths
        self.projection_paths = {
            'FULL': os.path.join(base_dir, dataset_type, 'OCTA(FULL)'),
            'ILM_OPL': os.path.join(base_dir, dataset_type, 'OCTA(ILM_OPL)'),
            'OPL_BM': os.path.join(base_dir, dataset_type, 'OCTA(OPL_BM)')
        }
        self.label_path = os.path.join(base_dir, 'Label', 'Label', 'GT_LargeVessel')
        
        # Normalization parameters
        self.norm_mean = (0.5, 0.5, 0.5)
        self.norm_std = (0.5, 0.5, 0.5)
        
        # Set up transforms
        if augment:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=0.7
                ),
                A.ElasticTransform(
                    p=0.5, 
                    alpha=120, 
                    sigma=120*0.05, 
                    alpha_affine=120*0.03
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_file = self.image_ids[idx]
        
        # Load 3 projections
        projections = []
        for key in ['FULL', 'ILM_OPL', 'OPL_BM']:
            p_path = os.path.join(self.projection_paths[key], sample_file)
            img = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback to zeros if file not found
                img = np.zeros(self.image_size, dtype=np.uint8)
            projections.append(img)
        
        # Load mask
        mask_path = os.path.join(self.label_path, sample_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.image_size, dtype=np.uint8)
        
        # Stack as RGB-like 3-channel image
        rgb_image = np.stack(projections, axis=2)
        vessel_mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        transformed = self.transform(image=rgb_image, mask=vessel_mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        
        # Generate initial points
        if self.point_generator == "hybrid":
            points, labels = generate_hybrid_points(
                mask.numpy().squeeze(), 
                self.max_points
            )
        else:
            points, labels = generate_random_points(
                mask.numpy().squeeze(), 
                self.max_points
            )
        
        return {
            'image': image,
            'mask': mask,
            'points': torch.from_numpy(points),
            'labels': torch.from_numpy(labels),
            'filename': sample_file
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    image_size: Tuple[int, int] = (512, 512),
    max_points: int = 10,
    dataset_type: str = "OCTA_3mm",
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        image_size: Target image size
        max_points: Max initial points
        dataset_type: OCTA_3mm or OCTA_6mm
        num_workers: DataLoader workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Find all images
    full_path = os.path.join(data_dir, dataset_type, 'OCTA(FULL)')
    all_image_ids = [
        f for f in os.listdir(full_path) 
        if f.lower().endswith(('.png', '.jpg', '.bmp'))
    ]
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_image_ids)
    
    train_size = int(train_ratio * len(all_image_ids))
    val_size = int(val_ratio * len(all_image_ids))
    
    train_ids = all_image_ids[:train_size]
    val_ids = all_image_ids[train_size:train_size + val_size]
    test_ids = all_image_ids[train_size + val_size:]
    
    # Create datasets
    train_dataset = OCTADataset(
        data_dir, train_ids, image_size, 
        augment=True, max_points=max_points,
        dataset_type=dataset_type
    )
    val_dataset = OCTADataset(
        data_dir, val_ids, image_size,
        augment=False, max_points=max_points,
        dataset_type=dataset_type
    )
    test_dataset = OCTADataset(
        data_dir, test_ids, image_size,
        augment=False, max_points=max_points,
        dataset_type=dataset_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size, 
        shuffle=True, num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Datasets created: {len(train_dataset)} train, "
          f"{len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader
