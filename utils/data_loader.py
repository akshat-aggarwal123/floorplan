# Path: utils/custom_data_loader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, List, Tuple, Optional

class FloorPlanDatasetCustom(Dataset):
    """
    Dataset class that handles your specific scattered class values
    """
    
    def __init__(
        self, 
        data_dir: str, 
        image_size: int = 512,
        transform: Optional[A.Compose] = None,
        mode: str = 'train'
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        # Your specific class mapping
        self.old_to_new_class = {
            0: 0,    # background
            1: 1,    # closet
            180: 2,  # bathroom
            181: 3,  # living_room
            195: 4,  # bedroom
            222: 5,  # hall
            224: 6,  # balcony
            233: 7,  # kitchen
            236: 8,  # dining_room
            238: 9   # laundry_room
        }
        
        # Get all image files (original floor plans)
        self.image_files = self._get_image_files()
        
    def _get_image_files(self) -> List[str]:
        """Get all original image files (not label files) with smarter filtering"""
        image_extensions = ['.jpg', '.jpeg', '.png']
        mask_suffixes = ['_wall', '_close', '_room', '_rooms', '_multi']
        image_files = []
        
        for file in os.listdir(self.data_dir):
            # Check if it's an image file first
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Get the name without the extension
                base_name = os.path.splitext(file)[0]
                
                # Check if the base name ends exactly with one of the mask suffixes
                if not any(base_name.endswith(suffix) for suffix in mask_suffixes):
                    image_files.append(file)
        
        return sorted(image_files)
    
    def _remap_room_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remap scattered class values to sequential indices"""
        remapped_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for old_class, new_class in self.old_to_new_class.items():
            remapped_mask[mask == old_class] = new_class
        
        return remapped_mask
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_masks(self, base_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load room and boundary masks with your naming convention"""
        # Room mask - handle both _room.png and _rooms.png
        room_mask_path = os.path.join(self.data_dir, f"{base_name}_rooms.png")
        if not os.path.exists(room_mask_path):
            room_mask_path = os.path.join(self.data_dir, f"{base_name}_room.png")
        
        if os.path.exists(room_mask_path):
            room_mask = cv2.imread(room_mask_path, cv2.IMREAD_GRAYSCALE)
            # Remap scattered classes to sequential indices
            room_mask = self._remap_room_mask(room_mask)
        else:
            room_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Create boundary mask from wall and door files
        boundary_mask = np.zeros((room_mask.shape[0], room_mask.shape[1]), dtype=np.uint8)
        
        # Load door/opening mask
        door_mask_path = os.path.join(self.data_dir, f"{base_name}_close.png")
        if os.path.exists(door_mask_path):
            door_mask = cv2.imread(door_mask_path, cv2.IMREAD_GRAYSCALE)
            boundary_mask[door_mask > 128] = 1  # doors/windows = 1
        
        # Load wall mask
        wall_mask_path = os.path.join(self.data_dir, f"{base_name}_wall.png")
        if os.path.exists(wall_mask_path):
            wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
            boundary_mask[wall_mask > 128] = 2  # walls = 2 (walls override doors)
            
        return room_mask, boundary_mask
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image filename and base name
        image_file = self.image_files[idx]
        base_name = os.path.splitext(image_file)[0]
        
        # Load image
        image_path = os.path.join(self.data_dir, image_file)
        image = self._load_image(image_path)
        
        if self.mode != 'test':
            # Load masks for training/validation
            room_mask, boundary_mask = self._load_masks(base_name)
            
            # Resize everything to target size
            image = cv2.resize(image, (self.image_size, self.image_size))
            room_mask = cv2.resize(room_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            boundary_mask = cv2.resize(boundary_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(
                    image=image,
                    mask=room_mask,
                    boundary_mask=boundary_mask
                )
                image = transformed['image']
                room_mask = transformed['mask']
                boundary_mask = transformed['boundary_mask']
            
            return {
                'image': image.float(),
                'room_mask': room_mask.long(),
                'boundary_mask': boundary_mask.long(),
                'filename': image_file
            }
        else:
            # Test mode - only image
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return {
                'image': image.float(),
                'filename': image_file
            }

def get_transforms_custom(image_size: int, mode: str = 'train') -> A.Compose:
    """Get augmentation transforms"""
    
    if mode == 'train':
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'boundary_mask': 'mask'})
    else:
        transforms = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'boundary_mask': 'mask'})
    
    return transforms

def create_data_loaders_custom(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 512,
    train_split: float = 0.7,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from single directory"""
    
    # Create full dataset
    full_dataset = FloorPlanDatasetCustom(
        data_dir,
        image_size=image_size,
        transform=get_transforms_custom(image_size, 'train'),
        mode='train'
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Update transforms for validation
    val_dataset.dataset.transform = get_transforms_custom(image_size, 'val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader