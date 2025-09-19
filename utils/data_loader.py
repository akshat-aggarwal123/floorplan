# Path: utils/data_loader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, List, Tuple, Optional

class FloorPlanDatasetCustom(Dataset):
    """
    Dataset class that handles your specific 16-class scattered values
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
        
        # Your specific class mapping - ALL 16 classes
        self.old_to_new_class = {
            0: 0,    # background
            1: 1,    # closet
            180: 2,  # bathroom
            181: 3,  # living_room
            194: 4,  # bedroom
            195: 5,  # hall
            221: 6,  # balcony
            222: 7,  # kitchen
            223: 8,  # dining_room
            224: 9,  # laundry_room
            232: 10, # office
            233: 11, # storage
            235: 12, # garage
            236: 13, # entrance
            237: 14, # corridor
            238: 15  # utility_room
        }
        
        # Get all original images (not label files)
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} image files in {data_dir}")
        
    def _get_image_files(self) -> List[str]:
        """Get all base names from room files since original images don't exist"""
        if not os.path.exists(self.data_dir):
            return []
        
        image_files = []
        all_files = os.listdir(self.data_dir)
        
        # Find all room files and extract base names
        for file in all_files:
            if file.endswith('_rooms.png'):
                base_name = file.replace('_rooms.png', '')
                # Create a virtual image filename
                image_files.append(f"{base_name}.png")
        
        return sorted(image_files)
    
    def _remap_room_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remap scattered class values to sequential indices"""
        remapped_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for old_class, new_class in self.old_to_new_class.items():
            remapped_mask[mask == old_class] = new_class
            
        return remapped_mask
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Use room mask as image since original images don't exist"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        room_path = os.path.join(self.data_dir, f"{base_name}_rooms.png")
        
        if os.path.exists(room_path):
            room_mask = cv2.imread(room_path, cv2.IMREAD_COLOR)  # Load as color
            if room_mask is not None:
                return cv2.cvtColor(room_mask, cv2.COLOR_BGR2RGB)
        
        raise ValueError(f"Could not load room mask for: {base_name}")
    
    def _load_masks(self, base_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load room and boundary masks"""
        # Room mask
        room_mask_path = os.path.join(self.data_dir, f"{base_name}_rooms.png")
        
        if os.path.exists(room_mask_path):
            room_mask = cv2.imread(room_mask_path, cv2.IMREAD_GRAYSCALE)
            if room_mask is not None:
                # Remap scattered classes to sequential indices
                room_mask = self._remap_room_mask(room_mask)
            else:
                room_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            room_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Create boundary mask from wall and door files
        boundary_mask = np.zeros((room_mask.shape[0], room_mask.shape[1]), dtype=np.uint8)
        
        # Load door/opening mask
        door_mask_path = os.path.join(self.data_dir, f"{base_name}_close.png")
        if os.path.exists(door_mask_path):
            door_mask = cv2.imread(door_mask_path, cv2.IMREAD_GRAYSCALE)
            if door_mask is not None:
                boundary_mask[door_mask > 128] = 1  # doors/windows = 1
        
        # Load wall mask - handle multiple wall label values
        wall_mask_path = os.path.join(self.data_dir, f"{base_name}_wall.png")
        if os.path.exists(wall_mask_path):
            wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
            if wall_mask is not None:
                # Your wall labels have values [0,242,245,246,255] - treat non-zero as walls
                boundary_mask[wall_mask > 128] = 2  # walls = 2 (walls override doors)
                
        return room_mask, boundary_mask
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
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
                else:
                    # Convert to tensor if no transforms
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    room_mask = torch.from_numpy(room_mask)
                    boundary_mask = torch.from_numpy(boundary_mask)
                
                return {
                    'image': image,
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
                else:
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
                return {
                    'image': image,
                    'filename': image_file
                }
                
        except Exception as e:
            print(f"Error loading sample {idx} ({image_file}): {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_room = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            dummy_boundary = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            
            return {
                'image': dummy_image,
                'room_mask': dummy_room,
                'boundary_mask': dummy_boundary,
                'filename': f'error_{idx}.jpg'
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
    
    print(f"Creating data loaders from: {data_dir}")
    
    # Create full dataset
    full_dataset = FloorPlanDatasetCustom(
        data_dir,
        image_size=image_size,
        transform=get_transforms_custom(image_size, 'train'),
        mode='train'
    )
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid image files found in {data_dir}")
    
    print(f"Total dataset size: {len(full_dataset)} images")
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Split: {train_size} train, {val_size} val, {test_size} test")
    
    # Create random splits
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    # Create validation dataset with different transforms
    val_dataset_wrapper = FloorPlanDatasetCustom(
        data_dir,
        image_size=image_size,
        transform=get_transforms_custom(image_size, 'val'),
        mode='train'
    )
    
    # Apply validation indices to validation dataset
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_wrapper, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader