# Path: utils/data_loader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

class FloorPlanDatasetCustom(Dataset):
    """
    Dataset class that handles your specific 16-class scattered values with CUDA optimization
    """
    
    def __init__(
        self, 
        data_dir: str, 
        image_size: int = 256,  # Reduced for better GPU memory usage
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
        
        # Cache for class remapping - speed up training
        self.class_lookup = np.zeros(256, dtype=np.uint8)
        for old_class, new_class in self.old_to_new_class.items():
            self.class_lookup[old_class] = new_class
        
        # Get all original images (not label files)
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} image files in {data_dir}")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No valid image files found in {data_dir}")
        
    def _get_image_files(self) -> List[str]:
        """Get all base names from room files since original images don't exist"""
        if not os.path.exists(self.data_dir):
            return []
        
        image_files = []
        all_files = os.listdir(self.data_dir)
        
        # Find all room files and extract base names
        room_files = [f for f in all_files if f.endswith('_rooms.png')]
        print(f"Found {len(room_files)} room files")
        
        for file in room_files:
            base_name = file.replace('_rooms.png', '')
            # Create a virtual image filename
            image_files.append(f"{base_name}.png")
        
        return sorted(image_files)
    
    def _remap_room_mask_fast(self, mask: np.ndarray) -> np.ndarray:
        """Fast vectorized remapping using lookup table"""
        # Clip values to valid range to prevent index errors
        mask_clipped = np.clip(mask, 0, 255)
        return self.class_lookup[mask_clipped].astype(np.uint8)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Use room mask as image since original images don't exist"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        room_path = os.path.join(self.data_dir, f"{base_name}_rooms.png")
        
        if os.path.exists(room_path):
            # Load room mask and convert to pseudo-RGB image
            room_mask = cv2.imread(room_path, cv2.IMREAD_GRAYSCALE)
            if room_mask is not None:
                # Create pseudo-RGB by replicating the mask across 3 channels
                # This gives the network visual information about room boundaries
                pseudo_image = cv2.applyColorMap(room_mask, cv2.COLORMAP_JET)
                return cv2.cvtColor(pseudo_image, cv2.COLOR_BGR2RGB)
        
        raise ValueError(f"Could not load room mask for: {base_name}")
    
    def _load_masks(self, base_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load room and boundary masks with error handling"""
        try:
            # Room mask
            room_mask_path = os.path.join(self.data_dir, f"{base_name}_rooms.png")
            
            if os.path.exists(room_mask_path):
                room_mask = cv2.imread(room_mask_path, cv2.IMREAD_GRAYSCALE)
                if room_mask is not None:
                    # Fast remap using lookup table
                    room_mask = self._remap_room_mask_fast(room_mask)
                else:
                    print(f"Warning: Could not read room mask: {room_mask_path}")
                    room_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                print(f"Warning: Room mask not found: {room_mask_path}")
                room_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            
            # Create boundary mask from wall and door files
            boundary_mask = np.zeros_like(room_mask, dtype=np.uint8)
            
            # Load door/opening mask
            door_mask_path = os.path.join(self.data_dir, f"{base_name}_close.png")
            if os.path.exists(door_mask_path):
                door_mask = cv2.imread(door_mask_path, cv2.IMREAD_GRAYSCALE)
                if door_mask is not None:
                    boundary_mask[door_mask > 128] = 1  # doors/openings = 1
            
            # Load wall mask - handle multiple wall label values
            wall_mask_path = os.path.join(self.data_dir, f"{base_name}_wall.png")
            if os.path.exists(wall_mask_path):
                wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
                if wall_mask is not None:
                    # Your wall labels have values [0,242,245,246,255] - treat non-zero as walls
                    boundary_mask[wall_mask > 128] = 2  # walls = 2 (walls override doors)
            
            return room_mask, boundary_mask
            
        except Exception as e:
            print(f"Error loading masks for {base_name}: {e}")
            # Return dummy masks
            return (np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                    np.zeros((self.image_size, self.image_size), dtype=np.uint8))
    
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
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                room_mask = cv2.resize(room_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                boundary_mask = cv2.resize(boundary_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                
                # Apply transformations
                if self.transform:
                    try:
                        transformed = self.transform(
                            image=image,
                            mask=room_mask,
                            boundary_mask=boundary_mask
                        )
                        image = transformed['image']
                        room_mask = transformed['mask']
                        boundary_mask = transformed['boundary_mask']
                    except Exception as e:
                        print(f"Transform error for {image_file}: {e}")
                        # Fallback to manual tensor conversion
                        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                        room_mask = torch.from_numpy(room_mask)
                        boundary_mask = torch.from_numpy(boundary_mask)
                else:
                    # Convert to tensor if no transforms
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    room_mask = torch.from_numpy(room_mask)
                    boundary_mask = torch.from_numpy(boundary_mask)
                
                # Ensure proper data types and ranges
                image = image.float()
                room_mask = room_mask.long()  # CrossEntropyLoss expects long
                boundary_mask = boundary_mask.long()
                
                # Clamp room_mask to valid range [0, 15]
                room_mask = torch.clamp(room_mask, 0, 15)
                boundary_mask = torch.clamp(boundary_mask, 0, 2)
                
                return {
                    'image': image,
                    'room_mask': room_mask,
                    'boundary_mask': boundary_mask,
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
                    'image': image.float(),
                    'filename': image_file
                }
                
        except Exception as e:
            print(f"Error loading sample {idx} ({self.image_files[idx] if idx < len(self.image_files) else 'unknown'}): {e}")
            # Return a dummy sample with correct tensor types for CUDA
            dummy_image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            dummy_room = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            dummy_boundary = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            
            return {
                'image': dummy_image,
                'room_mask': dummy_room,
                'boundary_mask': dummy_boundary,
                'filename': f'error_{idx}.png'
            }

def get_transforms_custom(image_size: int, mode: str = 'train') -> A.Compose:
    """Get augmentation transforms optimized for CUDA training"""
    
    if mode == 'train':
        transforms = A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ),
            
            # Color augmentations (lighter for pseudo-RGB images)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, 
                    contrast_limit=0.15, 
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=15, 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.3),
            
            # Normalize for pretrained models
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], additional_targets={'boundary_mask': 'mask'})
        
    else:  # validation/test
        transforms = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], additional_targets={'boundary_mask': 'mask'})
    
    return transforms

def create_data_loaders_custom(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 256,
    train_split: float = 0.7,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from single directory with CUDA optimization"""
    
    print(f"Creating data loaders from: {data_dir}")
    
    # Validate directory
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Check for required files
    all_files = os.listdir(data_dir)
    room_files = [f for f in all_files if f.endswith('_rooms.png')]
    
    if len(room_files) == 0:
        raise ValueError(f"No room mask files (*_rooms.png) found in {data_dir}")
    
    print(f"Found {len(room_files)} room files")
    
    # Create datasets with different transforms
    train_dataset = FloorPlanDatasetCustom(
        data_dir,
        image_size=image_size,
        transform=get_transforms_custom(image_size, 'train'),
        mode='train'
    )
    
    val_dataset = FloorPlanDatasetCustom(
        data_dir,
        image_size=image_size,
        transform=get_transforms_custom(image_size, 'val'),
        mode='train'
    )
    
    if len(train_dataset) == 0:
        raise ValueError(f"No valid image files found in {data_dir}")
    
    print(f"Total dataset size: {len(train_dataset)} images")
    
    # Split dataset indices
    total_size = len(train_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    
    # Create indices for splitting
    indices = list(range(total_size))
    
    # Use sklearn for consistent splitting
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(1 - train_split), 
        random_state=42,
        shuffle=True
    )
    
    if val_split > 0 and len(temp_indices) > 0:
        val_ratio = val_split / (val_split + (1 - train_split - val_split))
        val_indices, _ = train_test_split(
            temp_indices,
            test_size=(1 - val_ratio),
            random_state=42,
            shuffle=True
        )
    else:
        val_indices = temp_indices[:val_size] if val_size > 0 else []
    
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices) if val_indices else None
    
    # Custom collate function to handle potential errors
    def collate_fn(batch):
        """Custom collate function with error handling"""
        valid_samples = []
        for sample in batch:
            if 'error' not in sample['filename']:
                valid_samples.append(sample)
        
        if len(valid_samples) == 0:
            # Return dummy batch if all samples failed
            dummy_batch = {
                'image': torch.zeros(1, 3, image_size, image_size),
                'room_mask': torch.zeros(1, image_size, image_size, dtype=torch.long),
                'boundary_mask': torch.zeros(1, image_size, image_size, dtype=torch.long),
                'filename': ['dummy.png']
            }
            return dummy_batch
        
        # Default collate for valid samples
        return torch.utils.data.dataloader.default_collate(valid_samples)
    
    # Create data loaders with CUDA optimization
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(num_workers, 8),  # Cap num_workers for stability
        pin_memory=True,  # Always use pin_memory for CUDA
        drop_last=True,   # Consistent batch sizes
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset if val_subset else torch.utils.data.Subset(val_dataset, []),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, 4),  # Fewer workers for validation
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn
    ) if val_subset and len(val_indices) > 0 else DataLoader(
        torch.utils.data.Subset(val_dataset, []),
        batch_size=1,
        shuffle=False
    )
    
    print(f"Created train loader: {len(train_loader)} batches")
    print(f"Created val loader: {len(val_loader)} batches")
    
    # Test load one batch to verify everything works
    try:
        sample_batch = next(iter(train_loader))
        print("Sample batch shapes:")
        print(f"  Image: {sample_batch['image'].shape} dtype: {sample_batch['image'].dtype}")
        print(f"  Room mask: {sample_batch['room_mask'].shape} dtype: {sample_batch['room_mask'].dtype}")
        print(f"  Boundary mask: {sample_batch['boundary_mask'].shape} dtype: {sample_batch['boundary_mask'].dtype}")
        print(f"  Room mask unique values: {torch.unique(sample_batch['room_mask'])}")
        print(f"  Boundary mask unique values: {torch.unique(sample_batch['boundary_mask'])}")
    except Exception as e:
        print(f"Warning: Could not load sample batch: {e}")
    
    return train_loader, val_loader