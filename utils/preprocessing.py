# Path: utils/preprocessing.py

import os
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class FloorPlanPreprocessor:
    """
    Preprocessing utilities for floor plan images and labels
    Since your data is already split into train/val/test, this focuses on
    runtime preprocessing and augmentation
    """
    
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    
    def resize_with_padding(
        self, 
        image: np.ndarray, 
        target_size: int,
        fill_value: int = 255
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize image while maintaining aspect ratio using padding
        
        Returns:
            Resized image and metadata for reverse transformation
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            # Width is larger
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:
            # Height is larger or equal
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Apply padding
        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=[fill_value, fill_value, fill_value]
            )
        else:
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=fill_value
            )
        
        metadata = {
            'original_size': (h, w),
            'resized_size': (new_h, new_w),
            'padding': (top, bottom, left, right),
            'scale_factor': new_w / w
        }
        
        return padded, metadata
    
    def reverse_padding(self, image: np.ndarray, metadata: Dict) -> np.ndarray:
        """Remove padding and resize back to original size"""
        top, bottom, left, right = metadata['padding']
        original_h, original_w = metadata['original_size']
        
        # Remove padding
        if bottom == 0 and right == 0:
            cropped = image[top:, left:]
        elif bottom == 0:
            cropped = image[top:, left:-right]
        elif right == 0:
            cropped = image[top:-bottom, left:]
        else:
            cropped = image[top:-bottom, left:-right]
        
        # Resize back to original
        if len(image.shape) == 3:
            resized = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return resized
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Load and preprocess image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize with padding
        processed_image, metadata = self.resize_with_padding(image, self.target_size)
        
        # Normalize
        processed_image = self.normalize_image(processed_image)
        
        metadata['original_path'] = image_path
        
        return processed_image, metadata
    
    def load_and_preprocess_mask(
        self, 
        mask_path: str, 
        metadata: Dict,
        mask_type: str = 'room'
    ) -> np.ndarray:
        """Load and preprocess mask using same transformation as image"""
        if not os.path.exists(mask_path):
            # Return zero mask if file doesn't exist
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        
        # Convert mask values if needed
        if mask_type == 'room':
            # Room masks should have class indices (0-8)
            mask = self.process_room_mask(mask)
        elif mask_type in ['wall', 'door']:
            # Binary masks: convert to 0/1
            mask = (mask > 128).astype(np.uint8)
        
        # Apply same padding as image (but with 0 fill for masks)
        original_size = metadata['original_size']
        
        # First resize to match the image preprocessing
        if mask.shape[:2] != original_size:
            mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply same padding as image
        padded_mask, _ = self.resize_with_padding(mask, self.target_size, fill_value=0)
        
        return padded_mask
    
    def process_room_mask(self, mask: np.ndarray) -> np.ndarray:
        """Process room mask to ensure correct class indices"""
        # Handle different possible encodings
        unique_vals = np.unique(mask)
        
        if len(unique_vals) <= 9:
            # Already in correct format (0-8 indices)
            return mask
        
        # If using RGB encoding, convert to indices
        if mask.max() > 50:  # Likely RGB values
            # Create mapping from RGB values to class indices
            processed_mask = np.zeros_like(mask, dtype=np.uint8)
            
            # Define color mappings (you might need to adjust these based on your labels)
            color_to_class = {
                0: 0,      # background
                64: 1,     # closet
                128: 2,    # bathroom  
                192: 3,    # living room
                255: 4,    # bedroom
                # Add more mappings as needed
            }
            
            for color_val, class_id in color_to_class.items():
                processed_mask[mask == color_val] = class_id
            
            return processed_mask
        
        return mask
    
    def combine_boundary_masks(
        self, 
        wall_mask: np.ndarray, 
        door_mask: np.ndarray
    ) -> np.ndarray:
        """Combine wall and door masks into single boundary mask"""
        # 0: background, 1: door/window, 2: wall
        boundary_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        boundary_mask[door_mask > 0] = 1  # doors/windows
        boundary_mask[wall_mask > 0] = 2  # walls (walls override doors)
        
        return boundary_mask

class DatasetValidator:
    """
    Validate dataset integrity and provide statistics
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def validate_dataset(self) -> Dict:
        """Validate dataset and return statistics"""
        stats = {
            'total_images': 0,
            'images_with_room_labels': 0,
            'images_with_wall_labels': 0,
            'images_with_door_labels': 0,
            'missing_labels': [],
            'corrupt_files': [],
            'class_distribution': {},
            'image_sizes': [],
            'warnings': []
        }
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_dir.glob(f'*{ext}'))
            image_files.extend(self.data_dir.glob(f'*{ext.upper()}'))
        
        # Filter out label files
        image_files = [
            f for f in image_files 
            if not any(suffix in f.name for suffix in ['_wall', '_close', '_room', '_multi'])
        ]
        
        stats['total_images'] = len(image_files)
        
        for image_file in image_files:
            base_name = image_file.stem
            
            try:
                # Check image integrity
                img = cv2.imread(str(image_file))
                if img is None:
                    stats['corrupt_files'].append(str(image_file))
                    continue
                
                stats['image_sizes'].append(img.shape[:2])
                
                # Check for corresponding label files
                room_label = self.data_dir / f"{base_name}_room.png"
                wall_label = self.data_dir / f"{base_name}_wall.png"
                door_label = self.data_dir / f"{base_name}_close.png"
                
                if room_label.exists():
                    stats['images_with_room_labels'] += 1
                    # Analyze room class distribution
                    room_mask = cv2.imread(str(room_label), cv2.IMREAD_GRAYSCALE)
                    if room_mask is not None:
                        unique_classes = np.unique(room_mask)
                        for cls in unique_classes:
                            stats['class_distribution'][int(cls)] = stats['class_distribution'].get(int(cls), 0) + 1
                else:
                    stats['missing_labels'].append(f"{base_name}_room.png")
                
                if wall_label.exists():
                    stats['images_with_wall_labels'] += 1
                else:
                    stats['missing_labels'].append(f"{base_name}_wall.png")
                
                if door_label.exists():
                    stats['images_with_door_labels'] += 1
                else:
                    stats['missing_labels'].append(f"{base_name}_close.png")
                    
            except Exception as e:
                stats['corrupt_files'].append(f"{image_file}: {str(e)}")
        
        # Generate warnings
        if stats['images_with_room_labels'] < stats['total_images'] * 0.8:
            stats['warnings'].append("Less than 80% of images have room labels")
        
        if len(stats['missing_labels']) > stats['total_images'] * 0.1:
            stats['warnings'].append("More than 10% of labels are missing")
        
        # Analyze image size consistency
        if stats['image_sizes']:
            sizes = np.array(stats['image_sizes'])
            if len(np.unique(sizes[:, 0])) > 3 or len(np.unique(sizes[:, 1])) > 3:
                stats['warnings'].append("Highly variable image sizes detected")
        
        return stats
    
    def print_dataset_report(self):
        """Print detailed dataset validation report"""
        stats = self.validate_dataset()
        
        print("=== DATASET VALIDATION REPORT ===")
        print(f"Dataset directory: {self.data_dir}")
        print(f"Total images: {stats['total_images']}")
        print(f"Images with room labels: {stats['images_with_room_labels']} ({stats['images_with_room_labels']/max(stats['total_images'],1)*100:.1f}%)")
        print(f"Images with wall labels: {stats['images_with_wall_labels']} ({stats['images_with_wall_labels']/max(stats['total_images'],1)*100:.1f}%)")
        print(f"Images with door labels: {stats['images_with_door_labels']} ({stats['images_with_door_labels']/max(stats['total_images'],1)*100:.1f}%)")
        
        if stats['class_distribution']:
            print("\nRoom class distribution:")
            for class_id, count in sorted(stats['class_distribution'].items()):
                print(f"  Class {class_id}: {count} occurrences")
        
        if stats['image_sizes']:
            sizes = np.array(stats['image_sizes'])
            print(f"\nImage size statistics:")
            print(f"  Mean size: {sizes.mean(axis=0)}")
            print(f"  Min size: {sizes.min(axis=0)}")  
            print(f"  Max size: {sizes.max(axis=0)}")
        
        if stats['warnings']:
            print("\nWarnings:")
            for warning in stats['warnings']:
                print(f"  ⚠  {warning}")
        
        if stats['corrupt_files']:
            print(f"\nCorrupt files ({len(stats['corrupt_files'])}):")
            for corrupt_file in stats['corrupt_files'][:10]:  # Show first 10
                print(f"  ❌ {corrupt_file}")
            if len(stats['corrupt_files']) > 10:
                print(f"  ... and {len(stats['corrupt_files']) - 10} more")
        
        if stats['missing_labels']:
            print(f"\nMissing labels ({len(stats['missing_labels'])}):")
            for missing_label in stats['missing_labels'][:10]:  # Show first 10
                print(f"  📄 {missing_label}")
            if len(stats['missing_labels']) > 10:
                print(f"  ... and {len(stats['missing_labels']) - 10} more")

def create_data_statistics(data_dirs: List[str]) -> Dict:
    """Create comprehensive statistics for all dataset splits"""
    all_stats = {}
    
    for data_dir in data_dirs:
        split_name = Path(data_dir).name
        validator = DatasetValidator(data_dir)
        all_stats[split_name] = validator.validate_dataset()
    
    return all_stats

def main():
    """Example usage of preprocessing utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset validation and preprocessing utilities')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--validate', action='store_true',
                        help='Run dataset validation')
    
    args = parser.parse_args()
    
    if args.validate:
        validator = DatasetValidator(args.data_dir)
        validator.print_dataset_report()

if __name__ == '__main__':
    main()