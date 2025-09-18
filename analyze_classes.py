import cv2
import numpy as np
from pathlib import Path
import argparse

def analyze_room_classes(data_dir):
    """Analyze all room label files to find unique classes"""
    data_path = Path(data_dir)
    
    all_classes = set()
    class_counts = {}
    files_analyzed = 0
    
    # Look for room label files with your naming convention
    room_files = list(data_path.glob('*_rooms.png'))
    
    print(f"Found {len(room_files)} room label files")
    
    for room_file in room_files:
        try:
            # Load room mask
            room_mask = cv2.imread(str(room_file), cv2.IMREAD_GRAYSCALE)
            if room_mask is None:
                print(f"Could not load: {room_file}")
                continue
                
            # Get unique values
            unique_vals = np.unique(room_mask)
            all_classes.update(unique_vals)
            
            # Count occurrences
            for val in unique_vals:
                pixel_count = np.sum(room_mask == val)
                class_counts[val] = class_counts.get(val, 0) + pixel_count
            
            files_analyzed += 1
            
        except Exception as e:
            print(f"Error processing {room_file}: {e}")
    
    print(f"\nAnalyzed {files_analyzed} room label files")
    print(f"Found {len(all_classes)} unique class values: {sorted(all_classes)}")
    
    # Print class distribution
    print("\nClass distribution (by pixel count):")
    for class_val in sorted(class_counts.keys()):
        percentage = class_counts[class_val] / sum(class_counts.values()) * 100
        print(f"  Class {class_val}: {class_counts[class_val]:,} pixels ({percentage:.2f}%)")
    
    return sorted(all_classes), class_counts

def analyze_boundary_classes(data_dir):
    """Analyze boundary label files"""
    data_path = Path(data_dir)
    
    # Check wall files and door files with your naming convention
    wall_files = list(data_path.glob('*_wall.png'))
    door_files = list(data_path.glob('*_close.png'))  # close = door/window openings
    close_wall_files = list(data_path.glob('*_close_wall.png'))  # combined files
    
    wall_classes = set()
    door_classes = set()
    close_wall_classes = set()
    
    for wall_file in wall_files:
        try:
            mask = cv2.imread(str(wall_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                wall_classes.update(np.unique(mask))
        except:
            pass
    
    for door_file in door_files:
        try:
            mask = cv2.imread(str(door_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                door_classes.update(np.unique(mask))
        except:
            pass
            
    for close_wall_file in close_wall_files:
        try:
            mask = cv2.imread(str(close_wall_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                close_wall_classes.update(np.unique(mask))
        except:
            pass
    
    print(f"\nBoundary analysis:")
    print(f"Wall files found: {len(wall_files)}")
    print(f"Close/door files found: {len(door_files)}")
    print(f"Close_wall files found: {len(close_wall_files)}")
    print(f"Wall label values: {sorted(wall_classes)}")
    print(f"Door/opening label values: {sorted(door_classes)}")
    print(f"Close_wall label values: {sorted(close_wall_classes)}")
    
    return wall_classes, door_classes, close_wall_classes

def analyze_sample_files(data_dir, num_samples=5):
    """Analyze a few sample files in detail"""
    data_path = Path(data_dir)
    
    # Get some sample files
    room_files = sorted(list(data_path.glob('*_rooms.png')))[:num_samples]
    
    print(f"\nDetailed analysis of {len(room_files)} sample files:")
    print("=" * 50)
    
    for room_file in room_files:
        print(f"\nFile: {room_file.name}")
        
        try:
            # Load the room mask
            room_mask = cv2.imread(str(room_file), cv2.IMREAD_GRAYSCALE)
            if room_mask is None:
                print("  Could not load file")
                continue
                
            unique_vals = np.unique(room_mask)
            print(f"  Shape: {room_mask.shape}")
            print(f"  Unique values: {unique_vals}")
            print(f"  Value distribution:")
            
            for val in unique_vals:
                count = np.sum(room_mask == val)
                percentage = count / room_mask.size * 100
                print(f"    {val}: {count:,} pixels ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"  Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze YOUR dataset classes')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    
    # Check if we have the expected files
    room_files = list(data_path.glob('*_rooms.png'))
    wall_files = list(data_path.glob('*_wall.png'))
    close_files = list(data_path.glob('*_close.png'))
    
    if not room_files:
        print(f"No room label files (*_rooms.png) found in {args.data_dir}")
        return
    
    print(f"Dataset Analysis for: {data_path.absolute()}")
    print("=" * 60)
    print(f"Found file types:")
    print(f"  Room labels (*_rooms.png): {len(room_files)} files")
    print(f"  Wall labels (*_wall.png): {len(wall_files)} files") 
    print(f"  Door/opening labels (*_close.png): {len(close_files)} files")
    
    # Analyze room classes
    print(f"\n{'='*60}")
    print("ROOM CLASS ANALYSIS")
    print(f"{'='*60}")
    room_classes, room_counts = analyze_room_classes(args.data_dir)
    
    # Analyze boundary classes
    wall_classes, door_classes, close_wall_classes = analyze_boundary_classes(args.data_dir)
    
    # Sample analysis
    analyze_sample_files(args.data_dir)
    
    # Generate configuration
    if room_classes:
        max_room_class = max(room_classes)
        print(f"\n{'='*60}")
        print("CONFIGURATION RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"For your config.yaml file:")
        print(f"""
model:
  num_room_classes: {max_room_class + 1}  # Classes 0 to {max_room_class}
  num_boundary_classes: 3  # Background, wall, door/window

# Room type mapping:
room_types:""")
        
        # Generate room type mapping
        common_room_names = [
            "background", "closet", "bathroom", "living_room", "bedroom", 
            "hall", "balcony", "kitchen", "dining_room", "laundry_room",
            "office", "storage", "garage", "entrance", "corridor", "utility_room"
        ]
        
        for i, class_id in enumerate(sorted(room_classes)):
            if i < len(common_room_names):
                room_name = common_room_names[i]
            else:
                room_name = f"room_type_{class_id}"
            print(f"  {class_id}: \"{room_name}\"")
        
        print(f"\nNext steps:")
        print(f"1. Update your config.yaml with the settings above")
        print(f"2. Your data is already in the right location!")
        print(f"3. Run training with: python train.py --config config/config.yaml")

if __name__ == '__main__':
    main()