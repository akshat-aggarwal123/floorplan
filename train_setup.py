# Path: test_setup.py

"""
Quick test script to verify all imports and data loading work correctly
Run this before starting training to catch any issues early
"""

import sys
import os
import torch

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        from models.model import ModernFloorPlanNet
        print("✓ ModernFloorPlanNet imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ModernFloorPlanNet: {e}")
        return False
    
    try:
        from models.losses import MultiTaskLoss
        print("✓ MultiTaskLoss imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MultiTaskLoss: {e}")
        return False
    
    try:
        from utils.data_loader import create_data_loaders_custom, FloorPlanDatasetCustom
        print("✓ Data loader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data loader: {e}")
        return False
    
    return True

def test_data_loading(data_dir):
    """Test data loading with your dataset"""
    print(f"\nTesting data loading from: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory does not exist: {data_dir}")
        return False
    
    try:
        from utils.data_loader import FloorPlanDatasetCustom
        
        # Test dataset creation
        dataset = FloorPlanDatasetCustom(
            data_dir=data_dir,
            image_size=256,  # Small size for testing
            mode='train'
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("✗ No samples found in dataset")
            return False
        
        # Test loading one sample
        sample = dataset[0]
        
        print(f"✓ Sample loaded:")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Room mask shape: {sample['room_mask'].shape}")
        print(f"  - Boundary mask shape: {sample['boundary_mask'].shape}")
        print(f"  - Filename: {sample['filename']}")
        
        # Check class values
        room_classes = torch.unique(sample['room_mask'])
        boundary_classes = torch.unique(sample['boundary_mask'])
        
        print(f"  - Room classes found: {room_classes.tolist()}")
        print(f"  - Boundary classes found: {boundary_classes.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from models.model import ModernFloorPlanNet
        
        model = ModernFloorPlanNet(
            encoder_name='efficientnet-b4',
            num_room_classes=16,
            num_boundary_classes=3,
            dropout=0.1
        )
        
        print(f"✓ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful:")
        print(f"  - Room logits shape: {output['room_logits'].shape}")
        print(f"  - Boundary logits shape: {output['boundary_logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_function():
    """Test loss function"""
    print("\nTesting loss function...")
    
    try:
        from models.losses import MultiTaskLoss
        
        criterion = MultiTaskLoss(
            room_weight=1.0,
            boundary_weight=1.0,
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        # Create dummy predictions and targets
        batch_size, height, width = 2, 64, 64
        room_logits = torch.randn(batch_size, 16, height, width)
        boundary_logits = torch.randn(batch_size, 3, height, width)
        
        room_targets = torch.randint(0, 16, (batch_size, height, width))
        boundary_targets = torch.randint(0, 3, (batch_size, height, width))
        
        predictions = {
            'room_logits': room_logits,
            'boundary_logits': boundary_logits
        }
        
        targets = {
            'room_mask': room_targets,
            'boundary_mask': boundary_targets
        }
        
        loss_dict = criterion(predictions, targets)
        
        print(f"✓ Loss calculation successful:")
        print(f"  - Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"  - Room loss: {loss_dict['room_loss'].item():.4f}")
        print(f"  - Boundary loss: {loss_dict['boundary_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing loss function: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loaders(data_dir):
    """Test data loader creation"""
    print("\nTesting data loader creation...")
    
    try:
        from utils.data_loader import create_data_loaders_custom
        
        train_loader, val_loader = create_data_loaders_custom(
            data_dir=data_dir,
            batch_size=2,
            num_workers=0,  # Disable multiprocessing for testing
            image_size=256,
            train_split=0.7,
            val_split=0.2
        )
        
        print(f"✓ Data loaders created:")
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Val loader: {len(val_loader)} batches")
        
        # Test loading one batch
        for batch in train_loader:
            print(f"✓ Batch loaded successfully:")
            print(f"  - Batch size: {batch['image'].shape[0]}")
            print(f"  - Image shape: {batch['image'].shape}")
            print(f"  - Room mask shape: {batch['room_mask'].shape}")
            print(f"  - Boundary mask shape: {batch['boundary_mask'].shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("FLOOR PLAN TRAINING SETUP TEST")
    print("="*60)
    
    data_dir = "data/dataset"  # Update this to your actual data directory
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", lambda: test_data_loading(data_dir)),
        ("Model Creation", test_model_creation),
        ("Loss Function", test_loss_function),
        ("Data Loaders", lambda: test_data_loaders(data_dir)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running {test_name} Test")
        print(f"{'-'*40}")
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} test PASSED")
        else:
            print(f"✗ {test_name} test FAILED")
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to start training.")
        print("\nTo start training, run:")
        print(f"python train.py --data_dir {data_dir}")
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")
        
        print("\nCommon fixes:")
        print("1. Make sure all required packages are installed: pip install -r requirements.txt")
        print("2. Check that your data directory path is correct")
        print("3. Verify that image and label files exist in the data directory")

if __name__ == '__main__':
    main()