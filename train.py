# Path: train.py

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

# Fixed imports based on your folder structure
from models.model import ModernFloorPlanNet, get_stable_model_for_training
from utils.data_loader import create_data_loaders_custom

class FloorPlanTrainerCustom:
    """
    Custom trainer for your specific dataset format with 16 room classes
    """
    
    def __init__(self, data_dir: str, config_path: str = None):
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config for your 679 images with 16 classes
            self.config = {
                'data': {
                    'batch_size': 4,  # Increased for CUDA
                    'num_workers': 4,  # For CUDA parallel loading
                    'image_size': 256,
                    'train_split': 0.7,
                    'val_split': 0.2
                },
                'model': {
                    'backbone': 'efficientnet-b4',
                    'num_room_classes': 16,  # Your 16 classes
                    'num_boundary_classes': 3,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'weight_decay': 1e-4,
                    'mixed_precision': True,  # Enable for CUDA
                    'early_stopping_patience': 10
                },
                'output': {
                    'checkpoint_dir': 'checkpoints',
                    'log_dir': 'logs'
                }
            }
        
        # Setup device - Force CUDA
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check your GPU setup.")
        
        print(f"Using device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Create directories
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.log_dir = Path(self.config['output']['log_dir'])
        
        for dir_path in [self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = get_stable_model_for_training().to(self.device)
        
        # Initialize loss functions with CUDA support
        self.room_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.boundary_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=1e-6
        )
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Best validation score tracking
        self.best_val_score = 0.0
        self.patience_counter = 0
        
        # Mixed precision training for CUDA
        if self.config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
            print("Mixed precision disabled")
        
        # Data directory
        self.data_dir = data_dir
    
    def _calculate_iou(self, pred, target, num_classes=None):
        """Calculate IoU score with proper CUDA handling"""
        device = self.device
        
        # Ensure tensors are on CUDA
        pred = pred.to(device)
        target = target.to(device)
        
        if num_classes is None:
            num_classes = max(torch.max(pred).item(), torch.max(target).item()) + 1
        
        iou_scores = []
        for cls in range(1, num_classes):  # Skip background (class 0)
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                iou = intersection / union
            else:
                # Create tensor on correct device
                iou = torch.tensor(1.0 if intersection == 0 else 0.0, device=device)
            
            iou_scores.append(iou.to(device))
        
        if iou_scores:
            return torch.stack(iou_scores).mean()
        else:
            return torch.tensor(0.0, device=device)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with CUDA optimization"""
        self.model.train()
        total_loss = 0.0
        room_scores = []
        boundary_scores = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move all data to CUDA with non_blocking for speed
                images = batch['image'].to(self.device, non_blocking=True)
                room_masks = batch['room_mask'].to(self.device, non_blocking=True)
                boundary_masks = batch['boundary_mask'].to(self.device, non_blocking=True)
                
                # Ensure data types are correct
                room_masks = room_masks.long()
                boundary_masks = boundary_masks.long()
                
                self.optimizer.zero_grad()
                
                if self.scaler:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        
                        # Ensure predictions are on CUDA
                        room_logits = predictions['room_logits'].to(self.device)
                        boundary_logits = predictions['boundary_logits'].to(self.device)
                        
                        # Calculate losses
                        room_loss = self.room_criterion(room_logits, room_masks)
                        boundary_loss = self.boundary_criterion(boundary_logits, boundary_masks)
                        loss = room_loss + boundary_loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular training
                    predictions = self.model(images)
                    
                    # Ensure predictions are on CUDA
                    room_logits = predictions['room_logits'].to(self.device)
                    boundary_logits = predictions['boundary_logits'].to(self.device)
                    
                    # Calculate losses
                    room_loss = self.room_criterion(room_logits, room_masks)
                    boundary_loss = self.boundary_criterion(boundary_logits, boundary_masks)
                    loss = room_loss + boundary_loss
                    
                    loss.backward()
                    self.optimizer.step()
                
                # Calculate metrics with CUDA tensors
                with torch.no_grad():
                    room_pred = torch.softmax(room_logits, dim=1)
                    boundary_pred = torch.softmax(boundary_logits, dim=1)
                    
                    # Convert to class predictions
                    room_pred_class = torch.argmax(room_pred, dim=1)
                    boundary_pred_class = torch.argmax(boundary_pred, dim=1)
                    
                    room_iou = self._calculate_iou(room_pred_class, room_masks, self.config['model']['num_room_classes'])
                    boundary_iou = self._calculate_iou(boundary_pred_class, boundary_masks, self.config['model']['num_boundary_classes'])
                    
                    room_scores.append(room_iou.item())
                    boundary_scores.append(boundary_iou.item())
                    total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'room_iou': f'{room_iou.item():.4f}',
                    'boundary_iou': f'{boundary_iou.item():.4f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1024**2:.0f}MB'
                })
                
                # Clear cache periodically to prevent memory issues
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                print(f"Batch shapes - Images: {batch['image'].shape if 'image' in batch else 'None'}")
                print(f"Room masks: {batch['room_mask'].shape if 'room_mask' in batch else 'None'}")
                print(f"Boundary masks: {batch['boundary_mask'].shape if 'boundary_mask' in batch else 'None'}")
                import traceback
                traceback.print_exc()
                
                # Clear CUDA cache on error
                torch.cuda.empty_cache()
                continue
        
        # Calculate average metrics
        avg_loss = total_loss / max(len(train_loader), 1)
        avg_room_iou = np.mean(room_scores) if room_scores else 0.0
        avg_boundary_iou = np.mean(boundary_scores) if boundary_scores else 0.0
        
        return avg_loss, avg_room_iou, avg_boundary_iou
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch with CUDA optimization"""
        self.model.eval()
        total_loss = 0.0
        room_scores = []
        boundary_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                try:
                    # Move all data to CUDA
                    images = batch['image'].to(self.device, non_blocking=True)
                    room_masks = batch['room_mask'].to(self.device, non_blocking=True)
                    boundary_masks = batch['boundary_mask'].to(self.device, non_blocking=True)
                    
                    # Ensure data types are correct
                    room_masks = room_masks.long()
                    boundary_masks = boundary_masks.long()
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            predictions = self.model(images)
                    else:
                        predictions = self.model(images)
                    
                    # Ensure predictions are on CUDA
                    room_logits = predictions['room_logits'].to(self.device)
                    boundary_logits = predictions['boundary_logits'].to(self.device)
                    
                    # Calculate losses
                    room_loss = self.room_criterion(room_logits, room_masks)
                    boundary_loss = self.boundary_criterion(boundary_logits, boundary_masks)
                    loss = room_loss + boundary_loss
                    
                    # Calculate metrics
                    room_pred = torch.softmax(room_logits, dim=1)
                    boundary_pred = torch.softmax(boundary_logits, dim=1)
                    
                    room_pred_class = torch.argmax(room_pred, dim=1)
                    boundary_pred_class = torch.argmax(boundary_pred, dim=1)
                    
                    room_iou = self._calculate_iou(room_pred_class, room_masks, self.config['model']['num_room_classes'])
                    boundary_iou = self._calculate_iou(boundary_pred_class, boundary_masks, self.config['model']['num_boundary_classes'])
                    
                    room_scores.append(room_iou.item())
                    boundary_scores.append(boundary_iou.item())
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"\nError in validation batch: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        # Calculate average metrics
        avg_loss = total_loss / max(len(val_loader), 1)
        avg_room_iou = np.mean(room_scores) if room_scores else 0.0
        avg_boundary_iou = np.mean(boundary_scores) if boundary_scores else 0.0
        
        return avg_loss, avg_room_iou, avg_boundary_iou
    
    def train(self):
        """Main training loop"""
        print("Creating data loaders from single directory...")
        
        # Create data loaders from your single directory
        try:
            train_loader, val_loader = create_data_loaders_custom(
                data_dir=self.data_dir,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                image_size=self.config['data']['image_size'],
                train_split=self.config['data']['train_split'],
                val_split=self.config['data']['val_split']
            )
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            print("Make sure your utils/data_loader.py has the create_data_loaders_custom function")
            import traceback
            traceback.print_exc()
            return
        
        print(f"Created train loader with {len(train_loader)} batches")
        print(f"Created val loader with {len(val_loader)} batches")
        
        if len(train_loader) == 0:
            print("No training data found!")
            return
            
        print("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            train_loss, train_room_iou, train_boundary_iou = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            if len(val_loader) > 0:
                val_loss, val_room_iou, val_boundary_iou = self.validate_epoch(val_loader, epoch)
            else:
                val_loss, val_room_iou, val_boundary_iou = train_loss, train_room_iou, train_boundary_iou
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate combined validation score
            val_score = (val_room_iou + val_boundary_iou) / 2
            
            # Logging
            print(f'\nEpoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Room IoU: {train_room_iou:.4f}, Boundary IoU: {train_boundary_iou:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Room IoU: {val_room_iou:.4f}, Boundary IoU: {val_boundary_iou:.4f}')
            print(f'  LR: {current_lr:.6f}')
            print(f'  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB / {torch.cuda.memory_reserved()/1024**2:.0f}MB')
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('IoU/Train_Room', train_room_iou, epoch)
            self.writer.add_scalar('IoU/Train_Boundary', train_boundary_iou, epoch)
            self.writer.add_scalar('IoU/Val_Room', val_room_iou, epoch)
            self.writer.add_scalar('IoU/Val_Boundary', val_boundary_iou, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('GPU_Memory_MB', torch.cuda.memory_allocated()/1024**2, epoch)
            
            # Save best model
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_score': self.best_val_score,
                    'config': self.config,
                    'class_mapping': {
                        0: 0, 1: 1, 180: 2, 181: 3, 194: 4, 195: 5, 221: 6, 222: 7,
                        223: 8, 224: 9, 232: 10, 233: 11, 235: 12, 236: 13, 237: 14, 238: 15
                    }
                }
                
                torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
                print(f'  New best model saved! Val Score: {val_score:.4f}')
                
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_score': val_score,
                    'config': self.config
                }
                torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
            # Clear CUDA cache at the end of each epoch
            torch.cuda.empty_cache()
        
        print("\nTraining completed!")
        print(f"Best validation score: {self.best_val_score:.4f}")
        print(f"Best model saved at: {self.checkpoint_dir / 'best_model.pth'}")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train Floor Plan Recognition Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to your dataset directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available! Please install PyTorch with CUDA support.")
        return
    
    # Initialize trainer and start training
    try:
        trainer = FloorPlanTrainerCustom(args.data_dir, args.config)
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Clear CUDA cache on exit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()