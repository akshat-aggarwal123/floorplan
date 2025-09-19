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
from models.model import ModernFloorPlanNet
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
                    'batch_size': 2,  # Small batch for stability
                    'num_workers': 0,  # Set to 0 for Windows compatibility
                    'image_size': 256,  # Reduced for faster training
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
                    'mixed_precision': False,  # Disable for CPU
                    'early_stopping_patience': 10
                },
                'output': {
                    'checkpoint_dir': 'checkpoints',
                    'log_dir': 'logs'
                }
            }
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.log_dir = Path(self.config['output']['log_dir'])
        
        for dir_path in [self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = ModernFloorPlanNet(
            encoder_name=self.config['model']['backbone'],
            num_room_classes=self.config['model']['num_room_classes'],
            num_boundary_classes=self.config['model']['num_boundary_classes'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Initialize simple loss functions (instead of complex MultiTaskLoss)
        self.room_criterion = torch.nn.CrossEntropyLoss()
        self.boundary_criterion = torch.nn.CrossEntropyLoss()
        
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
        
        # Mixed precision training - only if using GPU
        if self.device.type == 'cuda' and self.config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            print("Mixed precision disabled (CPU or config setting)")
        
        # Data directory
        self.data_dir = data_dir
    
    def _calculate_iou(self, pred, target, num_classes=None):
        """Calculate IoU score"""
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
                iou = torch.tensor(1.0 if intersection == 0 else 0.0)
            
            iou_scores.append(iou)
        
        return torch.stack(iou_scores).mean() if iou_scores else torch.tensor(0.0)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        room_scores = []
        boundary_scores = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device)
                room_masks = batch['room_mask'].to(self.device)
                boundary_masks = batch['boundary_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.scaler:
                    # Mixed precision training (GPU only)
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        
                        # Calculate losses using simple criterion
                        room_loss = self.room_criterion(predictions['room_logits'], room_masks)
                        boundary_loss = self.boundary_criterion(predictions['boundary_logits'], boundary_masks)
                        loss = room_loss + boundary_loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular training
                    predictions = self.model(images)
                    
                    # Calculate losses using simple criterion
                    room_loss = self.room_criterion(predictions['room_logits'], room_masks)
                    boundary_loss = self.boundary_criterion(predictions['boundary_logits'], boundary_masks)
                    loss = room_loss + boundary_loss
                    
                    loss.backward()
                    self.optimizer.step()
                
                # Calculate metrics
                room_pred = torch.softmax(predictions['room_logits'], dim=1)
                boundary_pred = torch.softmax(predictions['boundary_logits'], dim=1)
                
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
                    'boundary_iou': f'{boundary_iou.item():.4f}'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate average metrics
        avg_loss = total_loss / max(len(train_loader), 1)
        avg_room_iou = np.mean(room_scores) if room_scores else 0.0
        avg_boundary_iou = np.mean(boundary_scores) if boundary_scores else 0.0
        
        return avg_loss, avg_room_iou, avg_boundary_iou
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        room_scores = []
        boundary_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                try:
                    images = batch['image'].to(self.device)
                    room_masks = batch['room_mask'].to(self.device)
                    boundary_masks = batch['boundary_mask'].to(self.device)
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            predictions = self.model(images)
                    else:
                        predictions = self.model(images)
                    
                    # Calculate losses using simple criterion
                    room_loss = self.room_criterion(predictions['room_logits'], room_masks)
                    boundary_loss = self.boundary_criterion(predictions['boundary_logits'], boundary_masks)
                    loss = room_loss + boundary_loss
                    
                    # Calculate metrics
                    room_pred = torch.softmax(predictions['room_logits'], dim=1)
                    boundary_pred = torch.softmax(predictions['boundary_logits'], dim=1)
                    
                    room_pred_class = torch.argmax(room_pred, dim=1)
                    boundary_pred_class = torch.argmax(boundary_pred, dim=1)
                    
                    room_iou = self._calculate_iou(room_pred_class, room_masks, self.config['model']['num_room_classes'])
                    boundary_iou = self._calculate_iou(boundary_pred_class, boundary_masks, self.config['model']['num_boundary_classes'])
                    
                    room_scores.append(room_iou.item())
                    boundary_scores.append(boundary_iou.item())
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"\nError in validation batch: {e}")
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
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('IoU/Train_Room', train_room_iou, epoch)
            self.writer.add_scalar('IoU/Train_Boundary', train_boundary_iou, epoch)
            self.writer.add_scalar('IoU/Val_Room', val_room_iou, epoch)
            self.writer.add_scalar('IoU/Val_Boundary', val_boundary_iou, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
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
    
    # Initialize trainer and start training
    try:
        trainer = FloorPlanTrainerCustom(args.data_dir, args.config)
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()