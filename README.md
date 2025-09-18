# Modern Floor Plan Recognition System

A modern implementation of deep learning-based floor plan recognition using PyTorch, featuring multi-task learning for room segmentation and boundary detection with JSON output format.

## Features

- **Modern Architecture**: Built with EfficientNet backbone and U-Net decoder
- **Multi-task Learning**: Simultaneous room segmentation and boundary detection
- **Attention Mechanism**: Room-boundary guided attention for improved accuracy
- **JSON Output**: Structured output compatible with 3D floor plan tools
- **Advanced Postprocessing**: Intelligent cleaning and validation of predictions
- **Flexible Training**: Support for mixed precision training, early stopping, and multiple optimizers

## Project Structure

```
floor_plan_recognition/
├── config/
│   └── config.yaml                 # Configuration file
├── data/
│   ├── train/                      # Training images and labels
│   ├── val/                        # Validation images and labels
│   └── test/                       # Test images
├── models/
│   ├── __init__.py
│   ├── model.py                    # Model architecture
│   └── losses.py                   # Loss functions
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset and data loading
│   ├── preprocessing.py            # Data preprocessing utilities
│   └── postprocessing.py           # Postprocessing and validation
├── train.py                        # Training script
├── predict.py                      # Prediction script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd floor_plan_recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   Based on your images, organize your data as follows:
   ```
   data/
   ├── train/
   │   ├── image1.jpg
   │   ├── image1_room.png          # Room segmentation labels
   │   ├── image1_wall.png          # Wall labels  
   │   ├── image1_close.png         # Door/window labels
   │   ├── image2.jpg
   │   └── ...
   ├── val/
   │   └── ... (same structure)
   └── test/
       ├── test1.jpg
       └── ...
   ```

## Label Format

Your dataset should include the following label types:
- **Room labels** (`*_room.png`): Segmentation masks with room types (0=background, 1=closet, 2=bathroom, 3=living_room, 4=bedroom, 5=hall, 6=balcony, 7=kitchen, 8=dining_room)
- **Wall labels** (`*_wall.png`): Binary masks marking walls (255=wall, 0=background)  
- **Opening labels** (`*_close.png`): Binary masks marking doors/windows (255=opening, 0=background)

## Configuration

Edit `config/config.yaml` to customize:

- **Data paths**: Update paths to your dataset
- **Model settings**: Choose backbone (efficientnet-b4, resnet50, etc.)
- **Training parameters**: Batch size, learning rate, epochs
- **Output settings**: Where to save results

Key parameters:
```yaml
data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: 512
  batch_size: 4

model:
  backbone: "efficientnet-b4"
  num_room_classes: 9
  num_boundary_classes: 3

training:
  epochs: 100
  learning_rate: 0.001
  mixed_precision: true
```

## Training

### Start Training
```bash
python train.py --config config/config.yaml
```

### Monitor Training
- View logs in `logs/` directory
- Use TensorBoard: `tensorboard --logdir logs`
- Optional: Enable Weights & Biases logging in config

### Training Features
- **Mixed Precision**: Faster training with lower memory usage
- **Early Stopping**: Prevents overfitting
- **Cosine Annealing**: Learning rate scheduling
- **Focal Loss**: Handles class imbalance
- **Multi-task Loss**: Balanced room and boundary learning

## Prediction

### Single Image Prediction
```bash
python predict.py \
  --config config/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/image.jpg \
  --output result.json
```

### Batch Prediction
```bash
python predict.py \
  --config config/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input data/test/ \
  --output outputs/ \
  --batch
```

## Output Format

The system outputs structured JSON similar to your design.json example:

```json
{
  "floorplanner": {
    "version": "2.0.1a",
    "corners": { "corner_0": {"x": -250, "y": 250, "elevation": 250} },
    "walls": [
      {
        "corner1": "corner_0",
        "corner2": "corner_1", 
        "frontTexture": {...},
        "wallType": "STRAIGHT"
      }
    ],
    "rooms": { "room_id": {"name": "bedroom"} },
    "newFloorTextures": { "room_id": {...} }
  },
  "items": [
    {
      "itemName": "Door",
      "itemType": 7,
      "position": [0, 105, 0]
    }
  ]
}
```

## Model Architecture

### Backbone
- **EfficientNet-B4**: Efficient and accurate feature extraction
- **Pre-trained**: ImageNet initialization for faster convergence

### Multi-task Heads
- **Room Branch**: 9-class segmentation for room types
- **Boundary Branch**: 3-class segmentation (background, opening, wall)

### Attention Mechanism
- **Cross-task Guidance**: Boundary features guide room segmentation
- **Residual Connections**: Stable training and gradient flow

### Loss Functions
- **Focal Loss**: Addresses class imbalance in segmentation
- **Multi-task Weighting**: Balanced learning across tasks

## Advanced Features

### Postprocessing Pipeline
1. **Mask Cleaning**: Remove noise and small components
2. **Morphological Operations**: Connect broken lines and fill holes
3. **Room Extraction**: Instance segmentation of individual rooms
4. **Geometry Processing**: Wall line extraction and simplification
5. **Validation**: Check floor plan consistency

### Data Augmentation
- **Geometric**: Rotation, flipping, scaling
- **Color**: Brightness, contrast, hue adjustments
- **Noise**: Gaussian and ISO noise simulation

## Performance Tips

### Training Optimization
- **Batch Size**: Use largest batch size that fits in GPU memory
- **Mixed Precision**: Enable for 40-50% speedup with minimal accuracy loss
- **Learning Rate**: Start with 1e-3, reduce if loss plateaus
- **Data Loading**: Increase num_workers for faster data loading

### Inference Optimization
- **Model Quantization**: Reduce model size for deployment
- **Batch Inference**: Process multiple images together
- **TensorRT**: NVIDIA GPU acceleration for production

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or image size
2. **No Convergence**: Lower learning rate or check data labels
3. **Poor Boundaries**: Increase boundary loss weight
4. **Over-segmentation**: Post-process with larger min_component_size

### Data Issues
- Ensure labels match image filenames exactly
- Check label value ranges (0-8 for rooms, 0-2 for boundaries)
- Verify image and label dimensions match

## Hyperparameter Recommendations

### For Small Datasets (<1000 images)
```yaml
training:
  learning_rate: 0.0005
  batch_size: 2
  epochs: 150
model:
  dropout: 0.2
```

### For Large Datasets (>5000 images)  
```yaml
training:
  learning_rate: 0.001
  batch_size: 8
  epochs: 80
model:
  dropout: 0.1
```

## Future Improvements

- **3D Room Height**: Add height prediction for 3D models
- **Furniture Detection**: Integrate object detection for room items  
- **Style Transfer**: Generate different architectural styles
- **Interactive Editing**: Tools for manual correction

## Citation

If you use this code, please cite the original paper:
```bibtex
@InProceedings{zeng2019deepfloor,
    author = {Zhiliang ZENG, Xianzhi LI, Ying Kin Yu, and Chi-Wing Fu},
    title = {Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```