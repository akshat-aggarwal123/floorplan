# Path: predict.py

import os
import json
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any

from models.model import ModernFloorPlanNet
from utils.data_loader import get_transforms
from utils.postprocessing import FloorPlanPostProcessor

class FloorPlanPredictor:
    """
    Floor Plan Recognition Predictor with JSON output
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ModernFloorPlanNet(
            encoder_name=self.config['model']['backbone'],
            num_room_classes=self.config['model']['num_room_classes'],
            num_boundary_classes=self.config['model']['num_boundary_classes'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize transforms
        self.transforms = get_transforms(
            self.config['data']['image_size'], 
            mode='val'
        )
        
        # Initialize postprocessor
        self.postprocessor = FloorPlanPostProcessor(
            room_types=self.config['room_types']
        )
        
        print("Model loaded successfully!")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation score: {checkpoint.get('best_val_score', 'N/A')}")
    
    def _preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess input image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Resize to model input size
        image = cv2.resize(image, (self.config['data']['image_size'], 
                                  self.config['data']['image_size']))
        
        # Apply transforms
        transformed = self.transforms(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def _predict_single_image(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Predict on single image"""
        with torch.no_grad():
            predictions = self.model(image_tensor.to(self.device))
            
            # Get predictions
            room_logits = predictions['room_logits']
            boundary_logits = predictions['boundary_logits']
            
            # Apply softmax and get predictions
            room_pred = torch.softmax(room_logits, dim=1)
            boundary_pred = torch.softmax(boundary_logits, dim=1)
            
            # Convert to numpy
            room_pred = room_pred.cpu().numpy()[0]  # Remove batch dimension
            boundary_pred = boundary_pred.cpu().numpy()[0]
            
            # Get class predictions
            room_mask = np.argmax(room_pred, axis=0)
            boundary_mask = np.argmax(boundary_pred, axis=0)
            
            return {
                'room_mask': room_mask,
                'boundary_mask': boundary_mask,
                'room_probabilities': room_pred,
                'boundary_probabilities': boundary_pred
            }
    
    def predict_image(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Predict floor plan layout from image and return JSON format
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save JSON output
            
        Returns:
            Dictionary containing floor plan data in design.json format
        """
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image = self._preprocess_image(image_path)
        
        # Get predictions
        predictions = self._predict_single_image(image_tensor)
        
        # Postprocess predictions
        processed_results = self.postprocessor.process_predictions(
            predictions,
            original_image.shape[:2]  # (height, width)
        )
        
        # Convert to design JSON format
        design_json = self._convert_to_design_format(
            processed_results,
            original_image.shape
        )
        
        # Save JSON output if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(design_json, f, indent=2)
            print(f"JSON output saved to: {output_path}")
        
        return design_json
    
    def _convert_to_design_format(self, processed_results: Dict, image_shape: Tuple) -> Dict[str, Any]:
        """
        Convert predictions to design.json format similar to your example
        """
        height, width = image_shape[:2]
        
        # Initialize design structure
        design = {
            "floorplanner": {
                "version": "2.0.1a",
                "corners": {},
                "walls": [],
                "rooms": {},
                "wallTextures": [],
                "floorTextures": {},
                "newFloorTextures": {},
                "carbonSheet": {},
                "boundary": {
                    "points": [
                        {"x": -width//40, "y": -height//40, "elevation": 5},
                        {"x": width//40, "y": -height//40, "elevation": 5},
                        {"x": width//40, "y": height//40, "elevation": 5},
                        {"x": -width//40, "y": height//40, "elevation": 5}
                    ],
                    "style": {"color": "", "type": "texture", "colormap": "", "repeat": 3000}
                },
                "units": "m"
            },
            "items": [],
            "lights": [],
            "sunlight": [{
                "name": "SunLight",
                "position": {"x": 0, "y": 1000, "z": 1000},
                "intensity": 5,
                "color": 16777215,
                "shadow": True,
                "target": {"x": width//8, "y": 0, "z": -height//8}
            }],
            "hemlight": [{
                "name": "HemisphereLight",
                "position": {"x": 0, "y": 0, "z": 0},
                "intensity": 0.6,
                "color": 16777215
            }],
            "amblight": [{
                "name": "AmbientLight",
                "position": {"x": 0, "y": 0, "z": 0},
                "intensity": 2.5,
                "color": 16775392
            }]
        }
        
        # Process detected rooms
        room_data = processed_results.get('rooms', {})
        
        # Generate corners and walls from room boundaries
        corners, walls = self._generate_corners_and_walls(
            processed_results.get('walls', []),
            processed_results.get('boundaries', [])
        )
        
        design["floorplanner"]["corners"] = corners
        design["floorplanner"]["walls"] = walls
        
        # Process rooms
        rooms_dict = {}
        floor_textures = {}
        
        for room_id, room_info in room_data.items():
            room_type = room_info.get('type', 'unknown')
            room_name = self.config['room_types'].get(room_info.get('class_id', 0), 'Unknown Room')
            
            # Create room entry
            rooms_dict[room_id] = {"name": room_name}
            
            # Add floor texture based on room type
            texture_config = self._get_room_texture(room_type)
            if texture_config:
                floor_textures[room_id] = texture_config
        
        design["floorplanner"]["rooms"] = rooms_dict
        design["floorplanner"]["newFloorTextures"] = floor_textures
        
        # Add detected doors and windows as items
        openings = processed_results.get('openings', [])
        items = self._generate_opening_items(openings)
        design["items"] = items
        
        return design
    
    def _generate_corners_and_walls(self, walls: List, boundaries: List) -> Tuple[Dict, List]:
        """Generate corners and walls from detected boundaries"""
        corners = {}
        walls = []
        
        # This is a simplified implementation
        # In practice, you'd need sophisticated wall/corner detection
        corner_id_counter = 0
        
        # Create basic rectangular layout as fallback
        basic_corners = [
            {"x": -250, "y": 250, "elevation": 250},
            {"x": -250, "y": -250, "elevation": 250},
            {"x": 250, "y": -250, "elevation": 250},
            {"x": 250, "y": 250, "elevation": 250}
        ]
        
        corner_ids = []
        for i, corner in enumerate(basic_corners):
            corner_id = f"corner_{corner_id_counter}"
            corners[corner_id] = corner
            corner_ids.append(corner_id)
            corner_id_counter += 1
        
        # Create walls connecting corners
        wall_texture = {
            "name": "painted_wall_brown",
            "repeat": 129,
            "colormap": "textures/Wall/painted_wall_brown/painted_wall-diffuse.jpg",
            "color": "#FFFFFF",
            "emissive": "#000000",
            "reflective": 0.5,
            "shininess": 0.5
        }
        
        for i in range(len(corner_ids)):
            next_i = (i + 1) % len(corner_ids)
            wall = {
                "corner1": corner_ids[i],
                "corner2": corner_ids[next_i],
                "frontTexture": wall_texture,
                "backTexture": wall_texture,
                "wallType": "STRAIGHT",
                "thickness": 0.1
            }
            walls.append(wall)
        
        return corners, walls
    
    def _generate_opening_items(self, openings: List) -> List[Dict]:
        """Generate door and window items"""
        items = []
        
        for opening in openings:
            if opening.get('type') == 'door':
                item = {
                    "itemName": "Door",
                    "thumbnail": "models/thumbnails/door1.png",
                    "itemType": 7,
                    "position": opening.get('position', [0, 105, 0]),
                    "rotation": opening.get('rotation', [0, 0, 0]),
                    "size": opening.get('size', [130, 210, 20]),
                    "modelURL": "models/glb/InWallFloorItem/door1.glb",
                    "fixed": False,
                    "resizable": False
                }
            elif opening.get('type') == 'window':
                item = {
                    "itemName": "Window",
                    "thumbnail": "models/thumbnails/windows1.png",
                    "itemType": 3,
                    "position": opening.get('position', [0, 140, 0]),
                    "rotation": opening.get('rotation', [0, 0, 0]),
                    "size": opening.get('size', [180, 120, 15]),
                    "modelURL": "models/glb/InWallItem/windows1.glb",
                    "fixed": False,
                    "resizable": False
                }
            else:
                continue
            
            items.append(item)
        
        return items
    
    def _get_room_texture(self, room_type: str) -> Dict:
        """Get appropriate floor texture for room type"""
        texture_mapping = {
            'bedroom': {
                "repeat": 500,
                "colormap": "textures/Floor/LightFineWood/light_fine_wood.jpg",
                "color": "#FFFFFF",
                "emissive": "#000000",
                "reflective": 0.1,
                "shininess": 0.5
            },
            'bathroom': {
                "name": "Herringbone_MarbleTiles",
                "repeat": 250,
                "colormap": "textures/Floor/HerringboneMarbleTiles/HerringboneMarbleTiles01_BaseColor.png",
                "normalmap": "textures/Floor/HerringboneMarbleTiles/HerringboneMarbleTiles01_Normal.png",
                "color": "#FFFFFF",
                "emissive": "#000000",
                "reflective": 0.5,
                "shininess": 0.5
            },
            'living_room': {
                "repeat": 400,
                "colormap": "textures/Floor/ParquetFloor/parquet_floor.jpg",
                "color": "#FFFFFF",
                "emissive": "#000000",
                "reflective": 0.3,
                "shininess": 0.4
            }
        }
        
        return texture_mapping.get(room_type, texture_mapping['living_room'])
    
    def predict_batch(self, input_dir: str, output_dir: str):
        """Predict on batch of images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            try:
                # Generate output filename
                output_file = output_path / f"{image_file.stem}_floorplan.json"
                
                # Process image
                result = self.predict_image(str(image_file), str(output_file))
                print(f"Processed: {image_file.name}")
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Floor Plan Recognition Prediction')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path or directory')
    parser.add_argument('--batch', action='store_true',
                        help='Process batch of images')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FloorPlanPredictor(args.config, args.checkpoint)
    
    if args.batch:
        # Process batch of images
        predictor.predict_batch(args.input, args.output)
    else:
        # Process single image
        predictor.predict_image(args.input, args.output)
        print("Prediction completed!")

if __name__ == '__main__':
    main()