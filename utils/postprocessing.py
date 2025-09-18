# Path: utils/postprocessing.py

import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology
from typing import Dict, List, Tuple, Any
import uuid

class FloorPlanPostProcessor:
    """
    Postprocessing utilities for floor plan predictions
    """
    
    def __init__(self, room_types: Dict[int, str]):
        self.room_types = room_types
        
    def process_predictions(
        self, 
        predictions: Dict[str, np.ndarray], 
        target_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Process raw model predictions into structured floor plan data
        
        Args:
            predictions: Dictionary containing model predictions
            target_size: Target output size (height, width)
            
        Returns:
            Dictionary with processed floor plan components
        """
        room_mask = predictions['room_mask']
        boundary_mask = predictions['boundary_mask']
        
        # Resize to target size
        room_mask = cv2.resize(
            room_mask.astype(np.uint8), 
            (target_size[1], target_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        boundary_mask = cv2.resize(
            boundary_mask.astype(np.uint8), 
            (target_size[1], target_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Clean up predictions
        room_mask = self._clean_room_mask(room_mask)
        boundary_mask = self._clean_boundary_mask(boundary_mask)
        
        # Extract components
        rooms = self._extract_rooms(room_mask)
        walls = self._extract_walls(boundary_mask)
        openings = self._extract_openings(boundary_mask)
        
        return {
            'room_mask': room_mask,
            'boundary_mask': boundary_mask,
            'rooms': rooms,
            'walls': walls,
            'openings': openings,
            'boundaries': self._combine_boundaries(walls, openings)
        }
    
    def _clean_room_mask(self, room_mask: np.ndarray) -> np.ndarray:
        """Clean up room segmentation mask"""
        cleaned_mask = room_mask.copy()
        
        # Remove small isolated regions for each room type
        for room_id in np.unique(room_mask):
            if room_id == 0:  # Skip background
                continue
                
            # Create binary mask for this room
            binary_mask = (room_mask == room_id).astype(np.uint8)
            
            # Remove small components
            binary_mask = self._remove_small_components(binary_mask, min_size=100)
            
            # Fill holes
            binary_mask = self._fill_holes(binary_mask)
            
            # Morphological opening to separate connected rooms
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Update cleaned mask
            cleaned_mask[binary_mask == 0] = np.where(
                cleaned_mask[binary_mask == 0] == room_id, 
                0, 
                cleaned_mask[binary_mask == 0]
            )
            
        return cleaned_mask
    
    def _clean_boundary_mask(self, boundary_mask: np.ndarray) -> np.ndarray:
        """Clean up boundary segmentation mask"""
        cleaned_mask = boundary_mask.copy()
        
        # Clean walls (class 2)
        wall_mask = (boundary_mask == 2).astype(np.uint8)
        wall_mask = self._clean_wall_lines(wall_mask)
        cleaned_mask[wall_mask == 1] = 2
        cleaned_mask[wall_mask == 0] = np.where(
            cleaned_mask[wall_mask == 0] == 2, 0, cleaned_mask[wall_mask == 0]
        )
        
        # Clean doors/windows (class 1)
        opening_mask = (boundary_mask == 1).astype(np.uint8)
        opening_mask = self._remove_small_components(opening_mask, min_size=50)
        cleaned_mask[opening_mask == 1] = 1
        cleaned_mask[opening_mask == 0] = np.where(
            cleaned_mask[opening_mask == 0] == 1, 0, cleaned_mask[opening_mask == 0]
        )
        
        return cleaned_mask
    
    def _remove_small_components(self, binary_mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small connected components"""
        # Label connected components
        labeled = measure.label(binary_mask, connectivity=2)
        
        # Remove small components
        cleaned = morphology.remove_small_objects(
            labeled, min_size=min_size, connectivity=2
        )
        
        return (cleaned > 0).astype(np.uint8)
    
    def _fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """Fill holes in binary mask"""
        return ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    
    def _clean_wall_lines(self, wall_mask: np.ndarray) -> np.ndarray:
        """Clean and connect wall lines"""
        # Morphological closing to connect nearby wall segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        
        # Skeletonize to get thin wall lines
        wall_mask = morphology.skeletonize(wall_mask.astype(bool)).astype(np.uint8)
        
        # Dilate slightly to ensure connectivity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
        
        return wall_mask
    
    def _extract_rooms(self, room_mask: np.ndarray) -> Dict[str, Dict]:
        """Extract individual rooms from room mask"""
        rooms = {}
        
        for room_id in np.unique(room_mask):
            if room_id == 0:  # Skip background
                continue
            
            # Create binary mask for this room
            binary_mask = (room_mask == room_id).astype(np.uint8)
            
            # Find connected components (separate room instances)
            labeled = measure.label(binary_mask, connectivity=2)
            
            for region_id in np.unique(labeled):
                if region_id == 0:  # Skip background
                    continue
                
                # Get region properties
                region_mask = (labeled == region_id).astype(np.uint8)
                props = measure.regionprops(region_mask)[0]
                
                # Generate unique room ID
                unique_id = str(uuid.uuid4())[:8]
                
                # Store room information
                rooms[unique_id] = {
                    'class_id': int(room_id),
                    'type': self.room_types.get(int(room_id), 'unknown'),
                    'area': int(props.area),
                    'centroid': [float(props.centroid[1]), float(props.centroid[0])],  # (x, y)
                    'bounding_box': [
                        int(props.bbox[1]), int(props.bbox[0]),  # min_x, min_y
                        int(props.bbox[3]), int(props.bbox[2])   # max_x, max_y
                    ],
                    'mask': region_mask.tolist() if region_mask.size < 10000 else None
                }
        
        return rooms
    
    def _extract_walls(self, boundary_mask: np.ndarray) -> List[Dict]:
        """Extract wall segments"""
        wall_mask = (boundary_mask == 2).astype(np.uint8)
        walls = []
        
        # Find wall contours
        contours, _ = cv2.findContours(
            wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50:  # Skip small contours
                continue
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of points
            points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            
            walls.append({
                'id': f'wall_{i}',
                'points': points,
                'length': float(cv2.arcLength(contour, False)),
                'type': 'wall'
            })
        
        return walls
    
    def _extract_openings(self, boundary_mask: np.ndarray) -> List[Dict]:
        """Extract doors and windows"""
        opening_mask = (boundary_mask == 1).astype(np.uint8)
        openings = []
        
        # Find opening contours
        contours, _ = cv2.findContours(
            opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 30:  # Skip very small openings
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify as door or window based on size and aspect ratio
            aspect_ratio = max(w, h) / min(w, h)
            opening_type = 'door' if area > 200 and aspect_ratio < 3 else 'window'
            
            # Calculate center point
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w//2, y + h//2
            
            openings.append({
                'id': f'{opening_type}_{i}',
                'type': opening_type,
                'position': [cx, cy],
                'size': [w, h],
                'area': float(area),
                'bounding_box': [x, y, x + w, y + h],
                'rotation': [0, 0, 0]  # Default rotation
            })
        
        return openings
    
    def _combine_boundaries(self, walls: List[Dict], openings: List[Dict]) -> List[Dict]:
        """Combine walls and openings into unified boundary list"""
        boundaries = []
        
        # Add walls
        for wall in walls:
            boundaries.append({
                'type': 'wall',
                'data': wall
            })
        
        # Add openings
        for opening in openings:
            boundaries.append({
                'type': opening['type'],
                'data': opening
            })
        
        return boundaries

class GeometryProcessor:
    """
    Advanced geometry processing for floor plan elements
    """
    
    @staticmethod
    def simplify_polygon(points: List[Tuple[float, float]], tolerance: float = 2.0) -> List[Tuple[float, float]]:
        """Simplify polygon using Douglas-Peucker algorithm"""
        if len(points) < 3:
            return points
        
        def perpendicular_distance(point, line_start, line_end):
            """Calculate perpendicular distance from point to line"""
            if line_start == line_end:
                return np.linalg.norm(np.array(point) - np.array(line_start))
            
            n = abs((line_end[1] - line_start[1]) * point[0] - 
                   (line_end[0] - line_start[0]) * point[1] + 
                   line_end[0] * line_start[1] - 
                   line_end[1] * line_start[0])
            d = np.sqrt((line_end[1] - line_start[1])**2 + (line_end[0] - line_start[0])**2)
            return n / d
        
        def douglas_peucker(points_list, tolerance):
            if len(points_list) < 3:
                return points_list
            
            # Find the point with maximum distance from line between first and last
            max_distance = 0
            max_index = 0
            
            for i in range(1, len(points_list) - 1):
                distance = perpendicular_distance(
                    points_list[i], points_list[0], points_list[-1]
                )
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            
            # If max distance is greater than tolerance, recursively simplify
            if max_distance > tolerance:
                # Recursive call
                left_points = douglas_peucker(points_list[:max_index+1], tolerance)
                right_points = douglas_peucker(points_list[max_index:], tolerance)
                
                # Combine results
                return left_points[:-1] + right_points
            else:
                return [points_list[0], points_list[-1]]
        
        return douglas_peucker(points, tolerance)
    
    @staticmethod
    def detect_rectangular_rooms(room_mask: np.ndarray) -> List[Dict]:
        """Detect and fit rectangles to room shapes"""
        rectangular_rooms = []
        
        # Find contours
        contours, _ = cv2.findContours(
            room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 100:
                continue
            
            # Fit minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Calculate rectangle properties
            center, (width, height), angle = rect
            
            rectangular_rooms.append({
                'id': f'rect_room_{i}',
                'center': [float(center[0]), float(center[1])],
                'width': float(width),
                'height': float(height),
                'angle': float(angle),
                'corners': box.tolist(),
                'area': float(width * height)
            })
        
        return rectangular_rooms

class FloorPlanValidator:
    """
    Validate and correct floor plan predictions
    """
    
    @staticmethod
    def validate_room_connectivity(rooms: Dict, walls: List) -> Dict[str, List[str]]:
        """Check which rooms are connected"""
        connections = {}
        
        for room_id in rooms:
            connections[room_id] = []
        
        # Simple connectivity check based on room centroids and walls
        # In practice, this would be more sophisticated
        room_centroids = {
            room_id: room_data['centroid'] 
            for room_id, room_data in rooms.items()
        }
        
        for room_id1, centroid1 in room_centroids.items():
            for room_id2, centroid2 in room_centroids.items():
                if room_id1 != room_id2:
                    # Check if rooms are close enough to be connected
                    distance = np.linalg.norm(
                        np.array(centroid1) - np.array(centroid2)
                    )
                    if distance < 200:  # Threshold for connectivity
                        connections[room_id1].append(room_id2)
        
        return connections
    
    @staticmethod
    def check_plan_validity(processed_results: Dict) -> Dict[str, Any]:
        """Perform comprehensive validation of floor plan"""
        validation_report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        rooms = processed_results.get('rooms', {})
        walls = processed_results.get('walls', [])
        openings = processed_results.get('openings', [])
        
        # Check minimum requirements
        if len(rooms) == 0:
            validation_report['errors'].append("No rooms detected")
            validation_report['is_valid'] = False
        
        if len(walls) == 0:
            validation_report['warnings'].append("No walls detected")
        
        # Check room sizes
        for room_id, room_data in rooms.items():
            area = room_data.get('area', 0)
            if area < 500:  # Very small room
                validation_report['warnings'].append(
                    f"Room {room_id} ({room_data.get('type', 'unknown')}) "
                    f"has very small area: {area} pixels"
                )
            elif area > 50000:  # Very large room
                validation_report['warnings'].append(
                    f"Room {room_id} ({room_data.get('type', 'unknown')}) "
                    f"has very large area: {area} pixels"
                )
        
        # Check for reasonable number of rooms
        if len(rooms) > 20:
            validation_report['warnings'].append(
                f"Large number of rooms detected: {len(rooms)}. "
                "This might indicate over-segmentation."
            )
        
        # Check opening placement
        door_count = len([op for op in openings if op.get('type') == 'door'])
        window_count = len([op for op in openings if op.get('type') == 'window'])
        
        if door_count == 0:
            validation_report['suggestions'].append(
                "No doors detected. Consider adding entrance doors."
            )
        
        validation_report['statistics'] = {
            'room_count': len(rooms),
            'wall_count': len(walls),
            'door_count': door_count,
            'window_count': window_count,
            'total_room_area': sum(room.get('area', 0) for room in rooms.values())
        }
        
        return validation_report