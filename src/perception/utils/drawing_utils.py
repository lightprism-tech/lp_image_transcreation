"""
Drawing Utilities
Functions for visualizing detection results
"""

import cv2
import numpy as np
from pathlib import Path
from perception.config import settings


class DebugVisualizer:
    """Visualizes detection results for debugging"""
    
    def __init__(self, save_dir=None):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save debug images
        """
        self.save_dir = save_dir or settings.DEBUG_IMAGES_DIR
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for different object classes
        self.colors = self._generate_colors(100)
    
    def _generate_colors(self, num_colors: int) -> list:
        """Generate distinct colors for visualization"""
        np.random.seed(42)
        colors = []
        for i in range(num_colors):
            colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
        return colors
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: list,
        label: str = "",
        confidence: float = None,
        color: tuple = None
    ) -> np.ndarray:
        """
        Draw bounding box on image
        
        Args:
            image: Input image (will be copied)
            bbox: [x1, y1, x2, y2]
            label: Label text
            confidence: Confidence score
            color: RGB color tuple
            
        Returns:
            Image with drawn bbox
        """
        image = image.copy()
        
        if color is None:
            color = (0, 255, 0)  # Green
        
        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        
        # Draw rectangle
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Draw label
        if label or confidence is not None:
            text = label
            if confidence is not None:
                text += f" {confidence:.2f}"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color_bgr,
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return image
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: list,
        title: str = "Detections"
    ) -> np.ndarray:
        """
        Draw all detections on image
        
        Args:
            image: Input image
            detections: List of detection dicts with bbox, class_name, confidence
            title: Title for the visualization
            
        Returns:
            Image with all detections drawn
        """
        vis_image = image.copy()
        
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            label = det.get('class_name', 'object')
            confidence = det.get('confidence', 0.0)
            
            # Use different colors for different objects
            color = self.colors[i % len(self.colors)]
            
            vis_image = self.draw_bbox(vis_image, bbox, label, confidence, color)
        
        return vis_image
    
    def save_visualization(
        self,
        image: np.ndarray,
        filename: str
    ):
        """
        Save visualization to file
        
        Args:
            image: Image to save (RGB)
            filename: Output filename
        """
        output_path = self.save_dir / filename
        
        # Convert RGB to BGR for saving
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)
    
    def visualize_pipeline_results(
        self,
        image: np.ndarray,
        objects: list,
        text_regions: list,
        image_name: str
    ):
        """
        Create comprehensive visualization of pipeline results
        
        Args:
            image: Input image
            objects: Detected objects
            text_regions: Detected text regions
            image_name: Name for output file
        """
        # Draw objects
        vis_objects = self.draw_detections(image, objects, "Objects")
        self.save_visualization(vis_objects, f"{image_name}_objects.jpg")
        
        # Draw text regions
        vis_text = image.copy()
        for region in text_regions:
            bbox = region.get('bbox', [])
            vis_text = self.draw_bbox(vis_text, bbox, "text", color=(255, 0, 0))
        self.save_visualization(vis_text, f"{image_name}_text.jpg")
        
        # Combined visualization
        vis_combined = self.draw_detections(image, objects)
        for region in text_regions:
            bbox = region.get('bbox', [])
            vis_combined = self.draw_bbox(vis_combined, bbox, "text", color=(255, 0, 0))
        self.save_visualization(vis_combined, f"{image_name}_combined.jpg")
