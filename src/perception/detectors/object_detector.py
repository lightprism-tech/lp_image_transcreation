"""
Object Detector using YOLOv8x
Detects objects and returns bounding boxes
"""

import numpy as np
from ultralytics import YOLO
from perception.config import settings


class ObjectDetector:
    """Detects objects in images using YOLOv8x"""
    
    def __init__(self, model_path=None):
        """
        Initialize YOLOv8x object detector
        
        Args:
            model_path: Path to YOLOv8x model weights
        """
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.threshold = settings.OBJECT_DETECTION_THRESHOLD
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8x model"""
        try:
            self.model = YOLO(str(self.model_path))
            print(f"✓ YOLOv8x model loaded from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load YOLOv8x model: {e}")
            print(f"  Download with: yolo download model=yolov8x.pt")
            raise
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect objects in image using YOLOv8x
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of detections, each containing:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str
            }
        """
        if self.model is None:
            raise RuntimeError("YOLOv8x model not loaded")
        
        # Run YOLOv8x inference
        results = self.model(image, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                
                # Filter by confidence threshold
                if confidence >= self.threshold:
                    detections.append({
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'confidence': confidence,
                        'class_id': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])]
                    })
        
        return detections
