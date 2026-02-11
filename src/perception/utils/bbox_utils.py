"""
Bounding Box Utilities
Helper functions for bounding box operations
"""

import numpy as np


def bbox_area(bbox: list) -> float:
    """
    Calculate bounding box area
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        Area in pixels
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_iou(bbox1: list, bbox2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def bbox_center(bbox: list) -> tuple:
    """
    Get bounding box center point
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        (center_x, center_y)
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_distance(bbox1: list, bbox2: list) -> float:
    """
    Calculate Euclidean distance between bbox centers
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        Distance in pixels
    """
    c1 = bbox_center(bbox1)
    c2 = bbox_center(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def bbox_contains(bbox_outer: list, bbox_inner: list) -> bool:
    """
    Check if one bbox contains another
    
    Args:
        bbox_outer: Potentially containing bbox [x1, y1, x2, y2]
        bbox_inner: Potentially contained bbox [x1, y1, x2, y2]
        
    Returns:
        True if bbox_outer contains bbox_inner
    """
    return (bbox_outer[0] <= bbox_inner[0] and
            bbox_outer[1] <= bbox_inner[1] and
            bbox_outer[2] >= bbox_inner[2] and
            bbox_outer[3] >= bbox_inner[3])


def normalize_bbox(bbox: list, image_width: int, image_height: int) -> list:
    """
    Normalize bbox coordinates to [0, 1] range
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_width: Image width
        image_height: Image height
        
    Returns:
        Normalized bbox [x1, y1, x2, y2]
    """
    return [
        bbox[0] / image_width,
        bbox[1] / image_height,
        bbox[2] / image_width,
        bbox[3] / image_height
    ]


def denormalize_bbox(bbox: list, image_width: int, image_height: int) -> list:
    """
    Convert normalized bbox back to pixel coordinates
    
    Args:
        bbox: Normalized [x1, y1, x2, y2]
        image_width: Image width
        image_height: Image height
        
    Returns:
        Pixel bbox [x1, y1, x2, y2]
    """
    return [
        bbox[0] * image_width,
        bbox[1] * image_height,
        bbox[2] * image_width,
        bbox[3] * image_height
    ]


def clip_bbox(bbox: list, image_width: int, image_height: int) -> list:
    """
    Clip bbox to image boundaries
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_width: Image width
        image_height: Image height
        
    Returns:
        Clipped bbox
    """
    return [
        max(0, min(bbox[0], image_width)),
        max(0, min(bbox[1], image_height)),
        max(0, min(bbox[2], image_width)),
        max(0, min(bbox[3], image_height))
    ]
