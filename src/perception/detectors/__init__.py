"""Detection modules for object, text, and image classification."""
from perception.detectors.object_detector import ObjectDetector
from perception.detectors.text_detector import TextDetector
from perception.detectors.image_type_classifier import ImageTypeClassifier

__all__ = ["ObjectDetector", "TextDetector", "ImageTypeClassifier"]
