#!/usr/bin/env python3
"""
model_logger.py - Model operations logging for SAKAR Vision AI

This module provides specialized logging for AI model operations including 
model loading, inference, and predictions.
"""

import json
import time
import numpy as np
from datetime import datetime
from functools import wraps
from typing import Dict, List, Any, Union, Optional

from utils.logging.logger_config import LoggerFactory, log_exception
from utils.logging.performance_logger import Timer

# Create dedicated model logger
logger = LoggerFactory.get_logger("model", detailed=True)


class ModelLogger:
    """Utility class for tracking AI model operations."""

    def __init__(self, model_name: str, model_type: str = "unknown"):
        """
        Initialize model logger for a specific model.

        Args:
            model_name (str): Name of the model
            model_type (str): Type of model (YOLO, MobileNet, etc.)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.inference_times = []

    def log_model_load(self, success: bool = True, error: Optional[Exception] = None,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log model loading event.

        Args:
            success (bool): Whether the model was loaded successfully
            error (Exception, optional): Error if loading failed
            details (dict, optional): Additional details about the model
        """
        if success:
            message = f"Model '{self.model_name}' ({self.model_type}) loaded successfully"
            if details:
                message += f" - Details: {details}"
            logger.info(message)
        else:
            logger.error(f"Failed to load model '{self.model_name}' ({self.model_type})")
            if error:
                log_exception(logger, error)

    def log_inference_start(self, batch_size: int = 1, input_shape: Optional[tuple] = None) -> None:
        """
        Log the start of an inference operation.

        Args:
            batch_size (int): Size of the batch
            input_shape (tuple, optional): Shape of the input
        """
        message = f"Starting inference with model '{self.model_name}' - batch size: {batch_size}"
        if input_shape:
            message += f", input shape: {input_shape}"

        logger.debug(message)

    def log_inference_end(self, inference_time: float, success: bool = True,
                          error: Optional[Exception] = None) -> None:
        """
        Log the completion of an inference operation.

        Args:
            inference_time (float): Time taken for inference in seconds
            success (bool): Whether the inference was successful
            error (Exception, optional): Error if inference failed
        """
        # Store inference time for statistics
        if success:
            self.inference_times.append(inference_time)
            # Keep only the last 100 inference times
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

        if success:
            logger.debug(
                f"Inference completed in {inference_time:.4f}s with model '{self.model_name}'")
        else:
            logger.error(f"Inference failed with model '{self.model_name}'")
            if error:
                log_exception(logger, error)

    def log_prediction_result(self, predictions: Any, confidence: Optional[float] = None,
                              prediction_time: Optional[float] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log prediction results.

        Args:
            predictions: Prediction results
            confidence (float, optional): Confidence score of the prediction
            prediction_time (float, optional): Time taken for prediction in seconds
            metadata (dict, optional): Additional metadata about the prediction
        """
        # Summarize predictions based on their type
        if isinstance(predictions, list):
            if len(predictions) == 0:
                summary = "No predictions"
            elif len(predictions) <= 3:
                summary = f"Predictions: {str(predictions)}"
            else:
                summary = f"{len(predictions)} predictions"
        else:
            summary = str(predictions)

        message = f"Model '{self.model_name}' prediction: {summary}"

        if confidence is not None:
            message += f", confidence: {confidence:.4f}"

        if prediction_time is not None:
            message += f", time: {prediction_time:.4f}s"

        if metadata:
            message += f", metadata: {metadata}"

        logger.info(message)

    def log_model_metrics(self) -> Dict[str, float]:
        """
        Log metrics about the model's performance.

        Returns:
            dict: Performance metrics
        """
        if not self.inference_times:
            logger.info(f"No inference metrics available for model '{self.model_name}'")
            return {}

        metrics = {
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "std_inference_time": np.std(self.inference_times),
            "count": len(self.inference_times),
        }

        logger.info(f"Model '{self.model_name}' metrics: "
                    f"avg_time={metrics['avg_inference_time']:.4f}s, "
                    f"min_time={metrics['min_inference_time']:.4f}s, "
                    f"max_time={metrics['max_inference_time']:.4f}s, "
                    f"count={metrics['count']}")

        return metrics

    def log_model_unload(self) -> None:
        """Log model unloading event."""
        logger.info(f"Model '{self.model_name}' ({self.model_type}) unloaded")

        # Log final metrics when unloading
        if self.inference_times:
            self.log_model_metrics()
        else:
            logger.info(f"Model '{self.model_name}' was not used for inference")


def track_model_inference(model_name=None, model_type=None):
    """
    Decorator for tracking model inference.

    Args:
        model_name (str, optional): Name of the model
        model_type (str, optional): Type of model (YOLO, MobileNet, etc.)

    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to determine model name from the instance if not provided
            nonlocal model_name, model_type
            if args and hasattr(args[0], 'model_name'):
                _model_name = args[0].model_name
            else:
                _model_name = model_name or "unknown"

            if args and hasattr(args[0], 'model_type'):
                _model_type = args[0].model_type
            else:
                _model_type = model_type or "unknown"

            # Create logger
            model_logger = ModelLogger(_model_name, _model_type)

            # Get input shape if available
            input_shape = None
            if len(args) > 1:
                if hasattr(args[1], 'shape'):
                    input_shape = args[1].shape
                elif isinstance(args[1], (list, tuple)) and all(hasattr(x, 'shape') for x in args[1]):
                    input_shape = [x.shape for x in args[1]]

            # Log inference start
            batch_size = kwargs.get('batch_size', 1)
            model_logger.log_inference_start(batch_size, input_shape)

            # Measure inference time
            start_time = time.time()
            try:
                # Execute the inference function
                predictions = func(*args, **kwargs)

                # Calculate time
                inference_time = time.time() - start_time

                # Log successful inference
                model_logger.log_inference_end(inference_time, success=True)

                # Try to extract confidence
                confidence = None
                if hasattr(predictions, 'confidence'):
                    confidence = predictions.confidence
                elif isinstance(predictions, dict) and 'confidence' in predictions:
                    confidence = predictions['confidence']

                # Log prediction result
                model_logger.log_prediction_result(
                    predictions=predictions,
                    confidence=confidence,
                    prediction_time=inference_time
                )

                return predictions
            except Exception as e:
                # Log failed inference
                inference_time = time.time() - start_time
                model_logger.log_inference_end(inference_time, success=False, error=e)
                raise

        return wrapper
    return decorator


class YOLOModelLogger(ModelLogger):
    """Specialized logger for YOLO model operations."""

    def __init__(self, model_name):
        """Initialize YOLO model logger."""
        super().__init__(model_name, "YOLO")

    def log_detection_results(self, detections, confidence_threshold=0.5,
                              inference_time=None, frame_id=None):
        """
        Log object detection results.

        Args:
            detections: Detection results from YOLO model
            confidence_threshold (float): Confidence threshold used
            inference_time (float, optional): Time taken for inference
            frame_id (any, optional): ID of the frame
        """
        try:
            # Extract useful information from detections
            if hasattr(detections, 'xyxy'):
                # Handle Ultralytics format
                num_objects = len(detections.xyxy[0])
                classes = detections.cls[0] if hasattr(detections, 'cls') else None
                confs = detections.conf[0] if hasattr(detections, 'conf') else None

                # Count by class if available
                class_counts = {}
                if classes is not None and len(classes) > 0:
                    for cls in classes:
                        cls_name = int(cls.item())
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                summary = f"{num_objects} objects detected"
                if class_counts:
                    summary += f" - Classes: {class_counts}"
            elif isinstance(detections, list):
                # Handle plain list format
                num_objects = len(detections)
                summary = f"{num_objects} objects detected"
            else:
                # Generic handling
                summary = f"Detections: {str(detections)[:100]}"

            message = f"YOLO model '{self.model_name}' detection: {summary}, threshold: {confidence_threshold:.2f}"

            if inference_time is not None:
                message += f", time: {inference_time:.4f}s"

            if frame_id is not None:
                message += f", frame: {frame_id}"

            logger.info(message)
        except Exception as e:
            logger.warning(f"Error logging detection results: {e}")


class MobileNetModelLogger(ModelLogger):
    """Specialized logger for MobileNet classification model operations."""

    def __init__(self, model_name):
        """Initialize MobileNet model logger."""
        super().__init__(model_name, "MobileNet")

    def log_classification_result(self, class_id, class_name=None, confidence=None,
                                  inference_time=None, top_k=None):
        """
        Log classification result.

        Args:
            class_id: ID of the predicted class
            class_name (str, optional): Name of the predicted class
            confidence (float, optional): Confidence score
            inference_time (float, optional): Time taken for inference
            top_k (list, optional): Top-K results if available
        """
        if class_name:
            result = f"Class: {class_id} ({class_name})"
        else:
            result = f"Class: {class_id}"

        message = f"Classification result: {result}"

        if confidence is not None:
            message += f", confidence: {confidence:.4f}"

        if inference_time is not None:
            message += f", time: {inference_time:.4f}s"

        if top_k:
            # Format top-k results
            top_k_str = ", ".join([f"{k}:{v:.4f}" for k, v in top_k])
            message += f", top results: [{top_k_str}]"

        logger.info(message)


# Demo function
def demo_model_logger():
    """Demonstrate model logging functionality."""
    # Basic model logger
    model_logger = ModelLogger("sakar_vision_model", "YOLO")
    model_logger.log_model_load(success=True, details={"version": "v1.0", "input_size": 640})

    # Log multiple inferences to build statistics
    for i in range(5):
        inference_time = 0.05 + (i * 0.01)  # Simulate varying inference times
        model_logger.log_inference_start(batch_size=1, input_shape=(640, 640, 3))
        model_logger.log_inference_end(inference_time)
        model_logger.log_prediction_result(
            predictions=["crack", "hole", "scratch"],
            confidence=0.92,
            prediction_time=inference_time
        )

    # Log metrics
    model_logger.log_model_metrics()

    # YOLO model logger
    yolo_logger = YOLOModelLogger("sakar_yolo_v8n")
    yolo_logger.log_model_load(True, details={"weights": "sakar_yolo_v8n.pt"})

    # Create mock detection results to simulate YOLO output
    class MockDetections:
        def __init__(self):
            self.xyxy = [np.array([[100, 200, 300, 400], [500, 600, 700, 800]])]
            self.cls = [np.array([0, 1])]
            self.conf = [np.array([0.95, 0.82])]

    # Log detection results
    yolo_logger.log_detection_results(
        MockDetections(),
        confidence_threshold=0.5,
        inference_time=0.12,
        frame_id="frame_001"
    )

    # MobileNet model logger
    mobilenet_logger = MobileNetModelLogger("sakar_cls")
    mobilenet_logger.log_model_load(True)
    mobilenet_logger.log_classification_result(
        class_id=1,
        class_name="MS",
        confidence=0.98,
        inference_time=0.03,
        top_k=[("MS", 0.98), ("Copper", 0.01), ("Brass", 0.01)]
    )

    # Demonstrate the inference tracking decorator
    @track_model_inference(model_name="test_model", model_type="test")
    def run_inference(input_data):
        # Simulate inference
        time.sleep(0.1)
        return {"class_id": 2, "confidence": 0.87}

    # Run decorated function
    result = run_inference(np.zeros((224, 224, 3)))
    print(f"Decorated inference result: {result}")

    print("Model operations logged. Check the logs directory.")


if __name__ == "__main__":
    demo_model_logger()
