#!/usr/bin/env python3
"""
SAKAR VISION AI - Enhanced Image Classification UI Module with Prediction Storage

OVERVIEW:
This module implements a sophisticated real-time image classification interface for the Sakar Vision AI platform, 
serving as an advanced quality control and validation system that combines live camera feeds with deep learning-based 
classification capabilities. It provides a professional dual-dashboard architecture for monitoring and analyzing 
classification results in real-time, enabling users to validate AI model performance and ensure accurate defect 
detection with comprehensive statistical tracking and session persistence for manufacturing inspection workflows.

ENHANCED FEATURES:
- Automatic loading of best_model.pth from current directory
- Comprehensive prediction storage system with JSON export
- Intelligent class name detection from model checkpoints
- Motion detection for stable frame prediction
- Dual dashboard interface for defective/non-defective monitoring
- Visual data export capabilities
- Session-based prediction tracking
"""

import io
import os
import sys
import time
import json
import glob
import uuid
import base64
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog,
                             QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QProgressDialog, QPushButton, QStackedWidget,
                             QTextEdit, QVBoxLayout, QWidget, QFrame, QGroupBox,
                             QSizePolicy, QScrollArea, QSlider, QSpinBox, QLineEdit,
                             QInputDialog)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms

# Try to import matplotlib for visual charts
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available. Visual charts will be text-based.")

# Environment setup
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration paths
REPORTS_STORAGE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "classification_reports.json")
SESSION_STATE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "session_state.json")
PREDICTIONS_STORAGE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "predictions_log.json")


class DefectClassifier(nn.Module):
    """
    Enhanced DefectClassifier class compatible with various architectures
    """
    def __init__(self, num_classes=2, architecture='resnet18'):
        super(DefectClassifier, self).__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        
        if architecture == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif architecture == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        else:
            # Default to ResNet18
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class PredictionStorage:
    """Enhanced prediction storage system with comprehensive logging and management"""
    
    def __init__(self, storage_path=PREDICTIONS_STORAGE_PATH, max_predictions=10000):
        self.storage_path = storage_path
        self.max_predictions = max_predictions
        self.session_id = str(uuid.uuid4())[:8]  # Unique session identifier
        self.session_start_time = datetime.now().isoformat()
        self.predictions_cache = []
        
        # Initialize storage file if it doesn't exist
        self.initialize_storage()
        
    def initialize_storage(self):
        """Initialize the predictions storage file"""
        try:
            if not os.path.exists(self.storage_path):
                initial_data = {
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "version": "1.0",
                        "description": "SAKAR Vision AI - Prediction Storage System"
                    },
                    "sessions": {}
                }
                with open(self.storage_path, 'w') as f:
                    json.dump(initial_data, f, indent=4)
                print(f"✓ Initialized prediction storage: {self.storage_path}")
            else:
                print(f"✓ Using existing prediction storage: {self.storage_path}")
                
        except Exception as e:
            print(f"✗ Error initializing prediction storage: {e}")
    
    def save_prediction(self, predicted_class, confidence, additional_data=None):
        """
        Save a single prediction with comprehensive metadata
        
        Args:
            predicted_class (str): The predicted class name
            confidence (float): Confidence score (0.0 to 1.0)
            additional_data (dict): Optional additional metadata
        """
        try:
            # Create prediction record
            prediction_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_percentage": f"{confidence:.2%}",
                "metadata": {
                    "device": str(device),
                    "session_start": self.session_start_time
                }
            }
            
            # Add any additional data
            if additional_data:
                prediction_record["additional_data"] = additional_data
            
            # Add to cache for batch saving
            self.predictions_cache.append(prediction_record)
            
            # Save immediately for critical predictions or when cache is full
            if len(self.predictions_cache) >= 50:  # Batch save every 50 predictions
                self.flush_cache()
                
            return prediction_record["id"]
            
        except Exception as e:
            print(f"✗ Error saving prediction: {e}")
            return None
    
    def flush_cache(self):
        """Flush cached predictions to storage file"""
        if not self.predictions_cache:
            return
            
        try:
            # Load existing data
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Initialize session data if it doesn't exist
            if self.session_id not in data["sessions"]:
                data["sessions"][self.session_id] = {
                    "session_info": {
                        "session_id": self.session_id,
                        "start_time": self.session_start_time,
                        "last_update": datetime.now().isoformat()
                    },
                    "predictions": []
                }
            
            # Add cached predictions to session
            data["sessions"][self.session_id]["predictions"].extend(self.predictions_cache)
            data["sessions"][self.session_id]["session_info"]["last_update"] = datetime.now().isoformat()
            data["sessions"][self.session_id]["session_info"]["total_predictions"] = len(data["sessions"][self.session_id]["predictions"])
            
            # Manage storage size - keep only recent sessions if needed
            self.manage_storage_size(data)
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"✓ Saved {len(self.predictions_cache)} predictions to storage")
            self.predictions_cache = []  # Clear cache
            
        except Exception as e:
            print(f"✗ Error flushing predictions cache: {e}")
    
    def manage_storage_size(self, data):
        """Manage storage size by removing old sessions if necessary"""
        try:
            total_predictions = sum(len(session["predictions"]) for session in data["sessions"].values())
            
            if total_predictions > self.max_predictions:
                # Sort sessions by start time and remove oldest
                sessions_by_time = sorted(
                    data["sessions"].items(),
                    key=lambda x: x[1]["session_info"]["start_time"]
                )
                
                while total_predictions > self.max_predictions and len(sessions_by_time) > 1:
                    # Remove oldest session (but keep current session)
                    oldest_session_id, oldest_session = sessions_by_time.pop(0)
                    if oldest_session_id != self.session_id:
                        predictions_removed = len(oldest_session["predictions"])
                        del data["sessions"][oldest_session_id]
                        total_predictions -= predictions_removed
                        print(f"✓ Removed old session {oldest_session_id} with {predictions_removed} predictions")
                        
        except Exception as e:
            print(f"✗ Error managing storage size: {e}")
    
    def get_session_statistics(self):
        """Get statistics for current session"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.session_id in data["sessions"]:
                session_data = data["sessions"][self.session_id]
                predictions = session_data["predictions"]
                
                if not predictions:
                    return {"total": 0, "classes": {}, "avg_confidence": 0}
                
                # Calculate statistics
                total_predictions = len(predictions)
                class_counts = {}
                total_confidence = 0
                
                for pred in predictions:
                    class_name = pred["predicted_class"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_confidence += pred["confidence"]
                
                avg_confidence = total_confidence / total_predictions
                
                return {
                    "session_id": self.session_id,
                    "total": total_predictions,
                    "classes": class_counts,
                    "avg_confidence": avg_confidence,
                    "start_time": session_data["session_info"]["start_time"]
                }
            else:
                return {"total": 0, "classes": {}, "avg_confidence": 0}
                
        except Exception as e:
            print(f"✗ Error getting session statistics: {e}")
            return {"total": 0, "classes": {}, "avg_confidence": 0}
    
    def export_session_data(self, export_path=None):
        """Export current session data to a separate file"""
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"session_{self.session_id}_{timestamp}.json"
            
            # Flush any cached predictions first
            self.flush_cache()
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.session_id in data["sessions"]:
                session_export = {
                    "export_info": {
                        "exported_at": datetime.now().isoformat(),
                        "session_id": self.session_id,
                        "source_file": self.storage_path
                    },
                    "session_data": data["sessions"][self.session_id]
                }
                
                with open(export_path, 'w') as f:
                    json.dump(session_export, f, indent=4)
                
                print(f"✓ Session data exported to: {export_path}")
                return export_path
            else:
                print(f"✗ Session {self.session_id} not found")
                return None
                
        except Exception as e:
            print(f"✗ Error exporting session data: {e}")
            return None
    
    def close_session(self):
        """Close current session and flush all cached data"""
        try:
            # Flush any remaining cached predictions
            self.flush_cache()
            
            # Update session end time
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.session_id in data["sessions"]:
                data["sessions"][self.session_id]["session_info"]["end_time"] = datetime.now().isoformat()
                
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f"✓ Session {self.session_id} closed successfully")
            
        except Exception as e:
            print(f"✗ Error closing session: {e}")


def save_session_state(ui_name="image_classification"):
    """Save session state when this UI is opened"""
    try:
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "user": {
                "username": "admin",
                "full_name": "Administrator"
            },
            "additional_data": {
                "opened_at": datetime.now().isoformat()
            }
        }

        with open(SESSION_STATE_PATH, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session state saved: {ui_name}")

    except Exception as e:
        print(f"Error saving session state: {e}")


def save_session_on_close(ui_name="image_classification"):
    """Save session state when this UI is closed"""
    try:
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "user": {
                "username": "admin",
                "full_name": "Administrator"
            },
            "additional_data": {
                "closed_at": datetime.now().isoformat()
            }
        }

        with open(SESSION_STATE_PATH, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session state saved on close: {ui_name}")

    except Exception as e:
        print(f"Error saving session state on close: {e}")


class ClassificationDashboard(QWidget):
    """Dashboard for displaying classification results and statistics"""

    def __init__(self, parent=None, dashboard_type="non_defective"):
        super().__init__(parent)
        self.dashboard_type = dashboard_type  # "non_defective" or "defective"
        self.setMinimumWidth(300)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            QLabel {
                padding: 6px;
                color: #495057;
                font-size: 11pt;
            }
            QLabel#heading {
                background-color: #ff914d;
                color: white;
                font-weight: bold;
                font-size: 14pt;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 10px;
                border-bottom: 1px solid #ff7730;
            }
            QLabel#subheading {
                background-color: #e9ecef;
                font-weight: bold;
                color: #343a40;
                padding: 8px;
                border-top: 1px solid #dee2e6;
                border-bottom: 1px solid #dee2e6;
                margin-top: 5px;
            }
            QScrollArea {
                border: none;
            }
            QWidget#classificationListWidget {
                background-color: #ffffff;
                border: none;
            }
        """)

        # Initialize recent classifications
        self.recent_classifications = []  # Store recent classifications for display

        self.init_ui()

    def init_ui(self):
        """Initialize the dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Dashboard title based on type
        if self.dashboard_type == "non_defective":
            title_text = "Non-Defective"
            self.title_color = "#28a745"  # Green
        else:
            title_text = "Defective"
            self.title_color = "#dc3545"  # Red

        self.title_label = QLabel(title_text)
        self.title_label.setObjectName("heading")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"""
            background-color: {self.title_color};
            color: white;
            font-weight: bold;
            font-size: 14pt;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            padding: 10px;
        """)
        layout.addWidget(self.title_label)

        # Main content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        # Recent classifications section
        recent_label = QLabel("Recent Classifications")
        recent_label.setObjectName("subheading")
        content_layout.addWidget(recent_label)

        # Scroll area for recent classifications
        self.recent_scroll_area = QScrollArea()
        self.recent_scroll_area.setWidgetResizable(True)
        self.recent_scroll_area.setMinimumHeight(150)
        self.recent_scroll_area.setMaximumHeight(200)
        self.recent_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.recent_list_widget = QWidget()
        self.recent_list_widget.setObjectName("classificationListWidget")
        self.recent_layout = QVBoxLayout(self.recent_list_widget)
        self.recent_layout.setContentsMargins(5, 5, 5, 5)
        self.recent_layout.setSpacing(5)
        self.recent_layout.addStretch(1)

        self.recent_scroll_area.setWidget(self.recent_list_widget)
        content_layout.addWidget(self.recent_scroll_area)

        content_layout.addStretch()
        layout.addWidget(content_widget)

    def update_classification(self, class_name, confidence):
        """Update classification based on class name"""
        self.add_recent_classification(class_name, confidence)
        self.update_ui()

    def add_recent_classification(self, class_name, confidence):
        """Add a recent classification to the list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.recent_classifications.append({
            "class": class_name,
            "confidence": confidence,
            "time": timestamp
        })

        # Keep only last 10 classifications
        if len(self.recent_classifications) > 10:
            self.recent_classifications = self.recent_classifications[-10:]

    def update_ui(self):
        """Update the UI"""
        # Clear recent classifications display
        while self.recent_layout.count() > 1:  # Keep the stretch
            child = self.recent_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add recent classifications
        if not self.recent_classifications:
            placeholder_label = QLabel("No classifications yet")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #6c757d; font-style: italic;")
            self.recent_layout.insertWidget(0, placeholder_label)
        else:
            for i, item in enumerate(reversed(self.recent_classifications)):
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 2, 5, 2)

                time_label = QLabel(item["time"])
                time_label.setStyleSheet("color: #6c757d; font-size: 9pt;")
                time_label.setMinimumWidth(60)

                class_label = QLabel(item["class"])
                class_label.setStyleSheet("color: #495057; font-size: 10pt;")

                conf_label = QLabel(f"{item['confidence']:.1%}")
                conf_label.setStyleSheet(
                    f"color: {self.title_color}; font-weight: bold; font-size: 9pt;")
                conf_label.setMinimumWidth(50)

                item_layout.addWidget(time_label)
                item_layout.addWidget(class_label, 1)
                item_layout.addWidget(conf_label)

                self.recent_layout.insertWidget(0, item_widget)

        self.recent_list_widget.updateGeometry()


class CameraWorker(QObject):
    """Enhanced camera worker with motion detection and stationary frame prediction"""
    finished = pyqtSignal()
    image_captured = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float)
    motion_status_changed = pyqtSignal(bool)  # New signal for motion status

    def __init__(self, model=None, transform=None, class_names=None, idx_to_class=None):
        super().__init__()
        self.running = True
        self.capture = None
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.idx_to_class = idx_to_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Motion detection parameters
        self.motion_threshold = 2000  # Threshold for motion detection
        self.min_stationary_frames = 10  # Minimum frames to be stationary before prediction
        self.motion_sensitivity = 0.02  # Sensitivity for motion detection (0.01-0.1)
        
        # Motion detection state
        self.previous_frame = None
        self.stationary_frame_count = 0
        self.is_motion_detected = False
        self.frame_count = 0
        self.last_prediction_frame = None

    def detect_motion(self, current_frame):
        """
        Detect motion between consecutive frames using frame difference
        
        Args:
            current_frame: Current OpenCV frame
            
        Returns:
            bool: True if motion detected, False if stationary
        """
        try:
            # Convert to grayscale for motion detection
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
            
            # If this is the first frame, store it and return no motion
            if self.previous_frame is None:
                self.previous_frame = gray_current
                return False
            
            # Compute the absolute difference between frames
            frame_diff = cv2.absdiff(self.previous_frame, gray_current)
            
            # Apply threshold to get binary image
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate the percentage of changed pixels
            total_pixels = thresh.shape[0] * thresh.shape[1]
            changed_pixels = cv2.countNonZero(thresh)
            motion_percentage = changed_pixels / total_pixels
            
            # Update previous frame
            self.previous_frame = gray_current
            
            # Determine if motion is detected based on threshold
            motion_detected = motion_percentage > self.motion_sensitivity
            
            return motion_detected
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return True  # Assume motion on error to be safe

    def is_frame_suitable_for_prediction(self):
        """
        Check if current frame is suitable for prediction
        
        Returns:
            bool: True if frame is suitable (camera is stationary)
        """
        return (not self.is_motion_detected and 
                self.stationary_frame_count >= self.min_stationary_frames)

    def run(self):
        """Main camera loop with motion detection"""
        try:
            print("📹 Starting camera feed...")

            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("Failed to open camera 0, trying camera 1")
                self.capture = cv2.VideoCapture(1)

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            print("📹 Camera started with motion detection enabled")
            print(f"🎯 Predictions will only occur when camera is stationary for {self.min_stationary_frames} frames")

            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    continue

                self.frame_count += 1

                # Detect motion in current frame
                motion_detected = self.detect_motion(frame)
                
                # Update motion state
                if motion_detected != self.is_motion_detected:
                    self.is_motion_detected = motion_detected
                    self.motion_status_changed.emit(motion_detected)
                    
                    if motion_detected:
                        self.stationary_frame_count = 0
                    
                # Update stationary frame count
                if not motion_detected:
                    self.stationary_frame_count += 1
                else:
                    self.stationary_frame_count = 0

                # Convert to RGB for display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add motion status overlay to the image
                rgb_image_with_overlay = self.add_motion_overlay(rgb_image, motion_detected)
                
                # Convert to Qt image and emit
                h, w, ch = rgb_image_with_overlay.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image_with_overlay.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_captured.emit(qt_image)

                # Perform prediction only when camera is stationary and model is available
                if (self.model is not None and self.transform is not None and
                        self.is_frame_suitable_for_prediction()):
                    
                    # Only predict every 30 frames when stationary to avoid too frequent predictions
                    if self.stationary_frame_count % 30 == 0:
                        self.predict_frame(frame)

                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            print(f"Error in camera feed: {e}")
        finally:
            if self.capture is not None:
                self.capture.release()
            self.finished.emit()

    def add_motion_overlay(self, image, motion_detected):
        """
        Add visual overlay to indicate motion status
        
        Args:
            image: RGB image array
            motion_detected: Boolean indicating if motion is detected
            
        Returns:
            Modified image with overlay
        """
        try:
            # Create a copy to avoid modifying original
            overlay_image = image.copy()
            
            # Define overlay parameters
            overlay_height = 30
            overlay_width = 200
            
            # Position overlay at top-right corner
            y_start = 10
            x_start = image.shape[1] - overlay_width - 10
            y_end = y_start + overlay_height
            x_end = x_start + overlay_width
            
            # Choose color and text based on motion status
            if motion_detected:
                color = (255, 100, 100)  # Red for motion
                text = "MOTION DETECTED"
            elif self.stationary_frame_count < self.min_stationary_frames:
                color = (255, 255, 100)  # Yellow for stabilizing
                text = f"STABILIZING ({self.stationary_frame_count}/{self.min_stationary_frames})"
            else:
                color = (100, 255, 100)  # Green for ready
                text = "READY FOR PREDICTION"
            
            # Draw semi-transparent overlay rectangle
            overlay = overlay_image.copy()
            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
            overlay_image = cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0)
            
            # Add text
            cv2.putText(overlay_image, text, (x_start + 5, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
            return overlay_image
            
        except Exception as e:
            print(f"Error adding motion overlay: {e}")
            return image

    def predict_frame(self, frame):
        """Make prediction on current frame (only called when stationary)"""
        try:
            if self.model is None or self.transform is None:
                return

            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                probability = probabilities[predicted_idx].item()

                # Use idx_to_class mapping if available
                if self.idx_to_class and predicted_idx in self.idx_to_class:
                    predicted_class = self.idx_to_class[predicted_idx]
                elif predicted_idx < len(self.class_names):
                    predicted_class = self.class_names[predicted_idx]
                else:
                    predicted_class = f"Class {predicted_idx}"

                print(f"🎯 Prediction: {predicted_class} ({probability:.2%}) - Frame: {self.frame_count}")
                self.prediction_ready.emit(predicted_class, probability)

        except Exception as e:
            print(f"Error in prediction: {e}")

    def stop(self):
        """Stop camera worker"""
        self.running = False


class ClassNamesDialog(QDialog):
    """Custom dialog for entering class names"""
    
    def __init__(self, num_classes, current_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Class Names")
        self.setModal(True)
        self.resize(400, min(600, 100 + num_classes * 35))
        
        layout = QVBoxLayout()
        
        # Instructions
        instruction = QLabel(f"Enter names for {num_classes} classes:")
        instruction.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instruction)
        
        # Scroll area for many classes
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Input fields
        self.line_edits = []
        for i in range(num_classes):
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Class {i} name...")
            line_edit.setText(current_names[i] if i < len(current_names) else f"class_{i}")
            self.line_edits.append(line_edit)
            scroll_layout.addWidget(line_edit)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        preset_button = QPushButton("Use Defect Presets")
        preset_button.clicked.connect(self.use_defect_presets)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        
        button_layout.addWidget(preset_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def use_defect_presets(self):
        """Fill with common defect detection class names"""
        if len(self.line_edits) == 10:
            presets = [
                'normal', 'clamp_miss', 'screw_miss', 'scratch', 'dent', 
                'crack', 'spot', 'hole', 'incomplete', 'damaged'
            ]
        elif len(self.line_edits) == 2:
            presets = ['normal', 'defective']
        elif len(self.line_edits) == 6:
            presets = ['normal', 'clamp_miss', 'screw_miss', 'scratch', 'dent', 'crack']
        elif len(self.line_edits) == 8:
            presets = ['normal', 'clamp_miss', 'screw_miss', 'scratch', 'dent', 'crack', 'spot', 'hole']
        else:
            presets = [f"defect_type_{i}" for i in range(len(self.line_edits))]
            presets[0] = 'normal'  # First class is usually normal
        
        for i, preset in enumerate(presets):
            if i < len(self.line_edits):
                self.line_edits[i].setText(preset)
    
    def get_class_names(self):
        return [line_edit.text().strip() or f"class_{i}" for i, line_edit in enumerate(self.line_edits)]


class ImageClassiUI(QWidget):
    """Enhanced Image Classification UI with sophisticated interface and prediction storage"""

    def __init__(self, parent=None, good_dir=None, bad_dir=None):
        super().__init__(parent)
        self.demo_feed_ui = parent
        self.good_dir = good_dir
        self.bad_dir = bad_dir
        self.model = None
        self.class_names = []
        self.class_mapping = {}
        self.idx_to_class = {}
        self.img_height, self.img_width = 224, 224
        self.camera_feed_running = False

        # Initialize prediction storage system
        self.prediction_storage = PredictionStorage()

        # Save session state when this UI is opened
        save_session_state("image_classification")

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.init_ui()
        self.auto_load_latest_model()

    def init_ui(self):
        """Initialize the enhanced UI"""
        self.setWindowTitle('SAKAR VISION AI - Image Classification')
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a80d2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Main content area with dashboards and camera
        content_main_layout = QHBoxLayout()
        content_main_layout.setSpacing(10)

        # Left dashboard (Non-defective)
        self.left_dashboard = ClassificationDashboard(dashboard_type="non_defective")
        content_main_layout.addWidget(self.left_dashboard, 1)

        # Camera feed group
        camera_group = QGroupBox("Live Camera Classification")
        camera_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #FFA500;
                border-radius: 10px;
                padding: 10px;
                background-color: #FFFFFF;
            }
        """)
        camera_layout = QVBoxLayout()

        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("""
            border: 3px solid #FFA500;
            border-radius: 10px;
            background-color: #222222;
            color: white;
            font-size: 16px;
        """)
        self.camera_label.setText("Camera feed will appear here when started")
        camera_layout.addWidget(self.camera_label, 0, Qt.AlignCenter)

        # Control panel
        controls_container = QWidget()
        controls_container.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 15px;
        """)
        controls_layout = QVBoxLayout(controls_container)

        # Prediction display
        self.camera_prediction_label = QLabel("Prediction: Camera not started")
        self.camera_prediction_label.setAlignment(Qt.AlignCenter)
        self.camera_prediction_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #333333;
            padding: 15px;
            margin: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 2px solid #e9ecef;
        """)
        controls_layout.addWidget(self.camera_prediction_label)

        # Session statistics display
        self.session_stats_label = QLabel("Session: 0 predictions")
        self.session_stats_label.setAlignment(Qt.AlignCenter)
        self.session_stats_label.setStyleSheet("""
            font-size: 14px;
            color: #6c757d;
            padding: 8px;
            margin: 5px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        """)
        controls_layout.addWidget(self.session_stats_label)

        # Camera control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        self.start_camera_button = QPushButton("Start Camera")
        self.start_camera_button.clicked.connect(self.start_camera_feed)
        self.start_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 16px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.stop_camera_button = QPushButton("Stop Camera")
        self.stop_camera_button.clicked.connect(self.stop_camera_feed)
        self.stop_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 5px;
                font-size: 16px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        # Export button for prediction data
        self.export_button = QPushButton("Export Session Data")
        self.export_button.clicked.connect(self.export_session_data)

        # Visual export button
        self.visual_export_button = QPushButton("Export Visual Report")
        self.visual_export_button.clicked.connect(self.export_visual_data)

        button_layout.addStretch(1)
        button_layout.addWidget(self.start_camera_button)
        button_layout.addWidget(self.stop_camera_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.visual_export_button)
        button_layout.addStretch(1)

        controls_layout.addLayout(button_layout)
        camera_layout.addWidget(controls_container)
        camera_group.setLayout(camera_layout)

        content_main_layout.addWidget(camera_group, 4)

        # Right dashboard (Defective)
        self.right_dashboard = ClassificationDashboard(dashboard_type="defective")
        content_main_layout.addWidget(self.right_dashboard, 1)

        content_layout.addLayout(content_main_layout)

        # Status bar
        status_bar = QWidget()
        status_bar.setStyleSheet("""
            background-color: #f8f9fa;
            border-top: 1px solid #e9ecef;
            padding: 5px;
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 5, 10, 5)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #6c757d; font-size: 12px;")
        status_layout.addWidget(self.status_label)

        content_layout.addWidget(status_bar)
        main_layout.addWidget(content_widget)

        # Timer for updating session statistics
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_session_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds

    def update_session_stats(self):
        """Update session statistics display"""
        try:
            stats = self.prediction_storage.get_session_statistics()
            total = stats.get("total", 0)
            avg_conf = stats.get("avg_confidence", 0)
            
            if total > 0:
                self.session_stats_label.setText(
                    f"Session: {total} predictions | Avg Confidence: {avg_conf:.1%}"
                )
            else:
                self.session_stats_label.setText("Session: 0 predictions")
                
        except Exception as e:
            print(f"Error updating session stats: {e}")

    def export_session_data(self):
        """Export current session data"""
        try:
            file_dialog = QFileDialog()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"session_export_{timestamp}.json"
            
            filepath, _ = file_dialog.getSaveFileName(
                self,
                "Export Session Data",
                default_filename,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if filepath:
                exported_file = self.prediction_storage.export_session_data(filepath)
                if exported_file:
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Session data exported successfully to:\n{exported_file}"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Failed to export session data. Please check the logs."
                    )
                    
        except Exception as e:
            print(f"Error in export dialog: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during export:\n{e}"
            )

    def display_message(self, message):
        """Display status message"""
        self.status_label.setText(message)
        print(message)  # Also print to console
        QApplication.processEvents()

    def auto_load_latest_model(self):
        """Automatically load the best_model.pth or latest trained model"""
        # First try to load best_model.pth
        if os.path.exists("best_model.pth"):
            self.display_message("Loading best_model.pth...")
            return self.load_model("best_model.pth")
        
        # If best_model.pth not found, look for other .pth files
        model_files = glob.glob("*.pth")
        if not model_files:
            self.display_message("No trained model (.pth files) found in current directory.")
            return False

        latest_model = max(model_files, key=os.path.getmtime)
        self.display_message(f"Loading latest model: {latest_model}")
        return self.load_model(latest_model)

    def load_model(self, model_path):
        """Enhanced model loading with best_model.pth support"""
        if not os.path.exists(model_path):
            self.display_message(f"Model file not found: {model_path}")
            return False

        try:
            self.display_message(f"Loading model from: {model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                
                # Check for class mapping (best_model.pth format)
                if 'class_mapping' in checkpoint:
                    self.display_message("✓ Found class_mapping in checkpoint (best_model.pth format)")
                    self.class_mapping = checkpoint['class_mapping']
                    self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
                    num_classes = len(self.class_mapping)
                    self.class_names = list(self.class_mapping.keys())
                    
                    # Load model state
                    if 'model_state_dict' in checkpoint:
                        model_state_dict = checkpoint['model_state_dict']
                    else:
                        model_state_dict = checkpoint
                        
                    # Display additional info
                    if 'epoch' in checkpoint:
                        self.display_message(f"Model from epoch: {checkpoint['epoch']}")
                    if 'val_accuracy' in checkpoint:
                        self.display_message(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
                
                # Handle other checkpoint formats
                elif 'model_state_dict' in checkpoint:
                    self.display_message("Loading from standard checkpoint format")
                    model_state_dict = checkpoint['model_state_dict']
                    
                    # Try to get number of classes from the model structure
                    num_classes = self.detect_num_classes_from_state_dict(model_state_dict)
                    
                    # Check for other class name keys
                    class_keys = ['class_names', 'classes', 'idx_to_class']
                    for key in class_keys:
                        if key in checkpoint:
                            if isinstance(checkpoint[key], list):
                                self.class_names = checkpoint[key]
                                self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                                self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                            elif isinstance(checkpoint[key], dict):
                                if key == 'idx_to_class':
                                    self.idx_to_class = checkpoint[key]
                                    self.class_mapping = {v: k for k, v in self.idx_to_class.items()}
                                    self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
                            break
                    
                    # If no class names found, create generic ones
                    if not self.class_names:
                        self.class_names = [f"class_{i}" for i in range(num_classes)]
                        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                        
                else:
                    # Assume entire checkpoint is state dict
                    model_state_dict = checkpoint
                    num_classes = self.detect_num_classes_from_state_dict(model_state_dict)
                    self.class_names = [f"class_{i}" for i in range(num_classes)]
                    self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                    self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                    
            else:
                # Entire model object
                self.model = checkpoint
                self.model.to(device)
                self.model.eval()
                
                # Try to get number of classes
                if hasattr(self.model, 'fc'):
                    num_classes = self.model.fc.out_features
                elif hasattr(self.model, 'classifier'):
                    if hasattr(self.model.classifier, '1'):
                        num_classes = self.model.classifier[1].out_features
                    else:
                        num_classes = self.model.classifier[-1].out_features
                else:
                    num_classes = 2
                
                self.class_names = [f"class_{i}" for i in range(num_classes)]
                self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                
                self.display_message(f"✓ Model loaded: {os.path.basename(model_path)} | Classes: {num_classes}")
                self.display_message(f"✓ Classes: {', '.join(self.class_names)}")
                self.start_camera_button.setEnabled(True)
                return True

            # Create model architecture
            self.model = DefectClassifier(num_classes)
            
            # Load state dict
            try:
                # Remove any 'backbone.' prefix if present
                cleaned_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('backbone.'):
                        cleaned_state_dict[key] = value
                    else:
                        cleaned_state_dict[f'backbone.{key}'] = value
                
                self.model.load_state_dict(cleaned_state_dict, strict=False)
            except:
                # Try direct loading
                self.model.load_state_dict(model_state_dict, strict=False)
                
            self.model.to(device)
            self.model.eval()

            # Display success message
            self.display_message(f"✓ Model loaded: {os.path.basename(model_path)}")
            self.display_message(f"✓ Classes ({num_classes}): {', '.join(self.class_names)}")
            
            self.start_camera_button.setEnabled(True)
            return True

        except Exception as e:
            self.model = None
            error_msg = f"✗ Error loading model: {e}"
            self.display_message(error_msg)
            print(f"Full error details: {e}")
            
            QMessageBox.critical(self, "Model Load Error", f"Failed to load model.\nError: {e}")
            return False

    def detect_num_classes_from_state_dict(self, state_dict):
        """Detect number of classes from model state dict"""
        try:
            # Look for final layer weights
            for key, tensor in state_dict.items():
                if any(layer_name in key for layer_name in ['fc.weight', 'classifier.weight', 'classifier.1.weight']):
                    return tensor.shape[0]
            return 2  # Default fallback
        except:
            return 2

    def start_camera_feed(self):
        """Start camera feed for real-time classification"""
        if self.camera_feed_running or self.model is None:
            if self.model is None:
                QMessageBox.warning(self, "No Model", "Please load a trained model first.")
            return

        # Disable the Start Camera button immediately
        self.start_camera_button.setEnabled(False)
        self.camera_feed_running = True
        self.display_message("Starting camera feed...")

        # Enable the Stop Camera button
        self.stop_camera_button.setEnabled(True)

        # Start camera worker thread
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(
            model=self.model,
            transform=self.transform,
            class_names=self.class_names,
            idx_to_class=self.idx_to_class
        )
        self.camera_worker.moveToThread(self.camera_thread)

        # Connect signals
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.image_captured.connect(self.update_camera_view)
        self.camera_worker.prediction_ready.connect(self.update_prediction)
        self.camera_worker.finished.connect(self.stop_camera_feed)
        self.camera_worker.motion_status_changed.connect(self.update_motion_status)

        self.camera_thread.start()

    def stop_camera_feed(self):
        """Stop camera feed"""
        if not self.camera_feed_running:
            return

        # Disable the Stop Camera button immediately
        self.stop_camera_button.setEnabled(False)
        self.display_message("Stopping camera feed...")
        self.camera_feed_running = False

        if hasattr(self, 'camera_worker'):
            self.camera_worker.stop()

        if hasattr(self, 'camera_thread'):
            self.camera_thread.quit()
            self.camera_thread.wait()

        # Enable the Start Camera button
        self.start_camera_button.setEnabled(True)
        self.display_message("Camera feed stopped")
        self.camera_prediction_label.setText("Prediction: Camera stopped")

    def update_camera_view(self, image):
        """Update camera view with captured image - maintain aspect ratio"""
        if self.camera_label:
            # Get the QImage dimensions
            img_width = image.width()
            img_height = image.height()

            # Get the label dimensions
            label_width = self.camera_label.width()
            label_height = self.camera_label.height()

            # Calculate scaling factor while maintaining aspect ratio
            if label_width > 0 and label_height > 0 and img_width > 0 and img_height > 0:
                aspect_ratio = img_width / img_height
                label_ratio = label_width / label_height

                if aspect_ratio > label_ratio:
                    # Image is wider than label
                    new_width = label_width
                    new_height = int(label_width / aspect_ratio)
                else:
                    # Image is taller than label
                    new_height = label_height
                    new_width = int(label_height * aspect_ratio)

                # Scale the image
                scaled_image = image.scaled(new_width, new_height,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pixmap = QPixmap.fromImage(scaled_image)
            else:
                pixmap = QPixmap.fromImage(image)

            self.camera_label.setPixmap(pixmap)

    def update_prediction(self, predicted_class, probability):
        """Update prediction display and store in JSON"""
        confidence_text = f"{probability:.2%}"
        self.camera_prediction_label.setText(f"Prediction: {predicted_class} ({confidence_text})")

        # Store prediction in JSON file
        additional_data = {
            "model_classes": self.class_names,
            "camera_feed": True,
            "ui_component": "real_time_classification"
        }
        
        prediction_id = self.prediction_storage.save_prediction(
            predicted_class, 
            probability, 
            additional_data
        )
        
        if prediction_id:
            print(f"✓ Prediction stored with ID: {prediction_id}")

        # Update only the relevant dashboard based on actual class names
        if predicted_class.lower() in ['normal', 'good', 'ok', 'non_defective']:
            self.left_dashboard.update_classification(predicted_class, probability)
        else:
            # All other classes (defects) go to right dashboard
            self.right_dashboard.update_classification(predicted_class, probability)

    def update_motion_status(self, motion_detected):
        """Update UI or status based on motion detection"""
        if motion_detected:
            self.display_message("Motion detected - predictions paused")
        else:
            self.display_message("Camera stationary - ready for prediction")

    def export_visual_data(self):
        """Export visual data - simple text-based export"""
        try:
            stats = self.prediction_storage.get_session_statistics()
            total = stats.get("total", 0)
            
            if total == 0:
                QMessageBox.information(
                    self,
                    "No Data",
                    "No predictions available to export. Start camera feed and make some predictions first."
                )
                return
            
            # Get file path from user
            file_dialog = QFileDialog()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"visual_export_{timestamp}.txt"
            
            filepath, _ = file_dialog.getSaveFileName(
                self,
                "Export Visual Report",
                default_filename,
                "Text Files (*.txt);;All Files (*)"
            )
            
            if not filepath:
                return
            
            # Load session data
            with open(self.prediction_storage.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.prediction_storage.session_id not in data["sessions"]:
                QMessageBox.warning(self, "Export Failed", "Session data not found.")
                return
            
            predictions = data["sessions"][self.prediction_storage.session_id]["predictions"]
            classes = [p["predicted_class"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            
            # Create simple visual text export
            with open(filepath, 'w') as f:
                f.write("SAKAR VISION AI - VISUAL EXPORT REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Summary
                normal_keywords = ['normal', 'good', 'ok', 'non_defective']
                normal_count = sum(1 for c in classes if c.lower() in normal_keywords)
                defective_count = len(classes) - normal_count
                
                f.write("SUMMARY:\n")
                f.write(f"Total Predictions: {len(classes)}\n")
                f.write(f"Normal/Good: {normal_count}\n")
                f.write(f"Defective: {defective_count}\n")
                f.write(f"Average Confidence: {np.mean(confidences):.2%}\n\n")
                
                # Class breakdown
                class_counts = {}
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                f.write("CLASS BREAKDOWN:\n")
                for cls, count in sorted(class_counts.items()):
                    percentage = (count / len(classes)) * 100
                    f.write(f"{cls}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Visual representation
                f.write("VISUAL REPRESENTATION:\n")
                f.write("(N=Normal/Good, D=Defective)\n\n")
                
                for i, cls in enumerate(classes):
                    if i % 10 == 0 and i > 0:
                        f.write("\n")
                    
                    if cls.lower() in normal_keywords:
                        f.write("[N] ")
                    else:
                        f.write("[D] ")
                
                f.write("\n\n")
                
                # Detailed list
                f.write("DETAILED PREDICTIONS:\n")
                f.write("-" * 40 + "\n")
                for i, (cls, conf) in enumerate(zip(classes, confidences), 1):
                    f.write(f"{i:3d}. {cls:15s} ({conf:.1%})\n")
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Visual report created:\n{filepath}"
            )
            
        except Exception as e:
            print(f"Error in visual export: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during export:\n{e}"
            )

    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Stop camera feed if running
            if hasattr(self, 'camera_worker'):
                self.camera_worker.stop()
            if hasattr(self, 'camera_thread'):
                self.camera_thread.quit()
                self.camera_thread.wait()
            
            # Close prediction storage session
            if hasattr(self, 'prediction_storage'):
                self.prediction_storage.close_session()
                
            # Save session state on close
            save_session_on_close("image_classification")
            
        except Exception as e:
            print(f"Error during close: {e}")
        finally:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassiUI()
    window.show()
    sys.exit(app.exec_())