#!/usr/bin/env python3
"""
SAKAR VISION AI - Deployment UI Module

OVERVIEW:
This module implements the production deployment interface for the Sakar Vision AI platform, serving as the primary 
real-time defect detection and analysis system for industrial manufacturing inspection. It provides a comprehensive 
live monitoring dashboard that combines advanced computer vision capabilities with intelligent object tracking, 
automated defect counting, and real-time database synchronization to deliver continuous quality control monitoring 
for manufacturing processes with professional-grade accuracy and reliability.

KEY FUNCTIONALITY:
The system features sophisticated real-time YOLO-based defect detection with configurable confidence thresholds and 
class filtering, advanced object tracking algorithms to prevent duplicate counting of the same defects across video 
frames, dual-dashboard architecture for comprehensive defect monitoring with synchronized threshold management and 
automatic load balancing, and integrated Azure MySQL database synchronization for persistent defect count storage 
with user-specific tracking and batch update optimization. It includes intelligent model management with automatic 
detection of latest trained models from multiple training directories, dynamic camera management with configurable 
camera index selection and frame processing optimization, comprehensive session management with user authentication 
integration and persistent state tracking, and real-time performance monitoring with FPS tracking, inference time 
measurement, and system status indicators.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced multi-threading architecture for non-blocking inference operations, the module employs 
OpenCV for camera operations with optimized frame buffering and processing pipelines, comprehensive PyTorch integration 
with YOLO model loading and GPU acceleration support, and sophisticated object tracking systems using DeepSORT or 
custom tracking algorithms for reliable defect counting. The architecture features modular UI components (CustomTitleBar, 
CountingDashboard, ModelInferenceThread) with professional styling and real-time status updates, intelligent model 
discovery and loading from multiple training result directories with automatic latest model selection, robust error 
handling and recovery mechanisms for camera failures and model loading issues, and comprehensive Azure database 
integration with transaction management, batch processing, and user-specific data isolation following Azure best 
practices for scalable industrial deployment scenarios.
"""

import json
import os
import sys
import threading
import time
import traceback
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict, Counter
from datetime import datetime
import socket
import requests
from PIL import Image
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QPropertyAnimation, QRectF
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient, QBrush
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QProgressBar, QScrollArea
from PyQt5.QtWidgets import QSlider, QSpinBox
from ultralytics import YOLO
from azure_database import update_defect_count

# Import DeepSORT tracker
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("DeepSORT not available. Install with: pip install deep-sort-realtime")

from utils import ClassManager
from utils import set_window_icon

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Fix Qt plugin issue - explicitly set plugin path
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'
if 'cv2/qt/plugins' in os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', ''):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'

# Fix thread issue by ensuring proper thread initialization
os.environ['QT_THREAD_PRIORITY'] = 'normal'

# Path to defects configuration file
DEFECTS_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "defects_config.json")

# Path to store dashboard reports
REPORTS_STORAGE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "reports_storage.json")

# Session state path for persistence
SESSION_STATE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "session_state.json")


def save_session_state(ui_name="deploy"):
    """Save session state when this UI is opened"""
    try:
        # Load existing session state to preserve user information
        existing_session = {}
        if os.path.exists(SESSION_STATE_PATH):
            with open(SESSION_STATE_PATH, 'r') as f:
                existing_session = json.load(f)

        # Preserve existing user information and just update the UI name and timestamp
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "additional_data": {
                "opened_at": datetime.now().isoformat()
            }
        }

        # Preserve existing user information if available
        if "user" in existing_session:
            session_data["user"] = existing_session["user"]
        if "user_info" in existing_session:
            session_data["user_info"] = existing_session["user_info"]
        if "current_user_id" in existing_session:
            session_data["current_user_id"] = existing_session["current_user_id"]
        if "current_username" in existing_session:
            session_data["current_username"] = existing_session["current_username"]

        with open(SESSION_STATE_PATH, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session state saved: {ui_name}")

    except Exception as e:
        print(f"Error saving session state: {e}")


def save_session_on_close(ui_name="deploy"):
    """Save session state when this UI is closed"""
    try:
        # Load existing session state to preserve user information
        existing_session = {}
        if os.path.exists(SESSION_STATE_PATH):
            with open(SESSION_STATE_PATH, 'r') as f:
                existing_session = json.load(f)

        # Preserve existing user information and just update the UI name and timestamp
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "additional_data": {
                "closed_at": datetime.now().isoformat()
            }
        }

        # Preserve existing user information if available
        if "user" in existing_session:
            session_data["user"] = existing_session["user"]
        if "user_info" in existing_session:
            session_data["user_info"] = existing_session["user_info"]
        if "current_user_id" in existing_session:
            session_data["current_user_id"] = existing_session["current_user_id"]
        if "current_username" in existing_session:
            session_data["current_username"] = existing_session["current_username"]

        with open(SESSION_STATE_PATH, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session state saved on close: {ui_name}")

    except Exception as e:
        print(f"Error saving session state on close: {e}")


def calculate_iou(boxA, boxB):

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxA[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  # Add epsilon for stability

    # Return the intersection over union value
    return iou


def case_insensitive_match(defect_name, defect_classes):
    """Helper function to match defect names case-insensitively."""
    return any(defect_name.lower() == cls.lower() for cls in defect_classes)


class CustomTitleBar(QFrame):
    def __init__(self, parent=None, demo_button=None):
        super().__init__(parent)
        self.setFixedHeight(70)  # Adjusted height for better header appearance
        self.setStyleSheet("""
            background-color: white;  /* White background */
            border: none;
        """)

        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        # Logo image label setup
        self.logo_label = QLabel()
        logo_path = "sakar_logo.png"  # Path to the logo image
        logo_pixmap = QPixmap(logo_path)
        logo_pixmap = logo_pixmap.scaledToHeight(200, Qt.SmoothTransformation)  # Increased size
        self.logo_label.setPixmap(logo_pixmap)

        # Add stretch before title to push it toward center
        layout.addWidget(self.logo_label)
        layout.addStretch(1)

        # Title label with center alignment
        self.title_label = QLabel("SAKAR VISION AI")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: black;  /* Black text for better contrast on white background */
            font-weight: bold;
            font-size: 32px;  /* Larger font size */
        """)

        # Add title to layout (without stretch factor)
        layout.addWidget(self.title_label)

        # Add stretch after title to push it toward center
        layout.addStretch(1)

        # Add system status indicator to the right side of title bar (same as other UIs)
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-size: 16px;
            }
        """)

        self.status_label = QLabel("System Online")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-size: 14px;
                font-weight: 500;
            }
        """)

        # Create right section for status indicators (matching other UIs)
        right_section = QHBoxLayout()
        right_section.setSpacing(10)
        right_section.addWidget(self.status_indicator)
        right_section.addWidget(self.status_label)

        layout.addLayout(right_section)

        # Add any additional button if provided
        if demo_button:
            layout.addWidget(demo_button)

        self.setLayout(layout)

        # Timer to check internet connectivity (same as other UIs)
        self.connectivity_timer = QTimer(self)
        self.connectivity_timer.timeout.connect(self.check_internet_connectivity)
        self.connectivity_timer.start(5000)  # Check every 5 seconds

        # Initial connectivity check
        self.check_internet_connectivity()

    def check_internet_connectivity(self):
        """
        Check if internet connection is available and update status indicator.
        """
        try:
            # Try to connect to Google's DNS server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.update_status(True)
            return True
        except OSError:
            try:
                # Fallback: Try HTTP request to Google
                response = requests.get("http://www.google.com", timeout=3)
                if response.status_code == 200:
                    self.update_status(True)
                    return True
            except:
                pass
            self.update_status(False)
            return False

    def update_status(self, is_online):
        """
        Update the visual status indicator based on connectivity.
        """
        if is_online:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #28a745;
                    font-size: 16px;
                }
            """)
            self.status_label.setText("System Online")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #28a745;
                    font-size: 14px;
                    font-weight: 500;
                }
            """)
        else:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    font-size: 16px;
                }
            """)
            self.status_label.setText("System Offline")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    font-size: 14px;
                    font-weight: 500;
                }
            """)


class ModelInferenceThread(QThread):
    """Thread for running model inference without blocking the UI."""
    inference_complete = pyqtSignal(
        object, float, object, object)  # Results, inference time, material classifications, tracking data

    def __init__(self, yolo_model, frame, selected_defects, mobilenet_model=None, transform=None, tracker=None, material_enabled=False, defect_classes=None):
        super().__init__()
        self.yolo_model = yolo_model
        self.frame = frame.copy()  # Create a copy to avoid race conditions
        self.selected_defects = selected_defects
        self.mobilenet_model = mobilenet_model
        self.transform = transform
        # Use the provided defect classes or fallback to default
        self.defect_classes = defect_classes if defect_classes is not None else [
            'burr', 'casting', 'hole', 'punch', 'scratch']
        self.classification_classes = ["MS", "Copper", "Brass"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = tracker
        self.material_enabled = material_enabled  # New flag to track if material classification is enabled

    def run(self):
        try:
            start_time = time.time()

            # Run YOLO model on the frame
            results = self.yolo_model(self.frame, conf=0.10, verbose=False)

            # Material classifications dict to store results for each detection
            material_predictions = {}

            # Direct tracking data - simplified without DeepSORT
            tracking_data = {}

            # Process each detection for material classification
            if len(results) > 0:
                result = results[0]  # Get first result
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                # Only proceed if there are detections with good confidence
                if len(boxes) > 0 and np.max(confidences) >= 0:  # Changed confidence threshold to 0
                    # First, run material classification on all detections
                    if self.material_enabled and self.mobilenet_model is not None and self.transform is not None:
                        for i, box in enumerate(boxes):
                            # Only classify detections with good confidence
                            if confidences[i] < 0.10:  # Skip low confidence detections
                                continue

                            # Extract the region
                            x1, y1, x2, y2 = map(int, box)

                            # Ensure coordinates are valid
                            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                                # Crop the region
                                cropped = self.frame[y1:y2, x1:x2]

                                if cropped.size > 0:
                                    try:
                                        # Convert to PIL for transform
                                        cropped_pil = Image.fromarray(
                                            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

                                        # Transform and classify
                                        cropped_tensor = self.transform(
                                            cropped_pil).unsqueeze(0).to(self.device)

                                        with torch.no_grad():
                                            output = self.mobilenet_model(cropped_tensor)
                                            predicted_class = torch.argmax(
                                                torch.softmax(output, dim=1), dim=1).item()
                                            material_name = self.classification_classes[predicted_class]

                                            # Store the prediction with box index as key
                                            material_predictions[i] = material_name
                                            print(f"Classified box {i} as {material_name}")
                                    except Exception as e:
                                        print(f"Error in material classification: {e}")

                    # Generate tracking IDs for detections with good confidence
                    for i, (bbox, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                        # Skip low confidence detections
                        if conf < 0.1:  # Higher threshold for counting
                            continue

                        x1, y1, x2, y2 = map(int, bbox)

                        # Skip detections not in selected defects
                        if class_id < len(self.defect_classes):
                            defect_name = self.defect_classes[class_id]
                        else:
                            defect_name = f"Class {class_id}"

                        if not any(defect_name.lower() == selected.lower() for selected in self.selected_defects):
                            continue

                        # Create a more stable unique ID based on detection attributes
                        # Use location and class only, but round to nearest 20px to stabilize tracking
                        # This helps track the same object across frames without counting it multiple times
                        center_x = (x1 + x2) // 2 // 20 * 20  # Center X rounded to nearest 20
                        center_y = (y1 + y2) // 2 // 20 * 20  # Center Y rounded to nearest 20
                        width = (x2 - x1) // 20 * 20  # Width rounded to nearest 20
                        height = (y2 - y1) // 20 * 20  # Height rounded to nearest 20

                        # Include all these attributes in the ID to make it more robust
                        track_id = f"{class_id}_{center_x}_{center_y}_{width}_{height}"

                        # Create tracking data entry with basic information
                        tracking_data[track_id] = {
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id,
                            'confidence': conf,
                            'age': 10  # Fixed age to ensure it's counted
                        }

                        # Only add material information if material classification is enabled
                        if self.material_enabled:
                            # Get the material if available
                            material = material_predictions.get(i, "MS")
                            tracking_data[track_id]['material'] = material

            inference_time = time.time() - start_time
            self.inference_complete.emit(results, inference_time,
                                         material_predictions, tracking_data)

        except Exception as e:
            print(f"Error in inference thread: {e}")
            traceback.print_exc()
            self.inference_complete.emit(None, 0, {}, {})


class DeploymentUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set window properties before UI initialization
        self.setWindowTitle("Model Deployment - Live Analysis")

        # Set proper window flags - use this single approach
        self.setWindowFlags(Qt.Window)

        # Set minimum size
        self.setMinimumSize(1024, 768)

        self.setGeometry(100, 100, 1000, 700)

        # Save session state when this UI is opened
        save_session_state("deploy")

        # Set stylesheet for consistent look
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
                font-size: 14px;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #333333;
                font-weight: bold;
            }
            QPushButton {
                background-color: #808080;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #a0a0a0;
            }
        """)

        # Current user ID from login - initialize early to avoid attribute errors
        self.current_user_id = None
        self.user_info = None

        # Initialize models directly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = None
        self.yolo_model_path = "best46.pt"
        self.mobilenet_model = None
        self.mobilenet_model_path = "sakar_cls.pth"
        set_window_icon(self)

        # Camera variables
        self.camera_capture = None
        self.is_running = False
        self.detection_thread = None
        self.confidence_threshold = 0

        # Current material classification
        self.current_material = None

        # Defect classes and selection
        class_manager = ClassManager()
        if class_manager.initialized or class_manager.load_from_file():
            self.defect_classes = class_manager.get_classes()
            print(
                f"Deployment UI initialized with classes from ClassManager: {self.defect_classes}")
        else:
            print("No classes found in ClassManager, using default classes")
            self.defect_classes = []

        self.selected_defects = []
        self.classification_classes = ["MS", "Copper", "Brass"]

        # Material classification transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Debug mode
        self.debug_mode = True

        # Inference queue management
        self.max_concurrent_inferences = 1
        self.inference_queue = []

        # Add a timer for batch updates
        self.batch_update_timer = QTimer(self)
        self.batch_update_timer.timeout.connect(self.batch_save_reports)
        self.batch_update_timer.start(5000)  # Save reports every 5 seconds

        # Initialize a buffer for batched detections
        self.detection_buffer = []

        # Initialize UI first, then load selected defects
        self.initUI()

        # After UI initialization, try to load user info from config and then load defects
        self._load_user_id_from_configs()
        self.load_selected_defects()

        # Current user ID from login
        self.current_user_id = None

    def get_current_user_id(self):
        """
        Returns the current user ID for defect tracking.

        Returns:
            str: The current user ID or None if not available
        """
        # Try to get user ID from user_info if available
        if hasattr(self, 'user_info') and isinstance(self.user_info, dict) and 'id' in self.user_info:
            return self.user_info.get('id')

        # Return the stored current_user_id
        return self.current_user_id

    def set_user_info(self, user_info):
        """
        Set user information including user ID

        Args:
            user_info (dict): Dictionary containing user information with at least 'id' key
        """
        self.user_info = user_info
        if isinstance(user_info, dict) and 'id' in user_info:
            self.current_user_id = user_info['id']
            print(f"User ID set to: {self.current_user_id}")

    def load_selected_defects(self):
        """Load selected defects from configuration file and distribute between dashboards."""
        try:
            print(f"Looking for config file at: {DEFECTS_CONFIG_PATH}")
            print(f"Config file exists: {os.path.exists(DEFECTS_CONFIG_PATH)}")

            if os.path.exists(DEFECTS_CONFIG_PATH):
                with open(DEFECTS_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    loaded_defects = config.get('selected_defects', [])

                    # Extract user information and store it
                    if "user" in config and config["user"]:
                        username = config["user"]
                        from azure_database import get_user_id_by_username
                        user_id = get_user_id_by_username(username)
                        if user_id:
                            print(f"Setting user ID from config: {user_id} (username: {username})")
                            self.set_user_info({"id": user_id, "username": username})
                        else:
                            print(f"Could not retrieve user ID for username: {username}")

                    # Normalize case format to match defect_classes
                    self.selected_defects = []
                    for loaded_defect in loaded_defects:
                        if loaded_defect in self.defect_classes:
                            self.selected_defects.append(loaded_defect)
                        else:
                            matching_defects = [dc for dc in self.defect_classes
                                                if dc.lower() == loaded_defect.lower()]
                            if matching_defects:
                                self.selected_defects.append(matching_defects[0])
                            else:
                                self.selected_defects.append(loaded_defect)

                    print(f"Loaded selected defects: {self.selected_defects}")

                    # Distribute defects between left and right dashboards
                    if hasattr(self, 'leftDashboard') and hasattr(self, 'rightDashboard') and self.selected_defects:
                        left_defects = []
                        right_defects = []
                        for i, defect in enumerate(self.selected_defects):
                            if i % 2 == 0:
                                left_defects.append(defect)
                            else:
                                right_defects.append(defect)

                        self.leftDashboard.pre_populated_defects = [
                            d.capitalize() for d in left_defects]
                        self.rightDashboard.pre_populated_defects = [
                            d.capitalize() for d in right_defects]

                        self.leftDashboard.selected_defects_shown = False
                        self.rightDashboard.selected_defects_shown = False
                        self.leftDashboard.update_ui()
                        self.rightDashboard.update_ui()

                    if not self.selected_defects:
                        self.selected_defects = self.defect_classes.copy()
                        print("No defects were selected, using all defect classes")
            else:
                self.selected_defects = self.defect_classes.copy()
                print("Defect config file not found, using all defect classes")
                print(f"Using all defect classes: {self.defect_classes}")

            # Update the selected defects label
            if hasattr(self, 'selectedDefectsLabel'):
                if len(self.selected_defects) <= 3:
                    text = "Selected defects: " + ", ".join(self.selected_defects)
                else:
                    text = f"Selected defects: {len(self.selected_defects)} types"
                self.selectedDefectsLabel.setText(text)

            # Also try to load user info from metal_sheet_location_config.json if not already set
            if self.current_user_id is None:
                metal_sheet_location_config_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "metal_sheet_location_config.json")

                if os.path.exists(metal_sheet_location_config_path):
                    try:
                        with open(metal_sheet_location_config_path, 'r') as f:
                            location_config = json.load(f)
                            if "user" in location_config and location_config["user"]:
                                username = location_config["user"]
                                from azure_database import get_user_id_by_username
                                user_id = get_user_id_by_username(username)
                                if user_id:
                                    print(
                                        f"Setting user ID from location config: {user_id} (username: {username})")
                                    self.set_user_info({"id": user_id, "username": username})
                                else:
                                    print(
                                        f"Could not retrieve user ID for username from location config: {username}")
                    except Exception as e:
                        print(f"Error reading location config file: {e}")

        except Exception as e:
            print(f"Error loading defect configuration: {e}")
            traceback.print_exc()
            self.selected_defects = self.defect_classes.copy()

    def initUI(self):
        """Initialize the user interface components."""
        # Main layout
        mainLayout = QVBoxLayout(self)
        mainLayout.setSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        # Add custom title bar
        self.title_bar = CustomTitleBar(self)
        mainLayout.addWidget(self.title_bar)

        # Content widget with margins
        contentWidget = QWidget()
        contentLayout = QVBoxLayout(contentWidget)
        contentLayout.setSpacing(10)
        contentLayout.setContentsMargins(10, 10, 10, 10)

        # Create horizontal layout for left dashboard, camera feed, and right dashboard
        cameraAndDashboardLayout = QHBoxLayout()
        cameraAndDashboardLayout.setSpacing(10)

        # Create the dashboard instances for left and right sides
        self.leftDashboard = CountingDashboard()
        self.dashboard = self.leftDashboard  # Keep the original dashboard reference for compatibility
        self.rightDashboard = CountingDashboard()

        # Camera feed group
        cameraGroup = QGroupBox("Live Defect Detection")
        cameraGroup.setStyleSheet(
            "QGroupBox { font-size: 16px; font-weight: bold; color: #333333; border: 2px solid #FFA500; border-radius: 10px; padding: 10px; background-color: #FFFFFF; }"
        )
        cameraLayout = QVBoxLayout()

        # Main camera feed display
        self.cameraFeedLabel = QLabel()
        self.cameraFeedLabel.setAlignment(Qt.AlignCenter)
        self.cameraFeedLabel.setMinimumSize(800, 600)
        self.cameraFeedLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cameraFeedLabel.setStyleSheet(
            "border: 3px solid #FFA500; border-radius: 10px;"
        )
        # Set the background image
        self.cameraFeedLabel.setPixmap(QPixmap("bg.png"))
        self.cameraFeedLabel.setScaledContents(True)
        self.cameraFeedLabel.setText("Camera feed will appear here when started")
        cameraLayout.addWidget(self.cameraFeedLabel, alignment=Qt.AlignCenter)

        # Add more margin at the bottom of the camera feed
        cameraLayout.addSpacing(20)

        # Create a container for all controls
        controlsContainer = QWidget()
        controlsContainer.setStyleSheet(
            "background-color: #FFFFFF; border-radius: 10px; padding: 15px;"
        )
        controlsContainerLayout = QVBoxLayout(controlsContainer)
        controlsContainerLayout.setSpacing(15)
        controlsContainerLayout.setContentsMargins(15, 15, 15, 15)

        # Keep model combobox in backend but hide from UI
        self.modelComboBox = QComboBox()
        self.modelComboBox.hide()  # Hide from UI
        self.loadAvailableModels()

        # Action buttons layout with better spacing
        actionButtonsLayout = QHBoxLayout()
        actionButtonsLayout.setSpacing(20)  # More spacing between elements

        # Simplified total rejected count - direct labels without container
        totalRejectedBox = QHBoxLayout()

        # Title label
        totalRejectedTitle = QLabel("TOTAL DEFECTS:")
        totalRejectedTitle.setStyleSheet("""
            font-weight: bold;
            font-size: 40px;
            color: #333333;
            margin-left: 300px;                             
                                         
        """)
        totalRejectedBox.addWidget(totalRejectedTitle)

        # Count label with large font
        self.totalRejectedCount = QLabel("0")
        self.totalRejectedCount.setStyleSheet("""
            font-weight: bold;
            font-size: 40px;
            color: #ff914d;
            margin-left: 10px;
        """)
        totalRejectedBox.addWidget(self.totalRejectedCount)

        # Add the total rejected box to the action buttons layout
        actionButtonsLayout.addLayout(totalRejectedBox)

        # Status label with fixed minimum width
        self.statusLabel = QLabel("Status: Ready")
        self.statusLabel.setMinimumWidth(200)
        self.statusLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.statusLabel.setStyleSheet("font-weight: bold; color: #333333;")
        actionButtonsLayout.addWidget(self.statusLabel, 1)

        # Add action buttons to controls container
        controlsContainerLayout.addLayout(actionButtonsLayout)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #d0d0d0;")
        controlsContainerLayout.addWidget(separator)

        # Add the controls container to the camera layout
        cameraLayout.addWidget(controlsContainer)

        cameraGroup.setLayout(cameraLayout)

        # Add both dashboards and camera group to the horizontal layout
        # Left dashboard takes 1/5 of width
        cameraAndDashboardLayout.addWidget(self.leftDashboard, 1)
        cameraAndDashboardLayout.addWidget(cameraGroup, 4)  # Camera feed takes 4/6 of width
        # Right dashboard takes 1/5 of width
        cameraAndDashboardLayout.addWidget(self.rightDashboard, 1)

        # Add the camera and dashboard layout to the content layout
        contentLayout.addLayout(cameraAndDashboardLayout)

        # Status bar
        statusBarFrame = QFrame()
        statusBarFrame.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px;")
        statusBarLayout = QHBoxLayout(statusBarFrame)
        statusBarLayout.setContentsMargins(10, 5, 10, 5)

        self.inferenceTimeLabel = QLabel("Inference time: --")
        self.fpsLabel = QLabel("FPS: --")
        self.selectedDefectsLabel = QLabel("Selected defects: --")
        statusBarLayout.addWidget(self.inferenceTimeLabel)
        statusBarLayout.addWidget(self.fpsLabel)
        statusBarLayout.addWidget(self.selectedDefectsLabel)
        statusBarLayout.addStretch(1)

        contentLayout.addWidget(statusBarFrame)

        # Add content widget to main layout
        mainLayout.addWidget(contentWidget)

        # Camera timer
        self.camera_feed_timer = QTimer(self)
        self.camera_feed_timer.timeout.connect(self.update_camera_feed)

        # Frame buffer and FPS variables
        self.frame_buffer_size = 2
        self.frame_buffer = []
        self.fps_frames = 0
        self.fps_start_time = 0
        self.last_fps_update = 0
        self.processing_width = 640
        self.processing_height = 480

        # Initialize DeepSORT tracker if available
        if DEEPSORT_AVAILABLE:
            self.object_tracker = DeepSort(
                max_age=30,                  # Max frames to keep track of objects
                n_init=3,                    # Frames needed to confirm a track
                nms_max_overlap=1.0,         # Non-maximum suppression threshold
                max_cosine_distance=0.4,     # Threshold for appearance similarity
                nn_budget=100                # Maximum samples per class
            )
        else:
            self.object_tracker = None

        # Now connect threshold signals after both dashboards are fully initialized
        self.leftDashboard.threshold_changed.connect(
            lambda value: self.rightDashboard.update_threshold_value(value))
        self.rightDashboard.threshold_changed.connect(
            lambda value: self.leftDashboard.update_threshold_value(value))

    def get_latest_trained_model(self):
        """
        Detects the latest trained model with comprehensive debugging.
        Prioritizes the highest numbered train folder (train, train1, train2, etc.).
        """
        print("\n" + "="*50)
        print("🔍 SEARCHING FOR LATEST TRAINED MODEL")
        print("="*50)

        # Primary location: combine/runs directory
        combine_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combine/runs")
        print(f"📁 Primary search location: {combine_runs_dir}")
        print(f"📁 Directory exists: {os.path.exists(combine_runs_dir)}")

        # First, try to find the latest train folder in combine/runs
        latest_model_path = self._find_latest_from_combine_runs(combine_runs_dir)
        if latest_model_path:
            print(f"✅ FOUND LATEST MODEL: {latest_model_path}")
            print("="*50 + "\n")
            return latest_model_path

        print("⚠️ No model found in combine/runs, checking fallback directories...")

        # Fallback: Check other possible directories
        other_possible_dirs = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "combine/runs/detect"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/trained")
        ]

        all_model_candidates = []

        # Search fallback directories
        for base_dir in other_possible_dirs:
            print(f"📁 Checking fallback: {base_dir}")
            if not os.path.exists(base_dir):
                print(f"   ❌ Directory not found")
                continue

            print(f"   ✅ Directory exists, scanning...")

            # Get all subdirectories in the base directory
            try:
                subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                           if os.path.isdir(os.path.join(base_dir, d))]
                print(f"   📂 Found {len(subdirs)} subdirectories")
            except PermissionError:
                print(f"   ❌ Permission denied")
                continue

            if not subdirs:
                print(f"   ℹ️ No subdirectories found")
                continue

            # Check each subdirectory for weights/best.pt
            for subdir in subdirs:
                subdir_name = os.path.basename(subdir)
                possible_weights = [
                    os.path.join(subdir, "weights", "best.pt"),
                    os.path.join(subdir, "weights", "last.pt"),
                    os.path.join(subdir, "best.pt"),
                    os.path.join(subdir, "last.pt")
                ]

                for weights_path in possible_weights:
                    if os.path.exists(weights_path):
                        mod_time = os.path.getmtime(weights_path)
                        mod_time_str = datetime.fromtimestamp(
                            mod_time).strftime("%Y-%m-%d %H:%M:%S")
                        all_model_candidates.append((weights_path, mod_time, subdir))
                        print(f"   🎯 Found: {weights_path} (modified: {mod_time_str})")
                        break

        # Sort all candidates by modification time (latest first)
        if all_model_candidates:
            all_model_candidates.sort(key=lambda x: x[1], reverse=True)
            latest_model_path = all_model_candidates[0][0]
            latest_subdir = all_model_candidates[0][2]

            print(f"✅ SELECTED FALLBACK MODEL: {latest_model_path}")
            print(f"📁 From directory: {latest_subdir}")
            print("="*50 + "\n")
            return latest_model_path

        print("❌ NO TRAINED MODELS FOUND ANYWHERE")
        print("="*50 + "\n")
        return None

    def _find_latest_from_combine_runs(self, combine_runs_dir):
        """Enhanced version with detailed debugging."""
        if not os.path.exists(combine_runs_dir):
            print(f"❌ Combine runs directory not found: {combine_runs_dir}")
            return None

        print(f"🔍 Scanning combine/runs directory...")

        try:
            # Get all items in the directory
            all_items = os.listdir(combine_runs_dir)
            print(f"📂 Found {len(all_items)} items: {all_items}")

            # Filter for train folders and extract their numbers
            train_folders = []
            for item in all_items:
                item_path = os.path.join(combine_runs_dir, item)
                print(f"   📁 Checking item: {item}")

                if not os.path.isdir(item_path):
                    print(f"      ❌ Not a directory")
                    continue

                if not item.startswith('train'):
                    print(f"      ❌ Doesn't start with 'train'")
                    continue

                # Extract the training run number
                if item == 'train':
                    train_number = 0  # Base train folder
                    print(f"      ✅ Base train folder (number: 0)")
                elif item.startswith('train') and len(item) > 5:
                    try:
                        # Extract number from train1, train2, etc.
                        number_part = item[5:]  # Remove 'train' prefix
                        train_number = int(number_part)
                        print(f"      ✅ Numbered train folder (number: {train_number})")
                    except ValueError:
                        print(f"      ❌ Can't parse number from '{number_part}'")
                        continue
                else:
                    print(f"      ❌ Doesn't match expected pattern")
                    continue

                train_folders.append((train_number, item, item_path))

            if not train_folders:
                print("❌ No valid train folders found")
                return None

            print(f"✅ Found {len(train_folders)} train folders")

            # Sort by train number (highest first)
            train_folders.sort(key=lambda x: x[0], reverse=True)
            print("📊 Sorted train folders:")
            for num, name, path in train_folders:
                print(f"   {num}: {name}")

            # Try each folder starting from the highest number
            for train_number, folder_name, folder_path in train_folders:
                print(f"\n🔍 Checking train folder: {folder_name} (number: {train_number})")

                # Look for weights/best.pt in this training folder
                possible_weights = [
                    os.path.join(folder_path, "weights", "best.pt"),
                    os.path.join(folder_path, "weights", "last.pt"),
                    os.path.join(folder_path, "best.pt"),
                    os.path.join(folder_path, "last.pt")
                ]

                for weights_path in possible_weights:
                    print(f"   📄 Checking: {weights_path}")
                    if os.path.exists(weights_path):
                        mod_time = os.path.getmtime(weights_path)
                        mod_time_str = datetime.fromtimestamp(
                            mod_time).strftime("%Y-%m-%d %H:%M:%S")
                        print(f"   🎯 FOUND MODEL: {weights_path}")
                        print(f"   📅 Modified: {mod_time_str}")
                        print(f"   ✅ Selected train folder: {folder_name}")
                        return weights_path
                    else:
                        print(f"   ❌ File not found")

                print(f"   ❌ No model weights found in {folder_name}")

            print("❌ No valid model weights found in any train folders")
            return None

        except Exception as e:
            print(f"❌ Error scanning combine/runs directory: {e}")
            import traceback
            traceback.print_exc()
            return None

    def loadAvailableModels(self):
        """Load available model files from standard locations with priority for latest trained model."""
        try:
            print("Scanning for models...")

            # Check if the UI has a model combo box - adjust this based on your actual UI component name
            if not hasattr(self, 'modelComboBox'):
                print("Warning: No modelComboBox found in DeploymentUI")
                return

            # Clear existing items first
            self.modelComboBox.clear()

            # Add select model option at index 0
            self.modelComboBox.addItem("Select Model", "")

            # Check for the latest trained model first
            latest_model_path = self.get_latest_trained_model()

            # Set default model path if not already set
            if not hasattr(self, 'yolo_model_path'):
                self.yolo_model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "sakar_ai.pt")

            if latest_model_path:
                # Extract folder info for display
                folder_name = self._extract_training_folder_name(latest_model_path)
                display_name = f"Latest Trained Model ({folder_name})"

                # Latest trained model found - add it and set as default
                self.modelComboBox.addItem(display_name, latest_model_path)
                self.modelComboBox.setCurrentIndex(1)  # Select the latest trained model
                self.model_path = latest_model_path
                print(f"Auto-selected latest trained model: {latest_model_path}")

                # Update status to show which model was auto-selected
                if hasattr(self, 'statusLabel'):
                    self.statusLabel.setText(f"Status: Auto-selected {folder_name}")

                # Also add the default model as an option if it exists
                if os.path.exists(self.yolo_model_path):
                    self.modelComboBox.addItem("Default Model (sakar_ai.pt)", self.yolo_model_path)
            else:
                # No latest trained model found - try to use default model
                if os.path.exists(self.yolo_model_path):
                    self.modelComboBox.addItem("Default Model (sakar_ai.pt)", self.yolo_model_path)
                    self.modelComboBox.setCurrentIndex(1)  # Select default model
                    self.model_path = self.yolo_model_path
                    print(f"No latest trained model found, using default: {self.yolo_model_path}")

                    if hasattr(self, 'statusLabel'):
                        self.statusLabel.setText("Status: Using default model")
                else:
                    # Neither latest trained nor default model exists
                    print(f"Warning: Default model not found at {self.yolo_model_path}")
                    if hasattr(self, 'statusLabel'):
                        self.statusLabel.setText(
                            "Status: No models found - please select manually")

            # Add additional models from the training runs
            self._add_additional_trained_models()

        except Exception as e:
            print(f"Error loading model list: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: ensure at least the default model is available if it exists
            if hasattr(self, 'modelComboBox') and self.modelComboBox.count() <= 1:  # Only "Select Model" option
                if hasattr(self, 'yolo_model_path') and os.path.exists(self.yolo_model_path):
                    self.modelComboBox.addItem("Default Model (sakar_ai.pt)", self.yolo_model_path)
                    self.modelComboBox.setCurrentIndex(1)
                    self.model_path = self.yolo_model_path
                elif hasattr(self, 'statusLabel'):
                    self.statusLabel.setText("Status: Error loading models")

    def _extract_training_folder_name(self, model_path):
        """Extract a user-friendly folder name from model path."""
        try:
            # Split path and find the relevant folder names
            path_parts = model_path.replace('\\', '/').split('/')

            # Look for train folders specifically
            for i, part in enumerate(path_parts):
                if part.startswith('train'):
                    # If it's in combine/runs/train pattern
                    if i > 0 and path_parts[i-1] == 'runs':
                        return part
                    # If it's just a train folder
                    return part

            # Look for runs/train or runs/detect patterns (fallback)
            if 'runs' in path_parts:
                runs_index = path_parts.index('runs')
                if runs_index + 2 < len(path_parts):
                    # Pattern: runs/detect/train or runs/train
                    if path_parts[runs_index + 1] == 'detect' and runs_index + 2 < len(path_parts):
                        return f"detect/{path_parts[runs_index + 2]}"
                    else:
                        return path_parts[runs_index + 1]

            # Fallback: use the parent directory name
            return os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        except:
            return "best.pt"

    def _add_additional_trained_models(self):
        """Add other available trained models to the dropdown for user choice."""
        try:
            # First, add models from combine/runs
            self._add_models_from_combine_runs()

            # Then add models from other locations
            runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
            if os.path.exists(runs_dir):
                self._add_models_from_runs_dir(runs_dir)

        except Exception as e:
            print(f"Error adding additional models: {e}")

    def _add_models_from_combine_runs(self):
        """Add all available models from combine/runs directory."""
        combine_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combine/runs")
        if not os.path.exists(combine_runs_dir):
            return

        try:
            # Check if modelComboBox exists
            if not hasattr(self, 'modelComboBox'):
                return

            # Get current paths to avoid duplicates
            current_paths = [self.modelComboBox.itemData(
                i) for i in range(self.modelComboBox.count())]

            # Get all train folders
            all_items = os.listdir(combine_runs_dir)
            train_folders = []

            for item in all_items:
                item_path = os.path.join(combine_runs_dir, item)
                if os.path.isdir(item_path) and item.startswith('train'):
                    # Extract the training run number for sorting
                    if item == 'train':
                        train_number = 0
                    elif item.startswith('train') and len(item) > 5:
                        try:
                            number_part = item[5:]
                            train_number = int(number_part)
                        except ValueError:
                            continue
                    else:
                        continue

                    # Look for model weights
                    weights_path = os.path.join(item_path, "weights", "best.pt")
                    if os.path.exists(weights_path):
                        try:
                            mod_time = os.path.getmtime(weights_path)
                            train_folders.append((train_number, item, weights_path, mod_time))
                        except:
                            pass

            # Sort by train number (highest first)
            train_folders.sort(key=lambda x: x[0], reverse=True)

            # Add to combobox (skip if already added)
            for train_number, folder_name, model_path, mod_time in train_folders:
                if model_path not in current_paths:
                    display_name = f"Combine/Runs Model ({folder_name})"
                    self.modelComboBox.addItem(display_name, model_path)
                    print(f"Added model option: {display_name}")

        except Exception as e:
            print(f"Error adding models from combine/runs: {e}")

    def _add_models_from_runs_dir(self, runs_dir):
        """Add models from the standard runs directory."""
        try:
            if not hasattr(self, 'modelComboBox'):
                return

            all_models = []

            # Collect all available models
            for item in os.listdir(runs_dir):
                item_path = os.path.join(runs_dir, item)
                if not os.path.isdir(item_path):
                    continue

                # Check train folders
                if item.startswith('train'):
                    self._collect_models_from_folder(item_path, all_models, item)

                # Check detect folders for train subfolders
                elif item.startswith('detect'):
                    try:
                        for detect_item in os.listdir(item_path):
                            detect_item_path = os.path.join(item_path, detect_item)
                            if (os.path.isdir(detect_item_path) and
                                    detect_item.startswith('train')):
                                self._collect_models_from_folder(
                                    detect_item_path, all_models, f"{item}/{detect_item}")
                    except:
                        continue

            # Sort by modification time and add to combobox (skip the latest one as it's already added)
            all_models.sort(key=lambda x: x[1], reverse=True)
            current_paths = [self.modelComboBox.itemData(
                i) for i in range(self.modelComboBox.count())]

            for model_path, mod_time, folder_name in all_models:
                if model_path not in current_paths:
                    self.modelComboBox.addItem(f"Trained Model ({folder_name})", model_path)

        except Exception as e:
            print(f"Error adding additional models from runs: {e}")

    def _collect_models_from_folder(self, folder_path, models_list, folder_name):
        """Collect model files from a specific training folder."""
        weights_path = os.path.join(folder_path, "weights", "best.pt")
        if os.path.exists(weights_path):
            try:
                mod_time = os.path.getmtime(weights_path)
                models_list.append((weights_path, mod_time, folder_name))
            except:
                pass

    def browseModelFile(self):
        """Opens a file dialog for users to select a model file."""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            os.path.expanduser("~"),
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )

        if file_path and os.path.exists(file_path):
            # Add to the combo box if it's not already there
            found = False
            for i in range(self.modelComboBox.count()):
                if self.modelComboBox.itemData(i) == file_path:
                    self.modelComboBox.setCurrentIndex(i)
                    found = True
                    break

            if not found:
                # Add the new model to the combo box
                model_name = os.path.basename(file_path)
                self.modelComboBox.addItem(f"(Custom) {model_name}", file_path)
                self.modelComboBox.setCurrentIndex(self.modelComboBox.count() - 1)

            # Set this as the current model path
            self.model_path = file_path
            self.statusLabel.setText(f"Status: Model selected - {os.path.basename(file_path)}")

            # Extract defect classes directly from the model file
            try:
                # Load the model temporarily just to access class names
                temp_model = YOLO(file_path)

                # Extract class names from the model
                if hasattr(temp_model, 'names'):
                    model_classes = []

                    # Handle both dictionary and list formats
                    if isinstance(temp_model.names, dict):
                        # For dictionary format, preserve order by index
                        model_classes = []
                        class_dict = temp_model.names
                        # Get the maximum index to make sure we don't miss any classes
                        max_index = max(int(key) for key in class_dict.keys()
                                        if isinstance(key, (int, str)) and str(key).isdigit())

                        # Create a list with all classes in order
                        for i in range(max_index + 1):
                            if i in class_dict:
                                model_classes.append(class_dict[i])
                            elif str(i) in class_dict:
                                model_classes.append(class_dict[str(i)])
                    else:
                        # For list format, just convert to list
                        model_classes = list(temp_model.names)

                    # Ensure class names are valid strings
                    model_classes = [str(cls) for cls in model_classes if cls]

                    # Update our defect classes with the ones from the model
                    self.defect_classes = model_classes

                    # Also update selected defects
                    self.selected_defects = model_classes.copy()

                    print(
                        f"Extracted {len(model_classes)} defect classes from model: {model_classes}")

                    # Update the selected defects label
                    if hasattr(self, 'selectedDefectsLabel'):
                        if len(self.selected_defects) <= 3:
                            text = "Selected defects: " + ", ".join(self.selected_defects)
                        else:
                            text = f"Selected defects: {len(self.selected_defects)} types"
                        self.selectedDefectsLabel.setText(text)

                    # Update dashboard pre-populated defects
                    if hasattr(self, 'dashboard'):
                        self.dashboard.pre_populated_defects = [
                            d.capitalize() for d in self.selected_defects]
                        self.dashboard.reset_counts()
                        self.dashboard.update_ui()

                    # Add all model classes to ClassManager to ensure consistency
                    class_manager = ClassManager()
                    class_manager.update_classes(
                        model_classes, source=f"model:{os.path.basename(file_path)}")

                    # Initialize all classes in the Azure database with 0 count
                    user_id = self.get_current_user_id()
                    if user_id is None:
                        self._load_user_id_from_configs()
                        user_id = self.get_current_user_id()

                    if user_id:
                        # Get current threshold
                        threshold = self.dashboard.threshold_value if hasattr(
                            self, 'dashboard') else 50

                        # Initialize each defect in the database with count 0
                        from azure_database import update_defect_count
                        for defect_name in model_classes:
                            try:
                                update_defect_count(
                                    defect_name,
                                    0,  # Initial count of 0
                                    threshold,
                                    is_initialization=True,  # Mark as initialization
                                    user_id=user_id
                                )
                                print(
                                    f"Initialized class in Azure database: {defect_name} for user {user_id}")
                            except Exception as e:
                                print(f"Error initializing class {defect_name} in database: {e}")
                    else:
                        print("Warning: No user ID available, classes not initialized in database")

                    # Save the updated defects to configuration file
                    try:
                        config = {
                            'selected_defects': self.selected_defects
                        }

                        # Add user info if available
                        if hasattr(self, 'user_info') and isinstance(self.user_info, dict) and 'username' in self.user_info:
                            config['user'] = self.user_info['username']

                        # Save to file
                        with open(DEFECTS_CONFIG_PATH, 'w') as f:
                            json.dump(config, f, indent=2)

                        print(f"Saved {len(self.selected_defects)} defect classes to config file")
                    except Exception as e:
                        print(f"Error saving defect configuration: {e}")
                        traceback.print_exc()

                    # Show a message about the updated defect classes
                    QMessageBox.information(
                        self,
                        "Defect Classes Updated",
                        f"Loaded {len(model_classes)} defect classes from model and initialized in database."
                    )
                else:
                    print("Model doesn't have class names attribute")
            except Exception as e:
                print(f"Error extracting classes from model: {e}")
                traceback.print_exc()

    def toggleAnalysis(self):
        """Toggle between starting and stopping the analysis."""
        if not self.is_running:
            # Update UI first to show we're processing
            self.statusLabel.setText("Status: Initializing analysis...")
            QApplication.processEvents()  # Update UI immediately

            # Start analysis
            selected_index = self.modelComboBox.currentIndex()
            if selected_index <= 0:  # "Select Model" is at index 0
                QMessageBox.warning(self, "Model Selection", "Please select a model first.")
                self.statusLabel.setText("Status: Ready")
                return

            self.model_path = self.modelComboBox.currentData()
            if not os.path.exists(self.model_path):
                QMessageBox.warning(self, "Model Error",
                                    f"Model file not found: {self.model_path}")
                self.statusLabel.setText("Status: Error - Model file not found")
                return

            # Load the models
            try:
                # Check if we need to update classes from the model
                class_manager = ClassManager()
                # We prioritize manually defined classes over model-extracted classes
                if class_manager.source not in ["manual_setClasses", "manual_addNewClass", "auto_annotation_run"]:
                    # If we haven't defined classes manually, extract from model
                    if class_manager.extract_from_model(self.model_path):
                        # Update our defect classes
                        self.defect_classes = class_manager.get_classes()
                        print(f"Updated defect classes from model: {self.defect_classes}")

                # Use the latest defect classes from ClassManager
                self.defect_classes = class_manager.get_classes()

                # Load YOLO model for defect detection
                self.statusLabel.setText("Status: Loading model...")
                QApplication.processEvents()

                try:
                    # Load YOLO model
                    self.yolo_model = YOLO(self.model_path).to(self.device)
                    if self.debug_mode:
                        print(f"model loaded: {self.model_path}")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    traceback.print_exc()
                    raise

                # Initialize camera
                self.statusLabel.setText("Status: Initializing camera...")
                QApplication.processEvents()

                # First check if the camera index is saved in settings
                camera_index = 0  # default
                settings_file = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "inspection_settings.json")
                if os.path.exists(settings_file):
                    try:
                        with open(settings_file, 'r') as f:
                            settings = json.load(f)
                            if 'selected_camera_index' in settings:
                                camera_index = int(settings['selected_camera_index'])
                                print(f"Using saved camera index from settings: {camera_index}")
                    except Exception as e:
                        print(f"Error reading camera index from settings: {e}")

                # Try to open camera with the saved index first
                self.camera_capture = cv2.VideoCapture(camera_index)
                if not self.camera_capture.isOpened():
                    # If that didn't work, try other indices as a fallback
                    print(f"Failed to open camera at index {camera_index}, trying alternatives...")
                    for cam_index in range(4):
                        if cam_index == camera_index:
                            continue  # Already tried this one
                        self.camera_capture = cv2.VideoCapture(cam_index)
                        if self.camera_capture.isOpened():
                            print(f"Successfully opened camera at index {cam_index}")
                            break

                # Final check if camera is available
                if self.camera_capture is None or not self.camera_capture.isOpened():
                    error_msg = "Could not initialize any camera. Please check your camera connection."
                    print(error_msg)
                    QMessageBox.critical(self, "Camera Error", error_msg)
                    self.statusLabel.setText("Status: Error - Could not open camera")
                    return

                # Set camera properties
                self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.processing_width)
                self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.processing_height)
                self.camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Clear any existing frame buffer
                self.frame_buffer = []
                self.inference_queue = []

                # Start the timer with optimized interval
                self.is_running = True
                # self.toggleButton.setText("Stop Analysis")
                # self.toggleButton.setStyleSheet("""
                #     QPushButton {
                #         background-color: #f44336;
                #         color: white;
                #         font-weight: bold;
                #         padding: 8px 16px;
                #     }
                #     QPushButton:hover {
                #         background-color: #e53935;
                #     }
                # """)
                self.statusLabel.setText("")  # Remove status text
                self.fps_start_time = time.time()
                self.fps_frames = 0
                self.camera_feed_timer.start(40)  # 40ms for better performance

            except Exception as e:
                error_msg = f"Error starting analysis: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                QMessageBox.critical(self, "Error", error_msg)
                self.statusLabel.setText(f"Status: Error - {str(e)}")
        else:
            # Stop analysis
            self.is_running = False
            # self.toggleButton.setText("Start Analysis")
            # self.toggleButton.setStyleSheet("""
            #     QPushButton {
            #         background-color: #4CAF50;
            #         color: white;
            #         font-weight: bold;
            #         padding: 8px 16px;
            #     }
            #     QPushButton:hover {
            #         background-color: #45a049;
            #     }
            # """)
            self.statusLabel.setText("Status: Stopped")

            if self.camera_feed_timer.isActive():
                self.camera_feed_timer.stop()

            if self.camera_capture and self.camera_capture.isOpened():
                self.camera_capture.release()
                self.camera_capture = None

            self.inferenceTimeLabel.setText("Inference time: --")
            self.fpsLabel.setText("FPS: --")
            self.yolo_model = None
            self.mobilenet_model = None

    def update_camera_feed(self):
        """Updates the camera feed with real-time analysis"""
        if not self.camera_capture or not self.is_running:
            return

        try:
            # Read from camera
            ret, frame = self.camera_capture.read()
            if not ret:
                self.statusLabel.setText("Status: Error reading from camera")
                self.toggleAnalysis()  # Stop analysis
                return

            # Add to frame buffer for smoother display
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > self.frame_buffer_size:
                self.frame_buffer.pop(0)

            # Update FPS counter
            self.fps_frames += 1
            elapsed_time = time.time() - self.fps_start_time

            # Update FPS every second
            if elapsed_time > 1.0 and time.time() - self.last_fps_update > 1.0:
                self.fpsLabel.setText(f"FPS: {self.fps_frames / elapsed_time:.1f}")
                self.fps_frames = 0
                self.fps_start_time = time.time()
                self.last_fps_update = time.time()

            # Only start a new inference if we don't have too many running
            active_threads = len([t for t in self.inference_queue if t.isRunning()])
            if active_threads < self.max_concurrent_inferences:
                # Use the latest frame for inference
                if self.frame_buffer:
                    inference_frame = self.frame_buffer[-1]

                    # Create and start a new inference thread
                    self.detection_thread = ModelInferenceThread(
                        self.yolo_model,
                        inference_frame,
                        self.selected_defects,
                        None,  # Removed mobilenet_model reference
                        self.transform,
                        self.object_tracker,  # Pass the tracker to the thread
                        False,  # Removed material classification status
                        defect_classes=self.defect_classes  # Pass current defect classes
                    )
                    self.detection_thread.inference_complete.connect(self.processResults)
                    self.detection_thread.start()

                    # Add to queue and clean up finished threads
                    self.inference_queue.append(self.detection_thread)
                    self.inference_queue = [t for t in self.inference_queue if t.isRunning()]

            # Always display the most recent frame if no annotated frame yet
            if self.frame_buffer and not hasattr(self, 'last_annotated_frame'):
                self.displayFrame(self.frame_buffer[-1])

        except Exception as e:
            print(f"Error in update_camera_feed: {e}")
            traceback.print_exc()
            self.statusLabel.setText(f"Status: Camera error - {str(e)}")

    def processResults(self, results, inference_time, material_predictions, tracking_data):
        """Process results from model inference."""
        # Update inference time display
        self.inferenceTimeLabel.setText(f"Inference time: {inference_time*1000:.1f} ms")

        try:
            # Only process if we have frames in buffer and valid results
            if not self.frame_buffer or results is None:
                return

            # Get the latest frame from buffer for annotation
            frame = self.frame_buffer[-1].copy()

            # Debug output for tracking_data
            if tracking_data:
                print(f"Received tracking data with {len(tracking_data)} tracked objects")

            # Create a set to track new defects that should be added to the database
            newly_detected_defects = set()

            # Debug: Print current user ID to help troubleshoot
            current_user_id = self.get_current_user_id()
            print(f"Current user ID in processResults: {current_user_id}")

            left_tracking_data = {}
            right_tracking_data = {}

            if tracking_data:
                for track_id, track_info in tracking_data.items():
                    class_id = track_info.get('class_id', -1)
                    if 0 <= class_id < len(self.defect_classes):
                        defect_name = self.defect_classes[class_id]

                        defect_index = -1
                        for i, d in enumerate(self.selected_defects):
                            if d.lower() == defect_name.lower():
                                defect_index = i
                                break

                        if defect_index >= 0:
                            if defect_index % 2 == 0:
                                left_tracking_data[track_id] = track_info
                            else:
                                right_tracking_data[track_id] = track_info
                        else:
                            left_tracking_data[track_id] = track_info
                            right_tracking_data[track_id] = track_info

            if hasattr(self, 'leftDashboard') and hasattr(self, 'rightDashboard'):
                previous_left_counts = self.leftDashboard.defect_counts.copy()
                previous_right_counts = self.rightDashboard.defect_counts.copy()

                if left_tracking_data:
                    self.leftDashboard.update_counts(left_tracking_data, self.defect_classes)

                if right_tracking_data:
                    self.rightDashboard.update_counts(right_tracking_data, self.defect_classes)

                combined_defect_counts = Counter()
                for defect, count in self.leftDashboard.defect_counts.items():
                    combined_defect_counts[defect] += count
                for defect, count in self.rightDashboard.defect_counts.items():
                    combined_defect_counts[defect] += count

                # Calculate total rejected count (sum of all defects from both dashboards)
                total_rejected = sum(combined_defect_counts.values())

                # Update the totalRejectedCount label with the combined total
                if hasattr(self, 'totalRejectedCount'):
                    self.totalRejectedCount.setText(str(total_rejected))

                    # Save the total rejected count to the database
                    try:
                        from azure_database import store_total_defect_count
                        user_id = self.get_current_user_id()
                        # Get the location_id if available (will be retrieved in the function if None)
                        location_id = None
                        # Store the total rejected count in the database
                        store_total_defect_count(
                            user_id=user_id, total_count=total_rejected, location_id=location_id)
                        print(
                            f"Saved total rejected count ({total_rejected}) to database for user {user_id}")
                    except Exception as db_error:
                        print(f"Error saving total rejected count to database: {db_error}")
                        import traceback
                        traceback.print_exc()

                if self.is_running:
                    for dashboard, prev_counts in [(self.leftDashboard, previous_left_counts),
                                                   (self.rightDashboard, previous_right_counts)]:
                        for defect_name, count in dashboard.defect_counts.items():
                            if count > prev_counts.get(defect_name, 0):
                                newly_detected_defects.add(defect_name)

                                current_threshold = dashboard.threshold_value / 100.0

                                increment = count - prev_counts.get(defect_name, 0)

                                if increment > 0:
                                    user_id = self.get_current_user_id()

                                    if user_id is None:
                                        self._load_user_id_from_configs()
                                        user_id = self.get_current_user_id()

                                    update_defect_count(
                                        defect_name,
                                        increment,
                                        int(current_threshold * 100),
                                        is_initialization=False,
                                        user_id=user_id
                                    )
                                    print(
                                        f"Real defect detected: {defect_name} with increment {increment}, threshold {int(current_threshold * 100)}%, user_id={user_id}")

            # Process results if they exist
            if len(results) > 0:
                result = results[0]  # Get first result

                # Extract detection data if boxes exist
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()

                    # Get current threshold from dashboard
                    current_threshold = self.leftDashboard.threshold_value / \
                        100.0 if hasattr(self, 'leftDashboard') else 0.5

                    # Process each detection
                    for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                        # Get class name
                        if class_id < len(self.defect_classes):
                            defect_name = self.defect_classes[class_id]
                        else:
                            defect_name = f"Class {class_id}"

                        # Skip this detection if it's not in the selected defects list
                        if not any(defect_name.lower() == selected.lower() for selected in self.selected_defects):
                            continue

                        # Skip detections with confidence below threshold
                        if conf < current_threshold:
                            continue

                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box)

                        # Draw bounding box (only for detections above threshold)
                        box_color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # Add defect label with confidence score
                        label_text = f"{defect_name}"
                        cv2.putText(frame, label_text, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Buffer the detection for batch saving
                        self.detection_buffer.append({
                            "defect_name": defect_name,
                            "confidence": conf,
                            "material": material_predictions.get(i, "Unknown"),
                            "user_id": self.get_current_user_id()  # Add user ID to the detection record
                        })

            # Display the annotated frame
            self.displayFrame(frame)
            self.last_annotated_frame = frame.copy()

        except Exception as e:
            print(f"Error processing results: {e}")
            traceback.print_exc()

            # If an error occurs, still display the original frame
            if self.frame_buffer:
                self.displayFrame(self.frame_buffer[-1])

        # Ensure the dashboard UI is updated after processing results
        if hasattr(self, 'dashboard'):
            self.dashboard.update_ui()
            if hasattr(self, 'rightDashboard'):
                self.rightDashboard.update_ui()

    def batch_save_reports(self):
        """Batch save reports to reduce frequent file I/O."""
        try:
            if self.detection_buffer:
                # Clear the buffer without saving to JSON
                self.detection_buffer = []
                print("Batch reports cleared successfully.")
        except Exception as e:
            print(f"Error during batch save: {e}")

    def displayFrame(self, frame):
        """Display the frame on the UI's camera feed label."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get dimensions
            frame_height, frame_width = rgb_frame.shape[:2]
            label_width = self.cameraFeedLabel.width()
            label_height = self.cameraFeedLabel.height()

            # Resize if needed
            if label_width > 0 and label_height > 0 and frame_width > 0 and frame_height > 0:
                # Keep aspect ratio
                aspect_ratio = frame_width / frame_height
                label_ratio = label_width / label_height

                if aspect_ratio > label_ratio:
                    new_width = label_width
                    new_height = int(label_width / aspect_ratio)
                else:
                    new_height = label_height
                    new_width = int(label_height * aspect_ratio)

                # Resize frame
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height),
                                       interpolation=cv2.INTER_AREA)

            # Convert to QImage and QPixmap
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Set pixmap to label
            self.cameraFeedLabel.setPixmap(pixmap)

        except Exception as e:
            print(f"Error displaying frame: {e}")
            traceback.print_exc()

    def closeEvent(self, event):
        """Handle window close event properly."""
        if self.is_running:
            self.toggleAnalysis()  # Stop analysis

        # Save session state when this UI is closed
        save_session_on_close("deploy")

        event.accept()

    def showEvent(self, event):
        """Called when the widget is shown."""
        super().showEvent(event)
        # Immediately maximize without delay
        self.showMaximized()
        print("Window shown and maximized")

        # Ensure only this title bar is visible
        self.ensure_single_title_bar()

        # Automatically start the camera when the tab is opened
        if not self.is_running:
            self.toggleAnalysis()

    def ensure_single_title_bar(self):
        """Ensures only this widget's title bar is visible."""
        parent = self.parent()
        if isinstance(parent, QStackedWidget):
            stacked_widget = parent
            # Hide all other title bars
            for i in range(stacked_widget.count()):
                widget = stacked_widget.widget(i)
                if widget != self and hasattr(widget, 'title_bar') and widget.title_bar:
                    widget.title_bar.hide()

                # Also check for nested UI components that might have title bars
                if widget != self and hasattr(widget, 'camera_feed_ui') and hasattr(widget.camera_feed_ui, 'title_bar'):
                    widget.camera_feed_ui.title_bar.hide()

    def isMaximized(self):
        """Check if window is maximized."""
        return bool(self.windowState() & Qt.WindowMaximized)

    def reset_counts(self):
        """Reset the counts to their initial state."""
        print("Resetting all counts")
        # Call the dashboard's reset_counts method
        if hasattr(self, 'dashboard') and self.dashboard:
            self.dashboard.reset_counts()
        # Reset any additional count tracking in this class if needed
        # Update the UI to reflect the reset counts
        QMessageBox.information(self, "Reset Complete", "All counts have been reset to zero.")

    def _load_user_id_from_configs(self):
        """
        Attempt to load the user ID from configuration files.
        This is a fallback method to ensure we always have a valid user ID.
        """
        try:
            # First try to load from session state
            session_state_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "session_state.json")
            if os.path.exists(session_state_path):
                with open(session_state_path, 'r') as f:
                    session_data = json.load(f)

                # Try multiple ways to get user ID from session state
                user_id = None
                if 'user_info' in session_data:
                    user_info = session_data['user_info']
                    user_id = user_info.get('user_id') or user_info.get('id')

                if not user_id:
                    user_id = session_data.get('current_user_id')

                if user_id:
                    self.current_user_id = user_id
                    self.user_info = session_data.get('user_info', {})
                    print(f"User ID loaded from session state: {user_id}")
                    return True

            # If that fails, try the defects config file
            if os.path.exists(DEFECTS_CONFIG_PATH):
                with open(DEFECTS_CONFIG_PATH, 'r') as f:
                    config_data = json.load(f)

                if 'user_info' in config_data and 'user_id' in config_data['user_info']:
                    user_id = config_data['user_info']['user_id']
                    self.current_user_id = user_id
                    self.user_info = config_data['user_info']
                    print(f"User ID loaded from defects config: {user_id}")
                    return True

            # If that fails, try the metal sheet location config
            metal_sheet_location_config_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "metal_sheet_location_config.json")

            if os.path.exists(metal_sheet_location_config_path):
                with open(metal_sheet_location_config_path, 'r') as f:
                    config_data = json.load(f)

                if 'user_info' in config_data and 'user_id' in config_data['user_info']:
                    user_id = config_data['user_info']['user_id']
                    self.current_user_id = user_id
                    self.user_info = config_data['user_info']
                    print(f"User ID loaded from metal sheet config: {user_id}")
                    return True

            # If we still don't have a user ID, use a hardcoded admin value as absolute last resort
            if self.get_current_user_id() is None:
                print("Warning: No user ID found in any config, using fallback admin ID")
                self.current_user_id = "A001"  # Fallback admin ID
                self.user_info = {"user_id": "A001", "id": "A001", "username": "admin"}
                return True

            return False
        except Exception as e:
            print(f"Error in user ID loading: {e}")
            traceback.print_exc()
            # Emergency fallback
            self.current_user_id = "A001"
            self.user_info = {"user_id": "A001", "id": "A001", "username": "admin"}
            return False


class CountingDashboard(QWidget):
    """
    A dashboard widget that displays counts of defects and material types.
    Uses object tracking to avoid double counting of the same object.
    """

    threshold_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)  # Increased minimum width
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa; /* Lighter background */
                border-radius: 8px;
                border: 1px solid #dee2e6; /* Lighter border */
            }
            QLabel {
                padding: 6px; /* Slightly more padding */
                color: #495057; /* Darker text color */
                font-size: 11pt; /* Slightly larger font */
            }
            QLabel#heading {
                background-color: #ff914d;
                color: white;
                font-weight: bold;
                font-size: 14pt; /* Larger heading font */
                border-top-left-radius: 7px; /* Match border radius */
                border-top-right-radius: 7px;
                padding: 10px;
                border-bottom: 1px solid #ff7730; /* Add subtle border */
            }
            QLabel#subheading {
                background-color: #e9ecef; /* Lighter subheading background */
                font-weight: bold;
                color: #343a40; /* Darker subheading text */
                padding: 8px;
                border-top: 1px solid #dee2e6;
                border-bottom: 1px solid #dee2e6;
                margin-top: 5px; /* Add some space above subheadings */
            }
            QScrollArea {
                border: none; /* Remove border from scroll area */
            }
            QWidget#defectListWidget { /* Style the container inside scroll area */
                 background-color: #ffffff;
                 border: none;
            }
        """)

        # Initialize counters
        self.defect_counts = Counter()
        self.material_counts = Counter()
        self.total_defects = 0
        self.threshold_value = 0  # Default threshold value (0%)
        self.pre_populated_defects = []  # Will be populated with selected defects
        self.selected_defects_shown = False

        # Enhanced tracking mechanism
        self.counted_tracks = set()  # Set of track IDs that have been counted
        self.spatial_memory = []     # List of bounding boxes we've already counted
        self.location_memory = []    # Remember center points of detected objects

        # Add a dictionary to store defect widgets for reuse
        self.defect_widgets = {}

        # Set up the UI
        self.init_ui()

    def init_ui(self):
        """Initialize the dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for full bleed heading
        layout.setSpacing(0)  # Remove spacing between sections initially

        # Dashboard title
        self.title_label = QLabel("Inspection Dashboard")
        self.title_label.setObjectName("heading")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Main content area with padding
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)  # Add padding inside
        content_layout.setSpacing(15)  # Add spacing between elements

        # --- Defect Breakdown Section ---
        self.defect_breakdown_label = QLabel("Defect Breakdown")
        self.defect_breakdown_label.setObjectName("subheading")
        content_layout.addWidget(self.defect_breakdown_label)

        # Scroll area for defect list - give it more space
        self.defect_scroll_area = QScrollArea()
        self.defect_scroll_area.setWidgetResizable(True)
        self.defect_scroll_area.setMinimumHeight(250)  # Increased height for more defects
        self.defect_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.defect_scroll_area.setStyleSheet("QScrollArea { border: none; }")

        self.defect_list_widget = QWidget()  # Container for defect items
        self.defect_list_widget.setObjectName("defectListWidget")
        self.defect_layout = QVBoxLayout(self.defect_list_widget)
        self.defect_layout.setContentsMargins(5, 5, 5, 5)
        self.defect_layout.setSpacing(10)  # Slightly increased spacing between defect items
        self.defect_layout.addStretch(1)  # Push items to the top

        self.defect_scroll_area.setWidget(self.defect_list_widget)
        content_layout.addWidget(self.defect_scroll_area)  # Add scroll area to main content

        # Large spacer to push content down
        content_layout.addSpacing(30)

        # Small spacer at the bottom
        content_layout.addSpacing(15)

        layout.addWidget(content_widget)  # Add content widget to main layout

        # Create a hidden label to maintain compatibility with other code that might reference total_rejected_count
        # This keeps the functionality without displaying the redundant UI element
        self.total_rejected_count = QLabel("0")
        self.total_rejected_count.setVisible(False)

    def save_report_to_json(self):
        """Save the current dashboard report and threshold value to a JSON file."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "defect_counts": dict(self.defect_counts),
                "material_counts": dict(self.material_counts),
                "total_defects": self.total_defects,
                "threshold_value": self.threshold_value
            }

            # Load existing reports if the file exists
            if os.path.exists(REPORTS_STORAGE_PATH):
                with open(REPORTS_STORAGE_PATH, 'r') as file:
                    try:
                        existing_reports = json.load(file)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        existing_reports = []
            else:
                existing_reports = []

            # Limit the number of reports to avoid file size issues
            if len(existing_reports) >= 1000:  # Keep the last 1000 reports
                existing_reports = existing_reports[-999:]

            # Append the new report
            existing_reports.append(report)

            # Save back to the file
            with open(REPORTS_STORAGE_PATH, 'w') as file:
                json.dump(existing_reports, file, indent=4)

            print("Dashboard report saved successfully.")
        except Exception as e:
            print(f"Error saving dashboard report: {e}")

    def update_counts(self, tracking_data, defect_classes):
        """Update the counts based on tracking data with improved duplicate detection"""
        new_objects_counted = False  # Flag to check if UI update is needed
        current_frame_bboxes = []    # Store current frame's bounding boxes

        # Convert defect classes to lowercase for case-insensitive comparison
        lowercase_defect_classes = [dc.lower() for dc in defect_classes]

        # Calculate confidence threshold from slider (0-100% to 0-1 scale)
        confidence_threshold = self.threshold_value / 100.0
        print(
            f"Using confidence threshold: {confidence_threshold:.2f} from threshold value: {self.threshold_value}%")

        # Debug: Print tracking data and defect classes
        print(f"Debug: Tracking data received: {tracking_data}")
        print(f"Debug: Defect classes: {defect_classes}")

        # Process each tracked object
        for track_id, track_info in tracking_data.items():
            # Skip if no bounding box information
            if 'bbox' not in track_info:
                continue

            # Skip if confidence is below the threshold
            confidence = track_info.get('confidence', 0.0)
            if confidence < confidence_threshold:
                print(
                    f"Skipping detection {track_id} with confidence {confidence:.2f} (below threshold {confidence_threshold:.2f})")
                continue

            current_bbox = track_info['bbox']
            current_frame_bboxes.append(current_bbox)

            # Skip if we've already counted this specific track_id
            if track_id in self.counted_tracks:
                print(f"Debug: Skipping already counted track_id {track_id}")
                continue

            # Check if this object is similar to any we've already counted
            already_counted = False
            for previous_bbox in self.spatial_memory:
                if self.is_same_object(current_bbox, previous_bbox):
                    already_counted = True
                    print(f"Debug: Skipping spatially similar object")
                    break

            if already_counted:
                # Mark as counted but don't increment counter again
                self.counted_tracks.add(track_id)
                continue

            # This is a new object - add to counted tracks
            self.counted_tracks.add(track_id)
            self.spatial_memory.append(current_bbox)  # Remember this bbox
            new_objects_counted = True  # Mark that a new object was counted

            # Store the center point for even more tracking stability
            center_x = (current_bbox[0] + current_bbox[2]) / 2
            center_y = (current_bbox[1] + current_bbox[3]) / 2
            self.location_memory.append((center_x, center_y))

            # Get defect name
            class_id = track_info.get('class_id', -1)
            if 0 <= class_id < len(lowercase_defect_classes):
                # Get the defect name but preserve the original capitalization from defect_classes
                lowercase_defect = lowercase_defect_classes[class_id]
                # Find original capitalization by matching lowercase version
                defect_name = next((d for d in defect_classes if d.lower()
                                   == lowercase_defect), lowercase_defect)

                print(f"Debug: Found defect {defect_name} with class_id {class_id}")

                # Make the first letter uppercase for display
                display_name = defect_name.capitalize()

                # Increment defect count
                self.defect_counts[display_name] += 1
                self.total_defects += 1  # Increment total only if it's a valid defect
                print(
                    f"Debug: Incremented count for {display_name}, new total: {self.total_defects}")
            else:
                # Handle unknown class_id if necessary
                defect_name = f"Unknown ({class_id})"
                print(f"Debug: Found unknown defect class_id {class_id}")
                # We don't count unknowns

        # Prune spatial memory (keep only the most recent objects)
        # This prevents the memory from growing indefinitely
        if len(self.spatial_memory) > 20:  # Keep last 20 objects
            self.spatial_memory = self.spatial_memory[-20:]

        if len(self.location_memory) > 20:  # Keep last 20 locations
            self.location_memory = self.location_memory[-20:]

        # Always update the UI after processing
        self.update_ui()

        # Save the report after updating counts
        self.save_report_to_json()

    def update_ui(self):
        """Update the UI with current counts"""
        # Update total rejected count
        self.total_rejected_count.setText(str(self.total_defects))

        # --- Update Defect Breakdown ---
        # First, create a set of all defects that should be displayed
        all_displayed_defects = set(self.pre_populated_defects)

        # Add all defects with counts to the set
        for defect_name in self.defect_counts:
            all_displayed_defects.add(defect_name)

        # No defects to display
        if not all_displayed_defects:
            # Check if the placeholder already exists
            if "placeholder" not in self.defect_widgets:
                # Create and store the placeholder widget
                placeholder_label = QLabel("No defects detected yet.")
                placeholder_label.setAlignment(Qt.AlignCenter)
                placeholder_label.setStyleSheet("color: #6c757d; font-style: italic;")
                self.defect_widgets["placeholder"] = placeholder_label
                self.defect_layout.insertWidget(0, placeholder_label,  # Insert at the top
                                                alignment=Qt.AlignTop)
        else:
            # Remove placeholder if it exists
            if "placeholder" in self.defect_widgets:
                self.defect_widgets["placeholder"].hide()
                self.defect_widgets.pop("placeholder")

            # Create a dictionary of defect counts with counts from defect_counts and 0 for pre-populated defects
            defect_display = {}
            for defect_name in all_displayed_defects:
                defect_display[defect_name] = self.defect_counts.get(defect_name, 0)

            # Sort by count (descending) and then by name (ascending)
            sorted_defects = sorted(defect_display.items(), key=lambda x: (-x[1], x[0]))

            # Keep track of widgets we've updated this cycle
            updated_widgets = set()

            # Display all defects in the sorted order
            for index, (defect_name, count) in enumerate(sorted_defects):
                widget_key = f"defect_{defect_name}"
                updated_widgets.add(widget_key)

                if widget_key in self.defect_widgets:
                    # Update existing widget
                    defect_layout = self.defect_widgets[widget_key]
                    defect_label = defect_layout.itemAt(0).widget()
                    count_label = defect_layout.itemAt(1).widget()

                    # Only update the count text, not the whole label
                    count_label.setText(str(count))

                    # Update styling based on count
                    if count > 0:
                        count_label.setStyleSheet(
                            "font-weight: bold; color: #dc3545; font-size: 11pt;")  # Red count
                    else:
                        count_label.setStyleSheet(
                            "font-weight: bold; color: #6c757d; font-size: 11pt;")  # Gray count for zero

                    # Move the layout to the correct position if needed
                    current_index = self.defect_layout.indexOf(defect_layout)
                    if current_index != index:
                        self.defect_layout.removeItem(defect_layout)
                        self.defect_layout.insertLayout(index, defect_layout)
                else:
                    # Create new widget
                    defect_item_layout = QHBoxLayout()
                    defect_label = QLabel(f"{defect_name}:")
                    defect_label.setStyleSheet("color: #495057;")  # Standard text color
                    defect_item_layout.addWidget(defect_label)

                    count_label = QLabel(str(count))
                    # Red for counts > 0, gray for zero counts
                    if count > 0:
                        count_label.setStyleSheet(
                            "font-weight: bold; color: #dc3545; font-size: 11pt;")  # Red count
                    else:
                        count_label.setStyleSheet(
                            "font-weight: bold; color: #6c757d; font-size: 11pt;")  # Gray count for zero

                    defect_item_layout.addWidget(count_label, 0, Qt.AlignRight)

                    # Store the layout in our widgets dictionary
                    self.defect_widgets[widget_key] = defect_item_layout

                    # Insert at the correct position
                    self.defect_layout.insertLayout(index, defect_item_layout)

            # Remove any widgets that are no longer needed
            widgets_to_remove = [
                key for key in self.defect_widgets if key not in updated_widgets and key != "placeholder"]
            for key in widgets_to_remove:
                layout = self.defect_widgets[key]
                # Remove and delete all widgets in the layout
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                # Remove the layout itself
                self.defect_layout.removeItem(layout)
                del self.defect_widgets[key]

        # Ensure the layout updates visually
        self.defect_list_widget.updateGeometry()
        self.updateGeometry()

        # Toggle scrollbar visibility based on content height
        defect_list_height = self.defect_list_widget.sizeHint().height()
        scroll_area_height = self.defect_scroll_area.height()

        # Only show scrollbars if content exceeds the visible area
        if defect_list_height > scroll_area_height:
            self.defect_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            self.defect_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Debug print for tracking
        print(
            f"Dashboard updated: {self.total_defects} total defects, {len(self.spatial_memory)} objects in memory")

    def update_threshold_value(self, value):
        """Update the threshold value, slider position, and emit the signal."""
        if self.threshold_value != value:  # Prevent recursive updates
            self.threshold_value = value
            self.threshold_slider.setValue(value)  # Update the slider's position
            self.threshold_value_label.setText(f"Threshold: {self.threshold_value}%")
            self.threshold_changed.emit(value)  # Emit signal only if value changes

    def is_same_object(self, bbox1, bbox2, iou_threshold=0.5, center_dist_threshold=50):
        """
        Check if two bounding boxes likely represent the same physical object
        using multiple criteria:
        1. IoU (Intersection over Union) - higher means more overlap
        2. Center point distance - lower means closer centers
        """
        # Calculate IoU
        iou = calculate_iou(bbox1, bbox2)

        # Calculate center points
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2

        center2_y = (bbox2[1] + bbox2[3]) / 2

        # Calculate distance between centers
        center_distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

        # Determine if it's the same object based on our criteria

        if iou > iou_threshold or center_distance < center_dist_threshold:
            return True

        return False

    def reset_counts(self):
        """Reset all counters"""
        self.defect_counts.clear()
        self.material_counts.clear()
        self.total_defects = 0
        self.counted_tracks.clear()
        self.spatial_memory.clear()  # Clear spatial memory too
        self.location_memory.clear()  # Clear location memory
        print("Dashboard counts reset.")                 # Add log
        self.update_ui()  # Update UI immediately after reset

    def save_defects_config(self):
        """Save the currently selected defects to the configuration file."""
        try:
            # Create config object
            config = {
                'selected_defects': self.selected_defects
            }

            # Add user info if available
            if hasattr(self, 'user_info') and isinstance(self.user_info, dict) and 'username' in self.user_info:
                config['user'] = self.user_info['username']

            # Save to file
            with open(DEFECTS_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Saved selected defects to config: {self.selected_defects}")
            return True
        except Exception as e:
            print(f"Error saving defect configuration: {e}")
            traceback.print_exc()
            return False

    def initialize_defects_in_database(self):
        """Initialize the defect classes in the Azure database with zero counts."""
        try:
            # Get current user ID - first try to get it from parent DeploymentUI instance
            user_id = None
            parent = self.parent()
            if parent and hasattr(parent, 'get_current_user_id'):
                user_id = parent.get_current_user_id()
                print(f"Got user ID from parent: {user_id}")

            # If not available from parent, load it from defects_config.json
            if not user_id:
                defects_config_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "defects_config.json")

                if os.path.exists(defects_config_path):
                    try:
                        with open(defects_config_path, 'r') as f:
                            config = json.load(f)
                            if "user" in config and config["user"]:
                                username = config["user"]
                                from azure_database import get_user_id_by_username
                                user_id = get_user_id_by_username(username)
                                if user_id:
                                    print(
                                        f"Using user ID from defects config: {user_id} (username: {username})")
                    except Exception as e:
                        print(f"Error reading defects config: {e}")

            # Then try to get it from metal_sheet_location_config.json
            if not user_id:
                metal_sheet_location_config_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "metal_sheet_location_config.json")

                if os.path.exists(metal_sheet_location_config_path):
                    try:
                        with open(metal_sheet_location_config_path, 'r') as f:
                            location_config = json.load(f)
                            if "user" in location_config and location_config["user"]:
                                username = location_config["user"]
                                from azure_database import get_user_id_by_username
                                user_id = get_user_id_by_username(username)
                                if user_id:
                                    print(
                                        f"Using user ID from location config: {user_id} (username: {username})")
                    except Exception as e:
                        print(f"Error reading location config: {e}")

            # If still no user ID, use a hardcoded admin value as fallback
            if not user_id:
                print("Warning: No user ID available, using admin fallback")
                user_id = "A001"  # Admin fallback

            print(f"Initializing defects in Azure database for user: {user_id}")

            # Get current threshold value
            threshold = self.threshold_value if hasattr(self, 'threshold_value') else 50

            # Import the database update function
            from azure_database import update_defect_count, get_latest_location_id

            # Get latest location_id for this user
            location_id = get_latest_location_id(user_id)
            print(f"Retrieved location_id={location_id} for user_id={user_id}")

            # Initialize each defect in the database with zero count
            for defect_name in self.pre_populated_defects:
                try:
                    # Call the database update function with initialization flag and location_id
                    update_defect_count(
                        defect_name,
                        0,  # Initial count of 0
                        threshold,
                        is_initialization=True,  # Mark as initialization to handle differently in database
                        user_id=user_id,
                        location_id=location_id  # Pass the location_id explicitly
                    )
                    print(
                        f"Initialized defect in database: {defect_name} for user {user_id} with location_id {location_id}")
                except Exception as e:
                    print(f"Error initializing defect {defect_name} in database: {e}")

            return True

        except Exception as e:
            print(f"Error initializing defects in database: {e}")
            traceback.print_exc()
            return False


class ConveyorVisualization(QWidget):
    """Widget that visualizes objects on a conveyor belt"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.positions = {}  # Dictionary of objects and their positions
        self.setStyleSheet(
            "background-color: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 5px;")

        # Setup animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(50)  # 50ms update interval

    def update_positions(self, positions_dict):
        """Update the positions of tracked objects"""
        self.positions = positions_dict
        self.update()

    def animate(self):
        """Animate objects moving along the conveyor"""
        if not self.positions:
            return

        # Move objects according to conveyor direction
        for track_id in list(self.positions.keys()):
            # Move object left by 2 pixels per frame
            pos = self.positions[track_id]['position']
            if isinstance(pos, (int, float)):
                self.positions[track_id]['position'] = pos - 2

                # Remove objects that have moved off the conveyor
                if pos < -50:
                    del self.positions[track_id]

        self.update()

    def paintEvent(self, event):
        """Paint the conveyor and objects"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Paint conveyor
        conveyor_rect = QRectF(0, 0, self.width(), self.height())

        # Draw conveyor belt background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(100, 100, 100))
        gradient.setColorAt(1, QColor(50, 50, 50))
        painter.fillRect(conveyor_rect, QBrush(gradient))

        # Draw rollers
        roller_spacing = 20
        roller_width = 4

        for x in range(0, self.width(), roller_spacing):
            painter.setPen(QPen(QColor(180, 180, 180), 1))
            painter.drawLine(x, 0, x, self.height())

        # Draw objects on conveyor
        for track_id, info in self.positions.items():
            # Get position
            pos = info.get('position')
            if not isinstance(pos, (int, float)):
                continue

            # Normalize position to widget width
            normalized_pos = int(pos) % (self.width() + 100)

            # Draw object
            painter.setBrush(QBrush(QColor(120, 120, 120)))
            painter.setPen(QPen(Qt.black, 1))

            # Check if object has defect
            has_defect = info.get('class_id', -1) >= 0

            if has_defect:
                # Draw with red border for defects
                painter.setPen(QPen(QColor(255, 0, 0), 2))

            # Draw the object as a rectangle
            obj_width = 30
            obj_height = self.height() - 20
            obj_y = 10

            # Draw a rectangle representing the material
            painter.drawRect(normalized_pos, obj_y, obj_width, obj_height)


if __name__ == '__main__':
    # Initialize Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("DeploymentUI")
    app.setOrganizationName("SakarVision")

    # Create window
    window = DeploymentUI()

    # Set maximized state first, then show
    window.setWindowState(Qt.WindowMaximized)
    window.show()

    sys.exit(app.exec_())
