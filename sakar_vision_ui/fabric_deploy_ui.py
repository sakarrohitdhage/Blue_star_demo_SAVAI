#!/usr/bin/env python3
"""
fabric_deploy_ui.py - Fabric Inspection Deployment UI

This module provides the interface for fabric defect detection using a camera feed.
"""

import json
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
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
from PyQt5.QtWidgets import QLineEdit
import torch

# Try to import YOLO, but don't fail fatally if it's not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: Ultralytics YOLO not available. Model loading will be disabled.")
    YOLO_AVAILABLE = False

from utils import set_window_icon

# Make sure CUDA is properly configured if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Fix Qt plugin issue - explicitly set plugin path
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
if 'cv2/qt/plugins' in os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', ''):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'

# Fix thread issue by ensuring proper thread initialization
os.environ['QT_THREAD_PRIORITY'] = 'normal'

# Path to location configuration file
LOCATION_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "location_config.json")


class CustomTitleBar(QFrame):
    def __init__(self, parent=None, demo_button=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet("""
            background-color: #ff914d; 
            border: none;
        """)

        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        # Logo image label setup
        self.logo_label = QLabel()
        logo_path = "Screenshot from 2025-03-13 12-58-07.jpeg"  # Update with actual logo path
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_pixmap = logo_pixmap.scaledToHeight(45, Qt.SmoothTransformation)
            self.logo_label.setPixmap(logo_pixmap)
        else:
            # Fallback if logo not found
            self.logo_label.setText("SAKAR")
            self.logo_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")

        # Title label
        self.title_label = QLabel("FABRIC INSPECTION")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: white;
            font-weight: bold;
            font-size: 28px;
            margin-left: 0px;
            margin-right: 195px;
        """)

        # Add logo and title to layout
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch(0)
        logo_layout.setSpacing(2)

        layout.addLayout(logo_layout)
        layout.addWidget(self.title_label, 1)
        layout.addStretch(0)

        # Add any additional button if provided
        if demo_button:
            layout.addWidget(demo_button)

        self.setLayout(layout)


class ModelInferenceThread(QThread):
    """Thread for running model inference without blocking the UI."""
    inference_complete = pyqtSignal(object, float)  # Results, inference time

    def __init__(self, yolo_model, frame, defect_classes):
        super().__init__()
        self.yolo_model = yolo_model
        self.frame = frame.copy()  # Create a copy to avoid race conditions
        self.defect_classes = defect_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            start_time = time.time()

            # Run YOLO model on the frame
            results = self.yolo_model(self.frame, conf=0.10, verbose=False)

            inference_time = time.time() - start_time
            self.inference_complete.emit(results, inference_time)

        except Exception as e:
            print(f"Error in inference thread: {e}")
            traceback.print_exc()
            self.inference_complete.emit(None, 0)


class FabricDeploymentUI(QWidget):
    def __init__(self, inspection_data=None):
        super().__init__()

        # Set window properties before UI initialization
        self.setWindowTitle("Fabric Inspection - Live Analysis")
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(1024, 768)
        self.setGeometry(100, 100, 1000, 700)
        set_window_icon(self)

        # Store inspection data
        self.inspection_data = inspection_data or {
            "organization": "Default Organization",
            "location": "Default Location",
            "user": {"username": "User", "full_name": "Default User"}
        }

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
            #backButton {
                background-color: #f44336;
                color: white;
            }
            #backButton:hover {
                background-color: #e53935;
            }
        """)

        # Print debug information
        print(f"FabricDeploymentUI initialized for organization: {self.inspection_data.get('organization')}, "
              f"location: {self.inspection_data.get('location')}")

        # Initialize models directly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = None
        self.yolo_model_path = "best.pt"  # Default model path

        # Camera variables
        self.camera_capture = None
        self.is_running = False
        self.detection_thread = None
        self.confidence_threshold = 0

        # Fabric defect classes - this is an editable list
        self.defect_classes = ["stain", "hole", "thread", "slub", "tear", "misalignment"]

        # Inference queue management
        self.max_concurrent_inferences = 1
        self.inference_queue = []

        # Initialize UI and load configuration
        self.initUI()
        self.load_config()

        # Frame buffer and FPS variables
        self.frame_buffer_size = 2
        self.frame_buffer = []
        self.fps_frames = 0
        self.fps_start_time = 0
        self.last_fps_update = 0
        self.processing_width = 640
        self.processing_height = 480

    def load_config(self):
        """Load location and organization configuration."""
        try:
            if os.path.exists(LOCATION_CONFIG_PATH):
                with open(LOCATION_CONFIG_PATH, 'r') as f:
                    config = json.load(f)

                # Update inspection data if available
                if "organization" in config:
                    self.inspection_data["organization"] = config["organization"]
                if "location" in config:
                    self.inspection_data["location"] = config["location"]

                # Update UI to reflect the loaded configuration
                if hasattr(self, 'orgLocationLabel'):
                    self.orgLocationLabel.setText(
                        f"Organization: {self.inspection_data['organization']} | "
                        f"Location: {self.inspection_data['location']}"
                    )
        except Exception as e:
            print(f"Error loading location configuration: {e}")

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

        # Organization and location info
        self.orgLocationLabel = QLabel(
            f"Organization: {self.inspection_data['organization']} | "
            f"Location: {self.inspection_data['location']}"
        )
        self.orgLocationLabel.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333333;
            background-color: #f8f9fa;
            padding: 5px 10px;
            border-radius: 3px;
        """)
        contentLayout.addWidget(self.orgLocationLabel)

        # Camera feed group
        cameraGroup = QGroupBox("Live Fabric Defect Detection")
        cameraGroup.setStyleSheet(
            "QGroupBox { font-size: 16px; font-weight: bold; color: #333333; }")
        cameraLayout = QVBoxLayout()

        # Main camera feed display
        self.cameraFeedLabel = QLabel()
        self.cameraFeedLabel.setAlignment(Qt.AlignCenter)
        self.cameraFeedLabel.setMinimumSize(800, 600)
        self.cameraFeedLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cameraFeedLabel.setStyleSheet(
            "background-color: #000; color: white; border: 1px solid #555555;")
        self.cameraFeedLabel.setText("Camera feed will appear here when started")
        cameraLayout.addWidget(self.cameraFeedLabel)

        # Add more margin at the bottom of the camera feed
        cameraLayout.addSpacing(10)

        # Create a container for all controls
        controlsContainer = QWidget()
        controlsContainer.setStyleSheet(
            "background-color: #FFFFFF; border-radius: 5px; padding: 10px;")
        controlsContainerLayout = QVBoxLayout(controlsContainer)
        controlsContainerLayout.setSpacing(10)
        controlsContainerLayout.setContentsMargins(10, 10, 10, 10)

        # Defect classes input - this is a new section for fabric inspection
        defectClassesLayout = QHBoxLayout()
        defectClassesLayout.setSpacing(15)

        defectClassesLabel = QLabel("Defect Classes:")
        defectClassesLabel.setStyleSheet("font-weight: bold;")
        defectClassesLabel.setFixedWidth(120)
        defectClassesLayout.addWidget(defectClassesLabel)

        self.defectClassesInput = QLineEdit()
        self.defectClassesInput.setText(", ".join(self.defect_classes))
        self.defectClassesInput.setPlaceholderText("Enter defect classes separated by commas")
        self.defectClassesInput.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        defectClassesLayout.addWidget(self.defectClassesInput, 1)

        self.updateClassesButton = QPushButton("Update Classes")
        self.updateClassesButton.setFixedWidth(150)
        self.updateClassesButton.clicked.connect(self.updateDefectClasses)
        self.updateClassesButton.setStyleSheet("background-color: #4a90e2; color: white;")
        defectClassesLayout.addWidget(self.updateClassesButton)

        # Add defect classes layout to controls container
        controlsContainerLayout.addLayout(defectClassesLayout)

        # Model selection layout - with improved spacing
        modelSelectionLayout = QHBoxLayout()
        modelSelectionLayout.setSpacing(15)

        modelLabel = QLabel("Model:")
        modelLabel.setStyleSheet("font-weight: bold;")
        modelLabel.setFixedWidth(60)
        modelSelectionLayout.addWidget(modelLabel)

        self.modelPathInput = QLineEdit()
        self.modelPathInput.setText(self.yolo_model_path)
        self.modelPathInput.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        modelSelectionLayout.addWidget(self.modelPathInput, 1)

        # Browse button
        self.browseModelButton = QPushButton("Browse...")
        self.browseModelButton.setFixedWidth(100)
        self.browseModelButton.clicked.connect(self.browseModelFile)
        self.browseModelButton.setStyleSheet("background-color: #4a90e2; color: white;")
        modelSelectionLayout.addWidget(self.browseModelButton)

        # Add model selection layout to controls container
        controlsContainerLayout.addLayout(modelSelectionLayout)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #d0d0d0;")
        controlsContainerLayout.addWidget(separator)

        # Action buttons layout with better spacing
        actionButtonsLayout = QHBoxLayout()
        actionButtonsLayout.setSpacing(20)

        # Start/Stop button with fixed width
        self.toggleButton = QPushButton("Start Analysis")
        self.toggleButton.setFixedWidth(150)
        self.toggleButton.clicked.connect(self.toggleAnalysis)
        self.toggleButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        actionButtonsLayout.addWidget(self.toggleButton)

        # Status label with fixed minimum width
        self.statusLabel = QLabel("Status: Ready")
        self.statusLabel.setMinimumWidth(200)
        self.statusLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.statusLabel.setStyleSheet("font-weight: bold; color: #333333;")
        actionButtonsLayout.addWidget(self.statusLabel, 1)

        # Add action buttons to controls container
        controlsContainerLayout.addLayout(actionButtonsLayout)

        # Add the controls container to the camera layout
        cameraLayout.addWidget(controlsContainer)

        cameraGroup.setLayout(cameraLayout)
        contentLayout.addWidget(cameraGroup)

        # Status bar
        statusBarFrame = QFrame()
        statusBarFrame.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px;")
        statusBarLayout = QHBoxLayout(statusBarFrame)
        statusBarLayout.setContentsMargins(10, 5, 10, 5)

        self.inferenceTimeLabel = QLabel("Inference time: --")
        self.fpsLabel = QLabel("FPS: --")
        self.defectsDetectedLabel = QLabel("Defects detected: --")
        statusBarLayout.addWidget(self.inferenceTimeLabel)
        statusBarLayout.addWidget(self.fpsLabel)
        statusBarLayout.addWidget(self.defectsDetectedLabel)
        statusBarLayout.addStretch(1)

        contentLayout.addWidget(statusBarFrame)

        # Add content widget to main layout
        mainLayout.addWidget(contentWidget)

        # Camera timer
        self.camera_feed_timer = QTimer(self)
        self.camera_feed_timer.timeout.connect(self.update_camera_feed)

    def updateDefectClasses(self):
        """Update the defect classes based on user input."""
        input_text = self.defectClassesInput.text().strip()
        if input_text:
            # Split by comma and strip whitespace
            new_classes = [cls.strip() for cls in input_text.split(',') if cls.strip()]
            if new_classes:
                self.defect_classes = new_classes
                QMessageBox.information(self, "Success", "Defect classes updated successfully.")
                print(f"Updated defect classes: {self.defect_classes}")
            else:
                QMessageBox.warning(self, "Invalid Input",
                                    "Please enter at least one defect class.")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter at least one defect class.")

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
            # Update model path
            self.yolo_model_path = file_path
            self.modelPathInput.setText(file_path)
            self.statusLabel.setText(f"Status: Model selected - {os.path.basename(file_path)}")

    def toggleAnalysis(self):
        """Toggle between starting and stopping the analysis."""
        if not self.is_running:
            # Check if YOLO is available
            if not YOLO_AVAILABLE:
                QMessageBox.warning(self, "Module Error",
                                    "The Ultralytics YOLO module is not available. "
                                    "Please install it using 'pip install ultralytics'.")
                return

            # Update UI first to show we're processing
            self.statusLabel.setText("Status: Initializing analysis...")
            QApplication.processEvents()  # Update UI immediately

            # Start analysis
            self.yolo_model_path = self.modelPathInput.text().strip()
            if not os.path.exists(self.yolo_model_path):
                QMessageBox.warning(self, "Model Error",
                                    f"Model file not found: {self.yolo_model_path}")
                self.statusLabel.setText("Status: Error - Model file not found")
                return

            # Load the model
            try:
                # Load YOLO model for defect detection
                self.statusLabel.setText("Status: Loading model...")
                QApplication.processEvents()

                try:
                    # Load YOLO model
                    self.yolo_model = YOLO(self.yolo_model_path).to(self.device)
                    print(f"Model loaded: {self.yolo_model_path}")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    traceback.print_exc()
                    raise

                # Initialize camera
                self.statusLabel.setText("Status: Initializing camera...")
                QApplication.processEvents()

                # Try to open camera
                self.camera_capture = cv2.VideoCapture(0)
                if not self.camera_capture.isOpened():
                    # Try other indices if 0 didn't work
                    for cam_index in range(1, 4):
                        self.camera_capture = cv2.VideoCapture(cam_index)
                        if self.camera_capture.isOpened():
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
                self.toggleButton.setText("Stop Analysis")
                self.toggleButton.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        font-weight: bold;
                        padding: 8px 16px;
                    }
                    QPushButton:hover {
                        background-color: #e53935;
                    }
                """)
                self.statusLabel.setText("Status: Running")
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
            self.toggleButton.setText("Start Analysis")
            self.toggleButton.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.statusLabel.setText("Status: Stopped")

            if self.camera_feed_timer.isActive():
                self.camera_feed_timer.stop()

            if self.camera_capture and self.camera_capture.isOpened():
                self.camera_capture.release()
                self.camera_capture = None

            self.inferenceTimeLabel.setText("Inference time: --")
            self.fpsLabel.setText("FPS: --")
            self.defectsDetectedLabel.setText("Defects detected: --")
            self.yolo_model = None

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
                        self.defect_classes
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

    def processResults(self, results, inference_time):
        """Process results from model inference."""
        # Update inference time display
        self.inferenceTimeLabel.setText(f"Inference time: {inference_time*1000:.1f} ms")

        try:
            # Only process if we have frames in buffer and valid results
            if not self.frame_buffer or results is None:
                return

            # Get the latest frame from buffer for annotation
            frame = self.frame_buffer[-1].copy()
            defects_detected = 0

            # Process results if they exist
            if len(results) > 0:
                result = results[0]  # Get first result

                # Extract detection data if boxes exist
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()

                    # Process each detection
                    for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                        # Get class name
                        if class_id < len(self.defect_classes):
                            defect_name = self.defect_classes[class_id]
                        else:
                            defect_name = f"Class {class_id}"

                        # Count this as a detection
                        defects_detected += 1

                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box)

                        # Draw bounding box - use blue color for fabric defects
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # Add defect label with confidence
                        label_text = f"{defect_name} ({conf:.2f})"
                        cv2.putText(frame, label_text, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Update defects detected counter
            self.defectsDetectedLabel.setText(f"Defects detected: {defects_detected}")

            # Display the annotated frame
            self.displayFrame(frame)
            self.last_annotated_frame = frame.copy()

        except Exception as e:
            print(f"Error processing results: {e}")
            traceback.print_exc()

            # If an error occurs, still display the original frame
            if self.frame_buffer:
                self.displayFrame(self.frame_buffer[-1])

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

    def goBackToLocationSelection(self):
        """Return to the location selection interface."""
        # Stop any running analysis
        if self.is_running:
            self.toggleAnalysis()

        try:
            # Go back to location selection
            print("Importing LocationSelectionUI...")
            try:
                from location_selection_ui import LocationSelectionUI
            except ImportError as e:
                print(f"Error importing LocationSelectionUI: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Import Error",
                                     f"Failed to load the location selection module: {str(e)}\n"
                                     "Please check that all required files are in place.")
                return

            print("Creating LocationSelectionUI instance...")
            location_ui = LocationSelectionUI(
                user_info=self.inspection_data.get("user"),
                organization=self.inspection_data.get("organization")
            )

            print("Showing LocationSelectionUI...")
            location_ui.show()
            location_ui.showMaximized()

            # Use a delayed hide to ensure the new window appears first
            QTimer.singleShot(200, self.hide)
        except Exception as e:
            print(f"Error in goBackToLocationSelection: {e}")
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Could not navigate to location selection: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event properly."""
        if self.is_running:
            self.toggleAnalysis()  # Stop analysis
        event.accept()

    def showEvent(self, event):
        """Called when the widget is shown."""
        super().showEvent(event)
        # Immediately maximize without delay
        self.showMaximized()
        print("FabricDeploymentUI window shown and maximized")


if __name__ == '__main__':
    # Print current directory and check for required files
    print(f"Current directory: {os.getcwd()}")

    # Initialize Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("FabricDeploymentUI")
    app.setOrganizationName("SakarVision")

    # Sample inspection data
    inspection_data = {
        "organization": "Test Organization",
        "location": "Pune",
        "user": {
            "username": "sakarrobotics",
            "full_name": "Sakar Robotics"
        }
    }

    # Create window
    window = FabricDeploymentUI(inspection_data)

    # Set maximized state first, then show
    window.setWindowState(Qt.WindowMaximized)
    window.show()

    sys.exit(app.exec_())
