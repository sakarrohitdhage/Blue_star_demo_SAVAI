"""
SAKAR VISION AI - Camera Feed Cloud UI Module

OVERVIEW:
This module implements a comprehensive computer vision interface for the Sakar Vision AI platform, serving as the main hub 
for real-time camera operations, image capture, and data management workflows. The application provides a unified interface 
that integrates live camera feeds with automated image capture, cloud storage synchronization, and seamless navigation 
between multiple AI-powered annotation tools (manual and automatic annotation, demo feeds, image classification, and model deployment).

KEY FUNCTIONALITY:
The system manages continuous camera capture sessions with configurable frame rates and duration settings, automatically 
distributing captured images between manual and automatic annotation workflows based on user-defined percentages. All captured 
images are stored locally with intelligent naming conventions (including user identification and timestamps) and are batch-uploaded 
to Azure cloud storage with offline resilience. The interface features a sophisticated user management system that tracks user 
sessions across multiple JSON configuration files, internet connectivity monitoring with visual status indicators, and a 
QStackedWidget-based navigation system that efficiently manages camera resources when switching between different UI modules 
(manual annotation, auto annotation, demo feeds, image classification, and deployment tools).

TECHNICAL ARCHITECTURE:
Built using PyQt5 with OpenCV for camera operations and integrated Azure storage capabilities, the application employs multiple 
QTimer instances for real-time camera feed updates, capture scheduling, session management, and status monitoring. The modular 
design includes embedded UI components (ImageAnnotationTool, DemoFeedUI, AutoAnnotateUI, ImageClassiUI, DeploymentUI) that are 
managed through a central stacked widget system, ensuring proper resource allocation and preventing camera conflicts between modules.
"""

from datetime import datetime
from datetime import timedelta
import json
import os
import random
import sys
import time
import logging
import socket
import requests

# Fix the logging import to use the correct module
from utils.logging import get_logger

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QSpinBox
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGroupBox
from ultralytics import YOLO

# Import the Azure storage module with offline resilience
import azure_storage

from auto_annotation import AutoAnnotateUI
from demo_feed_ui import DemoFeedUI
from deploy_ui import DeploymentUI
from image_classi_ui import ImageClassiUI
from manual_annotation_ui import ImageAnnotationTool
from utils import set_window_icon
# Import UI event logger
from utils.logging.ui_event_logger import UIEventTracker

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'

# Get a logger instance
logger = get_logger("camera_feed_cloud_ui")

# Default interval for captures (5 minutes = 300 seconds)
DEFAULT_UPLOAD_INTERVAL = 300


class CustomTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-bottom: 2px solid #e9ecef;
            }
        """)

        # Create main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(20)

        # Left section - Logo and SakarRobotics
        left_section = QHBoxLayout()
        left_section.setSpacing(15)

        # Logo
        self.logo_label = QLabel()
        logo_path = "logo.jpeg"
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_pixmap = logo_pixmap.scaledToHeight(40, Qt.SmoothTransformation)
            self.logo_label.setPixmap(logo_pixmap)

        # SakarRobotics text
        sakar_label = QLabel("")
        sakar_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 18px;
                font-weight: bold;
            }
        """)

        left_section.addWidget(self.logo_label)
        left_section.addWidget(sakar_label)
        left_section.addStretch()

        # Center section - Camera Feed title
        center_section = QHBoxLayout()
        self.title_label = QLabel("Camera Feed")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        center_section.addWidget(self.title_label)

        # Right section - System Status
        right_section = QHBoxLayout()
        right_section.setSpacing(15)

        # System Status indicators
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
                color: #333333;
                font-size: 14px;
                font-weight: 500;
            }
        """)

        right_section.addStretch()
        right_section.addWidget(self.status_indicator)
        right_section.addWidget(self.status_label)

        # Add all sections to main layout
        layout.addLayout(left_section, 1)
        layout.addLayout(center_section, 1)
        layout.addLayout(right_section, 1)

        # Timer to check internet connectivity
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


class CameraFeedUI(QWidget):
    """
    Main UI class for the Camera Feed application, managing camera capture,
    manual annotation, and demo feed functionalities.
    """

    def __init__(self, user_info=None, parent=None):
        """
        Initializes the CameraFeedUI, setting up UI components, timers,
        and embedding other UI modules.

        Args:
            user_info (dict): User information from login/defect selection (optional)
            parent: Parent widget
        """
        super().__init__(parent)

        print("CameraFeedUI initializing...")

        # Initialize user information storage first
        self.current_user_id = "A001"  # Safe default
        self.current_username = "admin"  # Safe default
        self.current_user_full_name = "Administrator"  # Safe default

        # Try to fetch user information from login system
        try:
            self._fetch_user_from_login(user_info)
            print(f"User loaded: {self.current_username} (ID: {self.current_user_id})")
        except Exception as e:
            print(f"Warning: Could not fetch user info: {e}")
            print("Using default admin user")

        # Initialize UI components
        self.camera_feed_label = QLabel(self)
        self.camera_feed_timer = QTimer(self)  # Timer for displaying camera feed
        self.camera_capture = None
        self.selected_camera_index = self.get_selected_camera_index()  # Get saved camera index
        self.capture_active = False
        self.capture_progress_label = QLabel("Ready", self)
        self.capture_storage_folder = None
        self.capture_folder_a = None  # Path for folder A
        self.capture_folder_b = None  # Path for folder B
        self.capture_count = 0
        self.capture_progress_bar = QProgressBar(self)
        self.progress_percent_label = QLabel("0%", self)  # Initialize here
        self.btn_capture_images_setup = QPushButton("Start Continuous Capture", self)
        self.btn_manual_annotation = QPushButton("Manual Annotation", self)
        set_window_icon(self)

        # Upload interval settings - now it's capture duration
        self.upload_interval_spinbox = QSpinBox(self)
        self.upload_interval_spinbox_label = QLabel("Capture Duration (seconds):", self)
        self.upload_interval = DEFAULT_UPLOAD_INTERVAL  # Default capture duration

        # Upload interval timer - now it's capture duration timer
        self.capture_duration_timer = QTimer(self)  # Timer for capture duration
        self.capture_duration_timer.timeout.connect(self.stop_and_upload_all)

        # Capture rate settings (frames per second)
        self.capture_rate_spinbox = QSpinBox(self)
        self.capture_rate_spinbox.setRange(1, 30)  # 1-30 frames per second
        self.capture_rate_spinbox.setValue(1)  # Default 1 fps
        self.capture_rate_spinbox_label = QLabel("Capture Rate (fps):", self)
        self.capture_rate = 1  # Default 1 frame per second

        # Capture timer
        self.capture_timer = QTimer(self)  # Timer for regular captures
        self.capture_timer.timeout.connect(self.capture_frame)

        # Status update timer for smooth progress bar updates
        self.status_update_timer = QTimer(self)
        self.status_update_timer.timeout.connect(self.update_status_display)
        self.status_update_timer.start(1000)  # Update every second

        # New buttons for pause and resume
        self.btn_pause_feed = QPushButton("Pause Camera", self)
        self.btn_resume_feed = QPushButton("Resume Camera", self)
        self.feed_paused = False  # Flag to track pause state

        # Lists to store captured images for batch upload
        self.pending_uploads = []  # List of local image paths

        # Upload batch tracking
        self.last_upload_time = None
        self.capture_end_time = None  # When capture session will end
        self.remaining_session_time = None  # Track remaining time when paused
        self.upload_count = 0
        self.current_batch_count = 0
        self.capture_session_active = False  # Track if we're in a capture session

        # New UI elements for data distribution slider
        self.data_distribution_slider_label = QLabel("Manual Input %", self)
        self.data_distribution_slider = QSlider(Qt.Horizontal, self)
        self.data_distribution_label = QLabel("50%", self)  # Label to display slider value
        self.data_distribution_percentage = 50  # Default distribution percentage

        # Configure slider
        self.data_distribution_slider.setRange(0, 100)
        self.data_distribution_slider.setValue(50)
        self.data_distribution_slider.setTickInterval(10)
        self.data_distribution_slider.setTickPosition(QSlider.TicksBelow)

        # Connect slider signal
        self.data_distribution_slider.valueChanged.connect(self.update_data_distribution_label)

        # Connect interval spinbox
        self.upload_interval_spinbox.valueChanged.connect(self.update_upload_interval)

        # Connect capture rate spinbox
        self.capture_rate_spinbox.valueChanged.connect(self.update_capture_rate)

        # Connect button signals to slots
        self.btn_capture_images_setup.clicked.connect(self.toggle_continuous_capture)
        self.btn_pause_feed.clicked.connect(self.pause_camera_feed)
        self.btn_resume_feed.clicked.connect(self.resume_camera_feed)
        self.btn_manual_annotation.clicked.connect(self.show_manual_annotation_ui)

        # Embed the manual annotation tool
        self.manual_annotation_tool = ImageAnnotationTool()
        print(f"Manual annotation tool object created: {self.manual_annotation_tool}")
        self.manual_annotation_tool.hide()

        # Embed the Demo Feed UI
        self.demo_feed_ui = DemoFeedUI(self)

        # Embed the Image Classification UI
        self.image_classi_ui = ImageClassiUI(self)

        # Embed the Auto Annotation UI
        self.auto_annoate_ui = AutoAnnotateUI()  # Instantiate AutoAnnotateUI

        # Embed the Deployment UI
        self.deployment_ui = DeploymentUI()

        # Connect the deploy button in Auto Annotation UI to show the deployment UI
        self.auto_annoate_ui.deployModelButton.clicked.connect(self.show_deployment_ui)

        # Use QStackedWidget to manage different UI views
        self.stacked_widget = QStackedWidget(self)
        self.camera_feed_widget = QWidget()
        self.annotation_tool_widget = self.manual_annotation_tool
        self.demo_feed_widget = self.demo_feed_ui  # Add DemoFeedUI to stacked widget
        self.image_classi_widget = self.image_classi_ui  # Add ImageClassiUI to stacked widget

        # Create custom title bar
        self.title_bar = CustomTitleBar(self)

        # Set up the User Interface
        self.init_ui()
        self.setup_camera_feed_ui()
        self.setup_stacked_widget()

        # Apply global stylesheet with isolated selectors
        self.setStyleSheet("""
            CameraFeedUI {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #ffffff;
            }
            CameraFeedUI QPushButton {
                background-color: #62FF4D;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                min-height: 20px;
            }
            CameraFeedUI QPushButton:hover {
                background-color: #BBFF4D ;
            }
            CameraFeedUI QPushButton:pressed {
                background-color: #7a6449;
            }
            CameraFeedUI QPushButton#resumeButton {
                background-color: #FFEA4D;
                color: #333333;
            }
            CameraFeedUI QPushButton#resumeButton:hover {
                background-color: #FFEA4D;
            }
            CameraFeedUI QProgressBar {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                text-align: center;
                height: 25px;
                background-color: #f8f9fa;
                font-size: 14px;
                font-weight: 600;
            }
            CameraFeedUI QProgressBar::chunk {
                background-color: #ff914d;
                border-radius: 6px;
                margin: 1px;
            }
            CameraFeedUI QLabel {
                color: #333333;
                font-size: 14px;
            }
            CameraFeedUI QSpinBox {
                border: 1px solid #e9ecef;
                border-radius: 15px;
                padding: 12px 20px;
                font-size: 14px;
                min-width: 100px;
                background-color: #f8f9fa;
                font-weight: 400;
                color: #333333;
            }
            CameraFeedUI QSpinBox:focus {
                border-color: #8b7355;
                background-color: white;
                outline: none;
            }
            CameraFeedUI QSpinBox::up-button {
                width: 0px;
                height: 0px;
                border: none;
            }
            CameraFeedUI QSpinBox::down-button {
                width: 0px;
                height: 0px;
                border: none;
            }
            CameraFeedUI QSpinBox::up-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            CameraFeedUI QSpinBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            CameraFeedUI QSlider::groove:horizontal {
                border: 1px solid #e9ecef;
                height: 6px;
                background: #f8f9fa;
                border-radius: 3px;
            }
            CameraFeedUI QSlider::handle:horizontal {
                background: #FF914D;
                border: 1px solid #FF914D;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            CameraFeedUI QSlider::sub-page:horizontal {
                background: #FF711A;
                border-radius: 3px;
            }
            CameraFeedUI QSlider::add-page:horizontal {
                background: #FFCFB2;
                border-radius: 3px;
            }
            CameraFeedUI QGroupBox {
                font-weight: 600;
                font-size: 16px;
                color: #333333;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 10px;
            }
            CameraFeedUI QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 10px 0 10px;
                background-color: white;
            }
        """)

        # Connect QStackedWidget's currentChanged signal
        self.stacked_widget.currentChanged.connect(self.on_stack_changed)
        self.stored_window_state = Qt.WindowNoState  # Initialize stored window state

        # Optional: Add debug output to see user info at startup
        self.debug_user_info()

    def _fetch_user_from_login(self, provided_user_info=None):
        """Fetch user information from existing login system with comprehensive fallbacks."""
        try:
            print("Camera Feed UI: Fetching user from login system...")

            # Method 1: Use provided user_info parameter (highest priority)
            if provided_user_info and isinstance(provided_user_info, dict):
                user_id = (provided_user_info.get('user_id') or
                           provided_user_info.get('id') or
                           provided_user_info.get('ID'))
                username = provided_user_info.get('username')

                if user_id and username:
                    self.current_user_id = str(user_id)
                    self.current_username = str(username)
                    self.current_user_full_name = provided_user_info.get('full_name', username)
                    print(f"✓ Using provided user info: {username} (ID: {user_id})")
                    self._update_defects_config_with_user_info()  # Update config file
                    return

            # Method 2: Read from session_state.json (created by login_ui.py)
            session_file = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "session_state.json")
            if os.path.exists(session_file):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)

                    print(f"Session data keys: {list(session_data.keys())}")

                    # Try multiple ways to get user info from session
                    user_id = None
                    username = None
                    full_name = None

                    # Method 2a: Check for direct 'user' key (most common)
                    if 'user' in session_data and session_data['user']:
                        user_data = session_data['user']

                        # Handle if 'user' is a dictionary with user info
                        if isinstance(user_data, dict):
                            username = user_data.get('username')
                            if not user_id:
                                user_id = user_data.get('user_id') or user_data.get('id')
                            if not full_name:
                                full_name = user_data.get('full_name')
                            print(f"Found user info dict in session 'user' key: {user_data}")
                        # Handle if 'user' is just a string username
                        elif isinstance(user_data, str) and user_data != "unknown":
                            username = user_data
                            print(f"Found username string in session 'user' key: {username}")
                        else:
                            print(f"Unexpected user data type in session: {type(user_data)}")

                    # Method 2b: Check for structured user_info
                    if 'user_info' in session_data and isinstance(session_data['user_info'], dict):
                        user_info = session_data['user_info']
                        if not username:
                            username = user_info.get('username')
                        if not user_id:
                            user_id = (user_info.get('user_id') or user_info.get('id'))
                        if not full_name:
                            full_name = user_info.get('full_name')

                    # Method 2c: Check for direct current_user fields
                    if not username:
                        username = session_data.get('current_username')
                    if not user_id:
                        user_id = session_data.get('current_user_id')

                    # Method 2d: Check additional_data if it exists
                    if 'additional_data' in session_data and isinstance(session_data['additional_data'], dict):
                        additional_data = session_data['additional_data']
                        if not username:
                            username = additional_data.get(
                                'username') or additional_data.get('user')
                        if not user_id:
                            user_id = additional_data.get('user_id') or additional_data.get('id')
                        if not full_name:
                            full_name = additional_data.get('full_name')

                    # If we have a username, try to get user_id from database
                    if username and username != "unknown" and not user_id:
                        try:
                            from azure_database import get_user_id_by_username
                            user_id = get_user_id_by_username(username)
                            print(f"✓ Retrieved user_id from database: {user_id}")
                        except Exception as db_error:
                            print(f"Database lookup failed for user '{username}': {db_error}")

                    # Set the user info if we have valid data
                    if username and username != "unknown":
                        self.current_username = str(username)
                        self.current_user_full_name = str(full_name) if full_name else username
                        if user_id:
                            self.current_user_id = str(user_id)
                            print(f"✓ Loaded from session file: {username} (ID: {user_id})")
                        else:
                            print(f"✓ Loaded username from session file: {username} (no ID found)")

                        # Update defects config with correct user info
                        self._update_defects_config_with_user_info()
                        return

                except Exception as e:
                    print(f"Error reading session file: {e}")

            # Method 3: Read from defects config file (but clean it if it has "unknown")
            defects_config_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "defects_config.json")
            if os.path.exists(defects_config_path):
                try:
                    with open(defects_config_path, 'r') as f:
                        config_data = json.load(f)

                    print(f"Defects config keys: {list(config_data.keys())}")

                    # Method 3a: Check for structured user_info
                    if 'user_info' in config_data and isinstance(config_data['user_info'], dict):
                        user_info = config_data['user_info']
                        user_id = user_info.get('user_id') or user_info.get('id')
                        username = user_info.get('username')
                        full_name = user_info.get('full_name', username)

                        if user_id and username and username != "unknown":
                            self.current_user_id = str(user_id)
                            self.current_username = str(username)
                            self.current_user_full_name = str(full_name) if full_name else username
                            print(f"✓ Loaded from defects config: {username} (ID: {user_id})")
                            return

                    # Method 3b: Check for direct 'user' key
                    if 'user' in config_data and config_data['user']:
                        user_data = config_data['user']

                        # Handle if 'user' is a dictionary with user info
                        if isinstance(user_data, dict):
                            username = user_data.get('username')
                            if not user_id:
                                user_id = user_data.get('user_id') or user_data.get('id')
                            if not full_name:
                                full_name = user_data.get('full_name')
                            print(
                                f"Found user info dict in defects config 'user' key: {user_data}")
                        # Handle if 'user' is just a string username
                        elif isinstance(user_data, str) and user_data != "unknown":
                            username = user_data
                            print(
                                f"Found username string in defects config 'user' key: {username}")

                        # Only proceed if we have a valid username string
                        if username and username != "unknown":
                            try:
                                from azure_database import get_user_id_by_username
                                if not user_id:  # Only lookup if we don't already have an ID
                                    user_id = get_user_id_by_username(username)
                                if user_id:
                                    self.current_user_id = str(user_id)
                                    self.current_username = str(username)
                                    self.current_user_full_name = str(
                                        full_name) if full_name else username
                                    print(
                                        f"✓ Loaded from defects config (username lookup): {username} (ID: {user_id})")
                                    return
                            except Exception as db_error:
                                print(f"Database lookup failed for user '{username}': {db_error}")
                                # Still use the username even if ID lookup fails
                                self.current_username = str(username)
                                self.current_user_full_name = str(
                                    full_name) if full_name else username
                                print(f"✓ Using username from defects config (no ID): {username}")
                                return

                except Exception as e:
                    print(f"Error reading defects config: {e}")

            # If we reach here, no user info was found
            print("⚠ No valid user info found, keeping admin defaults")

        except Exception as e:
            print(f"Error in _fetch_user_from_login: {e}")
            # Keep default values set in __init__

        finally:
            # Final validation - ensure username is never a dictionary
            if isinstance(self.current_username, dict):
                print(f"⚠ WARNING: Username was set to dict: {self.current_username}")
                if 'username' in self.current_username:
                    actual_username = self.current_username['username']
                    full_name = self.current_username.get('full_name', actual_username)
                    print(f"Extracting username from dict: {actual_username}")
                    self.current_username = str(actual_username)
                    if not hasattr(self, 'current_user_full_name') or self.current_user_full_name == actual_username:
                        self.current_user_full_name = str(full_name)
                else:
                    print("Dict doesn't contain 'username' key, resetting to admin")
                    self.current_username = "admin"
                    self.current_user_full_name = "Administrator"

            # Ensure all values are strings
            self.current_user_id = str(self.current_user_id)
            self.current_username = str(self.current_username)
            self.current_user_full_name = str(self.current_user_full_name)

            # Always update the defects config with clean user info
            self._update_defects_config_with_user_info()

    def _update_defects_config_with_user_info(self):
        """Update the defects_config.json file with correct user information."""
        try:
            defects_config_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "defects_config.json")

            # Read existing config or create new one
            config_data = {}
            if os.path.exists(defects_config_path):
                try:
                    with open(defects_config_path, 'r') as f:
                        config_data = json.load(f)
                except Exception as e:
                    print(f"Error reading existing defects config: {e}")
                    config_data = {}

            # Update with correct user information
            config_data['user_info'] = {
                'user_id': self.current_user_id,
                'username': self.current_username,
                'full_name': self.current_user_full_name
            }

            # Also update the 'user' key to prevent confusion
            config_data['user'] = self.current_username

            # Write back to file
            with open(defects_config_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            print(
                f"✓ Updated defects_config.json with user: {self.current_username} (ID: {self.current_user_id})")

        except Exception as e:
            print(f"Error updating defects_config.json: {e}")

    def pass_user_info_to_child_components(self):
        """Pass user information to child UI components to prevent them from reading stale config files."""
        try:
            user_info = {
                'user_id': self.current_user_id,
                'username': self.current_username,
                'full_name': self.current_user_full_name
            }

            # Pass to manual annotation tool if it has a method to set user info
            if hasattr(self.manual_annotation_tool, 'set_user_info'):
                self.manual_annotation_tool.set_user_info(user_info)

            # Pass to demo feed UI
            if hasattr(self.demo_feed_ui, 'set_user_info'):
                self.demo_feed_ui.set_user_info(user_info)

            # Pass to image classification UI
            if hasattr(self.image_classi_ui, 'set_user_info'):
                self.image_classi_ui.set_user_info(user_info)

            # Pass to auto annotation UI
            if hasattr(self.auto_annoate_ui, 'set_user_info'):
                self.auto_annoate_ui.set_user_info(user_info)

            # Pass to deployment UI
            if hasattr(self.deployment_ui, 'set_user_info'):
                self.deployment_ui.set_user_info(user_info)

            print(f"✓ User info passed to child components: {self.current_username}")

        except Exception as e:
            print(f"Error passing user info to child components: {e}")

    def get_current_user_id(self):
        """Get the current user ID with fallback."""
        return getattr(self, 'current_user_id', "A001")

    def get_current_username(self):
        """Get the current username with fallback."""
        return getattr(self, 'current_username', "admin")

    def get_current_user_full_name(self):
        """Get the current user's full name with fallback."""
        return getattr(self, 'current_user_full_name', "Administrator")

    def debug_user_info(self):
        """Debug method to see what user info was loaded."""
        print(f"\n=== Camera Feed UI User Info ===")
        print(f"User ID: {self.current_user_id}")
        print(f"Username: {self.current_username}")
        print(f"Full Name: {self.current_user_full_name}")

        # Check session file
        session_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "session_state.json")
        print(f"Session file exists: {os.path.exists(session_file)}")

        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                print(f"Session file contains: {list(session_data.keys())}")
            except:
                print("Could not read session file")

        print(f"===============================\n")

    def init_ui(self):
        """
        Initializes the main window's basic properties such as title and geometry.
        """
        self.setWindowTitle('SAKAR VISION AI - Camera Feed')
        self.setGeometry(200, 200, 1400, 800)

    def update_upload_interval(self, seconds):
        """
        Updates the capture duration and resets the timer if active.
        """
        print(f"Updating capture duration to {seconds} seconds")
        self.upload_interval = seconds

        # Reset timer if active (during an ongoing session)
        if self.capture_duration_timer.isActive():
            print(f"Session is active - restarting timer with new duration: {seconds} seconds")
            self.capture_duration_timer.stop()
            self.capture_duration_timer.start(seconds * 1000)  # Convert to milliseconds

            # Update the capture end time from NOW (not from original start time)
            self.capture_end_time = datetime.now() + timedelta(seconds=seconds)

            # Show notification about the change
            self.capture_progress_label.setText(
                f"Duration changed! Session will now end in {seconds} seconds")

            # Update status display
            self.update_status_display()

            print(f"Timer restarted. New end time: {self.capture_end_time}")
        else:
            print("No active session - duration change will apply to next session")

    def update_capture_rate(self, fps):
        """
        Updates the capture rate (frames per second) and resets the timer if active.
        """
        print(f"Updating capture rate to {fps} fps")
        self.capture_rate = fps

        # Reset timer if active
        if self.capture_timer.isActive():
            self.capture_timer.stop()
            # Convert fps to milliseconds between frames
            ms_between_frames = int(1000 / fps)
            self.capture_timer.start(ms_between_frames)

    def stop_and_upload_all(self):
        """
        Called when capture duration ends. Stops capturing and uploads all images.
        """
        print("Capture duration completed. Stopping capture and uploading all images...")

        # Stop capturing
        self.capture_active = False
        self.capture_session_active = False
        self.feed_paused = False
        self.remaining_session_time = None

        # Stop timers
        if self.capture_timer.isActive():
            self.capture_timer.stop()
        if self.capture_duration_timer.isActive():
            self.capture_duration_timer.stop()

        # DON'T stop camera feed here - keep it running for next session
        # self.stop_camera_feed()  # Commented out to prevent camera error

        # Update UI
        self.btn_capture_images_setup.setText("Start Continuous Capture")
        self.btn_pause_feed.setVisible(False)
        self.btn_resume_feed.setVisible(False)

        # Upload all pending images
        if len(self.pending_uploads) > 0:
            self.capture_progress_label.setText(
                f"Uploading all {len(self.pending_uploads)} captured images...")
            self.process_all_uploads()
        else:
            self.capture_progress_label.setText("Capture session completed - no images to upload")
            self.capture_progress_bar.setValue(0)

    def process_all_uploads(self):
        """
        Processes all pending uploads at once and shows completion popup.
        """
        if len(self.pending_uploads) == 0:
            print("No images to upload")
            self.capture_progress_label.setText("Ready")
            self.capture_progress_bar.setValue(0)
            return

        upload_total = len(self.pending_uploads)
        print(f"Processing final upload of {upload_total} images")

        # Update progress bar settings
        self.capture_progress_bar.setRange(0, upload_total)
        self.capture_progress_bar.setValue(0)

        # Check system connectivity status
        is_online = self.is_system_online()
        if is_online:
            self.capture_progress_label.setText(f"Uploading all {upload_total} images to Azure...")
        else:
            self.capture_progress_label.setText(
                f"System offline - all {upload_total} images stored locally")

        # Create Azure directories if network is available
        if hasattr(azure_storage, 'has_network_connectivity') and azure_storage.has_network_connectivity:
            # Ensure Azure directories exist with the same names as local folders
            azure_storage.create_directory_in_azure("manual_input")
            azure_storage.create_directory_in_azure("auto_input")
            azure_storage.create_directory_in_azure("manually_annotated")
            azure_storage.create_directory_in_azure("auto_annotated")

        # Prepare uploads with proper format for batch upload
        batch_uploads = []
        for local_path, azure_folder, filename in self.pending_uploads:
            if os.path.exists(local_path):
                batch_uploads.append((local_path, azure_folder, filename))

        # Process upload in batch and track success/failure
        successful, failed = 0, 0
        if batch_uploads:
            # Only attempt batch upload if we have files to upload
            successful, failed = azure_storage.batch_upload_to_azure(batch_uploads)

            # Update the UI with progress
            self.capture_progress_bar.setValue(upload_total)
            self.progress_percent_label.setText("100%")

            # Handle offline scenario gracefully - store files locally
            if failed > 0 and not is_online:
                self.capture_progress_label.setText(
                    f"All {upload_total} images stored locally (offline)")
            else:
                self.capture_progress_label.setText(
                    f"Successfully uploaded {successful} of {upload_total} images")

        # Clear the pending uploads list
        self.pending_uploads = []
        self.upload_count += successful
        self.current_batch_count = 0

        # Show final completion notification (only once!)
        self.show_final_upload_notification(successful, failed, upload_total)

        # Reset UI after upload completion
        self.capture_active = False
        self.capture_session_active = False
        self.feed_paused = False
        self.capture_end_time = None
        self.remaining_session_time = None
        self.btn_capture_images_setup.setText("Start Continuous Capture")
        self.btn_pause_feed.setVisible(False)
        self.btn_resume_feed.setVisible(False)
        self.capture_progress_label.setText("Ready")
        self.capture_progress_bar.setValue(0)
        self.progress_percent_label.setText("0%")

    def show_final_upload_notification(self, success_count, failed_count, total_count):
        """
        Shows a single notification popup after the entire capture session is complete.
        """
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Capture Session Complete")
        msg_box.setIcon(QMessageBox.Information)

        if failed_count == 0:
            msg_box.setText(
                f"✅ Capture session completed successfully!\n\nUploaded all {success_count} images at {time_str}")
        else:
            msg_box.setText(
                f"⚠️ Capture session completed with some issues\n\nUploaded {success_count} of {total_count} images. {failed_count} uploads failed.")

        msg_box.setInformativeText(
            f"📊 Session Summary:\n"
            f"• Total images captured: {self.capture_count}\n"
            f"• Images uploaded to Azure: {success_count}\n"
            f"• Images stored locally: {failed_count}\n"
            f"• Capture duration: {self.upload_interval} seconds ({self.upload_interval//60}m {self.upload_interval%60}s)\n"
            f"• Captured by user: {self.current_username} (ID: {self.current_user_id})\n\n"
            f"Ready to start a new capture session.")

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def update_status_display(self):
        """
        Updates the status display with current information.
        """
        if not self.capture_active:
            self.capture_progress_label.setText("Ready")
            self.capture_progress_bar.setValue(0)
            self.progress_percent_label.setText("0%")
            return

        now = datetime.now()

        if self.capture_end_time:
            # Calculate time until capture session ends
            time_remaining = self.capture_end_time - now
            if time_remaining.total_seconds() <= 0:
                return  # Session should end soon

            seconds_remaining = max(0, int(time_remaining.total_seconds()))
            minutes_remaining = seconds_remaining // 60
            seconds_remaining %= 60

            # Format the status message
            status = (
                f"Capturing images: {self.capture_count} captured, "
                f"{self.current_batch_count} pending upload. "
                f"Session ends in {minutes_remaining}m {seconds_remaining}s"
            )
            self.capture_progress_label.setText(status)

            # Update progress bar to show session progress
            if self.upload_interval > 0:
                total_session_seconds = self.upload_interval
                elapsed_seconds = total_session_seconds - \
                    (seconds_remaining + minutes_remaining * 60)
                percentage = min(100, max(0, int((elapsed_seconds / total_session_seconds) * 100)))
                self.capture_progress_bar.setRange(0, 100)
                self.capture_progress_bar.setValue(percentage)
                self.progress_percent_label.setText(f"{percentage}%")

    def setup_camera_feed_ui(self):
        """
        Sets up the UI elements specifically for the camera feed view.
        """
        # Camera feed setup
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setStyleSheet("""
            QLabel {
                background-color: #343a40;
                color: #ffffff;
                border: 3px solid #ff914d;
                border-radius: 12px;
                font-size: 16px;
                min-height: 400px;
            }
        """)
        self.camera_feed_label.setText("📷\n\nCamera Feed Here")
        self.camera_feed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_feed_timer.timeout.connect(self.update_camera_feed)

        # Progress bar setup
        self.capture_progress_bar.setRange(0, 100)
        self.capture_progress_bar.setValue(0)

        # Spinbox setup - back to seconds with lower minimum
        self.upload_interval_spinbox.setRange(1, 3600)  # 1 second to 1 hour
        self.upload_interval_spinbox.setValue(300)  # Default 5 minutes (300 seconds)
        self.upload_interval = 300  # Set default to 300 seconds (5 minutes)

        # Main horizontal layout
        main_h_layout = QHBoxLayout()
        main_h_layout.setSpacing(30)
        main_h_layout.setContentsMargins(30, 30, 30, 30)

        # Left side - Camera feed (takes 2/3 of space)
        camera_container = QFrame()
        camera_container.setStyleSheet("""
            QFrame {
                background-color: transparent;
            }
        """)
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.addWidget(self.camera_feed_label)

        # Right side - Controls (takes 1/3 of space)
        controls_container = QWidget()
        controls_container.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(25)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Capture Settings Group
        capture_group = QGroupBox("Capture Settings")
        capture_group_layout = QVBoxLayout()
        capture_group_layout.setSpacing(20)
        capture_group_layout.setContentsMargins(20, 25, 20, 20)

        # Capture Rate
        capture_rate_layout = QVBoxLayout()
        capture_rate_layout.setSpacing(8)
        self.capture_rate_spinbox_label.setStyleSheet("font-weight: 500; color: #555;")
        capture_rate_layout.addWidget(self.capture_rate_spinbox_label)
        capture_rate_layout.addWidget(self.capture_rate_spinbox)

        # Upload Interval
        upload_interval_layout = QVBoxLayout()
        upload_interval_layout.setSpacing(8)
        self.upload_interval_spinbox_label.setStyleSheet("font-weight: 500; color: #555;")
        upload_interval_layout.addWidget(self.upload_interval_spinbox_label)
        upload_interval_layout.addWidget(self.upload_interval_spinbox)

        capture_group_layout.addLayout(capture_rate_layout)
        capture_group_layout.addLayout(upload_interval_layout)
        capture_group.setLayout(capture_group_layout)

        # Data Distribution Group
        distribution_group = QGroupBox("Data Distribution")
        distribution_group_layout = QVBoxLayout()
        distribution_group_layout.setSpacing(15)
        distribution_group_layout.setContentsMargins(20, 25, 20, 20)

        # Slider with labels
        slider_container = QVBoxLayout()
        slider_container.setSpacing(8)

        # Slider label and percentage
        slider_header = QHBoxLayout()
        self.data_distribution_slider_label.setStyleSheet("font-weight: 500; color: #555;")
        slider_header.addWidget(self.data_distribution_slider_label)
        slider_header.addStretch()
        self.data_distribution_label.setStyleSheet(
            "font-weight: 600; color: #ff914d; font-size: 16px;")
        slider_header.addWidget(self.data_distribution_label)

        slider_container.addLayout(slider_header)
        slider_container.addWidget(self.data_distribution_slider)

        distribution_group_layout.addLayout(slider_container)
        distribution_group.setLayout(distribution_group_layout)

        # Action Buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)

        # Start/Stop button
        self.btn_capture_images_setup.setStyleSheet("""
            QPushButton {
                background-color: #ff914d;
                font-size: 16px;
                font-weight: 600;
                padding: 15px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #FF711A;
            }
        """)

        # Resume Camera button
        self.btn_resume_feed.setText("Resume Camera")
        self.btn_resume_feed.setObjectName("resumeButton")
        self.btn_resume_feed.setVisible(False)  # Initially hidden

        # Pause Camera button
        self.btn_pause_feed.setText("Pause Camera")
        self.btn_pause_feed.setVisible(False)  # Initially hidden

        # Manual Annotation button
        self.btn_manual_annotation.setStyleSheet("""
            QPushButton {
                background-color: #4DBBFF;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1AA8FF;
            }                              
        """)

        buttons_layout.addWidget(self.btn_capture_images_setup)
        buttons_layout.addWidget(self.btn_pause_feed)
        buttons_layout.addWidget(self.btn_resume_feed)
        buttons_layout.addWidget(self.btn_manual_annotation)

        # Add all to controls layout
        controls_layout.addWidget(capture_group)
        controls_layout.addWidget(distribution_group)
        controls_layout.addStretch()
        controls_layout.addLayout(buttons_layout)

        # Add to main layout
        main_h_layout.addWidget(camera_container, 2)  # 2/3 space
        main_h_layout.addWidget(controls_container, 0)  # Fixed width

        # Status bar at bottom
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(30, 10, 30, 20)
        status_layout.setSpacing(15)

        status_left = QLabel("Status:")
        status_left.setStyleSheet("font-weight: 500; color: #666;")

        self.capture_progress_label.setStyleSheet("color: #333; font-weight: 500;")

        # Make progress bar visible and properly sized
        self.capture_progress_bar.setMinimumHeight(35)  # Increased from 25 to 35
        self.capture_progress_bar.setMaximumWidth(500)  # Increased from 335 to 500
        self.capture_progress_bar.setMinimumWidth(400)  # Added minimum width
        self.capture_progress_bar.setVisible(True)

        self.progress_percent_label = QLabel("0%")
        self.progress_percent_label.setStyleSheet("font-weight: 500; color: #666;")

        status_layout.addWidget(status_left)
        status_layout.addWidget(self.capture_progress_label, 1)
        status_layout.addWidget(self.capture_progress_bar)
        status_layout.addWidget(self.progress_percent_label)

        # Main camera UI layout
        camera_ui_layout = QVBoxLayout()
        camera_ui_layout.setContentsMargins(0, 0, 0, 0)
        camera_ui_layout.setSpacing(0)

        camera_ui_layout.addLayout(main_h_layout, 1)
        camera_ui_layout.addLayout(status_layout, 0)

        self.camera_feed_widget.setLayout(camera_ui_layout)

    def toggle_continuous_capture(self):
        """
        Toggles the continuous capture mode on and off.
        """
        if not self.capture_active:
            # Start continuous capture session
            if not self.capture_storage_folder:
                self.choose_storage_location()
                if not self.capture_storage_folder:
                    return

            self.capture_count = 0
            self.current_batch_count = 0
            self.upload_count = 0
            self.pending_uploads = []
            self.remaining_session_time = None

            # Start camera only if not already running
            if not (self.camera_capture and self.camera_capture.isOpened()):
                print("Starting camera feed for new session")
                self.start_camera_feed()
            else:
                print("Camera already running, reusing existing feed")

            # Set up capture session
            self.upload_interval = self.upload_interval_spinbox.value()  # Now in seconds
            self.capture_rate = self.capture_rate_spinbox.value()

            # Calculate milliseconds between frames from fps
            ms_between_frames = int(1000 / self.capture_rate)

            # Start capture timer
            self.capture_timer.start(ms_between_frames)

            # Start capture duration timer (convert seconds to milliseconds)
            self.capture_duration_timer.start(self.upload_interval * 1000)

            # Set capture end time
            self.capture_end_time = datetime.now() + timedelta(seconds=self.upload_interval)

            # Update status
            self.capture_active = True
            self.capture_session_active = True
            self.feed_paused = False
            self.btn_capture_images_setup.setText("Stop Capture Session")

            # Show pause/resume buttons
            self.btn_pause_feed.setVisible(True)
            self.btn_resume_feed.setVisible(False)

            # Initialize progress
            self.capture_progress_bar.setRange(0, 100)
            self.capture_progress_bar.setValue(0)
            self.update_status_display()

            # Show confirmation to user
            QMessageBox.information(
                self, "Capture Session Started",
                f"📸 Capture session started by {self.current_username}!\n\n"
                f"• Capture rate: {self.capture_rate} frames per second\n"
                f"• Session duration: {self.upload_interval} seconds ({self.upload_interval//60}m {self.upload_interval%60}s)\n"
                f"• Images will be uploaded to Azure after session ends\n\n"
                f"The session will automatically stop and upload all images after {self.upload_interval} seconds.")
        else:
            # Stop continuous capture session manually
            self.stop_continuous_capture()

    def stop_continuous_capture(self):
        """
        Stops the continuous capture process manually.
        """
        # Stop timers
        if self.capture_timer.isActive():
            self.capture_timer.stop()
        if self.capture_duration_timer.isActive():
            self.capture_duration_timer.stop()

        # Ask user if they want to upload captured images
        if len(self.pending_uploads) > 0:
            reply = QMessageBox.question(
                self, "Upload Captured Images",
                f"You have {len(self.pending_uploads)} images captured.\n\n"
                f"Would you like to upload them to Azure now?",
                QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.process_all_uploads()
                return  # Let the upload process handle UI reset
            else:
                # User chose not to upload, just clear pending uploads
                self.pending_uploads = []

        # Reset status
        self.capture_active = False
        self.capture_session_active = False
        self.feed_paused = False
        self.capture_end_time = None
        self.remaining_session_time = None
        self.btn_capture_images_setup.setText("Start Continuous Capture")
        self.capture_progress_label.setText("Ready")
        self.capture_progress_bar.setValue(0)
        self.progress_percent_label.setText("0%")

        # Hide pause/resume buttons
        self.btn_pause_feed.setVisible(False)
        self.btn_resume_feed.setVisible(False)

    def setup_stacked_widget(self):
        """
        Sets up the QStackedWidget to manage and switch between different views.
        """
        self.stacked_widget.addWidget(self.camera_feed_widget)
        print(f"Camera feed widget index: {self.stacked_widget.indexOf(self.camera_feed_widget)}")

        self.stacked_widget.addWidget(self.demo_feed_widget)
        print(f"Demo feed widget index: {self.stacked_widget.indexOf(self.demo_feed_widget)}")

        self.stacked_widget.addWidget(self.image_classi_widget)
        print(
            f"Image classification widget index: {self.stacked_widget.indexOf(self.image_classi_widget)}")

        self.stacked_widget.addWidget(self.annotation_tool_widget)
        print(
            f"Annotation tool widget index: {self.stacked_widget.indexOf(self.annotation_tool_widget)}")

        self.stacked_widget.addWidget(self.auto_annoate_ui)
        print(f"Auto annotation widget index: {self.stacked_widget.indexOf(self.auto_annoate_ui)}")

        self.stacked_widget.addWidget(self.deployment_ui)
        print(f"Deployment UI widget index: {self.stacked_widget.indexOf(self.deployment_ui)}")

        self.stacked_widget.setCurrentIndex(0)  # Initially show the camera feed

        # Main layout with title bar
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Add title bar
        main_layout.addWidget(self.title_bar)

        # Add stacked widget
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

    def choose_storage_location(self):
        """
        Opens a file dialog to select a storage location for captured images.
        Also creates additional directories for annotation workflows.
        """
        # Specify the default directory (e.g., "/home/sakar02/CapturedImages")
        default_directory = "testing/"

        # Open the file dialog with the default directory
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Main Storage Folder for Captured Images', default_directory)

        if folder:
            self.capture_storage_folder = folder
            self.capture_folder_a = os.path.join(self.capture_storage_folder, "manual_input")
            self.capture_folder_b = os.path.join(self.capture_storage_folder, "auto_input")

            # Create additional folders for annotation workflows
            manual_annotated_path = os.path.join(self.capture_storage_folder, "manually_annotated")
            auto_annotated_path = os.path.join(self.capture_storage_folder, "auto_annotated")

            try:
                # Create local folders
                os.makedirs(self.capture_folder_a, exist_ok=True)
                os.makedirs(self.capture_folder_b, exist_ok=True)
                os.makedirs(manual_annotated_path, exist_ok=True)
                os.makedirs(auto_annotated_path, exist_ok=True)

                print(f"Storage folder selected: {self.capture_storage_folder}")
                print(f"Manual input folder created: {self.capture_folder_a}")
                print(f"Auto input folder created: {self.capture_folder_b}")
                print(f"Manual annotation folder created: {manual_annotated_path}")
                print(f"Auto annotation folder created: {auto_annotated_path}")

                QMessageBox.information(self, "Storage Location",
                                        f"Directory structure created successfully:\n\n"
                                        f"Root: {self.capture_storage_folder}\n"
                                        f"├── manual_input (for manual annotation input)\n"
                                        f"├── auto_input (for auto annotation input)\n"
                                        f"├── manually_annotated (output from manual annotation)\n"
                                        f"└── auto_annotated (output from auto annotation)\n\n"
                                        f"These folders will be automatically used by the tool.")
            except OSError as e:
                QMessageBox.critical(
                    self, "Folder Creation Error", f"Could not create subfolders in the selected directory.\nError: {e}")
                self.capture_storage_folder = None
                self.capture_folder_a = None
                self.capture_folder_b = None
        else:
            QMessageBox.warning(self, "Storage Location",
                                "No storage folder selected. Capture cancelled.")

        # If we have a valid folder structure, set the default paths in other UIs
        if self.capture_storage_folder:
            # Set the manual_input as the default path for manual annotation tool
            if hasattr(self.manual_annotation_tool, 'image_folder_on_start'):
                self.manual_annotation_tool.image_folder_on_start = self.capture_folder_a

    def start_camera_feed(self):
        """
        Initializes and starts the camera feed using the selected camera index.
        """
        # Try to get the selected camera index from settings
        settings_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "inspection_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    if 'selected_camera_index' in settings:
                        self.selected_camera_index = settings['selected_camera_index']
                        print(
                            f"Using saved camera index from settings: {self.selected_camera_index}")
            except Exception as e:
                print(f"Error reading camera index from settings: {e}")

        # Try to open the selected camera
        self.camera_capture = cv2.VideoCapture(self.selected_camera_index)
        if not self.camera_capture.isOpened():
            QMessageBox.critical(self, "Camera Error",
                                 f"Could not open camera index {self.selected_camera_index}")
            return
        self.camera_feed_timer.start(30)  # 30ms refresh rate for camera feed

    def stop_camera_feed(self):
        """
        Stops the camera feed.
        """
        if self.camera_capture and self.camera_capture.isOpened():
            self.camera_capture.release()
        self.camera_feed_timer.stop()
        self.camera_feed_label.clear()
        self.camera_feed_label.setText("📷\n\nCamera Feed Stopped")

    def get_unique_filename(self, folder, base_filename):
        """
        Generates a unique filename in the specified folder.
        """
        filepath = os.path.join(folder, base_filename)
        if not os.path.exists(filepath):
            return filepath

        name, ext = os.path.splitext(base_filename)
        counter = 1
        while True:
            new_filename = f"{name}_{counter}{ext}"
            filepath = os.path.join(folder, new_filename)
            if not os.path.exists(filepath):
                return filepath
            counter += 1

    def update_camera_feed(self):
        """
        Updates the camera feed display in the UI.
        """
        if self.camera_capture and self.camera_capture.isOpened():
            ret, frame = self.camera_capture.read()
            if ret:
                # Convert frame to format suitable for display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                # Scale pixmap to fit label
                pixmap_scaled = pixmap.scaled(
                    self.camera_feed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_feed_label.setPixmap(pixmap_scaled)
            else:
                # Handle camera error
                self.stop_camera_feed()
                QMessageBox.warning(self, "Camera Error", "Error reading frame from camera.")

                # If in continuous capture mode, stop capturing
                if self.capture_active:
                    self.stop_continuous_capture()

    def capture_frame(self):
        """
        Captures a single frame and adds it to pending uploads.
        This is called at the capture rate frequency.
        """
        if not self.capture_active or self.feed_paused:
            return

        if self.camera_capture and self.camera_capture.isOpened():
            ret, frame = self.camera_capture.read()
            if ret:
                # Increment counters
                self.capture_count += 1
                self.current_batch_count += 1

                # Create filename with timestamp and user info
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                user_prefix = self.current_username[:4] if self.current_username else "unkn"
                base_filename = f"captured_image_{user_prefix}_{timestamp}_{self.capture_count}.png"

                # Determine folder based on distribution slider
                if random.randint(0, 99) < self.data_distribution_percentage:
                    target_folder = self.capture_folder_a
                    azure_folder = "manual_input"
                else:
                    target_folder = self.capture_folder_b
                    azure_folder = "auto_input"

                # Save image locally
                image_path = self.get_unique_filename(target_folder, base_filename)
                cv2.imwrite(image_path, frame)

                # Add to pending uploads list with Azure folder information
                self.pending_uploads.append((image_path, azure_folder, base_filename))

                # Log capture with user info
                print(
                    f"Frame captured by user {self.current_username} (ID: {self.current_user_id}): {base_filename}")

                # Update status display
                self.update_status_display()
            else:
                print("Error capturing frame")

                # If in continuous capture mode, stop capturing
                if self.capture_active:
                    self.stop_continuous_capture()
                    QMessageBox.warning(self, "Camera Error",
                                        "Error capturing frame. Continuous capture stopped.")

    def show_defect_selection(self):
        """
        Navigate back to the defect selection UI.
        """
        # Find the parent that contains the DefectSelectionUI
        parent = self.parent()
        parent_with_defect_selection = None

        while parent is not None:
            print(f"Looking for DefectSelectionUI in parent: {parent.__class__.__name__}")
            if hasattr(parent, 'defect_ui') and parent.defect_ui:
                parent_with_defect_selection = parent
                print("Found parent with defect_ui attribute")
                break
            elif hasattr(parent, 'DefectSelectionUI'):
                parent_with_defect_selection = parent
                print("Found parent with DefectSelectionUI attribute")
                break
            parent = parent.parent()

        if parent_with_defect_selection:
            # Try to show the DefectSelectionUI
            if hasattr(parent_with_defect_selection, 'defect_ui'):
                parent_with_defect_selection.defect_ui.show()
                self.hide()
                return True

        # Try to go back through the stacked widget
        if hasattr(self, 'stacked_widget'):
            print(f"Checking stacked widget with {self.stacked_widget.count()} items")

            # Look for DefectSelectionUI in the stacked widget
            from defect_selection_ui import DefectSelectionUI
            for i in range(self.stacked_widget.count()):
                widget = self.stacked_widget.widget(i)
                print(f"Widget at index {i}: {widget.__class__.__name__}")
                if isinstance(widget, DefectSelectionUI):
                    print(f"Found DefectSelectionUI at index {i}")
                    self.stacked_widget.setCurrentIndex(i)
                    return True

        # If we can't find it, try to go to a parent application with its own navigation
        app_parent = self.window()
        if hasattr(app_parent, 'show_defect_selection'):
            print("Using parent's show_defect_selection method")
            app_parent.show_defect_selection()
            return True

        # Last resort - check if we should be calling a method on another object
        if hasattr(self, 'defect_selection_ui'):
            print("Found defect_selection_ui attribute in CameraFeedUI")
            defect_ui = self.defect_selection_ui
            if defect_ui:
                # If it's a widget, try to show it
                if isinstance(defect_ui, QWidget):
                    print("Showing defect_selection_ui widget")
                    defect_ui.show()
                    self.hide()
                    return True

        print("Failed to navigate to DefectSelectionUI through CameraFeedUI")
        return False

    def show_manual_annotation_ui(self):
        """
        Switches to the manual annotation UI and automatically loads images from manual_input folder.
        """
        print("show_manual_annotation_ui function called!")
        self.stacked_widget.setCurrentIndex(3)

        # Update the title to "Data Pre-Processing"
        if hasattr(self, 'title_bar'):
            self.title_bar.title_label.setText("Data Pre-Processing")

        if self.capture_storage_folder is not None:
            # Automatically use manual_input folder for manual annotation (updated from Folder_A)
            manual_input_path = os.path.join(self.capture_storage_folder, "manual_input")
            print(f"Automatically loading images from manual_input: {manual_input_path}")

            # Check if the folder exists before trying to load
            if os.path.exists(manual_input_path):
                # Set the path in the UI text field if it exists
                if hasattr(self.manual_annotation_tool, 'inputDirLineEdit'):
                    self.manual_annotation_tool.inputDirLineEdit.setText(manual_input_path)

                # Load the images from manual_input folder
                self.manual_annotation_tool.loadImagesFromFolder(manual_input_path)
            else:
                print(f"Warning: manual_input folder does not exist at {manual_input_path}")
                # Show a user-friendly message
                QMessageBox.information(
                    self, "Folder Not Found",
                    f"The manual_input folder doesn't exist yet.\n\n"
                    f"Expected location: {manual_input_path}\n\n"
                    f"Please start a capture session first to create the folder structure, "
                    f"or manually select a folder using the 'Open Folder' button in the annotation tool."
                )
        else:
            print("capture_storage_folder is None, not loading images automatically.")

    def show_auto_annotation_ui(self):
        """
        Switches to the auto annotation UI and prefills the directories.
        """
        print("show_auto_annotation_ui function called!")

        # Hide our title bar before switching
        if hasattr(self, 'title_bar'):
            self.title_bar.hide()

        self.stacked_widget.setCurrentIndex(4)  # Switch to auto annotation tool view

        # Show auto annotation title bar if it exists
        auto_annotate_ui = self.stacked_widget.widget(4)
        if hasattr(auto_annotate_ui, 'title_bar'):
            auto_annotate_ui.title_bar.show()

        if self.capture_storage_folder is not None:
            # Set input directory to auto_input folder (updated from Folder_B)
            auto_input_path = os.path.join(self.capture_storage_folder, "auto_input")
            print(f"Setting auto annotation input directory to: {auto_input_path}")

            # Check if the folder exists
            if os.path.exists(auto_input_path):
                self.auto_annoate_ui.inputDirLineEdit.setText(auto_input_path)
                print(f"Auto annotation input directory set to: {auto_input_path}")
            else:
                print(f"Warning: auto_input folder does not exist at {auto_input_path}")
                # Still set the path so user knows where it should be
                self.auto_annoate_ui.inputDirLineEdit.setText(auto_input_path)
                # Show a user-friendly message
                QMessageBox.information(
                    self, "Folder Not Found",
                    f"The auto_input folder doesn't exist yet.\n\n"
                    f"Expected location: {auto_input_path}\n\n"
                    f"Please start a capture session first to create the folder structure, "
                    f"or manually select a folder using the input directory selector in the auto annotation tool."
                )

            # Create and set output directory path (auto_annotated folder in root)
            auto_annotated_path = os.path.join(self.capture_storage_folder, "auto_annotated")
            os.makedirs(auto_annotated_path, exist_ok=True)
            self.auto_annoate_ui.outputDirLineEdit.setText(auto_annotated_path)
            # Also set the split directory to the output directory
            self.auto_annoate_ui.splitDirLineEdit.setText(auto_annotated_path)
            # Pre-configure the model path if specified
            model_path = "/home/sakar02/sakar-vision-ui/y11n.pt"
            if os.path.exists(model_path):
                self.auto_annoate_ui.modelLineEdit.setText(model_path)
                try:
                    from ultralytics import YOLO
                    self.auto_annoate_ui.MODEL_PATH = model_path
                    self.auto_annoate_ui.det_model = YOLO(model_path)
                except Exception as e:
                    print(f"Error loading model: {e}")
        else:
            print("capture_storage_folder is None, not setting auto annotation directories automatically.")

    def show_demo_feed_ui(self):
        """
        Switches to the demo feed UI.
        """
        print("showDemoFeedUI function is being called!")
        print(f"Type of self.demo_feed_widget in show_demo_feed_ui: {type(self.demo_feed_widget)}")
        print(f"Value of self.demo_feed_widget in show_demo_feed_ui: {self.demo_feed_widget}")
        self.stacked_widget.setCurrentIndex(1)

    def show_image_classi_ui(self):
        """
        Switches to the image classification UI.
        """
        print("showImageClassiUI function is being called!")
        print(
            f"Type of self.image_classi_widget in show_image_classi_ui: {type(self.image_classi_widget)}")
        print(
            f"Value of self.image_classi_widget in show_image_classi_ui: {self.image_classi_widget}")
        self.stacked_widget.setCurrentIndex(2)

    def show_camera_feed_ui(self):
        """
        Switches back to the camera feed UI.
        """
        self.stacked_widget.setCurrentIndex(0)

        # Update the title back to "Camera Feed"
        if hasattr(self, 'title_bar'):
            self.title_bar.title_label.setText("Camera Feed")

    def show_deployment_ui(self):
        """
        Switches to the deployment UI.
        """
        print("show_deployment_ui function called!")
        self.stacked_widget.setCurrentIndex(5)  # Switch to deployment UI (index 5)

    def closeEvent(self, event):
        """
        Handles application close event.
        """
        # Stop all ongoing processes
        self.stop_camera_feed()

        if self.capture_timer.isActive():
            self.capture_timer.stop()

        if self.capture_duration_timer.isActive():
            self.capture_duration_timer.stop()

        if self.status_update_timer.isActive():
            self.status_update_timer.stop()

        # Stop connectivity timer in title bar
        if hasattr(self.title_bar, 'connectivity_timer') and self.title_bar.connectivity_timer.isActive():
            self.title_bar.connectivity_timer.stop()

        # Handle pending uploads
        if len(self.pending_uploads) > 0:
            reply = QMessageBox.question(
                self, "Upload Captured Images",
                f"You have {len(self.pending_uploads)} captured images.\n\n"
                f"Would you like to upload them to Azure before closing?",
                QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.process_all_uploads()

        event.accept()

    def on_stack_changed(self, index):
        """
        Handles view switching in the stacked widget.
        Ensures proper camera resource management between views.
        """
        # Handle camera feed view
        camera_feed_index = self.stacked_widget.indexOf(self.camera_feed_widget)
        demo_feed_index = self.stacked_widget.indexOf(self.demo_feed_widget)

        # Store the previous camera state before switching
        was_camera_running = self.camera_capture and self.camera_capture.isOpened()

        # When leaving camera feed view
        if index != camera_feed_index:
            self.stored_window_state = self.windowState()
            # Store the camera state so we can restore it later
            self.camera_was_running_before_switch = was_camera_running
            if not self.capture_active:  # Only stop camera if not in continuous capture mode
                print("Camera Feed: Stopping camera before switching view")
                self.stop_camera_feed()

        # When returning to camera feed view
        elif index == camera_feed_index:
            print("Camera Feed: Returning to camera view")
            # Only restart camera if it was running before we switched AND we're not in continuous capture mode
            if not self.capture_active and hasattr(self, 'camera_was_running_before_switch') and self.camera_was_running_before_switch:
                print("Camera Feed: Restarting camera as it was running before switch")
                self.start_camera_feed()
            elif not self.capture_active:
                print("Camera Feed: Not restarting camera as it wasn't running before switch")
            self.setWindowState(self.stored_window_state)
            self.repaint()

        # Handle demo feed view
        if hasattr(self.demo_feed_ui, "is_active"):
            self.demo_feed_ui.is_active = (index == demo_feed_index)
            print(f"Demo Feed: Setting active state to {self.demo_feed_ui.is_active}")

        # When leaving demo feed view
        if index != demo_feed_index:
            if hasattr(self.demo_feed_ui, "pause_feed"):
                print("Demo Feed: Stopping camera before switching view")
                self.demo_feed_ui.pause_feed()
        # When entering demo feed view
        else:
            if hasattr(self.demo_feed_ui, "resume_feed"):
                print("Demo Feed: Starting camera in demo view")
                self.demo_feed_ui.resume_feed()

    def update_data_distribution_label(self, value):
        """
        Updates the data distribution slider label.
        """
        self.data_distribution_percentage = value
        self.data_distribution_label.setText(str(value) + "%")

    def pause_camera_feed(self):
        """
        Pauses the continuous capture and session timer.
        """
        if not self.capture_active:
            return

        print("pause_camera_feed called")
        self.feed_paused = True

        # Stop capture timer
        if self.capture_timer.isActive():
            self.capture_timer.stop()

        # Pause session duration timer and calculate remaining time
        if self.capture_duration_timer.isActive():
            self.capture_duration_timer.stop()

            # Calculate and store remaining time
            if self.capture_end_time:
                now = datetime.now()
                remaining = self.capture_end_time - now
                self.remaining_session_time = max(0, int(remaining.total_seconds()))
                print(f"Session paused with {self.remaining_session_time} seconds remaining")

        # Update UI
        self.btn_pause_feed.setVisible(False)
        self.btn_resume_feed.setVisible(True)
        self.capture_progress_label.setText(
            f"Capture session paused - {self.remaining_session_time}s remaining")

    def resume_camera_feed(self):
        """
        Resumes the continuous capture and session timer.
        """
        if not self.capture_active:
            return

        print("resume_camera_feed called")
        self.feed_paused = False

        # Restart capture timer
        if self.capture_active:
            ms_between_frames = int(1000 / self.capture_rate)
            self.capture_timer.start(ms_between_frames)

        # Restart session duration timer with remaining time
        if self.remaining_session_time and self.remaining_session_time > 0:
            print(f"Resuming session with {self.remaining_session_time} seconds remaining")
            self.capture_duration_timer.start(self.remaining_session_time * 1000)

            # Update end time based on remaining time
            self.capture_end_time = datetime.now() + timedelta(seconds=self.remaining_session_time)
            self.remaining_session_time = None  # Clear it

            # Update status
            self.update_status_display()

        # Update UI
        self.btn_resume_feed.setVisible(False)
        self.btn_pause_feed.setVisible(True)

    def get_selected_camera_index(self):
        """
        Gets the selected camera index from inspection_settings.json.
        Returns 0 if not found or if there's an error.
        """
        settings_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "inspection_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    if 'selected_camera_index' in settings:
                        return int(settings['selected_camera_index'])
                    print("No camera index in settings, using default")
            except Exception as e:
                print(f"Error reading camera index from settings: {e}")
        return 0  # Default to camera index 0

    def is_system_online(self):
        """
        Returns the current internet connectivity status.
        """
        if hasattr(self.title_bar, 'check_internet_connectivity'):
            return self.title_bar.check_internet_connectivity()
        return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraFeedUI()
    window.show()
    sys.exit(app.exec_())

    # Pass user information to child components after they're all created
    self.pass_user_info_to_child_components()

    # Initialize UI event tracking with proper user info
    try:
        self.ui_event_tracker = UIEventTracker(
            user_id=self.current_user_id,
            username=self.current_username
        )
        print(f"✓ UI Event Tracker initialized for user: {self.current_username}")
    except Exception as e:
        print(f"Warning: Could not initialize UI Event Tracker: {e}")
        self.ui_event_tracker = None
