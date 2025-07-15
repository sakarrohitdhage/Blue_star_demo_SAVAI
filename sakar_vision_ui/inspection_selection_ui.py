#!/usr/bin/env python3
"""
SAKAR VISION AI - Inspection Selection UI Module

OVERVIEW:
This module implements the initial inspection type selection interface for the Sakar Vision AI platform, serving as 
the primary entry point and workflow router that enables users to choose between different inspection modalities 
before authentication and system access. It provides a sophisticated, visually appealing interface that combines 
inspection type selection with comprehensive system connectivity validation, ensuring optimal hardware readiness 
and camera availability before proceeding to specific inspection workflows with seamless user experience and 
robust error handling for industrial deployment environments.

KEY FUNCTIONALITY:
The system features an elegant dual-card interface for selecting between Fabric Inspection and Metal Sheet Inspection 
workflows with visual feedback and professional styling, comprehensive system connectivity validation through integrated 
ConnectivityCheckDialog with real-time internet and camera detection capabilities, intelligent camera management with 
automatic detection, preview functionality, and configurable camera index selection with settings persistence, and 
sophisticated inspection workflow routing that automatically configures authentication credentials and system parameters 
based on selected inspection type. It includes robust session state management with persistent user preference storage, 
advanced troubleshooting guidance with contextual error messages and recovery recommendations, seamless integration 
with authentication systems through automatic credential pre-filling and inspection-specific configuration, and 
comprehensive error handling with graceful degradation for system connectivity issues.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced custom widget architecture and professional visual styling, the module employs 
sophisticated background management with BlurredBackgroundWidget for enhanced visual appeal and brand consistency, 
modular card-based interface design (InspectionCard) with hover animations and shadow effects for interactive feedback, 
and comprehensive connectivity validation system (ConnectivityCheckDialog) with real-time monitoring capabilities for 
internet and camera availability. The architecture features intelligent camera detection with multi-backend support 
including DirectShow compatibility for enhanced USB camera recognition, real-time preview functionality with optimized 
frame processing and automatic aspect ratio maintenance, persistent configuration management through JSON-based settings 
storage for user preferences and camera selections, and seamless workflow integration with automatic launch of appropriate 
authentication interfaces based on inspection type selection. The system includes robust error handling with detailed 
logging and user-friendly error messages for production deployment scenarios.
"""

from utils import set_window_icon
import cv2
import os
import sys
import json
import traceback
import socket
import urllib.request
from datetime import datetime
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QFont, QPixmap, QColor, QImage, QBrush, QLinearGradient, QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect, QMessageBox, QDialog, QProgressBar, QComboBox

# Define the path for storing settings
SETTINGS_FILE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "inspection_settings.json")

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'
# Import the LoginUI class
try:
    from login_ui import LoginUI
except ImportError as e:
    print(f"Error importing LoginUI: {e}")
    print("Make sure login_ui.py is in the same directory")


class ConnectivityCheckDialog(QDialog):
    """A dialog that checks for internet connectivity and available cameras."""

    def __init__(self, inspection_type, parent=None):
        super().__init__(parent)
        self.inspection_type = inspection_type
        self.setWindowTitle("System Connectivity Check")
        self.setMinimumWidth(900)  # Increased width for more space
        self.setFixedHeight(750)  # Increased height for more content
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 10px;
            }
            QLabel {
                color: #333;
                font-size: 13px;
            }
            QPushButton {
                background-color: #ff914d;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff7730;
            }
            QPushButton#cancelButton {
                background-color: #e0e0e0;
                color: #333;
            }
            QPushButton#retryButton {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton#retryButton:hover {
                background-color: #45a049;
            }
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #f0f0f0;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #ff914d;
                border-radius: 5px;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QLabel#troubleshootLabel {
                color: #e74c3c;
                font-size: 12px;
                background-color: #fdf2f2;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Set icon
        set_window_icon(self)

        # Variables to track check status
        self.has_internet = False
        self.camera_available = False
        self.selected_camera_index = 0
        self.available_cameras = []
        self.all_checks_passed = False
        self.is_monitoring = False

        # Camera preview variables
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.current_camera = None

        # Monitoring timers
        self.internet_monitor_timer = QTimer()
        self.internet_monitor_timer.timeout.connect(self.monitor_internet)

        self.camera_monitor_timer = QTimer()
        self.camera_monitor_timer.timeout.connect(self.monitor_cameras)

        # Create UI
        self.init_ui()

        # Start checks automatically
        QTimer.singleShot(100, self.start_checks)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title_label = QLabel("System Connectivity Check")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)

        # Subtitle
        subtitle_label = QLabel(f"Checking connectivity before proceeding to inspection")
        subtitle_label.setStyleSheet("font-size: 14px; color: #555;")
        subtitle_label.setAlignment(Qt.AlignCenter)

        # Internet connectivity section
        internet_header = QLabel("Internet Connectivity:")
        internet_header.setStyleSheet("font-size: 15px; font-weight: bold;")

        self.internet_status = QLabel("Checking...")
        self.internet_status.setStyleSheet("color: #888;")

        # Internet troubleshooting label with better sizing
        self.internet_troubleshoot = QLabel()
        self.internet_troubleshoot.setObjectName("troubleshootLabel")
        self.internet_troubleshoot.setWordWrap(True)
        self.internet_troubleshoot.setMaximumHeight(150)  # Limit height
        self.internet_troubleshoot.hide()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(16)

        # Camera section header
        camera_header = QLabel("Camera Detection:")
        camera_header.setStyleSheet("font-size: 15px; font-weight: bold;")

        self.camera_status = QLabel("Checking...")
        self.camera_status.setStyleSheet("color: #888;")

        # Camera troubleshooting label with better sizing
        self.camera_troubleshoot = QLabel()
        self.camera_troubleshoot.setObjectName("troubleshootLabel")
        self.camera_troubleshoot.setWordWrap(True)
        self.camera_troubleshoot.setMaximumHeight(150)  # Limit height
        self.camera_troubleshoot.hide()

        # Create preview and camera selection layout
        preview_camera_layout = QHBoxLayout()

        # Left side - Camera selection
        camera_selection_layout = QVBoxLayout()
        camera_selection_layout.setSpacing(10)

        camera_select_label = QLabel("Select Camera:")
        camera_select_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.camera_combo = QComboBox()
        self.camera_combo.setFixedHeight(40)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)

        # Status message
        self.status_message = QLabel()
        self.status_message.setStyleSheet("color: green; font-weight: bold;")
        self.status_message.setWordWrap(True)

        # Monitoring status - smaller and less prominent
        self.monitoring_status = QLabel()
        self.monitoring_status.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        self.monitoring_status.setWordWrap(True)

        camera_selection_layout.addWidget(camera_select_label)
        camera_selection_layout.addWidget(self.camera_combo)
        camera_selection_layout.addWidget(self.status_message)
        camera_selection_layout.addWidget(self.monitoring_status)
        camera_selection_layout.addStretch()

        # Right side - Camera preview
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 10px;
                background-color: #f5f5f5;
                padding: 5px;
            }
        """)
        self.preview_label.setText("Camera Preview")

        preview_camera_layout.addLayout(camera_selection_layout)
        preview_camera_layout.addWidget(self.preview_label)

        # Buttons layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        self.retry_button = QPushButton("Retry Check")
        self.retry_button.setObjectName("retryButton")
        self.retry_button.clicked.connect(self.retry_checks)
        self.retry_button.setFixedHeight(40)
        self.retry_button.setFixedWidth(120)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setFixedHeight(40)
        self.cancel_button.setFixedWidth(120)

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.accept)
        self.continue_button.setEnabled(False)
        self.continue_button.setDefault(True)
        self.continue_button.setFixedHeight(40)
        self.continue_button.setFixedWidth(120)

        button_layout.addWidget(self.retry_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.continue_button)

        # Add all elements to main layout
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(10)
        layout.addWidget(internet_header)
        layout.addWidget(self.internet_status)
        layout.addWidget(self.internet_troubleshoot)
        layout.addWidget(self.progress_bar)
        layout.addSpacing(15)
        layout.addWidget(camera_header)
        layout.addWidget(self.camera_status)
        layout.addWidget(self.camera_troubleshoot)
        layout.addSpacing(15)
        layout.addLayout(preview_camera_layout)
        layout.addSpacing(10)
        layout.addLayout(button_layout)

    def update_preview(self):
        """Update the camera preview with better error handling."""
        if self.current_camera is not None and self.current_camera.isOpened():
            try:
                ret, frame = self.current_camera.read()
                if ret and frame is not None:
                    # Convert frame to RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w

                    # Convert to QImage and scale to fit preview label
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    label_size = self.preview_label.size()
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.preview_label.setPixmap(scaled_pixmap)
                else:
                    # Camera read failed but camera is still "open"
                    # Don't immediately disconnect, could be temporary
                    self.preview_label.setText("Camera feed interrupted...")
                    self.preview_label.setStyleSheet("color: orange;")
            except Exception as e:
                # Exception during read - camera might be disconnected
                print(f"Camera preview exception: {e}")
                self.preview_label.setText("Camera Disconnected")
                self.preview_label.setStyleSheet("color: red;")
                # Let the monitoring function handle the reconnection
        else:
            self.preview_label.setText("No Camera")
            self.preview_label.setStyleSheet("color: #888;")

    def stop_current_preview(self):
        """Stop the current camera preview."""
        self.preview_timer.stop()
        if self.current_camera is not None:
            self.current_camera.release()
            self.current_camera = None
        self.preview_label.setText("Camera Preview")

    def start_preview(self, camera_index):
        """Start preview for the selected camera with better initialization."""
        # Stop any existing preview
        self.stop_current_preview()

        try:
            # Try to open the selected camera
            self.current_camera = cv2.VideoCapture(camera_index)

            # Set camera properties for better performance
            if self.current_camera.isOpened():
                # Set buffer size to reduce latency
                self.current_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Test if camera actually works
                ret, frame = self.current_camera.read()
                if ret and frame is not None:
                    # Start the preview timer
                    self.preview_timer.start(33)  # Update every 33ms (approx. 30 fps)
                    print(f"Started preview for camera {camera_index}")
                else:
                    # Camera opened but can't read frames
                    self.current_camera.release()
                    self.current_camera = None
                    self.preview_label.setText("Camera not responding")
                    print(f"Camera {camera_index} opened but not responding")
            else:
                self.preview_label.setText("Failed to open camera")
                print(f"Failed to open camera {camera_index}")

        except Exception as e:
            print(f"Exception starting preview for camera {camera_index}: {e}")
            self.preview_label.setText("Camera initialization error")
            if self.current_camera:
                self.current_camera.release()
                self.current_camera = None

    def start_checks(self):
        """Start the connectivity checks."""
        self.progress_bar.setValue(0)
        self.status_message.setText("Checking system connectivity...")
        self.status_message.setStyleSheet("color: #555;")
        self.internet_status.setText("Checking...")
        self.internet_status.setStyleSheet("color: #888;")
        self.camera_status.setText("Checking...")
        self.camera_status.setStyleSheet("color: #888;")
        self.continue_button.setEnabled(False)
        self.hide_troubleshoot_messages()

        # Start the checks with a slight delay between them
        QTimer.singleShot(200, self.check_internet)

    def retry_checks(self):
        """Retry the connectivity checks."""
        self.stop_monitoring()
        self.stop_current_preview()
        self.hide_troubleshoot_messages()
        self.start_checks()

    def hide_troubleshoot_messages(self):
        """Hide all troubleshoot messages."""
        self.internet_troubleshoot.hide()
        self.camera_troubleshoot.hide()

    def show_internet_troubleshoot(self):
        """Show internet troubleshooting message with proper formatting."""
        troubleshoot_text = """🔧 Internet Connection Troubleshooting:

• Check if network cable is properly connected
• Verify Wi-Fi connection is active and working
• Try restarting your router/modem
• Check if other devices can access the internet
• Disable VPN if currently active
• Try using a different network if available
• Contact network administrator if in corporate environment
• Check firewall settings"""

        self.internet_troubleshoot.setText(troubleshoot_text.strip())
        self.internet_troubleshoot.show()

    def show_camera_troubleshoot(self):
        """Show camera troubleshooting message with proper formatting."""
        troubleshoot_text = """🔧 Camera Connection Troubleshooting:

• Check if camera is properly connected to USB port
• Try different USB ports (USB 2.0 and USB 3.0)
• For blue USB 3.0 ports: Try USB 2.0 port if camera isn't USB 3.0 compatible
• Verify camera drivers are installed
• Close other applications that might be using the camera
• Unplug and reconnect the camera
• Test camera with webcam test on web browser
• Check Device Manager for camera recognition
• Restart the application if camera was working before"""

        self.camera_troubleshoot.setText(troubleshoot_text.strip())
        self.camera_troubleshoot.show()

    def check_internet(self):
        """Check for internet connectivity."""
        self.progress_bar.setValue(40)

        try:
            # Try to connect to Google's DNS server
            socket.create_connection(("8.8.8.8", 53), timeout=3)

            # Try to fetch a URL as a double-check
            try:
                urllib.request.urlopen("http://www.google.com", timeout=3)
                self.has_internet = True
                self.internet_status.setText("✓ Internet connection available")
                self.internet_status.setStyleSheet("color: green; font-weight: bold;")
                self.internet_troubleshoot.hide()
            except:
                self.has_internet = True  # DNS works, so we have basic connectivity
                self.internet_status.setText("✓ Internet connection available (limited)")
                self.internet_status.setStyleSheet("color: green; font-weight: bold;")
                self.internet_troubleshoot.hide()
        except:
            self.has_internet = False
            # Show troubleshoot immediately for initial check
            self.internet_status.setText("✗ No internet connection")
            self.internet_status.setStyleSheet("color: red; font-weight: bold;")
            self.show_internet_troubleshoot()

        self.progress_bar.setValue(70)

        # Continue to camera check
        QTimer.singleShot(500, self.check_cameras)

    def check_cameras(self):
        """Check for available cameras."""
        self.available_cameras = []

        # Try to access cameras with extended range for USB 3.0 ports
        max_cameras = 10  # Increased to check more camera indices
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Camera works
                    self.available_cameras.append(i)
                    print(f"Camera {i} detected and working")
                cap.release()

        # Also try DirectShow backend on Windows for better compatibility
        try:
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and i not in self.available_cameras:
                        self.available_cameras.append(i)
                        print(f"Camera {i} detected with DirectShow backend")
                    cap.release()
        except:
            pass

        self.progress_bar.setValue(100)

        # Update status based on camera availability
        if self.available_cameras:
            self.camera_available = True

            # Populate the combo box
            self.camera_combo.clear()
            for idx in self.available_cameras:
                self.camera_combo.addItem(f"Camera {idx}")

            self.selected_camera_index = self.available_cameras[0]
            self.camera_status.setText(f"✓ {len(self.available_cameras)} camera(s) detected")
            self.camera_status.setStyleSheet("color: green; font-weight: bold;")
            self.status_message.setText(
                f"All systems are ready! You can proceed with the inspection using Camera {self.selected_camera_index}.")
            self.status_message.setStyleSheet("color: green; font-weight: bold;")
            self.camera_troubleshoot.hide()

            # Start preview for the first camera
            self.start_preview(self.selected_camera_index)
        else:
            self.camera_available = False
            self.camera_combo.clear()
            self.camera_combo.addItem("No cameras available")
            self.camera_status.setText("✗ No cameras detected")
            self.camera_status.setStyleSheet("color: red; font-weight: bold;")
            self.status_message.setText(
                "No cameras detected. Please connect a camera to use the inspection features.")
            self.status_message.setStyleSheet("color: red; font-weight: bold;")
            # Show troubleshoot immediately for initial check
            self.show_camera_troubleshoot()

        # Update button states
        self.continue_button.setEnabled(self.camera_available)

        # Set check status
        self.all_checks_passed = self.camera_available

        # Start monitoring after initial checks
        self.start_monitoring()

    def start_monitoring(self):
        """Start real-time monitoring of connectivity."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_status.setText("🔄 Monitoring active...")
            # Longer intervals to reduce interference with active camera
            self.internet_monitor_timer.start(10000)  # Check internet every 10 seconds
            self.camera_monitor_timer.start(5000)    # Check cameras every 5 seconds

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        self.monitoring_status.setText("")
        self.internet_monitor_timer.stop()
        self.camera_monitor_timer.stop()

    def monitor_internet(self):
        """Monitor internet connectivity in real-time with silent retries."""
        if not self.is_monitoring:
            return

        previous_status = self.has_internet

        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            self.has_internet = True
            if not previous_status:  # Was disconnected, now connected
                self.internet_status.setText("✓ Internet connection restored")
                self.internet_status.setStyleSheet("color: green; font-weight: bold;")
                self.internet_troubleshoot.hide()
                print("Internet connection restored")
        except:
            self.has_internet = False
            # Only show error after multiple failed attempts
            if previous_status:  # Was connected, now disconnected
                print("Internet connection lost - starting background retry")
                # Start a background retry counter
                self.internet_retry_count = 0
                self.start_internet_retry()

    def start_internet_retry(self):
        """Start background retry for internet with error display after multiple failures."""
        if not hasattr(self, 'internet_retry_timer'):
            self.internet_retry_timer = QTimer()
            self.internet_retry_timer.timeout.connect(self.retry_internet_background)

        if not hasattr(self, 'internet_retry_count'):
            self.internet_retry_count = 0

        self.internet_retry_timer.start(1000)  # Retry every second

    def retry_internet_background(self):
        """Background retry for internet connection."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=1)
            # Connection restored
            self.has_internet = True
            self.internet_status.setText("✓ Internet connection restored")
            self.internet_status.setStyleSheet("color: green; font-weight: bold;")
            self.internet_troubleshoot.hide()
            self.internet_retry_timer.stop()
            print("Internet connection restored after retry")
        except:
            self.internet_retry_count += 1
            # Show error only after 10 failed attempts (10 seconds)
            if self.internet_retry_count >= 10:
                self.internet_status.setText("✗ No internet connection")
                self.internet_status.setStyleSheet("color: red; font-weight: bold;")
                self.show_internet_troubleshoot()
                self.internet_retry_timer.stop()
                print("Internet connection failed after multiple retries")

    def monitor_cameras(self):
        """Monitor camera connectivity in real-time with silent retries."""
        if not self.is_monitoring:
            return

        # Don't interfere with the current camera if it's working
        if self.current_camera is not None and self.current_camera.isOpened():
            # Quick check if current camera is still working
            try:
                ret, frame = self.current_camera.read()
                if ret:
                    # Current camera is working fine, don't scan for others
                    # Just reset retry count since we have a working camera
                    if hasattr(self, 'camera_retry_count'):
                        self.camera_retry_count = 0
                    return
                else:
                    # Current camera failed, proceed with full scan
                    print("Current camera failed, starting full scan")
            except:
                # Current camera failed, proceed with full scan
                print("Current camera exception, starting full scan")

        # Only do full camera scan if no camera is currently working
        current_cameras = []
        max_cameras = 10

        # Silent background check - but skip the currently used camera index
        for i in range(max_cameras):
            # Skip the currently active camera to avoid interference
            if (hasattr(self, 'selected_camera_index') and
                i == self.selected_camera_index and
                self.current_camera is not None and
                    self.current_camera.isOpened()):
                # Add current camera to list since we know it's working
                current_cameras.append(i)
                continue

            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        current_cameras.append(i)
                    cap.release()
            except:
                pass  # Silent failure, keep trying

        # Try DirectShow backend silently (also skip current camera)
        try:
            for i in range(5):
                # Skip the currently active camera
                if (hasattr(self, 'selected_camera_index') and
                    i == self.selected_camera_index and
                    self.current_camera is not None and
                        self.current_camera.isOpened()):
                    continue

                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and i not in current_cameras:
                        current_cameras.append(i)
                    cap.release()
        except:
            pass

        # Only act if there's a significant change
        if set(current_cameras) != set(self.available_cameras):
            previous_had_cameras = len(self.available_cameras) > 0
            current_has_cameras = len(current_cameras) > 0

            # Update available cameras
            self.available_cameras = current_cameras

            if current_has_cameras:
                # Cameras found - immediate UI update
                self.camera_available = True

                # Only update UI if this is a new detection or recovery
                if not previous_had_cameras:
                    print(f"Camera(s) detected: {self.available_cameras}")

                    # Update combo box
                    self.camera_combo.clear()
                    for idx in self.available_cameras:
                        self.camera_combo.addItem(f"Camera {idx}")

                    # Try to maintain previous selection if still available
                    if hasattr(self, 'selected_camera_index') and self.selected_camera_index in self.available_cameras:
                        idx = self.available_cameras.index(self.selected_camera_index)
                        self.camera_combo.setCurrentIndex(idx)
                    else:
                        # Only start new preview if we don't have a working one
                        if self.current_camera is None or not self.current_camera.isOpened():
                            self.selected_camera_index = self.available_cameras[0]
                            self.start_preview(self.selected_camera_index)

                    self.camera_status.setText(
                        f"✓ {len(self.available_cameras)} camera(s) detected")
                    self.camera_status.setStyleSheet("color: green; font-weight: bold;")
                    self.status_message.setText(
                        f"Camera detected! Using Camera {self.selected_camera_index}.")
                    self.status_message.setStyleSheet("color: green; font-weight: bold;")
                    self.camera_troubleshoot.hide()
                    self.continue_button.setEnabled(True)

                # Reset retry count
                if hasattr(self, 'camera_retry_count'):
                    self.camera_retry_count = 0

            else:
                # No cameras found
                if previous_had_cameras:
                    # Camera was disconnected
                    print("Camera disconnected - starting silent retry")
                    self.camera_available = False
                    self.stop_current_preview()

                    # Start background retry counter
                    if not hasattr(self, 'camera_retry_count'):
                        self.camera_retry_count = 0

                    # Don't immediately show error, start silent counting
                    self.camera_retry_count = 1  # Start counting from 1
                else:
                    # Still no cameras, increment counter if it exists
                    if hasattr(self, 'camera_retry_count'):
                        self.camera_retry_count += 1

                        # Show error only after multiple failed attempts (45 seconds = 15 attempts * 3 seconds)
                        if self.camera_retry_count >= 15:
                            self.camera_combo.clear()
                            self.camera_combo.addItem("No cameras available")
                            self.camera_status.setText("✗ No cameras detected")
                            self.camera_status.setStyleSheet("color: red; font-weight: bold;")
                            self.status_message.setText(
                                "No cameras detected after multiple attempts!")
                            self.status_message.setStyleSheet("color: red; font-weight: bold;")
                            self.show_camera_troubleshoot()
                            self.continue_button.setEnabled(False)
                            print("Camera detection failed after multiple retries")
                    else:
                        # First time no cameras detected
                        self.camera_retry_count = 1
                        print("No cameras detected - starting background retry")

    def on_camera_selected(self, index):
        """Handle camera selection change."""
        if 0 <= index < len(self.available_cameras):
            self.selected_camera_index = self.available_cameras[index]
            self.status_message.setText(
                f"All systems are ready! You can proceed with the inspection using Camera {self.selected_camera_index}.")
            print(f"Selected camera index: {self.selected_camera_index}")

            # Start preview for selected camera
            self.start_preview(self.selected_camera_index)

            # Save the camera index to settings
            try:
                settings = {}
                if os.path.exists(SETTINGS_FILE):
                    try:
                        with open(SETTINGS_FILE, 'r') as f:
                            settings = json.load(f)
                    except json.JSONDecodeError:
                        settings = {}

                settings['selected_camera_index'] = self.selected_camera_index

                with open(SETTINGS_FILE, 'w') as f:
                    json.dump(settings, f, indent=4)
                print(f"Saved camera index {self.selected_camera_index} to settings")
            except Exception as e:
                print(f"Error saving camera index to settings: {e}")
                traceback.print_exc()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.stop_monitoring()
        self.stop_current_preview()
        super().closeEvent(event)


class InspectionCard(QFrame):
    """A styled card for inspection type selection."""

    def __init__(self, title, description, image_path, parent=None):
        super().__init__(parent)
        self.setObjectName("inspectionCard")
        self.setStyleSheet("""
            #inspectionCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 1px solid rgba(200, 200, 200, 0.8);
            }
            QLabel#cardTitle {
                font-size: 18pt;
                font-weight: bold;
                color: #333333;
            }
            QLabel#cardDescription {
                font-size: 11pt;
                color: #555555;
            }
        """)

        # Set fixed size
        self.setFixedSize(350, 400)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Image
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap.scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Fallback if image not found
            image_label.setText("[IMAGE]")
            image_label.setStyleSheet(
                "font-size: 18pt; color: #cccccc; background-color: #f0f0f0; border-radius: 10px;")
            image_label.setFixedSize(300, 200)

        # Title
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setAlignment(Qt.AlignCenter)

        # Description
        description_label = QLabel(description)
        description_label.setObjectName("cardDescription")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)

        # Button
        self.select_button = QPushButton("Select")
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #ff914d;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff914d;
            }
            QPushButton:pressed {
                background-color: #ff914d;
            }
        """)
        self.select_button.setFixedHeight(45)

        # Add animation on hover
        self._animation = QPropertyAnimation(self.select_button, b"minimumHeight")
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

        self.select_button.enterEvent = lambda e: self._button_enter_event(e)
        self.select_button.leaveEvent = lambda e: self._button_leave_event(e)

        # Add widgets to layout
        layout.addWidget(image_label)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addStretch(1)
        layout.addWidget(self.select_button)

    def _button_enter_event(self, event):
        self._animation.stop()
        self._animation.setStartValue(self.select_button.minimumHeight())
        self._animation.setEndValue(self.select_button.minimumHeight() + 3)
        self._animation.start()

    def _button_leave_event(self, event):
        self._animation.stop()
        self._animation.setStartValue(self.select_button.minimumHeight())
        self._animation.setEndValue(max(45, self.select_button.minimumHeight() - 3))
        self._animation.start()


class BlurredBackgroundWidget(QWidget):
    """Widget that displays a blurred background image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.background_image = None
        self.blur_radius = 10

        # Load background image - can be customized
        bg_path = "bg.png"
        fallback_bg_path = "bg.png"

        if os.path.exists(bg_path):
            self.background_image = QImage(bg_path)
        elif os.path.exists(fallback_bg_path):
            self.background_image = QImage(fallback_bg_path)

        # Set background colors as fallback in case no image is available
        self.background_gradient = QLinearGradient(0, 0, 0, self.height())
        self.background_gradient.setColorAt(0, QColor("#1e3c72"))
        self.background_gradient.setColorAt(1, QColor("#2a5298"))

    def paintEvent(self, event):
        """Paint the background with a semi-transparent overlay instead of blur."""
        try:
            # Create safe painter with try/except
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            if self.background_image and not self.background_image.isNull():
                # Scale image safely - ensure no division by zero or null size
                if self.width() > 0 and self.height() > 0:
                    scaled_image = self.background_image.scaled(
                        self.size(),
                        Qt.KeepAspectRatioByExpanding,
                        Qt.SmoothTransformation
                    )

                    # Additional null check after scaling
                    if scaled_image.isNull():
                        raise Exception("Scaled image is null")

                    # Center the image
                    x = max(0, (scaled_image.width() - self.width()) // 2)
                    y = max(0, (scaled_image.height() - self.height()) // 2)

                    # Draw the original image without blur
                    painter.drawImage(0, 0, scaled_image, x, y, self.width(), self.height())

                    # Apply a semi-transparent overlay to simulate blur effect
                    overlay_color = QColor(255, 255, 255, 180)  # White with 70% opacity
                    painter.fillRect(self.rect(), overlay_color)

            # Always draw a semi-transparent overlay for better text contrast
            painter.fillRect(self.rect(), QColor(0, 0, 0, 80))

            # If no image, use gradient as fallback
            if self.background_image is None or self.background_image.isNull():
                # Update gradient size to match current widget size
                self.background_gradient = QLinearGradient(0, 0, 0, self.height())
                self.background_gradient.setColorAt(0, QColor("#1e3c72"))
                self.background_gradient.setColorAt(1, QColor("#2a5298"))

                painter.setBrush(QBrush(self.background_gradient))
                painter.drawRect(self.rect())

        except Exception as e:
            # Last resort - draw a solid color if everything else fails
            print(f"Error painting background: {e}")
            if painter.isActive():
                painter.fillRect(self.rect(), QColor("#1e3c72"))


class InspectionSelectionUI(QWidget):
    """
    Initial screen for selecting the type of inspection (Fabric or Metal Sheet)
    before proceeding to the login screen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAKAR VISION AI - Select Inspection Type")
        set_window_icon(self)
        self.setMinimumSize(800, 600)
        # Remove the window title bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Store references to other screens
        self.login_ui = None

        # Default selection (will be overridden by loaded settings if available)
        self.last_selection = None

        # Load any previous selection
        self.load_settings()

        # Initialize UI
        self.init_ui()

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12pt;
            }
            QLabel#headerLabel {
                font-size: 22pt;
                font-weight: bold;
                color: #ffffff;
            }
            QLabel#subHeaderLabel {
                font-size: 14pt;
                color: #ffffff;
            }
            QLabel#lastSelectionLabel {
                font-size: 12pt;
                color: #ffe0cc;
                font-style: italic;
            }
        """)

    def load_settings(self):
        """Load the previous inspection selection from settings file."""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    if 'last_selection' in settings:
                        self.last_selection = settings['last_selection']
                        print(f"Loaded last selection: {self.last_selection}")
        except Exception as e:
            print(f"Error loading settings: {e}")
            traceback.print_exc()

    def save_selection(self, selection_type):
        """Save the user's inspection type selection."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(SETTINGS_FILE)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Create or update settings
            settings = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, 'r') as f:
                        settings = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, start with empty settings
                    settings = {}

            # Update the settings
            settings['last_selection'] = selection_type
            settings['last_selection_time'] = datetime.now().isoformat()

            # Save to file
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)

            print(f"Saved selection: {selection_type}")
            return True
        except Exception as e:
            print(f"Error saving selection: {e}")
            traceback.print_exc()
            return False

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create blurred background widget
        self.background_widget = BlurredBackgroundWidget(self)

        # Content layout
        content_widget = QWidget()
        content_widget.setAttribute(Qt.WA_TranslucentBackground)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)

        # Header
        header_label = QLabel("SAKAR VISION AI")
        header_label.setObjectName("headerLabel")
        header_label.setAlignment(Qt.AlignCenter)

        # Subheader
        subheader_label = QLabel("Please select the type of inspection you want to perform")
        subheader_label.setObjectName("subHeaderLabel")
        subheader_label.setAlignment(Qt.AlignCenter)

        # Add headers to layout
        content_layout.addWidget(header_label)
        content_layout.addWidget(subheader_label)

        # Last selection label (if available)
        if self.last_selection:
            last_selection_label = QLabel(f"Last selected: {self.last_selection}")
            last_selection_label.setObjectName("lastSelectionLabel")
            last_selection_label.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(last_selection_label)

        content_layout.addSpacing(20)

        # Cards layout
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(30)
        cards_layout.setAlignment(Qt.AlignCenter)

        # Fabric inspection card
        self.fabric_card = InspectionCard(
            "Fabric Inspection",
            "Detect defects in fabric materials including weaving errors, stains, and texture inconsistencies.",
            "fabric.jpeg"  # Replace with actual image path
        )
        self.fabric_card.select_button.clicked.connect(self.select_fabric_inspection)

        # Metal sheet inspection card
        self.metal_card = InspectionCard(
            "Metal Sheet Inspection",
            "Detect defects in metal sheets including cracks, corrosion, dents, and surface irregularities.",
            "metal.jpeg"  # Replace with actual image path
        )
        self.metal_card.select_button.clicked.connect(self.select_metal_inspection)

        # Add cards to layout
        cards_layout.addWidget(self.fabric_card)
        cards_layout.addWidget(self.metal_card)

        # Add cards layout to main layout
        content_layout.addLayout(cards_layout)
        content_layout.addStretch(1)

        # Version info
        version_label = QLabel("Version 1.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #ffffff; font-size: 10pt;")
        content_layout.addWidget(version_label)

        # Make the background fill the entire widget
        self.background_widget.setGeometry(0, 0, self.width(), self.height())

        # Add widgets to main layout
        main_layout.addWidget(self.background_widget)
        main_layout.addWidget(content_widget)

    def select_fabric_inspection(self):
        """Handle fabric inspection selection."""
        try:
            # Save the selection
            self.save_selection("Fabric Inspection")

            # Show connectivity check dialog first
            connectivity_dialog = ConnectivityCheckDialog("Fabric Inspection", self)
            result = connectivity_dialog.exec_()

            if result == QDialog.Accepted:
                # Get the selected camera index
                selected_camera = connectivity_dialog.selected_camera_index

                # Launch login UI with fabric credentials
                self.login_ui = LoginUI()
                self.login_ui.is_fabric_inspection = True
                self.login_ui.username_edit.setText("sakarrobotics")  # Pre-fill username

                # Store the selected camera index in login_ui for later use
                if hasattr(self.login_ui, 'selected_camera_index'):
                    self.login_ui.selected_camera_index = selected_camera

                # Show the login window
                self.login_ui.show()
                self.login_ui.showMaximized()  # First show then maximize

                # Hide this window immediately for smooth transition
                self.hide()

                # Debug info
                print(
                    f"Fabric inspection selected, launching login UI with sakarrobotics (Camera: {selected_camera})")
            else:
                print("Connectivity check cancelled or failed")

        except Exception as e:
            print(f"Error launching fabric login UI: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not start login window: {str(e)}")

    def select_metal_inspection(self):
        """Handle metal sheet inspection selection."""
        try:
            # Save the selection
            self.save_selection("Metal Sheet Inspection")

            # Show connectivity check dialog first
            connectivity_dialog = ConnectivityCheckDialog("Metal Sheet Inspection", self)
            result = connectivity_dialog.exec_()

            if result == QDialog.Accepted:
                # Get the selected camera index
                selected_camera = connectivity_dialog.selected_camera_index

                # Launch login UI with metal sheet credentials
                self.login_ui = LoginUI()
                self.login_ui.is_fabric_inspection = False
                self.login_ui.username_edit.setText("admin")  # Pre-fill username

                # Store the selected camera index in login_ui for later use
                if hasattr(self.login_ui, 'selected_camera_index'):
                    self.login_ui.selected_camera_index = selected_camera

                # Show the login window
                self.login_ui.show()
                self.login_ui.showMaximized()  # First show then maximize

                # Hide this window immediately for smooth transition
                self.hide()

                # Debug info
                print(
                    f"Metal sheet inspection selected, launching login UI with admin (Camera: {selected_camera})")
            else:
                print("Connectivity check cancelled or failed")

        except Exception as e:
            print(f"Error launching metal sheet login UI: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not start login window: {str(e)}")

    def resizeEvent(self, event):
        """Handle resize events for the background."""
        self.background_widget.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def showEvent(self, event):
        """Override show event to maximize the window."""
        super().showEvent(event)
        self.showMaximized()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply global font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Print current directory and check for login_ui.py
    print(f"Current directory: {os.getcwd()}")
    if os.path.exists("login_ui.py"):
        print("login_ui.py found in current directory")
    else:
        print("WARNING: login_ui.py not found in current directory!")

    # Create and show selection window
    selection_window = InspectionSelectionUI()
    selection_window.showMaximized()

    sys.exit(app.exec_())
