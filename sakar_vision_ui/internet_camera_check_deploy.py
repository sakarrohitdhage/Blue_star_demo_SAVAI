#!/usr/bin/env python3
"""
SAKAR VISION AI - Internet Camera Check Deploy Module

OVERVIEW:
This module implements a comprehensive connectivity validation system for the Sakar Vision AI platform's deployment 
operations, serving as a critical pre-flight check interface that ensures optimal system readiness before proceeding 
with production deployment workflows. It provides a sophisticated dialog-based interface that combines real-time 
internet connectivity monitoring with advanced camera detection and validation capabilities, ensuring all hardware 
and network requirements are met for successful deployment operations with comprehensive troubleshooting guidance 
and automatic recovery mechanisms for industrial manufacturing environments.

KEY FUNCTIONALITY:
The system features advanced dual-connectivity validation with real-time internet connectivity testing through multiple 
protocols including DNS resolution and HTTP requests, sophisticated camera detection and validation with support for 
multiple camera backends including DirectShow compatibility for enhanced USB camera recognition, intelligent camera 
preview functionality with optimized frame processing and automatic aspect ratio maintenance for visual verification, 
and comprehensive real-time monitoring with automatic recovery mechanisms for both internet and camera connectivity. 
It includes sophisticated troubleshooting guidance with contextual error messages and step-by-step recovery instructions, 
persistent camera configuration management with automatic settings storage and user preference retention, advanced 
retry mechanisms with background monitoring to minimize user intervention, and seamless integration with deployment 
workflows through proper status reporting and configuration handoff to deployment interfaces.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced dialog-based architecture and professional visual styling, the module employs 
sophisticated timer-based monitoring systems with configurable intervals for both internet and camera connectivity 
validation, comprehensive OpenCV integration with multi-backend camera detection supporting standard and DirectShow 
APIs for maximum hardware compatibility, and intelligent preview management with proper resource acquisition and 
release patterns to prevent conflicts with deployment operations. The architecture features modular connectivity 
validation with separate monitoring threads for internet and camera systems, robust error handling with graceful 
degradation and automatic recovery mechanisms, persistent configuration management through JSON-based settings 
storage for camera preferences and system state, and comprehensive troubleshooting systems with contextual guidance 
and automatic diagnostic capabilities. The system includes advanced camera resource management with proper initialization 
patterns and conflict prevention mechanisms to ensure smooth handoff to deployment interfaces.
"""

import cv2
import os
import sys
import json
import traceback
import socket
import urllib.request
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QMessageBox, QDialog,
                             QProgressBar, QComboBox)

try:
    from utils import set_window_icon
except ImportError:
    def set_window_icon(widget):
        """Fallback function if utils is not available"""
        pass

# Define the path for storing settings
SETTINGS_FILE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "inspection_settings.json")


class ConnectivityCheckDialog(QDialog):
    """A dialog that checks for internet connectivity and available cameras for deployment."""

    def __init__(self, inspection_type="Deployment", parent=None):
        super().__init__(parent)
        self.inspection_type = inspection_type
        self.setWindowTitle("System Connectivity Check - Deployment")
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
        title_label = QLabel("System Connectivity Check - Deployment")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)

        # Subtitle
        subtitle_label = QLabel(
            f"Checking connectivity before proceeding to deployment operations")
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
• Check firewall settings that might block connections"""

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
• Restart the application if camera was working before
• Ensure camera permissions are granted to the application"""

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
                f"All systems are ready! You can proceed with deployment using Camera {self.selected_camera_index}.")
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
                "No cameras detected. Please connect a camera to use the deployment features.")
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
                f"All systems are ready! You can proceed with deployment using Camera {self.selected_camera_index}.")
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


# Test function to run the dialog standalone
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply global font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show the connectivity check dialog
    dialog = ConnectivityCheckDialog("Deployment Test")
    result = dialog.exec_()

    if result == QDialog.Accepted:
        print(f"Connectivity check passed! Selected camera: {dialog.selected_camera_index}")
    else:
        print("Connectivity check cancelled or failed")

    sys.exit(0)
