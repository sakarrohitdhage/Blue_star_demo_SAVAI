#!/usr/bin/env python3
"""
SAKAR VISION AI - Demo Feed UI Module

OVERVIEW:
This module implements a comprehensive demonstration and training interface for the Sakar Vision AI platform, serving as 
an interactive laboratory for real-time image classification, dataset creation, and model training workflows. It provides 
a professional, modern UI that combines live camera feeds with intelligent data collection capabilities, enabling users to 
capture and categorize images for defective and non-defective samples while simultaneously providing model training 
functionality with real-time progress monitoring and dataset management tools.

KEY FUNCTIONALITY:
The system features a sophisticated camera management system with configurable positioning and sizing capabilities, 
automatic and manual image capture modes with intelligent file naming and organization, advanced dataset splitting 
functionality with visual range sliders for train/validation/test ratios, and integrated PyTorch-based model training 
with real-time progress tracking and GPU acceleration support. It includes modern status card displays for real-time 
statistics tracking, continuous capture modes for both defective and non-defective samples with automatic interval-based 
saving, comprehensive dataset management with folder structure creation and file organization, and seamless navigation 
between different application modules through QStackedWidget integration.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced custom widget styling and a modular component-based architecture, the module employs 
OpenCV for camera operations with configurable camera index selection and frame processing, comprehensive PyTorch integration 
for deep learning model training with MobileNetV2 architecture and transfer learning capabilities, and sophisticated timer 
management for camera feeds, capture intervals, and UI updates. The architecture features custom UI components (StatusCard, 
ModernRangeSlider, TrainingProgressDialog) with professional styling and shadow effects, intelligent camera resource 
management with proper acquisition and release patterns to prevent conflicts, dynamic UI state management that adapts 
based on camera availability and capture modes, and robust error handling with user-friendly notifications and automatic 
recovery mechanisms for camera and training operations.
"""
import shutil
import os
import sys
import cv2
import random
import time
import json
import logging
from PyQt5.QtCore import pyqtSignal, QPoint, QRect, Qt, QTimer, QPropertyAnimation, QEasingCurve, QThread
from PyQt5.QtGui import QFont, QIcon, QImage, QPainter, QPixmap, QPalette, QLinearGradient, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QStackedWidget, QVBoxLayout, QWidget, QFrame,
                             QGridLayout, QSpinBox, QSizePolicy, QGraphicsDropShadowEffect, QDialog, QProgressBar)

# PyTorch imports for training functionality
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    PYTORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Training functionality will be disabled.")

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'


class TrainingProgressDialog(QDialog):
    """Custom dialog to show training progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training In Progress")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)  # Remove close button
        self.setMinimumSize(400, 150)
        self.setModal(True)

        # Create layout
        layout = QVBoxLayout(self)

        # Add label
        self.status_label = QLabel("Training model, please wait...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.status_label)

        # Add a progress bar (indeterminate)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        # Add epoch status label
        self.epoch_label = QLabel("Initializing...")
        self.epoch_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.epoch_label)

        # Set layout
        self.setLayout(layout)

    def update_status(self, text):
        """Update the status text of the dialog."""
        self.epoch_label.setText(text)
        QApplication.processEvents()


class StatusCard(QFrame):
    """Modern status card widget"""

    def __init__(self, title, value="0", color_scheme="blue", parent=None):
        super().__init__(parent)
        self.title = title
        self.value_text = value
        self.color_scheme = color_scheme
        self.setupUI()

    def setupUI(self):
        self.setFixedSize(346, 150)
        self.setFrameStyle(QFrame.NoFrame)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(20, 15, 20, 15)

        # Title label
        self.title_label = QLabel(self.title.upper())
        self.title_label.setAlignment(Qt.AlignCenter)

        # Value label
        self.value_label = QLabel(self.value_text)
        self.value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

        self.setStyleSheet(self.get_card_style())

    def get_card_style(self):
        colors = {
            "green": {"border": "#10b981", "value": "#059669", "bg": "#f0fdf4"},
            "red": {"border": "#dc2626", "value": "#b91c1c", "bg": "#fef2f2"},
            "blue": {"border": "#3b82f6", "value": "#1e40af", "bg": "#eff6ff"},
            "orange": {"border": "#f59e0b", "value": "#d97706", "bg": "#fffbeb"}
        }

        scheme = colors.get(self.color_scheme, colors["blue"])

        return f"""
            StatusCard {{
                background-color: white;
                border-left: 5px solid {scheme["border"]};
                border-radius: 12px;
            }}
            StatusCard:hover {{
                background-color: {scheme["bg"]};
            }}
            QLabel {{
                background-color: transparent;
                border: none;
            }}
            QLabel:first-child {{
                font-size: 11px;
                font-weight: 600;
                color: #6b7280;
                letter-spacing: 0.5px;
            }}
            QLabel:last-child {{
                font-size: 28px;
                font-weight: 700;
                color: {scheme["value"]};
            }}
        """

    def update_value(self, value):
        self.value_text = str(value)
        self.value_label.setText(self.value_text)


class ModernRangeSlider(QWidget):
    """Enhanced dual-handle range slider"""
    lowerValueChanged = pyqtSignal(int)
    upperValueChanged = pyqtSignal(int)
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self._minimum = 0
        self._maximum = 100
        self._lower = 70   # Train boundary
        self._upper = 90   # Valid boundary
        self._handle_width = 20
        self._handle_height = 20
        self.setMinimumHeight(60)
        self.setMouseTracking(True)
        self._moving_handle = None
        self._track_height = 12

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def lowerValue(self):
        return self._lower

    def upperValue(self):
        return self._upper

    def setLowerValue(self, value):
        value = max(self._minimum + 10, min(value, self._upper - 10))
        if value != self._lower:
            self._lower = value
            self.lowerValueChanged.emit(value)
            self.rangeChanged.emit(self._lower, self._upper)
            self.update()

    def setUpperValue(self, value):
        value = max(self._lower + 10, min(value, self._maximum - 10))
        if value != self._upper:
            self._upper = value
            self.upperValueChanged.emit(value)
            self.rangeChanged.emit(self._lower, self._upper)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate positions
        rect = self.rect()
        track_rect = QRect(
            self._handle_width // 2,
            (rect.height() - self._track_height) // 2,
            rect.width() - self._handle_width,
            self._track_height
        )

        span = self._maximum - self._minimum
        if span == 0:
            span = 1

        lower_pos = track_rect.left() + ((self._lower - self._minimum) / span) * track_rect.width()
        upper_pos = track_rect.left() + ((self._upper - self._minimum) / span) * track_rect.width()

        # Draw track background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#e5e7eb"))
        painter.drawRoundedRect(track_rect, self._track_height // 2, self._track_height // 2)

        # Draw train section (blue)
        train_rect = QRect(track_rect.left(), track_rect.top(),
                           int(lower_pos - track_rect.left()), track_rect.height())
        gradient = QLinearGradient(0, 0, 1, 0)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0, QColor("#3b82f6"))
        gradient.setColorAt(1, QColor("#2563eb"))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(train_rect, self._track_height // 2, self._track_height // 2)

        # Draw valid section (green)
        valid_rect = QRect(int(lower_pos), track_rect.top(),
                           int(upper_pos - lower_pos), track_rect.height())
        gradient = QLinearGradient(0, 0, 1, 0)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0, QColor("#10b981"))
        gradient.setColorAt(1, QColor("#059669"))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(valid_rect, self._track_height // 2, self._track_height // 2)

        # Draw test section (orange)
        test_rect = QRect(int(upper_pos), track_rect.top(),
                          track_rect.right() - int(upper_pos), track_rect.height())
        gradient = QLinearGradient(0, 0, 1, 0)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0, QColor("#f59e0b"))
        gradient.setColorAt(1, QColor("#d97706"))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(test_rect, self._track_height // 2, self._track_height // 2)

        # Draw handles
        handle_y = rect.height() // 2

        # Lower handle (train/valid boundary)
        lower_handle_rect = QRect(int(lower_pos - self._handle_width // 2),
                                  handle_y - self._handle_height // 2,
                                  self._handle_width, self._handle_height)
        painter.setBrush(QColor("white"))
        painter.setPen(QColor("#3b82f6"))
        painter.drawEllipse(lower_handle_rect)

        # Upper handle (valid/test boundary)
        upper_handle_rect = QRect(int(upper_pos - self._handle_width // 2),
                                  handle_y - self._handle_height // 2,
                                  self._handle_width, self._handle_height)
        painter.setBrush(QColor("white"))
        painter.setPen(QColor("#10b981"))
        painter.drawEllipse(upper_handle_rect)

    def mousePressEvent(self, event):
        pos = event.pos().x()
        rect = self.rect()
        track_rect = QRect(self._handle_width // 2, 0,
                           rect.width() - self._handle_width, rect.height())

        span = self._maximum - self._minimum
        lower_pos = track_rect.left() + ((self._lower - self._minimum) / span) * track_rect.width()
        upper_pos = track_rect.left() + ((self._upper - self._minimum) / span) * track_rect.width()

        if abs(pos - lower_pos) < abs(pos - upper_pos):
            self._moving_handle = "lower"
        else:
            self._moving_handle = "upper"
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self._moving_handle is None:
            return

        pos = event.pos().x()
        rect = self.rect()
        track_rect = QRect(self._handle_width // 2, 0,
                           rect.width() - self._handle_width, rect.height())

        ratio = (pos - track_rect.left()) / track_rect.width()
        value = self._minimum + ratio * (self._maximum - self._minimum)
        value = int(round(value))

        if self._moving_handle == "lower":
            self.setLowerValue(value)
        elif self._moving_handle == "upper":
            self.setUpperValue(value)

    def mouseReleaseEvent(self, event):
        self._moving_handle = None


class ModernDemoFeedUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_feed_ui = parent
        self.base_dir = None
        self.non_defective_dir = None
        self.defective_dir = None
        self.non_defective_count = 0
        self.defective_count = 0
        self.capture = None
        self.is_active = True  # Set to True by default so camera starts
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.reset_status)

        # Continuous capture timers and states
        self.defective_capture_timer = QTimer()
        self.defective_capture_timer.timeout.connect(self.capture_defective_image)
        self.non_defective_capture_timer = QTimer()
        self.non_defective_capture_timer.timeout.connect(self.capture_non_defective_image)

        # Capture states
        self.is_capturing_defective = False
        self.is_capturing_non_defective = False
        self.capture_interval = 1000  # 1 second in milliseconds

        self.selected_camera_index = self.get_selected_camera_index()

        # Store reference to camera section frame for manipulation
        self.camera_section_frame = None

        # Training related attributes
        self.model = None
        self.class_names = []
        self.img_height, self.img_width = 224, 224
        self.training_in_progress = False
        self.progress_dialog = None

        # Navigation Buttons (from original code)
        self.image_classi_button = QPushButton("Check Result →", self)
        self.image_classi_button.clicked.connect(self.showImageClassiUI)

        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('SAKAR VISION AI - Demo Feed')
        self.setGeometry(100, 100, 1200, 800)

        # Set main window style
        self.setStyleSheet("""
            QWidget {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Status cards
        status_layout = self.create_status_cards()
        main_layout.addLayout(status_layout)

        # Main content area
        content_layout = self.create_main_content()
        main_layout.addLayout(content_layout)

        self.setLayout(main_layout)

    def create_status_cards(self):
        layout = QHBoxLayout()
        layout.setSpacing(50)

        # Add stretch before cards to push them to the right - must be an integer
        layout.addStretch(1)

        self.camera_card = StatusCard("Camera Feed", "Inactive", "blue")
        self.non_defective_card = StatusCard("Non-defective", "0", "green")
        self.defective_card = StatusCard("Defective", "0", "red")
        self.total_card = StatusCard("Total Images", "0", "orange")

        layout.addWidget(self.camera_card)
        layout.addWidget(self.non_defective_card)
        layout.addWidget(self.defective_card)
        layout.addWidget(self.total_card)
        layout.addStretch(1)  # Add stretch after cards to fill space
        return layout

    def create_main_content(self):
        layout = QHBoxLayout()
        layout.setSpacing(10)

        # Camera section
        camera_section = self.create_camera_section()
        layout.addWidget(camera_section, 7)

        # Controls section
        controls_section = self.create_controls_section()
        layout.addWidget(controls_section, 3)
        layout.setContentsMargins(20, 0, 50, 0)

        return layout

    def create_camera_section(self):
        frame = QFrame()
        # Store reference to the frame for later manipulation
        self.camera_section_frame = frame

        # Set initial size and position - now adjustable
        frame.setFixedSize(1008, 688)
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 2px dashed #e5e7eb;
            }
            QFrame:hover {
                border-color: #ff914d;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Create a container widget for the initial UI elements
        self.initial_ui_container = QWidget()
        initial_layout = QVBoxLayout()
        initial_layout.setSpacing(20)

        # Folder icon (using text for simplicity)
        self.icon_label = QLabel("📁")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(QFont("Segoe UI", 48))
        self.icon_label.setStyleSheet("color: #9ca3af; background: transparent; border: none;")

        # Title text
        self.title_label = QLabel("Select a folder to start camera feed")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.title_label.setStyleSheet("color: #374151; background: transparent; border: none;")

        # Description
        self.desc_label = QLabel("Choose a directory where captured images will be saved")
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setFont(QFont("Segoe UI", 14))
        self.desc_label.setStyleSheet("color: #6b7280; background: transparent; border: none;")

        # Choose folder button
        self.choose_dir_btn = QPushButton('Choose Folder')
        self.choose_dir_btn.clicked.connect(self.choose_directory)
        self.choose_dir_btn.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.choose_dir_btn.setFixedSize(200, 50)
        self.choose_dir_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3b82f6, stop:1 #1d4ed8);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2563eb, stop:1 #1e40af);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1d4ed8, stop:1 #1e3a8a);
            }
        """)

        initial_layout.addWidget(self.icon_label)
        initial_layout.addWidget(self.title_label)
        initial_layout.addWidget(self.desc_label)
        initial_layout.addWidget(self.choose_dir_btn, alignment=Qt.AlignCenter)
        initial_layout.addStretch()

        self.initial_ui_container.setLayout(initial_layout)

        # Camera feed label - initially hidden
        self.feed_label = QLabel()
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setFixedSize(800, 500)
        self.feed_label.setStyleSheet("""
            QLabel {
                background: black;
                border: none;
                border-radius: 10px;
            }
        """)
        self.feed_label.hide()  # Initially hidden

        layout.addWidget(self.initial_ui_container)
        layout.addWidget(self.feed_label, alignment=Qt.AlignCenter)

        frame.setLayout(layout)
        return frame

    # NEW METHODS FOR ADJUSTING CAMERA CONTAINER
    def set_camera_container_size(self, width, height):
        """Set the size of the camera container frame"""
        if self.camera_section_frame:
            self.camera_section_frame.setFixedSize(width, height)
            print(f"Camera container resized to: {width}x{height}")

            # Adjust feed label size proportionally
            feed_width = int(width * 0.8)  # 80% of container width
            feed_height = int(height * 0.7)  # 70% of container height
            self.feed_label.setFixedSize(feed_width, feed_height)
            print(f"Feed label resized to: {feed_width}x{feed_height}")

    def set_camera_container_position(self, x, y):
        """Set the position of the camera container frame"""
        if self.camera_section_frame:
            # Remove from layout to enable absolute positioning
            self.camera_section_frame.setParent(self)
            self.camera_section_frame.move(x, y)
            self.camera_section_frame.show()
            print(f"Camera container moved to position: ({x}, {y})")

    def adjust_camera_container(self, x=None, y=None, width=None, height=None):
        """Convenience method to adjust both position and size of camera container"""
        if x is not None and y is not None:
            self.set_camera_container_position(x, y)

        if width is not None and height is not None:
            self.set_camera_container_size(width, height)

    def get_camera_container_geometry(self):
        """Get current geometry of the camera container"""
        if self.camera_section_frame:
            geometry = self.camera_section_frame.geometry()
            return {
                'x': geometry.x(),
                'y': geometry.y(),
                'width': geometry.width(),
                'height': geometry.height()
            }
        return None

    def scale_camera_container(self, scale_factor):
        """Scale the camera container by a factor (e.g., 1.2 for 20% larger)"""
        if self.camera_section_frame:
            current_size = self.camera_section_frame.size()
            new_width = int(current_size.width() * scale_factor)
            new_height = int(current_size.height() * scale_factor)
            self.set_camera_container_size(new_width, new_height)

    # EXAMPLE USAGE METHODS - You can call these to test the functionality
    def make_container_larger(self):
        """Example: Make container 20% larger"""
        self.scale_camera_container(1.2)

    def make_container_smaller(self):
        """Example: Make container 20% smaller"""
        self.scale_camera_container(0.8)

    def reset_container_size(self):
        """Example: Reset to original size"""
        self.set_camera_container_size(1008, 688)

    def set_custom_container_size(self):
        """Example: Set custom size - modify these values as needed"""
        # You can change these values to whatever you want
        custom_width = 1000   # Change this value
        custom_height = 800   # Change this value
        self.set_camera_container_size(custom_width, custom_height)

    def set_my_container_position(self):
        """Set container to my preferred position (40, 190)"""
        self.set_camera_container_position(300, 190)

    def create_controls_section(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Image classification panel
        classification_panel = self.create_classification_panel()
        layout.addWidget(classification_panel)

        # Dataset split panel
        split_panel = self.create_split_panel()
        layout.addWidget(split_panel)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_classification_panel(self):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
            }
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 30))
        frame.setGraphicsEffect(shadow)

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Title
        title = QLabel("Image Classification")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #374151; background: transparent; border: none;")

        # Description
        desc = QLabel("Click to save the current image to the respective folder:")
        desc.setFont(QFont("Segoe UI", 12))
        desc.setStyleSheet("color: #6b7280; background: transparent; border: none;")

        # Buttons
        self.good_btn = QPushButton('Save as Non-Defective')
        self.good_btn.clicked.connect(self.save_good)
        self.good_btn.setEnabled(False)
        self.good_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.good_btn.setFixedHeight(50)
        self.good_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #059669, stop:1 #047857);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #047857, stop:1 #065f46);
            }
            QPushButton:disabled {
                background-color: #e8f5e9;
                color: #808080;
            }
        """)

        self.bad_btn = QPushButton('Save as Defective')
        self.bad_btn.clicked.connect(self.save_bad)
        self.bad_btn.setEnabled(False)
        self.bad_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.bad_btn.setFixedHeight(50)
        self.bad_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #dc2626, stop:1 #b91c1c);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #b91c1c, stop:1 #991b1b);
            }
            QPushButton:disabled {
                background-color: #ffebee;
                color: #808080;
            }
        """)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(self.bad_btn)
        layout.addWidget(self.good_btn)

        frame.setLayout(layout)
        return frame

    def create_split_panel(self):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
            }
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 30))
        frame.setGraphicsEffect(shadow)

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)

        # Title
        title = QLabel("Dataset Split Configuration")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #374151; background: transparent; border: none;")

        # Percentage cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)

        self.train_card = QFrame()
        self.train_card.setFixedSize(165, 80)
        self.train_card.setStyleSheet("""
            QFrame {
                background-color: #eff6ff;
                border-left: 4px solid #3b82f6;
                border-radius: 8px;
            }
        """)
        train_layout = QVBoxLayout()
        train_layout.setContentsMargins(8, 8, 8, 8)
        train_layout.addStretch()
        train_label = QLabel("TRAIN")
        train_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        train_label.setStyleSheet("color: #6b7280; background: transparent; border: none;")
        train_label.setAlignment(Qt.AlignCenter)
        self.train_value = QLabel("70%")
        self.train_value.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.train_value.setStyleSheet("color: #3b82f6; background: transparent; border: none;")
        self.train_value.setAlignment(Qt.AlignCenter)
        train_layout.addWidget(train_label)
        train_layout.addWidget(self.train_value)
        train_layout.addStretch()
        self.train_card.setLayout(train_layout)

        self.valid_card = QFrame()
        self.valid_card.setFixedSize(165, 80)
        self.valid_card.setStyleSheet("""
            QFrame {
                background-color: #f0fdf4;
                border-left: 4px solid #10b981;
                border-radius: 8px;
            }
        """)
        valid_layout = QVBoxLayout()
        valid_layout.setContentsMargins(8, 8, 8, 8)
        valid_layout.addStretch()
        valid_label = QLabel("VALID")
        valid_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        valid_label.setStyleSheet("color: #6b7280; background: transparent; border: none;")
        valid_label.setAlignment(Qt.AlignCenter)
        self.valid_value = QLabel("20%")
        self.valid_value.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.valid_value.setStyleSheet("color: #10b981; background: transparent; border: none;")
        self.valid_value.setAlignment(Qt.AlignCenter)
        valid_layout.addWidget(valid_label)
        valid_layout.addWidget(self.valid_value)
        valid_layout.addStretch()
        self.valid_card.setLayout(valid_layout)

        self.test_card = QFrame()
        self.test_card.setFixedSize(165, 80)
        self.test_card.setStyleSheet("""
            QFrame {
                background-color: #fffbeb;
                border-left: 4px solid #f59e0b;
                border-radius: 8px;
            }
        """)
        test_layout = QVBoxLayout()
        test_layout.setContentsMargins(8, 8, 8, 8)
        test_layout.addStretch(1)
        test_label = QLabel("TEST")
        test_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        test_label.setStyleSheet("color: #6b7280; background: transparent; border: none;")
        test_label.setAlignment(Qt.AlignCenter)
        self.test_value = QLabel("10%")
        self.test_value.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.test_value.setStyleSheet("color: #f59e0b; background: transparent; border: none;")
        self.test_value.setAlignment(Qt.AlignCenter)
        test_layout.addWidget(test_label)
        test_layout.addWidget(self.test_value)
        test_layout.addStretch(1)
        self.test_card.setLayout(test_layout)

        cards_layout.addWidget(self.train_card)
        cards_layout.addWidget(self.valid_card)
        cards_layout.addWidget(self.test_card)
        cards_layout.addStretch()

        # Range slider
        self.range_slider = ModernRangeSlider(Qt.Horizontal, self)
        self.range_slider.lowerValueChanged.connect(self.update_ratio_labels)
        self.range_slider.upperValueChanged.connect(self.update_ratio_labels)

        # Input controls
        input_layout = QHBoxLayout()
        input_layout.setSpacing(15)

        train_input_frame = QFrame()
        train_input_frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border-radius: 8px;
                border: 2px solid transparent;
            }
            QFrame:focus-within {
                border-color: #3b82f6;
            }
        """)
        train_input_layout = QHBoxLayout()
        train_input_layout.setContentsMargins(12, 8, 12, 8)
        train_input_label = QLabel("Train:")
        train_input_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        train_input_label.setStyleSheet("color: #374151; background: transparent; border: none;")
        self.train_input = QSpinBox()
        self.train_input.setRange(10, 80)
        self.train_input.setValue(70)
        self.train_input.setSuffix("%")
        self.train_input.valueChanged.connect(self.update_from_input)
        self.train_input.setStyleSheet("""
            QSpinBox {
                border: none;
                background: transparent;
                font-weight: 600;
                color: #374151;
            }
        """)
        train_input_layout.addWidget(train_input_label)
        train_input_layout.addWidget(self.train_input)
        train_input_frame.setLayout(train_input_layout)

        valid_input_frame = QFrame()
        valid_input_frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border-radius: 8px;
                border: 2px solid transparent;
            }
            QFrame:focus-within {
                border-color: #10b981;
            }
        """)
        valid_input_layout = QHBoxLayout()
        valid_input_layout.setContentsMargins(12, 8, 12, 8)
        valid_input_label = QLabel("Valid:")
        valid_input_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        valid_input_label.setStyleSheet("color: #374151; background: transparent; border: none;")
        self.valid_input = QSpinBox()
        self.valid_input.setRange(10, 40)
        self.valid_input.setValue(20)
        self.valid_input.setSuffix("%")
        self.valid_input.valueChanged.connect(self.update_from_input)
        self.valid_input.setStyleSheet("""
            QSpinBox {
                border: none;
                background: transparent;
                font-weight: 600;
                color: #374151;
            }
        """)
        valid_input_layout.addWidget(valid_input_label)
        valid_input_layout.addWidget(self.valid_input)
        valid_input_frame.setLayout(valid_input_layout)

        input_layout.addWidget(train_input_frame)
        input_layout.addWidget(valid_input_frame)

        # Split button
        self.split_dataset_btn = QPushButton("Split Dataset")
        self.split_dataset_btn.clicked.connect(self.split_dataset_dialog)
        self.split_dataset_btn.setEnabled(False)
        self.split_dataset_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.split_dataset_btn.setFixedHeight(50)
        self.split_dataset_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff914d, stop:1 #FFC099);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff9351, stop:1 #ff9351);
            }
            QPushButton:disabled {
                background-color: #ffe0b2;
                color: #808080;
            }
        """)

        # NEW TRAIN BUTTON (added between Split Dataset and Check Result)
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_model_process)
        self.train_model_btn.setEnabled(True)  # Always enabled - independent
        self.train_model_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.train_model_btn.setFixedHeight(50)
        self.train_model_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4DFF91, stop:1 #99FFC0);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #51FF93, stop:1 #51FF93);
            }
            QPushButton:disabled {
                background-color: #9dc5e8;
                color: #808080;
            }
        """)

        # Check Result button (moved from header)
        self.image_classi_button.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.image_classi_button.setFixedHeight(50)
        self.image_classi_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4DBBFF, stop:1 #99D8FF);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #51BDFF, stop:1 #51BDFF);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #51BDFF, stop:1 #51BDFF);
            }
        """)

        layout.addWidget(title)
        layout.addLayout(cards_layout)
        layout.addWidget(self.range_slider)
        layout.addLayout(input_layout)
        layout.addWidget(self.split_dataset_btn)
        layout.addWidget(self.train_model_btn)  # NEW TRAIN BUTTON
        layout.addWidget(self.image_classi_button)

        frame.setLayout(layout)
        return frame

    # TRAINING FUNCTIONALITY (extracted from image classification UI)
    def train_model_process(self):
        """Handles the model training process."""
        if not PYTORCH_AVAILABLE:
            QMessageBox.critical(self, "PyTorch Not Available",
                                 "PyTorch is not installed. Please install PyTorch to use training functionality.")
            return

        if self.training_in_progress:
            QMessageBox.information(self, "Training In Progress",
                                    "Model training is already running.")
            return

        # Pause camera feed to prevent UI lag during training
        print("Pausing camera feed for model training...")
        if self.capture:
            self.timer.stop()
            self.capture.release()
            self.capture = None
            self.camera_card.update_value("Paused")
            self.feed_label.setText("Camera paused for training")
            self.feed_label.setStyleSheet("""
                QLabel {
                    color: #374151;
                    background: transparent;
                    border: none;
                    font-size: 18px;
                    font-weight: bold;
                }
            """)

        # Open file dialog to select training directory
        training_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Training Directory (should contain train/valid/test folders)",
            "",
            QFileDialog.ShowDirsOnly
        )

        if not training_dir:
            self.show_notification("No training directory selected.")
            # Resume camera feed if no directory selected
            print("Resuming camera feed after training cancelled...")
            if self.base_dir:
                self.start_camera()
                if self.capture:
                    self.camera_card.update_value("Active")
                    self.feed_label.setStyleSheet("""
                        QLabel {
                            background: black;
                            border: none;
                            border-radius: 10px;
                        }
                    """)
            return

        # Check if the training directory structure exists
        train_dir = os.path.join(training_dir, 'train')

        if not os.path.exists(train_dir):
            QMessageBox.warning(self, "Invalid Directory Structure",
                                f"Training directory not found: {train_dir}\n\nPlease select a directory that contains:\n- train/ folder\n- valid/ folder (optional)\n- test/ folder (optional)")
            # Resume camera feed on error
            print("Resuming camera feed after training error...")
            if self.base_dir:
                self.start_camera()
                if self.capture:
                    self.camera_card.update_value("Active")
                    self.feed_label.setStyleSheet("""
                        QLabel {
                            background: black;
                            border: none;
                            border-radius: 10px;
                        }
                    """)
            return

        # Load class names from the training directory
        try:
            self.class_names = [d for d in os.listdir(train_dir)
                                if os.path.isdir(os.path.join(train_dir, d))]
            if not self.class_names:
                QMessageBox.warning(self, "No Classes Found",
                                    "No class directories found in training data.\n\nMake sure your train/ folder contains subdirectories for each class (e.g., 'defective', 'non_defective').")
                # Resume camera feed on error
                print("Resuming camera feed after training error...")
                if self.base_dir:
                    self.start_camera()
                    if self.capture:
                        self.camera_card.update_value("Active")
                        self.feed_label.setStyleSheet("""
                            QLabel {
                                background: black;
                                border: none;
                                border-radius: 10px;
                            }
                        """)
                return

            # Show confirmation dialog with directory info
            class_list = ", ".join(self.class_names)
            reply = QMessageBox.question(self, "Confirm Training",
                                         f"Training directory: {training_dir}\n"
                                         f"Classes found: {class_list}\n\n"
                                         f"Do you want to start training?",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)

            if reply != QMessageBox.Yes:
                # Resume camera feed if training cancelled
                print("Resuming camera feed after training cancelled...")
                if self.base_dir:
                    self.start_camera()
                    if self.capture:
                        self.camera_card.update_value("Active")
                        self.feed_label.setStyleSheet("""
                            QLabel {
                                background: black;
                                border: none;
                                border-radius: 10px;
                            }
                        """)
                return

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading class names: {e}")
            # Resume camera feed on error
            print("Resuming camera feed after training error...")
            if self.base_dir:
                self.start_camera()
                if self.capture:
                    self.camera_card.update_value("Active")
                    self.feed_label.setStyleSheet("""
                        QLabel {
                            background: black;
                            border: none;
                            border-radius: 10px;
                        }
                    """)
            return

        # Create and show the progress dialog
        self.progress_dialog = TrainingProgressDialog(self)
        self.progress_dialog.show()
        QApplication.processEvents()

        self.training_in_progress = True
        self.train_model_btn.setText("Training...")
        self.train_model_btn.setEnabled(False)
        QApplication.processEvents()  # Update UI immediately

        try:
            self.train_model(training_dir)

            # Close the progress dialog
            if self.progress_dialog:
                self.progress_dialog.accept()
                self.progress_dialog = None

            QMessageBox.information(self, "Training Complete",
                                    "Model training finished and saved as model_best.pth")
            self.show_notification("Training completed successfully!")

        except Exception as e:
            # Close the progress dialog on error
            if self.progress_dialog:
                self.progress_dialog.accept()
                self.progress_dialog = None

            error_message = f"Error during training: {e}"
            QMessageBox.critical(self, "Training Error", error_message)
            self.show_notification(f"Training failed: {e}")

        finally:
            self.training_in_progress = False
            self.train_model_btn.setText("Train Model")
            self.train_model_btn.setEnabled(True)

            # Resume camera feed after training completes
            print("Resuming camera feed after training completion...")
            if self.base_dir:
                self.start_camera()
                if self.capture:
                    self.camera_card.update_value("Active")
                    self.feed_label.setStyleSheet("""
                        QLabel {
                            background: black;
                            border: none;
                            border-radius: 10px;
                        }
                    """)
                else:
                    self.camera_card.update_value("Error")
                    self.feed_label.setText("Error: Could not restart camera")
            else:
                self.camera_card.update_value("Inactive")

    def train_model(self, dataset_root_dir):
        """Trains the PyTorch model using the selected dataset directory."""
        train_dir = os.path.join(dataset_root_dir, 'train')
        validation_dir = os.path.join(dataset_root_dir, 'valid')
        model_path = 'model_best.pth'

        # Create data transformations
        train_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets and loaders
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        print(f"Train dataset created with {len(train_dataset)} images")

        # Update progress dialog
        if self.progress_dialog:
            self.progress_dialog.update_status(
                f"Loaded training dataset: {len(train_dataset)} images")

        # Setup validation dataset
        if os.path.exists(validation_dir):
            valid_dataset = ImageFolder(validation_dir, transform=valid_transform)
            print(f"Validation dataset created with {len(valid_dataset)} images")
            if self.progress_dialog:
                self.progress_dialog.update_status(
                    f"Loaded validation dataset: {len(valid_dataset)} images")
        else:
            from torch.utils.data import random_split
            train_size = int(0.8 * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
            print(f"Split into {train_size} training and {valid_size} validation images")
            if self.progress_dialog:
                self.progress_dialog.update_status(
                    f"Split data: {train_size} training, {valid_size} validation images")

        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=1)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=1)

        # Create model
        num_classes = len(self.class_names)
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model = model.to(device)

        if self.progress_dialog:
            self.progress_dialog.update_status("Created model with pretrained weights")

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)

        # Training parameters
        num_epochs = 10  # Reduced for demo
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"Epoch {epoch+1}/{num_epochs} - Training in progress...")
            if self.progress_dialog:
                self.progress_dialog.update_status(f"Training Epoch {epoch+1}/{num_epochs}")

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {i+1}/{len(train_loader)}")
                    if self.progress_dialog and i % 20 == 0:
                        self.progress_dialog.update_status(
                            f"Epoch {epoch+1}/{num_epochs} - Batch {i+1}/{len(train_loader)}")

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            print(f"Epoch {epoch+1}/{num_epochs} - Validating...")
            if self.progress_dialog:
                self.progress_dialog.update_status(f"Validating Epoch {epoch+1}/{num_epochs}")

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(valid_loader.dataset)
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} COMPLETED - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.progress_dialog:
                self.progress_dialog.update_status(
                    f"Epoch {epoch+1}/{num_epochs} Complete - Accuracy: {val_acc:.2f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'num_classes': len(self.class_names),
                    'class_names': self.class_names
                }, model_path)
                print(f"Saved improved model to {model_path}")
                if self.progress_dialog:
                    self.progress_dialog.update_status(
                        f"Saved improved model (Epoch {epoch+1}) - Val Acc: {val_acc:.4f}")

        # Load the best model
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if self.progress_dialog:
            self.progress_dialog.update_status("Training complete! Loading best model...")

        print("Model training complete. Saved as model_best.pth.")

    def get_selected_camera_index(self):
        """Gets the selected camera index from inspection_settings.json"""
        settings_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "inspection_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    if 'selected_camera_index' in settings:
                        return int(settings['selected_camera_index'])
            except Exception as e:
                print(f"Error reading camera index from settings: {e}")
        return 0

    def start_camera(self):
        """Start the camera feed using the selected camera index"""
        if self.capture is not None and self.capture.isOpened():
            print("Camera is already open")
            return True

        print(f"Starting camera with index: {self.selected_camera_index}")

        # Try to open the camera
        self.capture = cv2.VideoCapture(self.selected_camera_index)

        if not self.capture.isOpened():
            print(f"Failed to open camera at index {self.selected_camera_index}")
            QMessageBox.critical(self, "Camera Error",
                                 f"Could not open camera at index {self.selected_camera_index}")
            return False

        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Test if we can read a frame
        ret, frame = self.capture.read()
        if not ret:
            print("Could not read frame from camera")
            self.capture.release()
            self.capture = None
            QMessageBox.critical(self, "Camera Error", "Camera opened but cannot read frames")
            return False

        # Start the timer for frame updates
        if not self.timer.isActive():
            self.timer.start(30)  # 30ms = ~33 FPS
            print("Camera timer started")

        return True

    def choose_directory(self):
        """Prompt user to select a directory and start camera if successful"""
        default_directory = "testing2"
        if not os.path.exists(default_directory):
            os.makedirs(default_directory, exist_ok=True)

        base_dir = QFileDialog.getExistingDirectory(self, "Select Directory", default_directory)
        if base_dir:
            self.base_dir = base_dir
            self.non_defective_dir = os.path.join(self.base_dir, "non_defective")
            self.defective_dir = os.path.join(self.base_dir, "defective")
            os.makedirs(self.non_defective_dir, exist_ok=True)
            os.makedirs(self.defective_dir, exist_ok=True)

            self.non_defective_count = self.get_next_count(
                self.non_defective_dir, "non_defective_")
            self.defective_count = self.get_next_count(self.defective_dir, "defective_")

            # Update camera status
            self.camera_card.update_value("Starting...")

            # Enable buttons
            self.good_btn.setEnabled(True)
            self.bad_btn.setEnabled(True)
            self.split_dataset_btn.setEnabled(True)

            # Start camera - this is the key fix
            if not self.start_camera():
                # If camera fails, show error but keep initial UI
                self.camera_card.update_value("Error")
                self.show_notification("Error: Could not open camera")
                return
            else:
                # Camera started successfully - hide initial UI elements and show camera feed
                self.initial_ui_container.hide()
                self.feed_label.show()
                self.camera_card.update_value("Active")
                self.show_notification(f"Directory set: {self.base_dir}")

                # Update the camera section styling to remove dashed border
                self.camera_section_frame.setStyleSheet("""
                    QFrame {
                        background-color: white;
                        border-radius: 15px;
                        border: 1px solid #e5e7eb;
                    }
                """)
        else:
            self.show_notification("No directory selected.")

    @staticmethod
    def get_next_count(directory, prefix):
        """Determine the next available number for image naming."""
        existing_files = [f for f in os.listdir(
            directory) if f.startswith(prefix) and f.endswith('.jpg')]
        numbers = []
        for f in existing_files:
            try:
                num = int(f[len(prefix):-4])
                numbers.append(num)
            except ValueError:
                pass
        return max(numbers) + 1 if numbers else 1

    def update_frame(self):
        """Update the camera feed display"""
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                # Scale pixmap to fit the label while maintaining aspect ratio
                pixmap_scaled = pixmap.scaled(self.feed_label.size(),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.feed_label.setPixmap(pixmap_scaled)
            else:
                # If no frame is available, show error message
                self.feed_label.setText("Camera feed unavailable")
                self.camera_card.update_value("Error")

    def save_good(self):
        """Save the current frame to the 'non_defective' folder or toggle continuous capture."""
        # Check if defective capture is already running
        if self.is_capturing_defective:
            QMessageBox.warning(
                self, "Capture in Progress",
                "Defective image capture is currently running.\n\n"
                "Please stop the defective capture first before starting non-defective capture."
            )
            return

        if self.is_capturing_non_defective:
            # Currently capturing, so stop
            self.stop_non_defective_capture()
        else:
            # Not capturing, so start continuous capture
            self.start_non_defective_capture()

    def save_bad(self):
        """Save the current frame to the 'defective' folder or toggle continuous capture."""
        # Check if non-defective capture is already running
        if self.is_capturing_non_defective:
            QMessageBox.warning(
                self, "Capture in Progress",
                "Non-defective image capture is currently running.\n\n"
                "Please stop the non-defective capture first before starting defective capture."
            )
            return

        if self.is_capturing_defective:
            # Currently capturing, so stop
            self.stop_defective_capture()
        else:
            # Not capturing, so start continuous capture
            self.start_defective_capture()

    def update_status_cards(self):
        """Update the status cards with current counts"""
        self.non_defective_card.update_value(self.non_defective_count)
        self.defective_card.update_value(self.defective_count)
        self.total_card.update_value(self.non_defective_count + self.defective_count)

    def update_ratio_labels(self, _):
        """Update the train, valid, and test ratio labels based on the RangeSlider values."""
        train = self.range_slider.lowerValue()
        valid = self.range_slider.upperValue() - self.range_slider.lowerValue()
        test = self.range_slider.maximum() - self.range_slider.upperValue()

        self.train_value.setText(f"{train}%")
        self.valid_value.setText(f"{valid}%")
        self.test_value.setText(f"{test}%")

        # Update input boxes without triggering signals
        self.train_input.blockSignals(True)
        self.valid_input.blockSignals(True)
        self.train_input.setValue(train)
        self.valid_input.setValue(valid)
        self.train_input.blockSignals(False)
        self.valid_input.blockSignals(False)

    def update_from_input(self):
        """Update slider from input boxes"""
        train_val = self.train_input.value()
        valid_val = self.valid_input.value()
        test_val = 100 - train_val - valid_val

        # Ensure test is at least 10%
        if test_val < 10:
            test_val = 10
            if train_val > valid_val:
                train_val = 90 - valid_val
                self.train_input.setValue(train_val)
            else:
                valid_val = 90 - train_val
                self.valid_input.setValue(valid_val)

        # Update slider
        self.range_slider.setLowerValue(train_val)
        self.range_slider.setUpperValue(train_val + valid_val)

    def split_dataset_dialog(self):
        """Open a dialog to choose a destination directory for the dataset split."""
        # Pause camera feed to prevent UI lag during file operations
        print("Pausing camera feed for dataset split operation...")
        if self.capture:
            self.timer.stop()
            self.capture.release()
            self.capture = None
            self.camera_card.update_value("Paused")
            self.feed_label.setText("Camera paused for dataset operation")
            self.feed_label.setStyleSheet("""
                QLabel {
                    color: #374151;
                    background: transparent;
                    border: none;
                    font-size: 18px;
                    font-weight: bold;
                }
            """)

        dest_dir = QFileDialog.getExistingDirectory(
            self, "Choose Destination Directory for Dataset Split")

        if dest_dir:
            # Show processing status
            self.split_dataset_btn.setText("Processing...")
            self.split_dataset_btn.setEnabled(False)
            QApplication.processEvents()  # Update UI immediately

            try:
                self.split_dataset(dest_dir)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Dataset split failed: {str(e)}")
            finally:
                # Restore button state
                self.split_dataset_btn.setText("Split Dataset")
                self.split_dataset_btn.setEnabled(True)

        # Resume camera feed after operation
        print("Resuming camera feed after dataset split operation...")
        if self.base_dir:
            self.start_camera()
            if self.capture:
                self.camera_card.update_value("Active")
                self.feed_label.setStyleSheet("""
                    QLabel {
                        background: black;
                        border: none;
                        border-radius: 10px;
                    }
                """)
            else:
                self.camera_card.update_value("Error")
                self.feed_label.setText("Error: Could not restart camera")
        else:
            self.camera_card.update_value("Inactive")

    def split_dataset(self, destination_dir):
        """Split the dataset based on the slider ratios"""
        if not self.base_dir:
            QMessageBox.warning(self, "Dataset Split Error", "Please choose a directory first.")
            return

        dataset_base_dir = os.path.join(destination_dir, "dataset_split")
        train_dir = os.path.join(dataset_base_dir, "train")
        valid_dir = os.path.join(dataset_base_dir, "valid")
        test_dir = os.path.join(dataset_base_dir, "test")

        # Create directories
        for folder in [train_dir, valid_dir, test_dir]:
            os.makedirs(os.path.join(folder, "non_defective"), exist_ok=True)
            os.makedirs(os.path.join(folder, "defective"), exist_ok=True)

        # Get images
        non_defective_images = [f for f in os.listdir(self.non_defective_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        defective_images = [f for f in os.listdir(self.defective_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        random.shuffle(non_defective_images)
        random.shuffle(defective_images)

        # Calculate ratios
        train_ratio = self.range_slider.lowerValue() / 100.0
        valid_ratio = (self.range_slider.upperValue() - self.range_slider.lowerValue()) / 100.0

        def split_and_copy(images, source_dir, train_folder, valid_folder, test_folder, train_ratio, valid_ratio):
            train_split = int(train_ratio * len(images))
            valid_split = train_split + int(valid_ratio * len(images))
            train_images = images[:train_split]
            valid_images = images[train_split:valid_split]
            test_images = images[valid_split:]

            for img in train_images:
                shutil.copy(os.path.join(source_dir, img), os.path.join(train_folder, img))
            for img in valid_images:
                shutil.copy(os.path.join(source_dir, img), os.path.join(valid_folder, img))
            for img in test_images:
                shutil.copy(os.path.join(source_dir, img), os.path.join(test_folder, img))

        split_and_copy(non_defective_images, self.non_defective_dir,
                       os.path.join(train_dir, "non_defective"),
                       os.path.join(valid_dir, "non_defective"),
                       os.path.join(test_dir, "non_defective"),
                       train_ratio, valid_ratio)

        split_and_copy(defective_images, self.defective_dir,
                       os.path.join(train_dir, "defective"),
                       os.path.join(valid_dir, "defective"),
                       os.path.join(test_dir, "defective"),
                       train_ratio, valid_ratio)

        # Store the split directory for future reference (optional)
        self.last_split_dir = destination_dir

        QMessageBox.information(self, "Dataset Split Complete",
                                f"Dataset split into train, valid, and test folders inside: {dataset_base_dir}")
        self.show_notification("Dataset split completed successfully!")

    def show_notification(self, message):
        """Show a notification message"""
        print(f"Notification: {message}")

        # Create a temporary status message (you can enhance this with actual notification widgets)
        if hasattr(self, 'status_timer') and self.status_timer:
            self.status_timer.start(3000)  # Hide after 3 seconds

    def showCameraFeedUI(self):
        """Navigate back to the Camera Feed UI."""
        print("showCameraFeedUI in DemoFeedUI called!")
        if self.camera_feed_ui:
            print(f"self.camera_feed_ui is: {self.camera_feed_ui}")
            if isinstance(self.parentWidget(), QStackedWidget):
                self.parentWidget().setCurrentIndex(0)
                print("Switched to Camera Feed UI (index 0)")
            else:
                print("Parent is not a QStackedWidget, cannot switch views this way.")
        else:
            print("self.camera_feed_ui is NOT set!")

    def release_camera_completely(self):
        """Simple camera release"""
        print("Releasing camera...")

        # Stop timers
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        # Release camera
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
            self.capture = None
            print("Camera released")

        # Update UI
        if hasattr(self, 'feed_label'):
            self.feed_label.setText("Camera Released")

    def showImageClassiUI(self):
        """Navigate to the Image Classification UI."""

        print("Switching to Image Classification...")

        # RELEASE CAMERA FIRST
        self.release_camera_completely()

        # Small delay
        import time
        time.sleep(0.5)
        print("showImageClassiUI in DemoFeedUI called!")

        # First, check if we're in a stacked widget
        if isinstance(self.parentWidget(), QStackedWidget):
            stacked_widget = self.parentWidget()

            # Find the image classification UI in the stacked widget
            image_classi_index = -1

            # Try importing the ImageClassiUI class
            try:
                from image_classi_ui import ImageClassiUI
            except ImportError:
                print("Could not import ImageClassiUI")
                return

            # Look through all widgets in the stack
            for i in range(stacked_widget.count()):
                if isinstance(stacked_widget.widget(i), ImageClassiUI):
                    image_classi_index = i
                    break

            if image_classi_index != -1:
                # Found the image classification UI, switch to it
                print(f"Switching to Image Classification UI (index {image_classi_index})")
                QTimer.singleShot(100, lambda: stacked_widget.setCurrentIndex(image_classi_index))
                return
            else:
                print("ImageClassiUI not found in stacked widget, creating a new instance")

                # Create a new instance of ImageClassiUI with good and bad directories
                try:
                    image_classi_ui = ImageClassiUI(
                        self.camera_feed_ui,
                        good_dir=self.non_defective_dir,
                        bad_dir=self.defective_dir
                    )

                    # Add it to the stacked widget
                    new_index = stacked_widget.addWidget(image_classi_ui)
                    print(f"Added ImageClassiUI to stacked widget at index {new_index}")

                    # Switch to it
                    stacked_widget.setCurrentIndex(new_index)
                    return
                except Exception as e:
                    print(f"Error creating ImageClassiUI: {e}")
                    import traceback
                    traceback.print_exc()

        # If we're not in a stacked widget, try to find the parent with a stacked widget
        parent = self.camera_feed_ui
        if parent and hasattr(parent, 'stacked_widget'):
            stacked_widget = parent.stacked_widget

            # Try to find the image classification UI in the parent's stacked widget
            for i in range(stacked_widget.count()):
                widget = stacked_widget.widget(i)
                if hasattr(widget, '__class__') and widget.__class__.__name__ == 'ImageClassiUI':
                    print(f"Found ImageClassiUI in parent's stacked widget at index {i}")
                    stacked_widget.setCurrentIndex(i)
                    return

            print("ImageClassiUI not found in parent's stacked widget")
        else:
            print("Parent does not have a stacked_widget attribute or parent is None")

    def pause_feed(self):
        """Pause the demo feed"""
        print("DemoFeedUI: Pausing feed")
        self.is_active = False
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        # Show camera paused message
        self.feed_label.clear()
        self.feed_label.setText("Camera Feed Paused")
        self.feed_label.setStyleSheet("""
            QLabel {
                color: #374151;
                background: transparent;
                border: none;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.camera_card.update_value("Paused")

    def resume_feed(self):
        """Resume the demo feed"""
        print("DemoFeedUI: Resuming feed")
        self.is_active = True
        if not self.base_dir:
            # If no directory selected, show initial UI
            self.feed_label.hide()
            self.initial_ui_container.show()
            return

        if self.start_camera():
            # Hide initial UI and show camera feed
            self.initial_ui_container.hide()
            self.feed_label.show()
            self.feed_label.setStyleSheet("""
                QLabel {
                    background: black;
                    border: none;
                    border-radius: 10px;
                }
            """)
            if not self.timer.isActive():
                self.timer.start(30)
            self.camera_card.update_value("Active")
        else:
            self.feed_label.setText("Error: Could not start camera")
            self.feed_label.setStyleSheet("""
                QLabel {
                    color: #dc2626;
                    background: transparent;
                    border: none;
                    font-size: 18px;
                    font-weight: bold;
                }
            """)
            self.camera_card.update_value("Error")

    def reset_status(self):
        """Reset status after temporary message"""
        self.status_timer.stop()

    def closeEvent(self, event):
        self.release_camera_completely()
        event.accept()

    def capture_defective_image(self):
        """Capture defective image automatically at set intervals"""
        if not self.capture:
            self.stop_defective_capture()
            return

        ret, frame = self.capture.read()
        if ret:
            filename = os.path.join(self.defective_dir, f"defective_{self.defective_count}.jpg")
            success = cv2.imwrite(filename, frame)
            if success:
                self.defective_count += 1
                self.update_status_cards()
                self.show_notification(f"Auto-captured defective_{self.defective_count-1}.jpg")
                print(f"Auto-captured {filename}")
            else:
                self.show_notification("Failed to auto-capture defective image")
        else:
            self.show_notification("Could not capture defective frame")
            self.stop_defective_capture()

    def capture_non_defective_image(self):
        """Capture non-defective image automatically at set intervals"""
        if not self.capture:
            self.stop_non_defective_capture()
            return

        ret, frame = self.capture.read()
        if ret:
            filename = os.path.join(self.non_defective_dir,
                                    f"non_defective_{self.non_defective_count}.jpg")
            success = cv2.imwrite(filename, frame)
            if success:
                self.non_defective_count += 1
                self.update_status_cards()
                self.show_notification(
                    f"Auto-captured non_defective_{self.non_defective_count-1}.jpg")
                print(f"Auto-captured {filename}")
            else:
                self.show_notification("Failed to auto-capture non-defective image")
        else:
            self.show_notification("Could not capture non-defective frame")
            self.stop_non_defective_capture()

    def start_defective_capture(self):
        """Start continuous defective image capture"""
        if not self.capture:
            self.show_notification("Camera not available")
            return

        self.is_capturing_defective = True
        self.defective_capture_timer.start(self.capture_interval)
        self.bad_btn.setText('Stop Defective Capture')
        self.bad_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ef4444, stop:1 #dc2626);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #dc2626, stop:1 #b91c1c);
            }
        """)
        self.show_notification("Started continuous defective capture (1 sec intervals)")
        print("Started continuous defective capture")

    def stop_defective_capture(self):
        """Stop continuous defective image capture"""
        self.is_capturing_defective = False
        self.defective_capture_timer.stop()
        self.bad_btn.setText('Save as Defective')
        self.bad_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #dc2626, stop:1 #b91c1c);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #b91c1c, stop:1 #991b1b);
            }
            QPushButton:disabled {
                background-color: #ffebee;
                color: #808080;
            }
        """)
        self.show_notification("Stopped defective capture")
        print("Stopped continuous defective capture")

    def start_non_defective_capture(self):
        """Start continuous non-defective image capture"""
        if not self.capture:
            self.show_notification("Camera not available")
            return

        self.is_capturing_non_defective = True
        self.non_defective_capture_timer.start(self.capture_interval)
        self.good_btn.setText('Stop Non-Defective Capture')
        self.good_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #16a34a, stop:1 #15803d);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #15803d, stop:1 #166534);
            }
        """)
        self.show_notification("Started continuous non-defective capture (1 sec intervals)")
        print("Started continuous non-defective capture")

    def stop_non_defective_capture(self):
        """Stop continuous non-defective image capture"""
        self.is_capturing_non_defective = False
        self.non_defective_capture_timer.stop()
        self.good_btn.setText('Save as Non-Defective')
        self.good_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #059669, stop:1 #047857);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #047857, stop:1 #065f46);
            }
            QPushButton:disabled {
                background-color: #e8f5e9;
                color: #808080;
            }
        """)
        self.show_notification("Stopped non-defective capture")
        print("Stopped continuous non-defective capture")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModernDemoFeedUI()
    window.set_custom_container_size()
    window.set_my_container_position()
    window.show()

    # EXAMPLE USAGE - Uncomment any of these to test the functionality:

    # Resize the container to custom dimensions
    # window.set_camera_container_size(1200, 800)

    # Scale the container (1.2 = 20% larger, 0.8 = 20% smaller)
    # window.scale_camera_container(1.2)

    # Set both size and position at once
    # window.adjust_camera_container(x=50, y=50, width=1100, height=750)

    # Get current geometry
    # geometry = window.get_camera_container_geometry()
    # print(f"Current container geometry: {geometry}")

    sys.exit(app.exec_())

# Create an alias for backward compatibility
DemoFeedUI = ModernDemoFeedUI
