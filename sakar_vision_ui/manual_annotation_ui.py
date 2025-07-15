#!/usr/bin/env python3
"""
SAKAR VISION AI - Manual Annotation UI Module

OVERVIEW:
This module implements a sophisticated manual image annotation interface for the Sakar Vision AI platform, serving 
as a comprehensive annotation laboratory that enables users to create high-quality training datasets through advanced 
bounding box annotation capabilities with intelligent class management and professional visual design. It combines 
advanced computer vision tools with modern UI design principles to provide an efficient annotation workflow that 
supports both original defect classes from the platform's defect selection system and user-defined custom classes, 
ensuring seamless dataset preparation for AI model training in industrial manufacturing inspection environments.

KEY FUNCTIONALITY:
The system features advanced bounding box annotation with intuitive click-and-drag functionality, real-time visual 
feedback, and precise coordinate mapping between display and image coordinates, sophisticated class management with 
dual-category support for original defects from the platform's defect selection UI and custom classes added during 
annotation sessions, intelligent annotation persistence with YOLO format export, automatic image resizing capabilities, 
and comprehensive dataset organization, and professional UI design with modern sidebar layout, customizable zoom controls, 
efficient navigation between images, and responsive visual feedback for all user interactions. It includes comprehensive 
dataset splitting functionality with configurable train/validation/test ratios using an interactive range slider, 
integrated model training capabilities with progress monitoring and error handling, advanced image preprocessing with 
configurable resize options and automatic folder structure creation, and robust annotation history management with 
undo/redo capabilities and state persistence across annotation sessions.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced custom widget architecture featuring ModernButton, RangeSlider, and TrainingProgressDialog 
components with professional styling, the module employs sophisticated annotation management with YOLO format coordinate 
conversion, real-time bounding box rendering with class-specific color coding, and intelligent mouse event handling for 
precise annotation control. The architecture features comprehensive class management integration with the platform's 
ClassManager system, dual-category class support with visual distinction between original and custom classes, persistent 
metadata storage for class categorization, and seamless synchronization across multiple configuration files including 
classes.txt, shared_classes.json, and defects_config.json. It includes advanced image processing with OpenCV integration 
for image loading, display scaling, and coordinate transformations, efficient memory management with pixmap caching and 
optimized rendering pipelines, robust file I/O operations with error handling and atomic write operations, and comprehensive 
dataset export functionality with automatic folder structure creation and batch processing capabilities. The system features 
modular architecture enabling easy integration with auto-annotation workflows and seamless navigation between manual and 
automated annotation modes for complete dataset preparation workflows.
"""

import datetime
import math
import os
import random
import re
import shutil
import sys
import threading
import json
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QPoint, Qt, QTimer, QSize, QRect, QEvent
from PyQt5.QtGui import QBrush, QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
                             QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QShortcut,
                             QSpinBox, QStackedWidget, QVBoxLayout, QWidget, QFrame, QScrollArea,
                             QSizePolicy, QSlider, QGroupBox, QGridLayout)

from utils import ClassManager
from utils import set_window_icon

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
DEFECTS_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "defects_config.json")


def natural_sort_key(string_to_sort, _nsre=re.compile('([0-9]+)')):
    """Generates a natural sort key for strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(string_to_sort)]


class ModernButton(QPushButton):
    """Custom modern button with consistent styling"""

    def __init__(self, text="", icon=None, primary=False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        if icon:
            self.setIcon(icon)
        self.setStyleSheet(self.get_style())

    def get_style(self):
        if self.primary:
            return """
                QPushButton {
                    background-color: #ff914d;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    min-height: 32px;
                }
                QPushButton:hover {
                    background-color: #ff8a42;
                }
                QPushButton:pressed {
                    background-color: #ff6303;
                }
                QPushButton:disabled {
                    background-color: #9CA3AF;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #F3F4F6;
                    color: #374151;
                    border: 1px solid #D1D5DB;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    min-height: 32px;
                }
                QPushButton:hover {
                    background-color: #E5E7EB;
                    border-color: #9CA3AF;
                }
                QPushButton:pressed {
                    background-color: #D1D5DB;
                }
                QPushButton:disabled {
                    background-color: #F9FAFB;
                    color: #9CA3AF;
                }
            """


class RangeSlider(QWidget):
    valueChanged = pyqtSignal(int, int)  # Emits (lowerValue, upperValue)

    def __init__(self, minimum=0, maximum=100, lowerValue=60, upperValue=80, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._lowerValue = lowerValue
        self._upperValue = upperValue
        self._handleRadius = 8
        self._movingHandle = None
        self.setMinimumHeight(30)
        self.setMinimumWidth(200)
        self.setStyleSheet("background: transparent;")

    def lowerValue(self):
        return self._lowerValue

    def upperValue(self):
        return self._upperValue

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        line_y = rect.center().y()
        start_x = self._handleRadius
        end_x = rect.width() - self._handleRadius

        # Draw background track
        track_pen = QPen(QColor("#E5E7EB"), 6, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(track_pen)
        painter.drawLine(start_x, line_y, end_x, line_y)

        # Draw selected range track
        lower_x = self._valueToPos(self._lowerValue)
        upper_x = self._valueToPos(self._upperValue)
        range_pen = QPen(QColor("#ff914d"), 6, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(range_pen)
        painter.drawLine(lower_x, line_y, upper_x, line_y)

        # Draw handles
        handle_brush = QBrush(QColor("#ff914d"))
        handle_pen = QPen(QColor("#FFFFFF"), 2)
        painter.setPen(handle_pen)
        painter.setBrush(handle_brush)
        painter.drawEllipse(QPoint(lower_x, line_y), self._handleRadius, self._handleRadius)
        painter.drawEllipse(QPoint(upper_x, line_y), self._handleRadius, self._handleRadius)

    def _valueToPos(self, value):
        rect = self.rect()
        span = rect.width() - 2 * self._handleRadius
        return int(self._handleRadius + (value - self._min) / (self._max - self._min) * span)

    def _posToValue(self, pos):
        rect = self.rect()
        span = rect.width() - 2 * self._handleRadius
        value = self._min + (pos - self._handleRadius) / span * (self._max - self._min)
        return int(round(value))

    def mousePressEvent(self, event):
        pos = event.pos().x()
        lower_x = self._valueToPos(self._lowerValue)
        upper_x = self._valueToPos(self._upperValue)
        if abs(pos - lower_x) < abs(pos - upper_x):
            self._movingHandle = 'lower'
        else:
            self._movingHandle = 'upper'

    def mouseMoveEvent(self, event):
        if self._movingHandle is None:
            return
        pos = event.pos().x()
        value = self._posToValue(pos)
        if self._movingHandle == 'lower':
            if value < self._min:
                value = self._min
            if value > self._upperValue:
                value = self._upperValue
            self._lowerValue = value
        elif self._movingHandle == 'upper':
            if value > self._max:
                value = self._max
            if value < self._lowerValue:
                value = self._lowerValue
            self._upperValue = value
        self.valueChanged.emit(self._lowerValue, self._upperValue)
        self.update()

    def mouseReleaseEvent(self, event):
        self._movingHandle = None


class TrainingProgressDialog(QDialog):
    """Modern training progress dialog"""

    def __init__(self, is_retrain=False, parent=None):
        super().__init__(parent)
        self.is_retrain = is_retrain
        self.training_type = "Retraining" if is_retrain else "Training"
        self.setWindowTitle(f"Model {self.training_type} Progress")
        self.setMinimumSize(400, 200)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 8px;
            }
            QLabel {
                color: #374151;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(20)

        # Status message
        self.status_label = QLabel(
            f"Model {self.training_type.lower()} is in progress.\nThis may take a while depending on your dataset size and parameters.")
        self.status_label.setStyleSheet("font-size: 14px; color: #6B7280;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # Progress animation
        self.progress_label = QLabel("Working...")
        self.progress_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4F46E5;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.progress_label)

        # Instructions
        instruction_text = (
            "Please wait for the training to complete.\n"
            "This window will close automatically when finished.\n\n"
            "You can check the console for detailed progress information."
        )
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        self.layout.addWidget(self.instruction_label)

        # Animation timer
        self.animation_counter = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(500)

        self.finished = False

    def update_animation(self):
        """Updates the animation text to show ongoing activity."""
        self.animation_counter += 1
        dots = "." * (self.animation_counter % 4 + 1)
        self.progress_label.setText(f"{self.training_type} in progress{dots}")

    def set_complete(self, output_dir):
        """Called when training completes successfully."""
        self.animation_timer.stop()
        self.progress_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #10B981;")
        self.progress_label.setText(f"{self.training_type} Complete!")
        self.status_label.setText(
            f"Model {self.training_type.lower()} has completed successfully.\nResults saved in: {output_dir}")
        self.instruction_label.setText("You can close this window now.")

        # Add a close button
        self.close_button = ModernButton("Close", primary=True)
        self.close_button.clicked.connect(self.accept)
        self.layout.addWidget(self.close_button)

        self.finished = True

    def set_error(self, error_message):
        """Called when training encounters an error."""
        self.animation_timer.stop()
        self.progress_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #EF4444;")
        self.progress_label.setText(f"{self.training_type} Failed!")
        self.status_label.setText(f"An error occurred during model {self.training_type.lower()}:")

        # Add error details
        self.error_text = QLabel(error_message)
        self.error_text.setStyleSheet(
            "color: #EF4444; background-color: #FEF2F2; padding: 10px; border: 1px solid #FECACA; border-radius: 6px;")
        self.error_text.setWordWrap(True)
        self.layout.addWidget(self.error_text)

        # Add a close button
        self.close_button = ModernButton("Close", primary=True)
        self.close_button.clicked.connect(self.accept)
        self.layout.addWidget(self.close_button)

        self.finished = True

    def closeEvent(self, event):
        """Override close event to prevent closing until training is done."""
        if not self.finished:
            event.ignore()
        else:
            event.accept()

    def keyPressEvent(self, event):
        """Override key press to prevent Escape key from closing dialog."""
        if not self.finished and event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)


class ImageAnnotationTool(QWidget):
    """
    Modern Image Annotation Tool with sidebar layout similar to Vision AI Studio
    """
    annotation_updated = pyqtSignal()

    def __init__(self):
        """
        Initializes the ImageAnnotationTool with modern UI layout.
        """
        super().__init__()

        # Initialize data attributes
        self.image_folder = None
        self.image_list = []  # List of (filename, path) tuples
        self.current_index = 0
        self.boxes = []  # Each box is stored as (x1, y1, x2, y2, label)
        self.drawing = False
        self.start_point = QPoint()
        self.current_box = None
        self.original_image = None
        self.pixmap_original = QPixmap()
        self.zoom_factor = 1.0
        self.classes = []
        self.classes_file = "classes.txt"
        self.class_colors = {}
        self.annotation_history = []
        self.history_index = -1
        self.annotation_mode = "bbox"  # Only bbox mode supported
        self.selected_box_index = None  # To track which box is selected for editing
        self.selected_class_name = None
        self.original_classes = []  # Classes from defects_config.json (cannot be deleted)
        self.new_classes = []       # Classes added in manual annotation (can be deleted)
        # Attributes for moving and resizing
        self.moving_box = False
        self.resizing_box = False
        self.resize_edges = None
        self.box_original = None
        self.mouse_press_pos = None
        self.MAX_DISPLAY_CLASSES = 50  # Limit visible classes
        self.class_search_text = ""  # For future search functionality

        # Attributes for image resizing feature
        self.resize_width = 640
        self.resize_height = 640
        self.apply_resize = False
        self.output_folder_path = None

        set_window_icon(self)

        # Initialize UI
        self.initUI()
        self.loadClassesFromFile()
        print(f"Loaded classes at start: {self.classes}")
        self.annotation_updated.connect(self.updateHistory)
        self.image_folder_on_start = None
        self.updateClassDropdown()
        self.checkbox_resize.setChecked(True)
        self.apply_resize = True

    def initUI(self):
        """
        Initializes the modern user interface with sidebar layout.
        """
        self.setWindowTitle('Manual Annotation')
        self.setGeometry(100, 100, 1400, 900)

        # Set main application stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #F9FAFB;
                color: #374151;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            QLabel {
                color: #374151;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #ff914d;
                outline: none;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #9CA3AF;
            }
            QComboBox:focus {
                border-color: #ff914d;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #D1D5DB;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #6B7280;
                margin-right: 6px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #ff914d;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #D1D5DB;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #ff914d;
                border-color: #ff914d;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #374151;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
        """)

        # Create main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main content area first (will appear on left now)
        self.main_content = self.create_main_content()
        main_layout.addWidget(self.main_content, 1)  # Give main content area more space

        # Create sidebar second (will appear on right now)
        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)

        # Set up shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undoAnnotation)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redoAnnotation)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redoAnnotation)

    def create_sidebar(self):
        """Creates the left sidebar with all controls"""
        sidebar = QFrame()
        sidebar.setFixedWidth(550)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: white;
                border-right: 1px solid #E5E7EB;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(20)

        # Class Management Section
        class_group = self.create_class_management_section()
        sidebar_layout.addWidget(class_group)

        # Image Settings Section
        image_group = self.create_image_settings_section()
        sidebar_layout.addWidget(image_group)

        # Dataset Split Section
        dataset_group = self.create_dataset_split_section()
        sidebar_layout.addWidget(dataset_group)

        # Add stretch to push everything to top
        sidebar_layout.addStretch()

        # Auto Annotation button at bottom
        self.btn_auto_annotation = ModernButton("Go to Auto Annotation", primary=True)
        self.btn_auto_annotation.clicked.connect(self.show_auto_annotation_ui)
        sidebar_layout.addWidget(self.btn_auto_annotation)

        return sidebar

    def create_class_management_section(self):
        """Creates the class management section with improved class tag display"""
        group = QGroupBox("Class Management")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        # Add new class input
        add_class_layout = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText(
            "Add class (or multiple: class1, class2, class3)")
        self.btn_add_class = ModernButton("Add", primary=True)
        self.btn_add_class.clicked.connect(self.add_class_from_input)

        add_class_layout.addWidget(self.new_class_input)
        add_class_layout.addWidget(self.btn_add_class)
        layout.addLayout(add_class_layout)

        # Select class label and dropdown
        layout.addWidget(QLabel("Select Class:"))
        self.combo_classes = QComboBox()
        self.combo_classes.addItem("Choose a class")
        layout.addWidget(self.combo_classes)

        # Classes display area with tags
        classes_label = QLabel("Classes (0):")
        classes_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        layout.addWidget(classes_label)
        self.classes_label = classes_label

        # Create scroll area for class tags
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # Container widget inside scroll area
        self.classes_container = QWidget()
        self.classes_container.setStyleSheet("background-color: transparent;")

        # Use grid layout for more efficient layout of class tags
        self.classes_layout = QGridLayout(self.classes_container)
        self.classes_layout.setContentsMargins(0, 0, 0, 0)
        self.classes_layout.setSpacing(4)

        # Set up scroll area
        self.scroll_area.setWidget(self.classes_container)
        self.scroll_area.setFixedHeight(200)  # Set a fixed height for the scroll area
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.scroll_area)

        # Cache for class tag widgets to avoid recreating them
        self.class_tag_cache = {}

        return group

    def create_image_settings_section(self):
        """Creates the image settings section"""
        group = QGroupBox("Image Settings")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        # Resize checkbox
        self.checkbox_resize = QCheckBox("Resize Images")
        self.checkbox_resize.stateChanged.connect(self.toggleResize)
        layout.addWidget(self.checkbox_resize)

        # Width and Height inputs
        size_layout = QHBoxLayout()

        width_layout = QVBoxLayout()
        width_layout.addWidget(QLabel("Width"))
        self.spinbox_resize_width = QSpinBox()
        self.spinbox_resize_width.setRange(100, 2048)
        self.spinbox_resize_width.setValue(self.resize_width)
        # Disable spin buttons
        self.spinbox_resize_width.setButtonSymbols(QSpinBox.NoButtons)
        # Disable scroll wheel
        self.spinbox_resize_width.installEventFilter(self)
        width_layout.addWidget(self.spinbox_resize_width)

        height_layout = QVBoxLayout()
        height_layout.addWidget(QLabel("Height"))
        self.spinbox_resize_height = QSpinBox()
        self.spinbox_resize_height.setRange(100, 2048)
        self.spinbox_resize_height.setValue(self.resize_height)
        # Disable spin buttons
        self.spinbox_resize_height.setButtonSymbols(QSpinBox.NoButtons)
        # Disable scroll wheel
        self.spinbox_resize_height.installEventFilter(self)
        height_layout.addWidget(self.spinbox_resize_height)

        size_layout.addLayout(width_layout)
        size_layout.addLayout(height_layout)
        layout.addLayout(size_layout)

        # Apply button
        self.btn_apply_resize_settings = ModernButton("Apply Resize Settings", primary=True)
        self.btn_apply_resize_settings.clicked.connect(self.applyResizeSettings)
        layout.addWidget(self.btn_apply_resize_settings)

        return group

    def selectClass(self, class_name):
        """Selects the class in the dropdown when a class tag is clicked"""
        index = self.combo_classes.findText(class_name)
        if index >= 0:
            self.combo_classes.setCurrentIndex(index)
            self.selected_class_name = class_name  # Store the selected class
            print(f"Selected class: {class_name}")

            # Update the class tags display to show the selection
            self.update_class_display()

    def create_dataset_split_section(self):
        """Creates the dataset split section"""
        group = QGroupBox("Dataset Split")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        # Range slider for splits
        self.range_slider = RangeSlider(0, 100, 60, 80)
        self.range_slider.valueChanged.connect(self.updateRangeSliderLabel)
        layout.addWidget(self.range_slider)

        # Split labels
        self.label_range_slider = QLabel("Train: 60% | Valid: 20% | Test: 20%")
        self.label_range_slider.setStyleSheet(
            "font-size: 12px; color: #6B7280; text-align: center;")
        self.label_range_slider.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_range_slider)

        # Action buttons
        buttons_layout = QHBoxLayout()
        self.btn_export_dataset = ModernButton("Split Dataset", primary=True)
        self.btn_export_dataset.clicked.connect(self.splitDatasets)

        self.btn_train_model = ModernButton("Train Model", primary=True)
        self.btn_train_model.clicked.connect(self.trainModel)
        buttons_layout.addWidget(self.btn_export_dataset)
        buttons_layout.addWidget(self.btn_train_model)

        layout.addLayout(buttons_layout)

        return group

    def create_main_content(self):
        """Creates the main content area with image display"""
        main_content = QFrame()
        main_content.setStyleSheet("""
            QFrame {
                background-color: #F9FAFB;
            }
        """)

        layout = QVBoxLayout(main_content)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        # Main image display area - now includes header within the container
        image_area = self.create_image_display_area()
        layout.addWidget(image_area, 1)  # Give most space to image area

        # Controls section
        controls_layout = self.create_controls_section()
        layout.addLayout(controls_layout)

        return main_content

    def create_image_display_area(self):
        """Creates the main image display area"""
        # Container frame
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header section with title and controls
        header_layout = QHBoxLayout()

        # Left side - camera icon and title
        left_layout = QHBoxLayout()

        # Center - image counter display
        center_layout = QHBoxLayout()
        self.image_counter_label = QLabel("Image 1 of 150")
        self.annotations_counter_label = QLabel("0 annotations")

        for label in [self.image_counter_label, self.annotations_counter_label]:
            label.setStyleSheet("color: #6B7280; font-size: 14px;")

        center_layout.addWidget(self.image_counter_label)
        center_layout.addWidget(self.annotations_counter_label)
        center_layout.addStretch()

        # Right side - buttons
        right_layout = QHBoxLayout()

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(6)

        # Zoom out button
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedSize(36, 36)
        self.btn_zoom_out.setStyleSheet("""
            QPushButton {
                background-color: #F3F4F6; 
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E5E7EB;
            }
        """)
        self.btn_zoom_out.clicked.connect(self.zoomOut)

        # Zoom percentage
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        self.zoom_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #374151;
                padding: 4px;
            }
        """)
        self.zoom_label.setAlignment(Qt.AlignCenter)

        # Zoom in button
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(36, 36)
        self.btn_zoom_in.setStyleSheet("""
            QPushButton {
                background-color: #F3F4F6; 
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E5E7EB;
            }
        """)
        self.btn_zoom_in.clicked.connect(self.zoomIn)

        # Reset button
        self.btn_zoom_reset = QPushButton("Reset")
        self.btn_zoom_reset.setFixedSize(80, 36)
        self.btn_zoom_reset.setStyleSheet("""
            QPushButton {
                background-color: #F3F4F6; 
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #E5E7EB;
            }
        """)
        self.btn_zoom_reset.clicked.connect(self.zoomReset)

        # Navigation buttons
        self.btn_prev_image = QPushButton("<")
        self.btn_prev_image.setFixedSize(36, 36)
        self.btn_prev_image.setStyleSheet("""
            QPushButton {
                background-color: #F3F4F6; 
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #E5E7EB;
            }
        """)
        self.btn_prev_image.clicked.connect(self.prevImage)

        self.btn_next_image = QPushButton(">")
        self.btn_next_image.setFixedSize(36, 36)
        self.btn_next_image.setStyleSheet("""
            QPushButton {
                background-color: #F3F4F6; 
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #E5E7EB;
            }
        """)
        self.btn_next_image.clicked.connect(self.nextImage)

        # Save annotations button
        self.btn_save_annot = QPushButton("💾 Save")
        self.btn_save_annot.setFixedSize(120, 36)  # Set both width and height smaller
        self.btn_save_annot.setStyleSheet("""
            QPushButton {
                background-color: #10B981; 
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 18px;
                font-weight: 500;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.btn_save_annot.clicked.connect(self.saveAnnotations)

        # Add all zoom controls
        zoom_layout.addWidget(self.btn_zoom_out)
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.btn_zoom_in)
        zoom_layout.addWidget(self.btn_zoom_reset)

        # Add navigation and save buttons
        right_layout.addLayout(zoom_layout)
        right_layout.addWidget(self.btn_prev_image)
        right_layout.addWidget(self.btn_next_image)
        right_layout.addWidget(self.btn_save_annot)

        # Assemble the complete header
        header_layout.addLayout(left_layout, 3)
        header_layout.addLayout(center_layout, 2)
        header_layout.addLayout(right_layout, 5)

        # Add the header to the container
        layout.addLayout(header_layout)

        # Main image display
        self.label_image_display = QLabel()
        self.label_image_display.setAlignment(Qt.AlignCenter)
        self.label_image_display.setMinimumHeight(300)
        self.label_image_display.setStyleSheet("""
            QLabel {
                background-color: #1F2937;
                border: 1px solid #374151;
                border-radius: 6px;
                color: #9CA3AF;
            }
        """)
        self.label_image_display.setText(
            "Draw bounding boxes by selecting a class and clicking + dragging")
        self.label_image_display.setMouseTracking(True)
        self.label_image_display.installEventFilter(self)

        layout.addWidget(self.label_image_display, 1)

        return container

    def create_controls_section(self):
        """Creates the bottom controls section"""
        controls_layout = QHBoxLayout()

        # Left side - folder selection
        left_layout = QHBoxLayout()

        folder_label = QLabel("Select Folder:")
        folder_label.setStyleSheet("font-weight: 500;")

        self.inputDirLineEdit = QLineEdit()
        self.inputDirLineEdit.setPlaceholderText("Image directory path")

        self.btn_open_folder = ModernButton("Open Folder", primary=True)
        self.btn_open_folder.clicked.connect(self.openFolder)

        left_layout.addWidget(folder_label)
        left_layout.addWidget(self.inputDirLineEdit)
        left_layout.addWidget(self.btn_open_folder)

        controls_layout.addLayout(left_layout)
        controls_layout.addStretch()

        # Right side - REMOVE OR COMMENT OUT THIS SECTION
        # right_layout = QHBoxLayout()
        #
        # # Status info
        # self.status_label = QLabel("Zoom: 100%  Size: 640x640  Selected: None")
        # self.status_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        #
        # # Undo/Redo buttons
        # self.btn_undo = ModernButton("↶")
        # self.btn_undo.setFixedSize(32, 32)
        # self.btn_undo.setToolTip("Undo")
        # self.btn_undo.clicked.connect(self.undoAnnotation)
        # self.btn_undo.setEnabled(False)
        #
        # self.btn_redo = ModernButton("↷")
        # self.btn_redo.setFixedSize(32, 32)
        # self.btn_redo.setToolTip("Redo")
        # self.btn_redo.clicked.connect(self.redoAnnotation)
        # self.btn_redo.setEnabled(False)
        #
        # # Export button
        # self.btn_export = ModernButton("Export")
        # self.btn_export.clicked.connect(self.exportAnnotations)
        #
        # right_layout.addWidget(self.status_label)
        # right_layout.addWidget(self.btn_undo)
        # right_layout.addWidget(self.btn_redo)
        # right_layout.addWidget(self.btn_export)
        #
        # controls_layout.addLayout(right_layout)

        # NOTE: You'll need to define these properties as empty objects since they're used elsewhere
        # This prevents errors when other methods try to use them
        self.status_label = QLabel("")
        self.status_label.hide()  # Hide it

        self.btn_undo = ModernButton("")
        self.btn_undo.hide()  # Hide it

        self.btn_redo = ModernButton("")
        self.btn_redo.hide()  # Hide it

        self.btn_export = ModernButton("")
        self.btn_export.hide()  # Hide it

        return controls_layout

    # === CORE FUNCTIONALITY METHODS ===

    def setClasses(self):
        """Sets classes based on user input from a dialog."""
        text, ok = QInputDialog.getText(
            self, "Set Classes", "Enter class names (comma-separated):", text=", ".join(self.classes))
        if ok:
            new_classes_str = text.strip()
            new_classes = [cls.strip() for cls in new_classes_str.split(",") if cls.strip()]
            if not new_classes:
                QMessageBox.warning(self, "Warning", "Class names cannot be empty.")
                return

            self.classes = new_classes
            self.updateClassDropdown()
            self.generateClassColors()
            self.update_class_display()

            # Save to classes.txt
            self.saveClassesToFile()

            # Save to shared_classes.json via ClassManager
            class_manager = ClassManager()
            class_manager.update_classes(self.classes, "manual_setClasses")

            # Remove boxes with invalid classes
            self.boxes = [box for box in self.boxes if box[4] in self.classes]
            self.selected_box_index = None
            self.displayImage()

            print(f"Set classes saved to both classes.txt and shared_classes.json: {self.classes}")

    def onClassDropdownChanged(self, text):
        """Handles changes to the class dropdown selection"""
        if text != "Choose a class":
            self.selected_class_name = text
            self.update_class_display()  # Update tags to show the selection

    def updateClassDropdown(self):
        """Updates the class dropdown menu with current classes - optimized version"""
        current_selection = self.combo_classes.currentText()

        # Disconnect signals temporarily to prevent multiple calls (only if connected)
        try:
            self.combo_classes.currentTextChanged.disconnect()
        except TypeError:
            # Signal wasn't connected yet, which is fine
            pass

        # Clear and rebuild efficiently
        self.combo_classes.clear()
        self.combo_classes.addItem("Choose a class")

        # Add classes in batches for better performance
        if self.classes:
            self.combo_classes.addItems(self.classes)

        # Restore selection efficiently
        if current_selection and current_selection != "Choose a class":
            index = self.combo_classes.findText(current_selection)
            if index >= 0:
                self.combo_classes.setCurrentIndex(index)

        # Reconnect signal
        self.combo_classes.currentTextChanged.connect(self.onClassDropdownChanged)

        # Update display and generate colors
        self.update_class_display()
        self.generateClassColors()

    def loadUserSelectedDefects(self):
        """Loads user-selected defects from defects_config.json."""
        try:
            if os.path.exists(DEFECTS_CONFIG_PATH):
                with open(DEFECTS_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    selected_defects = config.get("selected_defects", [])

                    # Check if this config was created by manual annotation (has metadata)
                    metadata_path = os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "class_metadata.json")
                    if os.path.exists(metadata_path):
                        print(f"DEBUG: Found metadata, selected defects: {selected_defects}")
                    else:
                        print(
                            f"DEBUG: No metadata found, treating as original defects: {selected_defects}")

                    return selected_defects
        except Exception as e:
            print(f"Error loading user-selected defects: {e}")
        return []

    def loadClassesFromFile(self):
        """Loads classes from defects_config.json first, then falls back to other sources."""
        # Load class metadata first to know which are original vs new
        stored_original, stored_new = self.loadClassMetadata()

        # First priority: Load from defects_config.json (from defect selection UI)
        defects_config_classes = self.loadUserSelectedDefects()
        if defects_config_classes:
            # Separate the classes based on stored metadata
            if stored_original or stored_new:
                # We have metadata, use it to classify classes
                self.original_classes = [
                    cls for cls in defects_config_classes if cls in stored_original]
                self.new_classes = [cls for cls in defects_config_classes if cls in stored_new]

                # Any class not in metadata but in defects_config is treated as original
                unclassified = [
                    cls for cls in defects_config_classes if cls not in stored_original and cls not in stored_new]
                self.original_classes.extend(unclassified)
            else:
                # No metadata, treat all as original (first time loading)
                self.original_classes = defects_config_classes.copy()
                self.new_classes = []

            self.classes = defects_config_classes.copy()
            self.updateClassDropdown()
            self.generateClassColors()
            print(f"Classes loaded from defects_config.json")
            print(f"Original classes: {self.original_classes}")
            print(f"New classes: {self.new_classes}")
            return

        # Second priority: Try to get from ClassManager (shared_classes.json)
        class_manager = ClassManager()
        if class_manager.initialized or class_manager.load_from_file():
            shared_classes = class_manager.get_classes()
            if shared_classes:
                # Use metadata to classify classes
                if stored_original or stored_new:
                    self.original_classes = [
                        cls for cls in shared_classes if cls in stored_original]
                    self.new_classes = [cls for cls in shared_classes if cls in stored_new]

                    # Any unclassified class is treated as new (since it's not from defect selection)
                    unclassified = [
                        cls for cls in shared_classes if cls not in stored_original and cls not in stored_new]
                    self.new_classes.extend(unclassified)
                else:
                    # No metadata, treat all as new classes
                    self.original_classes = []
                    self.new_classes = shared_classes.copy()

                self.classes = shared_classes
                self.updateClassDropdown()
                self.generateClassColors()
                print(f"Classes loaded from ClassManager (shared_classes.json)")
                print(f"Original classes: {self.original_classes}")
                print(f"New classes: {self.new_classes}")
                return

        # Third priority: Load from local classes.txt file
        if os.path.exists(self.classes_file):
            try:
                with open(self.classes_file, 'r') as f:
                    loaded_classes = [line.strip() for line in f.readlines() if line.strip()]
                if loaded_classes:
                    # Use metadata to classify classes
                    if stored_original or stored_new:
                        self.original_classes = [
                            cls for cls in loaded_classes if cls in stored_original]
                        self.new_classes = [cls for cls in loaded_classes if cls in stored_new]

                        # Any unclassified class is treated as new
                        unclassified = [
                            cls for cls in loaded_classes if cls not in stored_original and cls not in stored_new]
                        self.new_classes.extend(unclassified)
                    else:
                        # No metadata, treat all as new classes
                        self.original_classes = []
                        self.new_classes = loaded_classes.copy()

                    self.classes = loaded_classes
                    # Update the ClassManager with these classes
                    class_manager.update_classes(self.classes, "classes_file")
                    self.updateClassDropdown()
                    self.generateClassColors()
                    print(f"Classes loaded from {self.classes_file}")
                    print(f"Original classes: {self.original_classes}")
                    print(f"New classes: {self.new_classes}")
                    return
            except Exception as e:
                print(f"Error loading classes from {self.classes_file}: {e}")

        # If no classes found anywhere, start with empty list
        self.original_classes = []
        self.classes = []
        self.new_classes = []
        self.updateClassDropdown()
        print("No classes found, starting with empty lists")

    def saveClassMetadata(self):
        """Saves metadata about which classes are original vs new"""
        try:
            metadata = {
                "original_classes": self.original_classes.copy(),
                "new_classes": self.new_classes.copy(),
                "timestamp": datetime.datetime.now().isoformat()
            }

            metadata_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "class_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(
                f"Class metadata saved: original={len(self.original_classes)}, new={len(self.new_classes)}")
            return True

        except Exception as e:
            print(f"Error saving class metadata: {e}")
            return False

    def loadClassMetadata(self):
        """Loads metadata about which classes are original vs new"""
        try:
            metadata_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "class_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get("original_classes", []), metadata.get("new_classes", [])
        except Exception as e:
            print(f"Error loading class metadata: {e}")
        return [], []

    def setOriginalClassesFromDefectUI(self, selected_classes):
        """Call this method when classes are selected from defect selection UI"""
        self.original_classes = selected_classes.copy()
        self.new_classes = []
        self.classes = selected_classes.copy()

        # Save metadata immediately
        self.saveClassMetadata()

        print(f"Set original classes from defect UI: {self.original_classes}")

    def saveClassesToFile(self):
        """Saves classes to both the local file and ClassManager."""
        try:
            # Save to local file
            with open(self.classes_file, 'w') as f:
                for cls in self.classes:
                    f.write(cls + '\n')
            print(f"Classes saved to {self.classes_file}")

            # Update ClassManager
            class_manager = ClassManager()
            class_manager.update_classes(self.classes, "manual_saveClasses")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save classes to file: {e}")

    def generateClassColors(self):
        """Generates colors efficiently with caching"""
        # Only generate colors for classes that don't have them
        existing_classes = set(self.class_colors.keys())
        new_classes = set(self.classes) - existing_classes

        # Remove colors for classes that no longer exist
        self.class_colors = {k: v for k, v in self.class_colors.items() if k in self.classes}

        # Generate colors only for new classes
        for class_name in new_classes:
            # Use a more deterministic color generation based on class name hash
            import hashlib
            hash_obj = hashlib.md5(class_name.encode())
            hash_hex = hash_obj.hexdigest()

            # Extract RGB values from hash
            r = int(hash_hex[0:2], 16) % 156 + 100  # 100-255
            g = int(hash_hex[2:4], 16) % 156 + 100  # 100-255
            b = int(hash_hex[4:6], 16) % 156 + 100  # 100-255

            self.class_colors[class_name] = QColor(r, g, b)

    def color_distance(self, color1, color2):
        """Calculate distance between two colors"""
        return abs(color1.red() - color2.red()) + abs(color1.green() - color2.green()) + abs(color1.blue() - color2.blue())

    def add_class_from_input(self):
        """Optimized class addition with batch processing"""
        input_text = self.new_class_input.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Warning", "Please enter class name(s).")
            return

        # Parse and validate input
        new_classes = [cls.strip() for cls in input_text.split(",") if cls.strip()]
        if not new_classes:
            QMessageBox.warning(self, "Warning", "Please enter valid class name(s).")
            return

        # Filter out duplicates
        added_classes = [cls for cls in new_classes if cls not in self.classes]
        duplicate_classes = [cls for cls in new_classes if cls in self.classes]

        if added_classes:
            # Batch add classes
            self.classes.extend(added_classes)
            self.new_classes.extend(added_classes)

            # Clear input
            self.new_class_input.clear()

            # Batch update all systems
            self.generateClassColors()  # Only generates for new classes now
            self.updateClassDropdown()

            # Save operations
            self.saveClassesToFile()

            # Update other config files
            class_manager = ClassManager()
            class_manager.update_classes(self.classes, "manual_addFromInput")
            self.updateDefectsConfigWithAllClasses()
            self.saveClassMetadata()

            # Show success message
            message = f"Added class: {added_classes[0]}" if len(
                added_classes) == 1 else f"Added {len(added_classes)} classes"
            self.autoCloseMessage("Success", f"{message} to all config files")

        if duplicate_classes:
            QMessageBox.information(
                self, "Info", f"These classes already exist: {', '.join(duplicate_classes)}")

    def update_class_display(self):
        """Updates the class tags display efficiently with virtual scrolling"""
        # Update classes count
        self.classes_label.setText(f"Classes ({len(self.classes)}):")

        # Clear existing layout efficiently
        while self.classes_layout.count():
            child = self.classes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear cache
        self.class_tag_cache.clear()

        # Limit display to first 50 classes for performance
        MAX_DISPLAY_CLASSES = 50
        display_classes = self.classes[:MAX_DISPLAY_CLASSES]

        if display_classes:
            cols = 4
            for i, class_name in enumerate(display_classes):
                row, col = divmod(i, cols)
                tag = self.create_class_tag(class_name)
                self.classes_layout.addWidget(tag, row, col)
                self.class_tag_cache[class_name] = tag

        # Show message if there are more classes
        if len(self.classes) > MAX_DISPLAY_CLASSES:
            info_label = QLabel(f"... and {len(self.classes) - MAX_DISPLAY_CLASSES} more classes")
            info_label.setStyleSheet("color: #6B7280; font-style: italic; padding: 8px;")
            row = (MAX_DISPLAY_CLASSES // cols) + 1
            self.classes_layout.addWidget(info_label, row, 0, 1, cols)

    def update_tag_style(self, tag, class_name, is_selected):
        """Updates the styling of a class tag based on its type and selection state"""
        # Find the name label in the tag
        name_label = None
        remove_btn = None
        for i in range(tag.layout().count()):
            widget = tag.layout().itemAt(i).widget()
            if isinstance(widget, QLabel):
                name_label = widget
            elif isinstance(widget, QPushButton):
                remove_btn = widget

        if name_label is None:
            return  # Can't find the label

        # Apply styling based on class type and selection state
        if class_name in self.original_classes:
            # Original classes styling
            if is_selected:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #4DBBFF;
                        border: 1px solid #4DBBFF;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
                name_label.setStyleSheet("""
                    QLabel {
                        color: white;
                        font-size: 12px;
                        font-weight: 600;
                        background: transparent;
                        border: none;
                    }
                """)
            else:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #E3F2FD;
                        border: 1px solid #4DBBFF;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
                name_label.setStyleSheet("""
                    QLabel {
                        color: #1976D2;
                        font-size: 12px;
                        font-weight: 600;
                        background: transparent;
                        border: none;
                    }
                """)
        else:
            # New classes styling
            if is_selected:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #C7D2FE;
                        border: 1px solid #818CF8;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
                name_label.setStyleSheet("""
                    QLabel {
                        color: white;
                        font-size: 12px;
                        font-weight: 500;
                        background: transparent;
                        border: none;
                    }
                """)
            else:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #EEF2FF;
                        border: 1px solid #C7D2FE;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
                name_label.setStyleSheet("""
                    QLabel {
                        color: #3730A3;
                        font-size: 12px;
                        font-weight: 500;
                        background: transparent;
                        border: none;
                    }
                """)

    def create_class_tag(self, class_name):
        """Creates a removable class tag with visual distinction for original vs new classes"""
        tag = QFrame()

        # Check if this class is the selected one
        is_selected = (class_name == self.selected_class_name)

        # Apply different styling based on whether the class is original, new, or selected
        if class_name in self.original_classes:
            # Original classes - blue theme, no remove button
            if is_selected:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #4DBBFF;  /* Darker blue for selected state */
                        border: 1px solid #4DBBFF;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
            else:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #E3F2FD;
                        border: 1px solid #4DBBFF;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
        else:
            # New classes - purple theme, with remove button
            if is_selected:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #C7D2FE;  /* Darker purple for selected state */
                        border: 1px solid #818CF8;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)
            else:
                tag.setStyleSheet("""
                    QFrame {
                        background-color: #EEF2FF;
                        border: 1px solid #C7D2FE;
                        border-radius: 12px;
                        padding: 4px 8px;
                    }
                """)

        layout = QHBoxLayout(tag)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(4)

        # Class name label
        name_label = QLabel(class_name)
        if class_name in self.original_classes:
            if is_selected:
                name_label.setStyleSheet("""
                    QLabel {
                        color: white;  /* White text for better visibility on dark background */
                        font-size: 12px;
                        font-weight: 600;
                        background: transparent;
                        border: none;
                    }
                """)
            else:
                name_label.setStyleSheet("""
                    QLabel {
                        color: #1976D2;
                        font-size: 12px;
                        font-weight: 600;
                        background: transparent;
                        border: none;
                    }
                """)
        else:
            if is_selected:
                name_label.setStyleSheet("""
                    QLabel {
                        color: white;  /* White text for better visibility on dark background */
                        font-size: 12px;
                        font-weight: 500;
                        background: transparent;
                        border: none;
                    }
                """)
            else:
                name_label.setStyleSheet("""
                    QLabel {
                        color: #3730A3;
                        font-size: 12px;
                        font-weight: 500;
                        background: transparent;
                        border: none;
                    }
                """)

        layout.addWidget(name_label)

        # Make the tag clickable and select the class when clicked
        tag.mousePressEvent = lambda event, cn=class_name: self.selectClass(cn)
        name_label.mousePressEvent = lambda event, cn=class_name: self.selectClass(cn)
        tag.setCursor(Qt.PointingHandCursor)
        name_label.setCursor(Qt.PointingHandCursor)

        # Add remove button only for new classes
        if class_name not in self.original_classes:
            remove_btn = QPushButton("×")
            remove_btn.setFixedSize(16, 16)
            remove_btn.setStyleSheet("""
                QPushButton {
                    background-color: #C7D2FE;
                    border: none;
                    border-radius: 8px;
                    color: #3730A3;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #A5B4FC;
                }
            """)
            remove_btn.clicked.connect(lambda: self.remove_class(class_name))
            layout.addWidget(remove_btn)

        # Set tooltip
        tooltip_text = f"Original class from Defect Selection UI\n(Cannot be deleted here)" if class_name in self.original_classes else f"New class added in Manual Annotation\n(Can be deleted)"
        tag.setToolTip(tooltip_text)

        return tag

    def select_class_from_tag(self, class_name):
        """Selects the class in the dropdown when the class tag is clicked"""
        index = self.combo_classes.findText(class_name)
        if index >= 0:
            self.combo_classes.setCurrentIndex(index)
            print(f"Selected class: {class_name}")

    def cleanup_class_widgets(self):
        """Cleanup function to prevent memory leaks"""
        # Clear the layout properly
        while self.classes_layout.count():
            child = self.classes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear cache
        self.class_tag_cache.clear()

        # Force garbage collection
        import gc
        gc.collect()

    def remove_class(self, class_name):
        """Optimized class removal with proper cleanup"""
        if class_name not in self.classes:
            return

        if class_name in self.original_classes:
            QMessageBox.warning(
                self, "Cannot Delete Class",
                f"You cannot delete the class '{class_name}' because it was selected from the Defect Selection UI."
            )
            return

        if class_name in self.new_classes:
            reply = QMessageBox.question(
                self,
                "Remove New Class",
                f"Are you sure you want to remove the class '{class_name}'?\n\n"
                f"This will remove it from all configuration files:\n"
                f"• classes.txt\n"
                f"• shared_classes.json\n"
                f"• defects_config.json",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Remove from all lists
                self.classes.remove(class_name)
                self.new_classes.remove(class_name)

                # Remove from color cache
                self.class_colors.pop(class_name, None)

                # Remove any boxes with this class
                self.boxes = [box for box in self.boxes if box[4] != class_name]

                # Reset selection if this class was selected
                if self.selected_class_name == class_name:
                    self.selected_class_name = None
                    self.combo_classes.setCurrentIndex(0)

                # Batch update UI
                self.updateClassDropdown()

                # Save changes
                self.saveClassesToFile()

                # Update other systems
                class_manager = ClassManager()
                class_manager.update_classes(self.classes, "manual_removeClass")
                self.updateDefectsConfigWithAllClasses()
                self.saveClassMetadata()

                # Refresh display
                self.displayImage()

                self.autoCloseMessage(
                    "Success", f"New class '{class_name}' removed from all config files")
                print(f"Removed new class '{class_name}' from all three config files")
                print(f"Remaining original classes: {self.original_classes}")
                print(f"Remaining new classes: {self.new_classes}")
        else:
            QMessageBox.information(
                self,
                "Class Not Found",
                f"Class '{class_name}' not found in removable classes."
            )
    # === IMAGE HANDLING METHODS ===

    def openFolder(self):
        """Opens a folder selection dialog and loads images from the selected folder."""
        # Check if we have a parent with capture_folder_a set
        parent = self.parent()
        folder_a_path = None

        while parent is not None:
            if hasattr(parent, 'capture_folder_a') and parent.capture_folder_a:
                folder_a_path = parent.capture_folder_a
                break
            parent = parent.parent()

        # Use folder dialog with initial path if available
        if folder_a_path and os.path.exists(folder_a_path):
            initial_dir = folder_a_path
        else:
            initial_dir = ""

        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder', initial_dir)
        if folder:
            self.inputDirLineEdit.setText(folder)
            self.loadImagesFromFolder(folder)

    def loadImagesFromFolder(self, folder):
        """Loads image filenames and paths from the selected folder."""
        if folder:
            self.image_folder = folder
            # Create a better naming scheme for the output folder
            base_folder_name = os.path.basename(self.image_folder)
            parent_folder = os.path.dirname(self.image_folder)

            # If it's Folder A, create "manually_annotated" in parent dir
            if base_folder_name == "Folder_A":
                self.output_folder_path = os.path.join(parent_folder, "manually_annotated")
            else:
                # Default case - append "_annotated"
                self.output_folder_path = os.path.join(
                    parent_folder, f"{base_folder_name}_annotated")

            os.makedirs(self.output_folder_path, exist_ok=True)
            print(f"Output folder path set to: {self.output_folder_path}")

            # Load images
            self.image_list = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(folder, f)
                    original_img = cv2.imread(image_path)
                    if original_img is not None:
                        self.image_list.append((f, image_path))
                    else:
                        print(f"Warning: Could not load image {f}, skipping.")

            self.image_list.sort(key=lambda item: natural_sort_key(item[0]))
            if not self.image_list:
                QMessageBox.warning(
                    self, "Warning", "No valid images found in the selected folder.")
                self.image_folder = None
                return

            self.current_index = 0
            self.boxes = self.loadAnnotations()
            self.resetHistory()
            self.loadImage()
            self.updateUI()

    def loadImage(self):
        """Loads the current image, annotations, and displays the image."""
        if not self.image_list:
            self.clearImageDisplay("No images in folder.")
            return

        current_image_filename, image_path = self.image_list[self.current_index]
        print(f"loadImage: Trying to load image from path: {image_path}")

        if not os.path.exists(image_path):
            self.clearImageDisplay(f"File not found: {os.path.basename(current_image_filename)}")
            QMessageBox.critical(
                self, "Error", f"File not found: {os.path.basename(current_image_filename)} at path: {image_path}")
            return

        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            self.clearImageDisplay(
                f"Failed to load image: {os.path.basename(current_image_filename)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load image: {os.path.basename(current_image_filename)}. Please check if the image file is valid.")
            return

        self.resetHistory()
        self.boxes = self.loadAnnotations()
        self.annotation_updated.emit()
        self.displayImage()
        self.zoomReset()
        self.updateUI()

    def clearImageDisplay(self, message="No Image Loaded"):
        """Clears the image display and shows a message."""
        self.original_image = None
        self.pixmap_original = QPixmap()
        self.label_image_display.clear()
        self.label_image_display.setText(
            f"{message}\nDraw bounding boxes by selecting a class and clicking + dragging")
        self.label_image_display.setStyleSheet("""
            QLabel {
                background-color: #1F2937;
                border: 1px solid #374151;
                border-radius: 6px;
                color: #9CA3AF;
            }
        """)

    def updateUI(self):
        """Updates UI elements with current state"""
        if self.image_list:
            # Update image counter
            self.image_counter_label.setText(
                f"Image {self.current_index + 1} of {len(self.image_list)}")

            # Update annotations counter
            self.annotations_counter_label.setText(f"{len(self.boxes)} annotations")

            # Update zoom label
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")

            # Update status
            if self.original_image is not None:
                h, w = self.original_image.shape[:2]
                selected_text = f"Box {self.selected_box_index + 1}" if self.selected_box_index is not None else "None"
                self.status_label.setText(
                    f"Zoom: {int(self.zoom_factor * 100)}%  Size: {w}x{h}  Selected: {selected_text}")
            else:
                self.status_label.setText("Zoom: 100%  Size: --  Selected: None")
        else:
            self.image_counter_label.setText("Image 0 of 0")
            self.annotations_counter_label.setText("0 annotations")
            self.zoom_label.setText("100%")
            self.status_label.setText("Zoom: 100%  Size: --  Selected: None")

    def displayImage(self):
        """Displays the current image with bounding boxes."""
        if self.original_image is None:
            return

        image = self.original_image.copy()

        # Draw bounding boxes with proper colors
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2, label = box

            # Get color for this class, or use default green
            if label in self.class_colors:
                color = self.class_colors[label]
            else:
                color = QColor(0, 255, 0)  # Default green

            # Highlight selected box in yellow
            if self.selected_box_index == i:
                box_color = QColor(255, 255, 0)  # Yellow for selected
            else:
                box_color = color

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2),
                          (box_color.blue(), box_color.green(), box_color.red()), 2)

            # Draw label
            cv2.putText(image, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (box_color.blue(), box_color.green(), box_color.red()), 2, cv2.LINE_AA)

        # Draw the box currently being drawn
        if self.drawing and self.annotation_mode == 'bbox' and self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()
        self.pixmap_original = QPixmap.fromImage(q_img)
        self.scaleImage()

    def scaleImage(self):
        """Scales the displayed image based on the zoom factor."""
        if not self.pixmap_original.isNull():
            size = self.pixmap_original.size() * self.zoom_factor
            scaled_pixmap = self.pixmap_original.scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_image_display.setPixmap(scaled_pixmap)
        else:
            self.label_image_display.clear()

    def zoomIn(self):
        """Zooms the image in."""
        self.zoom_factor = min(3.0, self.zoom_factor * 1.20)
        self.scaleImage()
        self.updateUI()

    def zoomOut(self):
        """Zooms the image out."""
        self.zoom_factor = max(0.1, self.zoom_factor * 0.80)
        self.scaleImage()
        self.updateUI()

    def zoomReset(self):
        """Resets the zoom level to 1.0."""
        self.zoom_factor = 1.0
        self.scaleImage()
        self.updateUI()

    def nextImage(self):
        """Loads the next image in the image list."""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.saveAnnotations()
            self.current_index += 1
            self.boxes = self.loadAnnotations()
            self.resetHistory()
            self.loadImage()
            self.selected_box_index = None
        else:
            QMessageBox.information(self, "Info", "You have reached the last image.")

    def prevImage(self):
        """Loads the previous image in the image list."""
        if self.image_list and self.current_index > 0:
            self.saveAnnotations()
            self.current_index -= 1
            self.boxes = self.loadAnnotations()
            self.resetHistory()
            self.loadImage()
            self.selected_box_index = None
        else:
            QMessageBox.information(self, "Info", "Reached the beginning of the image list.")

    def applyResizeSettings(self):
        """Applies resize settings from spinboxes and reloads images if folder is open."""
        self.resize_width = self.spinbox_resize_width.value()
        self.resize_height = self.spinbox_resize_height.value()
        if self.image_folder:
            self.loadImagesFromFolder(self.image_folder)

    def toggleResize(self, state):
        """Toggles image resizing on or off and reloads images if folder is open."""
        if state == Qt.Checked:
            self.apply_resize = True
            if self.image_folder:
                self.loadImagesFromFolder(self.image_folder)
        else:
            self.apply_resize = False
            if self.image_folder:
                self.loadImagesFromFolder(self.image_folder)

    def exportAnnotations(self):
        """Exports current annotations"""
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        # For now, just save current annotations
        self.saveAnnotations()
        self.autoCloseMessage("Export Complete", "Annotations exported successfully!")

    def autoCloseMessage(self, title, message, timeout=2000):
        """Displays an auto-closing message box with an "OK" button."""
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        QTimer.singleShot(timeout, msg.close)
        msg.exec_()

    # === ANNOTATION METHODS ===

    def saveAnnotations(self):
        """Saves annotations for the current image to a text file in YOLO format."""
        if self.image_list and self.original_image is not None:
            current_image_filename, original_image_path = self.image_list[self.current_index]
            image_name = os.path.splitext(current_image_filename)[0]
            if self.output_folder_path is None:
                QMessageBox.critical(
                    self, "Error", "Output folder path is not initialized. Please open an image folder first.")
                return
            annotation_path = os.path.join(self.output_folder_path, image_name + '.txt')
            height, width, _ = self.original_image.shape
            try:
                with open(annotation_path, 'w') as f:
                    for box in self.boxes:
                        x1, y1, x2, y2, label = box
                        class_id = self.classes.index(label) if label in self.classes else 0
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        w = abs(x2 - x1) / width
                        h = abs(y2 - y1) / height
                        f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n')
                print(f'Annotations saved: {annotation_path}')

                if self.apply_resize:
                    resized_image = cv2.resize(
                        self.original_image, (self.resize_width, self.resize_height))
                    resized_image_filename = os.path.join(
                        self.output_folder_path, current_image_filename)
                    cv2.imwrite(resized_image_filename, resized_image)
                    print(f'Resized image saved: {resized_image_filename}')
                else:
                    # Copy the original image to the output folder if not resizing
                    output_image_path = os.path.join(
                        self.output_folder_path, current_image_filename)
                    if not os.path.exists(output_image_path) or os.path.getmtime(original_image_path) > os.path.getmtime(output_image_path):
                        shutil.copy2(original_image_path, output_image_path)
                        print(f'Original image copied to: {output_image_path}')

                self.resetHistory()
                self.btn_undo.setEnabled(False)
                self.btn_redo.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save annotations: {e}")

    def loadAnnotations(self):
        """Loads annotations for the current image from a text file (if exists)."""
        if not self.image_list:
            return []
        current_image_filename, original_image_path = self.image_list[self.current_index]
        image_name = os.path.splitext(current_image_filename)[0]
        annotation_path = os.path.join(self.output_folder_path, image_name + '.txt')
        boxes = []
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) == 5:
                            class_id, x_center, y_center, w, h = map(float, parts)
                            class_id = int(class_id)
                            label = self.classes[class_id] if class_id < len(
                                self.classes) else f"class_{class_id}"
                            H, W, _ = self.original_image.shape if self.original_image is not None else (
                                1, 1, 1)
                            x1 = int((x_center - w/2) * W)
                            y1 = int((y_center - h/2) * H)
                            x2 = int((x_center + w/2) * W)
                            y2 = int((y_center + h/2) * H)
                            boxes.append((x1, y1, x2, y2, label))
                        else:
                            print(f"Warning: Skipping unrecognized format: {line.strip()}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Error loading annotations from {os.path.basename(annotation_path)}: {e}")
        return boxes

    # === MOUSE AND INTERACTION METHODS ===

    def is_inside_box(self, point, box):
        """Checks if a point is inside a bounding box."""
        x1, y1, x2, y2, _ = box
        return x1 <= point.x() <= x2 and y1 <= point.y() <= y2

    def delete_selected_box(self):
        """Deletes the currently selected bounding box."""
        if self.selected_box_index is not None:
            del self.boxes[self.selected_box_index]
            self.selected_box_index = None
            self.annotation_updated.emit()
            self.displayImage()
            self.updateUI()

    def edit_selected_box_class(self):
        """Edits the class label of the currently selected bounding box."""
        if self.selected_box_index is not None:
            current_box = self.boxes[self.selected_box_index]
            current_label = current_box[4]

            # Create list of available classes
            available_classes = self.classes.copy()
            if not available_classes:
                QMessageBox.warning(
                    self, "Warning", "No classes available. Please add classes first.")
                return

            current_index = 0
            if current_label in available_classes:
                current_index = available_classes.index(current_label)

            new_label, ok = QInputDialog.getItem(
                self, "Edit Class", "Choose new class:", available_classes,
                current=current_index, editable=False)

            if ok and new_label:
                updated_box = list(current_box)
                updated_box[4] = new_label
                self.boxes[self.selected_box_index] = tuple(updated_box)
                self.annotation_updated.emit()
                self.displayImage()
                self.updateUI()

    def add_new_class(self):
        """Adds a new class to the class list through user input."""
        text, ok = QInputDialog.getText(self, "Add New Class", "Enter new class name:")
        if ok and text:
            new_class_name = text.strip()
            if new_class_name and new_class_name not in self.classes:
                # Add to local classes
                self.classes.append(new_class_name)
                self.updateClassDropdown()
                self.saveClassesToFile()
                self.generateClassColors()

                # Update selected box if needed
                if self.selected_box_index is not None:
                    updated_box = list(self.boxes[self.selected_box_index])
                    updated_box[4] = new_class_name
                    self.boxes[self.selected_box_index] = tuple(updated_box)
                    self.selected_box_index = None
                    self.annotation_updated.emit()
                    self.displayImage()
                    self.updateUI()

                # Update the global ClassManager
                class_manager = ClassManager()
                class_manager.add_class(new_class_name, "manual_addNewClass")

            elif not new_class_name:
                QMessageBox.warning(self, "Warning", "Class name cannot be empty.")
            elif new_class_name in self.classes:
                QMessageBox.warning(self, "Warning", f"Class '{new_class_name}' already exists.")

    def contextMenuEvent(self, event):
        """Handles right-click context menu events for bounding box editing."""
        if self.selected_box_index is not None:
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: white;
                    border: 1px solid #E5E7EB;
                    border-radius: 6px;
                    padding: 4px;
                }
                QMenu::item {
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #4F46E5;
                    color: white;
                }
            """)

            # Show current class info
            current_box = self.boxes[self.selected_box_index]
            current_class = current_box[4]

            delete_action = menu.addAction(f"Delete Box ({current_class})")
            edit_class_action = menu.addAction("Change Class")
            menu.addSeparator()
            add_class_action = menu.addAction("Add New Class")

            action = menu.exec_(event.globalPos())

            if action == delete_action:
                self.delete_selected_box()
            elif action == edit_class_action:
                self.edit_selected_box_class()
            elif action == add_class_action:
                self.add_new_class()

            self.selected_box_index = None
            self.displayImage()
            self.updateUI()

    def mousePressEvent(self, event):
        """Handles mouse press events for bounding box annotation and selection."""
        if self.label_image_display.underMouse() and self.original_image is not None:
            # Get position relative to the image label widget
            label_pos = self.label_image_display.mapFromGlobal(event.globalPos())
            image_pos = self.mapToImageCoords(label_pos)

            # Check if a valid class is selected when starting to draw
            if event.button() == Qt.LeftButton and self.combo_classes.currentText() == "Choose a class":
                QMessageBox.warning(
                    self, "Warning", "Please select a class first before drawing a bounding box.")
                return

            if event.button() == Qt.RightButton:
                if self.selected_box_index is not None:
                    self.contextMenuEvent(event)
                return

            if self.annotation_mode == 'bbox':
                if event.button() == Qt.LeftButton:
                    box_clicked = False
                    for i, box in enumerate(self.boxes):
                        if self.is_inside_box(image_pos, box):
                            self.selected_box_index = i
                            box_clicked = True
                            x1, y1, x2, y2, _ = box
                            threshold = 10
                            resize_edges = {'left': abs(image_pos.x() - x1) <= threshold,
                                            'right': abs(image_pos.x() - x2) <= threshold,
                                            'top': abs(image_pos.y() - y1) <= threshold,
                                            'bottom': abs(image_pos.y() - y2) <= threshold}
                            if any(resize_edges.values()):
                                self.resizing_box = True
                                self.resize_edges = resize_edges
                                self.box_original = box
                            else:
                                self.moving_box = True
                                self.box_original = box
                                self.mouse_press_pos = image_pos
                            self.displayImage()
                            self.updateUI()
                            break

                    if not box_clicked:
                        self.selected_box_index = None
                        self.drawing = True
                        self.start_point = image_pos
                        self.current_box = (self.start_point.x(), self.start_point.y(),
                                            self.start_point.x(), self.start_point.y())
                        self.displayImage()

    def mouseMoveEvent(self, event):
        """Handles mouse move events for drawing, moving, and resizing bounding boxes."""
        if self.label_image_display.underMouse():
            # Get position relative to the image label widget
            label_pos = self.label_image_display.mapFromGlobal(event.globalPos())
            current_img_pos = self.mapToImageCoords(label_pos)

            if self.annotation_mode == 'bbox':
                if self.resizing_box and self.selected_box_index is not None:
                    x1, y1, x2, y2, label = self.box_original
                    if self.resize_edges.get('left'):
                        x1 = current_img_pos.x()
                        if x1 > x2 - 5:
                            x1 = x2 - 5
                    if self.resize_edges.get('right'):
                        x2 = current_img_pos.x()
                        if x2 < x1 + 5:
                            x2 = x1 + 5
                    if self.resize_edges.get('top'):
                        y1 = current_img_pos.y()
                        if y1 > y2 - 5:
                            y1 = y2 - 5
                    if self.resize_edges.get('bottom'):
                        y2 = current_img_pos.y()
                        if y2 < y1 + 5:
                            y2 = y1 + 5
                    self.boxes[self.selected_box_index] = (x1, y1, x2, y2, label)
                    self.displayImage()
                elif self.moving_box and self.selected_box_index is not None:
                    delta = current_img_pos - self.mouse_press_pos
                    x1, y1, x2, y2, label = self.box_original
                    new_box = (x1 + delta.x(), y1 + delta.y(), x2 +
                               delta.x(), y2 + delta.y(), label)
                    self.boxes[self.selected_box_index] = new_box
                    self.displayImage()
                elif self.drawing:
                    self.current_box = (self.start_point.x(), self.start_point.y(),
                                        current_img_pos.x(), current_img_pos.y())
                    self.displayImage()

    def mouseReleaseEvent(self, event):
        """Handles mouse release events to finalize bounding box drawing, moving, or resizing."""
        if self.label_image_display.underMouse():
            if self.annotation_mode == 'bbox' and event.button() == Qt.LeftButton:
                if self.resizing_box or self.moving_box:
                    self.resizing_box = False
                    self.moving_box = False
                    self.box_original = None
                    self.mouse_press_pos = None
                    self.annotation_updated.emit()
                    self.displayImage()
                    self.updateUI()
                elif self.drawing:
                    self.drawing = False
                    # Get position relative to the image label widget
                    label_pos = self.label_image_display.mapFromGlobal(event.globalPos())
                    end_img_pos = self.mapToImageCoords(label_pos)

                    x1 = self.start_point.x()
                    y1 = self.start_point.y()
                    x2 = end_img_pos.x()
                    y2 = end_img_pos.y()
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                        self.current_box = None
                        self.displayImage()
                        return

                    chosen_label = self.combo_classes.currentText()
                    print(f"Chosen Box Label: '{chosen_label}'")
                    new_box = (x1, y1, x2, y2, chosen_label)
                    self.boxes.append(new_box)
                    self.current_box = None
                    self.annotation_updated.emit()
                    self.displayImage()
                    self.updateUI()

    def eventFilter(self, source, event):
        """Filters events for the image display label, specifically for zoom using Ctrl+Wheel."""
        # Check if object has the attributes before comparing
        has_spinbox_width = hasattr(self, 'spinbox_resize_width')
        has_spinbox_height = hasattr(self, 'spinbox_resize_height')

        # Block wheel events for spin boxes if they exist
        if ((has_spinbox_width and source == self.spinbox_resize_width) or
                (has_spinbox_height and source == self.spinbox_resize_height)) and event.type() == QEvent.Wheel:
            return True  # Block the wheel event

        # Keep the rest of your existing event filter code
        if source == self.label_image_display and event.type() == event.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoomIn()
                else:
                    self.zoomOut()
                return True
        return super().eventFilter(source, event)

    def mapToImageCoords(self, widget_pos):
        """Maps widget coordinates to image coordinates - FIXED VERSION"""
        if not self.pixmap_original or self.pixmap_original.isNull():
            return QPoint(0, 0)

        # Get the current displayed pixmap
        displayed_pixmap = self.label_image_display.pixmap()
        if not displayed_pixmap:
            return QPoint(0, 0)

        # Get the label size and displayed pixmap size
        label_size = self.label_image_display.size()
        pixmap_size = displayed_pixmap.size()

        # Calculate the position of the image within the label (centered)
        x_offset = (label_size.width() - pixmap_size.width()) // 2
        y_offset = (label_size.height() - pixmap_size.height()) // 2

        # Adjust the click position by removing the offset
        adjusted_x = widget_pos.x() - x_offset
        adjusted_y = widget_pos.y() - y_offset

        # Check if the click is within the image bounds
        if (adjusted_x < 0 or adjusted_y < 0 or
                adjusted_x >= pixmap_size.width() or adjusted_y >= pixmap_size.height()):
            return QPoint(0, 0)

        # Scale the coordinates back to original image size
        scale_x = self.pixmap_original.width() / pixmap_size.width()
        scale_y = self.pixmap_original.height() / pixmap_size.height()

        original_x = int(adjusted_x * scale_x)
        original_y = int(adjusted_y * scale_y)

        # Ensure coordinates are within original image bounds
        original_x = max(0, min(original_x, self.pixmap_original.width() - 1))
        original_y = max(0, min(original_y, self.pixmap_original.height() - 1))

        return QPoint(original_x, original_y)

    def keyPressEvent(self, event):
        """Handles key press events for nudging selected bounding boxes."""
        if self.selected_box_index is not None:
            step = 2
            x1, y1, x2, y2, label = self.boxes[self.selected_box_index]
            if event.key() == Qt.Key_Left:
                if event.modifiers() & Qt.ControlModifier:
                    x1 -= step
                else:
                    x1 -= step
                    x2 -= step
            elif event.key() == Qt.Key_Right:
                if event.modifiers() & Qt.ControlModifier:
                    x2 += step
                else:
                    x1 += step
                    x2 += step
            elif event.key() == Qt.Key_Up:
                if event.modifiers() & Qt.ControlModifier:
                    y1 -= step
                else:
                    y1 -= step
                    y2 -= step
            elif event.key() == Qt.Key_Down:
                if event.modifiers() & Qt.ControlModifier:
                    y2 += step
                else:
                    y1 -= step
                    y2 += step
            if x2 - x1 < 5:
                x2 = x1 + 5
            if y2 - y1 < 5:
                y2 = y1 + 5
            self.boxes[self.selected_box_index] = (x1, y1, x2, y2, label)
            self.displayImage()
            self.updateUI()
        super().keyPressEvent(event)

    # === HISTORY METHODS ===

    def resetHistory(self):
        """Resets the annotation history for undo/redo."""
        self.annotation_history = []
        self.history_index = -1
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

    def updateHistory(self):
        """Updates the annotation history with the current state of bounding boxes."""
        if self.history_index < len(self.annotation_history) - 1:
            self.annotation_history = self.annotation_history[:self.history_index + 1]
        current_state = (self.boxes[:])
        if not self.annotation_history or self.annotation_history[-1] != current_state:
            self.annotation_history.append(current_state)
            self.history_index += 1
            self.btn_undo.setEnabled(True)
            self.btn_redo.setEnabled(False)
            if len(self.annotation_history) > 100:
                self.annotation_history = self.annotation_history[-100:]
                self.history_index = len(self.annotation_history) - 1

    def undoAnnotation(self):
        """Undoes the last annotation action."""
        if self.history_index > 0:
            self.history_index -= 1
            self.boxes = self.annotation_history[self.history_index]
            self.displayImage()
            self.updateUI()
            self.btn_redo.setEnabled(True)
            if self.history_index == 0:
                self.btn_undo.setEnabled(False)

    def redoAnnotation(self):
        """Redoes the last undone annotation action."""
        if self.history_index < len(self.annotation_history) - 1:
            self.history_index += 1
            self.boxes = self.annotation_history[self.history_index]
            self.displayImage()
            self.updateUI()
            self.btn_undo.setEnabled(True)
            if self.history_index == len(self.annotation_history) - 1:
                self.btn_redo.setEnabled(False)

    # === DATASET AND TRAINING METHODS ===

    def updateRangeSliderLabel(self, lower, upper):
        """Updates the range slider label with current split percentages."""
        train_percent = lower
        valid_percent = upper - lower
        test_percent = 100 - upper
        self.label_range_slider.setText(
            f"Train: {train_percent}% | Valid: {valid_percent}% | Test: {test_percent}%")

    def splitDatasets(self):
        """Splits datasets and defaults to the manually_annotated folder when available."""
        # Check if we have a parent with capture_storage_folder set
        parent = self.parent()
        root_folder = None

        while parent is not None:
            if hasattr(parent, 'capture_storage_folder') and parent.capture_storage_folder:
                root_folder = parent.capture_storage_folder
                break
            parent = parent.parent()

        # Default to manually_annotated if exists
        default_folder = None
        if root_folder:
            manually_annotated = os.path.join(root_folder, "manually_annotated")
            if os.path.exists(manually_annotated):
                default_folder = manually_annotated

        # Use output_folder_path as another option if available
        if not default_folder and self.output_folder_path and os.path.exists(self.output_folder_path):
            default_folder = self.output_folder_path

        # Open folder dialog with default path if available
        if default_folder:
            folder = QFileDialog.getExistingDirectory(
                self, "Select Folder to Split Datasets", default_folder)
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Folder to Split Datasets")

        if not folder:
            return

        # Process the dataset splitting
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        file_list = os.listdir(folder)
        basenames = []
        for file in file_list:
            name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                if (name + '.txt') in file_list:
                    basenames.append(name)

        basenames = list(set(basenames))
        if not basenames:
            QMessageBox.warning(
                self, "Warning", "No image and label pairs found in the selected folder.")
            return

        random.shuffle(basenames)
        n = len(basenames)
        lower = self.range_slider._lowerValue
        upper = self.range_slider._upperValue
        train_count = int(n * lower / 100)
        valid_count = int(n * (upper - lower) / 100)
        test_count = n - train_count - valid_count

        # Create destination subfolders for each split inside the same folder
        for split in ['train', 'valid', 'test']:
            split_folder = os.path.join(folder, split)
            images_folder = os.path.join(split_folder, "images")
            labels_folder = os.path.join(split_folder, "labels")
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)

        def move_pair(name, split):
            src_image = None
            for file in file_list:
                file_name, ext = os.path.splitext(file)
                if file_name == name and ext.lower() in image_extensions:
                    src_image = os.path.join(folder, file)
                    break
            src_label = os.path.join(folder, name + '.txt')
            dest_image = os.path.join(folder, split, "images",
                                      os.path.basename(src_image)) if src_image else None
            dest_label = os.path.join(folder, split, "labels", name + '.txt')
            if src_image and os.path.exists(src_image):
                shutil.move(src_image, dest_image)
            if os.path.exists(src_label):
                shutil.move(src_label, dest_label)

        for name in basenames[:train_count]:
            move_pair(name, "train")
        for name in basenames[train_count:train_count+valid_count]:
            move_pair(name, "valid")
        for name in basenames[train_count+valid_count:]:
            move_pair(name, "test")

        # Create .yaml file
        yaml_content = f"""
train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(self.classes)}
names: {self.classes}
"""
        yaml_path = os.path.join(folder, 'dataset.yaml')
        with open(yaml_path, 'w') as yaml_file:
            yaml_file.write(yaml_content)

        self.autoCloseMessage(
            "Export Complete", f"Datasets split into Train: {train_count}, Valid: {valid_count}, Test: {test_count}")

    def trainModel(self):
        """Opens the training dialog for initial model training."""
        dialog = TrainDialog(self, is_retrain=False)

        # Pre-fill model path
        default_model_path = "sakar_vision.pt"
        if os.path.exists(default_model_path):
            dialog.model_path_input.setText(default_model_path)

        # Find a suitable dataset.yaml file
        yaml_path = self.find_dataset_yaml()
        if yaml_path:
            dialog.yaml_path_input.setText(yaml_path)
        else:
            self.prompt_for_yaml_file(dialog)

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_params()
            self.runTraining(params)

    def find_dataset_yaml(self):
        """Finds a suitable dataset.yaml file"""
        yaml_path = None
        print("DEBUG: Starting search for dataset.yaml file...")

        # First check in output_folder_path
        if self.output_folder_path:
            possible_yaml = os.path.join(self.output_folder_path, 'dataset.yaml')
            print(f"DEBUG: Checking for YAML at output path: {possible_yaml}")
            if os.path.exists(possible_yaml):
                yaml_path = possible_yaml
                print(f"DEBUG: Found YAML at output path: {yaml_path}")

        # Check parent folders for YAML if not found
        if not yaml_path:
            parent = self.parent()
            root_folder = None

            while parent is not None:
                if hasattr(parent, 'capture_storage_folder') and parent.capture_storage_folder:
                    root_folder = parent.capture_storage_folder
                    print(f"DEBUG: Found root folder: {root_folder}")
                    break
                parent = parent.parent()

            if root_folder:
                # First check manually_annotated folder
                manually_annotated = os.path.join(root_folder, "manually_annotated")
                possible_yaml = os.path.join(manually_annotated, 'dataset.yaml')
                print(f"DEBUG: Checking for YAML at manually_annotated: {possible_yaml}")

                if os.path.exists(possible_yaml):
                    yaml_path = possible_yaml
                    print(f"DEBUG: Found YAML at manually_annotated: {yaml_path}")
                else:
                    # Try recursive search through all subdirectories of root_folder
                    print(f"DEBUG: Starting recursive search for dataset.yaml in {root_folder}")
                    for root, dirs, files in os.walk(root_folder):
                        for file in files:
                            if file == 'dataset.yaml':
                                possible_yaml = os.path.join(root, file)
                                print(f"DEBUG: Found dataset.yaml at: {possible_yaml}")
                                if not yaml_path:  # Take the first one we find
                                    yaml_path = possible_yaml
                                    print(f"DEBUG: Using YAML: {yaml_path}")

        return yaml_path

    def find_latest_best_pt(self):
        """Finds the latest best.pt file from training runs."""
        latest_model = None
        highest_train_num = 0

        print("DEBUG: Starting search for latest best.pt file...")

        # Check for runs directories in various possible locations
        search_dirs = []

        # First, check in output_folder_path if available
        if self.output_folder_path and os.path.exists(self.output_folder_path):
            runs_dir = os.path.join(self.output_folder_path, "runs")
            search_dirs.append(runs_dir)
            print(f"DEBUG: Adding search directory: {runs_dir}")

        # Check in the parent's storage folder structure
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'capture_storage_folder') and parent.capture_storage_folder:
                root_folder = parent.capture_storage_folder
                print(f"DEBUG: Found root folder: {root_folder}")

                # Look in manually_annotated/runs
                manual_runs = os.path.join(root_folder, "manually_annotated", "runs")
                if os.path.exists(manual_runs):
                    search_dirs.append(manual_runs)
                    print(f"DEBUG: Adding search directory: {manual_runs}")

                # Look in auto_annotated/runs
                auto_runs = os.path.join(root_folder, "auto_annotated", "runs")
                if os.path.exists(auto_runs):
                    search_dirs.append(auto_runs)
                    print(f"DEBUG: Adding search directory: {auto_runs}")

                break
            parent = parent.parent()

        # Add current directory as fallback
        current_dir_runs = os.path.join(os.getcwd(), "runs")
        if os.path.exists(current_dir_runs):
            search_dirs.append(current_dir_runs)
            print(f"DEBUG: Adding search directory: {current_dir_runs}")

        # Search in all potential runs directories for best.pt files
        for runs_dir in search_dirs:
            if os.path.exists(runs_dir):
                print(f"DEBUG: Searching in: {runs_dir}")
                try:
                    dir_contents = os.listdir(runs_dir)
                    print(f"DEBUG: Directory contents: {dir_contents}")

                    train_dirs = [d for d in dir_contents
                                  if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("train")]

                    train_dirs.sort(reverse=True)
                    print(f"DEBUG: Found train directories (sorted): {train_dirs}")

                    for train_dir in train_dirs:
                        match = re.search(r'train(\d+)', train_dir)
                        if match:
                            train_num = int(match.group(1))
                            print(f"DEBUG: Processing train{train_num}")

                            weights_dir = os.path.join(runs_dir, train_dir, "weights")
                            best_pt_path = os.path.join(weights_dir, "best.pt")

                            print(f"DEBUG: Checking for best.pt at: {best_pt_path}")
                            if os.path.exists(best_pt_path):
                                if train_num > highest_train_num:
                                    highest_train_num = train_num
                                    latest_model = best_pt_path
                                    print(f"DEBUG: Found newer model: {latest_model}")
                except Exception as e:
                    print(f"DEBUG: Error while searching directory {runs_dir}: {e}")

        if latest_model:
            print(f"DEBUG: Found latest model at: {latest_model}")
            return latest_model

        print("DEBUG: No best.pt models found")
        return None

    def prompt_for_yaml_file(self, dialog):
        """Prompts user to select a YAML file"""
        print("DEBUG: No suitable YAML file was found")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("YAML File Not Found")
        msg_box.setText("Could not automatically find a dataset.yaml file.")
        msg_box.setInformativeText("Would you like to manually select a YAML file?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)

        if msg_box.exec_() == QMessageBox.Yes:
            yaml_path, _ = QFileDialog.getOpenFileName(
                self, "Select YAML File", "", "YAML Files (*.yaml)")
            if yaml_path and os.path.exists(yaml_path):
                dialog.yaml_path_input.setText(yaml_path)
                print(f"DEBUG: Using manually selected YAML: {yaml_path}")

    def prompt_for_model_file(self, dialog):
        """Prompts user to select a model file"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Model Not Found")
        msg_box.setText("Could not automatically find the latest trained model.")
        msg_box.setInformativeText("Would you like to manually select a model file?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)

        if msg_box.exec_() == QMessageBox.Yes:
            model_path, _ = QFileDialog.getOpenFileName(
                self, "Select Model File", "", "PyTorch Models (*.pt)")
            if model_path and os.path.exists(model_path):
                dialog.model_path_input.setText(model_path)
                dialog.setWindowTitle("Retrain Model - Using Selected Weights")
            else:
                # Fall back to default model
                default_model_path = "/home/sakar02/sakar-vision-ui/yolo11n.pt"
                if os.path.exists(default_model_path):
                    dialog.model_path_input.setText(default_model_path)
                    dialog.setWindowTitle("Retrain Model - Using Default Weights")

    def runTraining(self, params):
        """Runs the training process in a separate thread"""
        import threading
        from ultralytics import YOLO

        model_path = params['model_path']
        yaml_path = params['yaml_path']
        epochs = params['epochs']
        imgsz = params['imgsz']
        batch = params['batch']
        lr0 = params['lr0']
        lrf = params['lrf']
        dropout = params['dropout']
        device = params['device']
        seed = params['seed']

        # Determine if this is a new training run or retraining
        is_retraining = "retrain" in model_path.lower() or "best.pt" in model_path.lower()

        # Create and show the progress dialog
        progress_dialog = TrainingProgressDialog(is_retraining, self)

        # Define the training function to run in a separate thread
        def train_thread_func():
            try:
                # Load the YOLO model
                model = YOLO(model_path)

                # Ensure the run folder is created in the selected dataset folder
                run_folder = os.path.join(os.path.dirname(yaml_path), 'runs')
                os.makedirs(run_folder, exist_ok=True)

                # Train the model with specified parameters
                results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    optimizer='auto',
                    lr0=lr0,
                    lrf=lrf,
                    dropout=dropout,
                    device=device,
                    seed=seed,
                    project=run_folder
                )

                print(f"Training complete.")

                # Get the output directory of the training run
                if hasattr(results, 'save_dir'):
                    output_dir = results.save_dir
                else:
                    output_dir = "unknown"

                # Update the dialog with success
                progress_dialog.set_complete(output_dir)

            except Exception as e:
                error_msg = str(e)
                print(f"Error during training: {error_msg}")
                progress_dialog.set_error(error_msg)

        # Start the training in a separate thread
        training_thread = threading.Thread(target=train_thread_func)
        training_thread.daemon = True
        training_thread.start()

        # Show the dialog
        progress_dialog.exec_()

    def show_auto_annotation_ui(self):
        """Navigates to the auto annotation UI."""
        # Synchronize classes before navigating
        self.syncClassesToAutoAnnotation()

        parent = self.parent()
        stacked_widget = None
        while parent is not None:
            if isinstance(parent, QWidget) and hasattr(parent, 'stacked_widget') and isinstance(getattr(parent, 'stacked_widget'), QStackedWidget):
                stacked_widget = parent.stacked_widget
                break
            parent = parent.parent()

        if stacked_widget is not None:
            stacked_widget.setCurrentIndex(4)  # Switch to auto annotation tool view

    def syncClassesToAutoAnnotation(self):
        """Synchronizes the current classes to the Auto Annotation UI if it exists."""
        parent = self.parent()
        auto_annotate_ui = None

        print(f"Starting sync of classes to Auto Annotation UI: {self.classes}")

        # First try to find the auto_annotate_ui directly in the parent's children
        while parent is not None:
            if hasattr(parent, 'auto_annoate_ui'):
                auto_annotate_ui = parent.auto_annoate_ui
                print(f"Found auto_annoate_ui object in parent: {parent}")
                break
            else:
                print(f"Parent does not have auto_annoate_ui: {parent}")
            parent = parent.parent()

        # If found, update its class names
        if auto_annotate_ui and hasattr(auto_annotate_ui, 'classNamesLineEdit'):
            classes_str = ", ".join(self.classes)
            auto_annotate_ui.classNamesLineEdit.setText(classes_str)
            print(f"Successfully synchronized classes to Auto Annotation UI: {classes_str}")
        else:
            if auto_annotate_ui:
                print(
                    f"Auto Annotation UI found but it doesn't have classNamesLineEdit: {auto_annotate_ui}")
            else:
                print("Auto Annotation UI not found in parent hierarchy")

    def updateDefectsConfigWithAllClasses(self):
        """Updates defects_config.json with both original and new classes."""
        try:
            # Read existing config to preserve user and timestamp info
            config = {
                "selected_defects": [],
                "user": "unknown",
                "timestamp": datetime.datetime.now().isoformat()
            }

            if os.path.exists(DEFECTS_CONFIG_PATH):
                try:
                    with open(DEFECTS_CONFIG_PATH, 'r') as f:
                        existing_config = json.load(f)
                        config["user"] = existing_config.get("user", "unknown")
                        config["timestamp"] = existing_config.get(
                            "timestamp", datetime.datetime.now().isoformat())
                except Exception as e:
                    print(f"Error reading existing defects config: {e}")

            # Update with all classes (original + new)
            config["selected_defects"] = self.classes.copy()

            # Write updated config
            with open(DEFECTS_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"Updated defects_config.json with all {len(self.classes)} classes")
            return True

        except Exception as e:
            print(f"Error updating defects config with all classes: {e}")
            return False


# Training Dialog Class
# Training Dialog Class - FIXED VERSION
class TrainDialog(QDialog):
    def __init__(self, parent=None, is_retrain=False):
        super().__init__(parent)
        self.setWindowTitle("Train Model" if not is_retrain else "Retrain Model")
        self.setMinimumSize(550, 650)  # Changed from fixed to minimum size
        self.resize(550, 650)  # Set initial size
        self.is_retrain = is_retrain

        # Apply modern styling with better spacing
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            QLabel {
                color: #374151;
                font-size: 14px;
                font-weight: 500;
                padding: 2px 0px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                min-height: 20px;
                min-width: 200px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #ff914d;
                outline: none;
            }
            QPushButton {
                background-color: #ff914d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                min-width: 100px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #ff8a42;
            }
            QPushButton:pressed {
                background-color: #ff6303;
            }
        """)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # Header
        header = QLabel("Model Training Configuration")
        header.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #1F2937;
                margin-bottom: 8px;
                padding: 0px;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Create scroll area for the form
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # Form widget
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(16)
        form_layout.setContentsMargins(0, 0, 0, 0)

        # File paths section
        paths_group = QGroupBox("Model & Data Paths")
        paths_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #374151;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: white;
            }
        """)
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.setSpacing(12)

        # Model path
        model_path_layout = QVBoxLayout()
        model_path_layout.setSpacing(6)
        model_path_label = QLabel("Model Path:")
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Select a model file (.pt)")

        model_button_layout = QHBoxLayout()
        self.model_path_button = QPushButton("Choose Model File")
        self.model_path_button.clicked.connect(self.choose_model_path)
        model_button_layout.addWidget(self.model_path_button)
        model_button_layout.addStretch()

        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_input)
        model_path_layout.addLayout(model_button_layout)
        paths_layout.addLayout(model_path_layout)

        # Data YAML path
        yaml_path_layout = QVBoxLayout()
        yaml_path_layout.setSpacing(6)
        yaml_path_label = QLabel("Dataset YAML Path:")
        self.yaml_path_input = QLineEdit()
        self.yaml_path_input.setPlaceholderText("Select dataset configuration file (.yaml)")

        yaml_button_layout = QHBoxLayout()
        self.yaml_path_button = QPushButton("Choose YAML File")
        self.yaml_path_button.clicked.connect(self.choose_yaml_path)
        yaml_button_layout.addWidget(self.yaml_path_button)
        yaml_button_layout.addStretch()

        yaml_path_layout.addWidget(yaml_path_label)
        yaml_path_layout.addWidget(self.yaml_path_input)
        yaml_path_layout.addLayout(yaml_button_layout)
        paths_layout.addLayout(yaml_path_layout)

        form_layout.addWidget(paths_group)

        # Training parameters section
        params_group = QGroupBox("Training Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #374151;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: white;
            }
        """)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(12)

        # Create parameter input fields with better layout
        param_fields = [
            ("Epochs:", "epochs_input", QSpinBox, (1, 1000, 100)),
            ("Image Size:", "imgsz_input", QSpinBox, (32, 2048, 640)),
            ("Batch Size:", "batch_input", QSpinBox, (1, 128, 16)),
            ("Initial Learning Rate:", "lr0_input", QDoubleSpinBox, (1e-6, 1.0, 0.0001)),
            ("Final Learning Rate Multiplier:", "lrf_input", QDoubleSpinBox, (1e-6, 1.0, 0.01)),
            ("Dropout:", "dropout_input", QDoubleSpinBox, (0.0, 1.0, 0.25)),
            ("Device (GPU ID):", "device_input", QSpinBox, (0, 8, 0)),
            ("Random Seed:", "seed_input", QSpinBox, (0, 10000, 42))
        ]

        for label_text, attr_name, widget_class, (min_val, max_val, default_val) in param_fields:
            param_layout = QHBoxLayout()
            param_layout.setSpacing(12)

            label = QLabel(label_text)
            label.setMinimumWidth(180)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            widget = widget_class()
            widget.setRange(min_val, max_val)
            if widget_class == QDoubleSpinBox:
                widget.setDecimals(6)
                widget.setSingleStep(0.0001 if attr_name in ['lr0_input', 'lrf_input'] else 0.01)
            widget.setValue(default_val)
            widget.setMinimumWidth(120)
            widget.setButtonSymbols(QSpinBox.NoButtons if widget_class ==
                                    QSpinBox else QDoubleSpinBox.NoButtons)
            widget.installEventFilter(self)  # Install event filter to block wheel events

            setattr(self, attr_name, widget)

            param_layout.addWidget(label)
            param_layout.addWidget(widget)
            param_layout.addStretch()

            params_layout.addLayout(param_layout)

        form_layout.addWidget(params_group)
        form_layout.addStretch()

        # Set the form widget to scroll area
        scroll_area.setWidget(form_widget)
        main_layout.addWidget(scroll_area, 1)

        # Buttons at bottom
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                min-width: 100px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)

        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.accept)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.start_button)

        main_layout.addLayout(button_layout)

    def eventFilter(self, source, event):
        # Block wheel events for spinboxes
        if isinstance(source, (QSpinBox, QDoubleSpinBox)) and event.type() == QEvent.Wheel:
            return True  # Block wheel events
        return super().eventFilter(source, event)

    def choose_model_path(self):
        """Opens file dialog to choose model path"""
        initial_dir = ""
        if self.model_path_input.text():
            initial_dir = os.path.dirname(self.model_path_input.text())

        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            initial_dir,
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if model_path:
            self.model_path_input.setText(model_path)

    def choose_yaml_path(self):
        """Opens file dialog to choose YAML path"""
        initial_dir = ""
        if self.yaml_path_input.text():
            initial_dir = os.path.dirname(self.yaml_path_input.text())

        yaml_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset YAML File",
            initial_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if yaml_path:
            self.yaml_path_input.setText(yaml_path)

    def get_params(self):
        """Returns training parameters as dictionary"""
        return {
            'model_path': self.model_path_input.text(),
            'yaml_path': self.yaml_path_input.text(),
            'epochs': self.epochs_input.value(),
            'imgsz': self.imgsz_input.value(),
            'batch': self.batch_input.value(),
            'lr0': self.lr0_input.value(),
            'lrf': self.lrf_input.value(),
            'dropout': self.dropout_input.value(),
            'device': self.device_input.value(),
            'seed': self.seed_input.value()
        }

    def accept(self):
        """Override accept to validate inputs"""
        # Validate model path
        if not self.model_path_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Please select a model file.")
            return

        if not os.path.exists(self.model_path_input.text()):
            QMessageBox.warning(self, "Validation Error", "Model file does not exist.")
            return

        # Validate YAML path
        if not self.yaml_path_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Please select a dataset YAML file.")
            return

        if not os.path.exists(self.yaml_path_input.text()):
            QMessageBox.warning(self, "Validation Error", "Dataset YAML file does not exist.")
            return

        # All validations passed
        super().accept()

        # Main application entry point
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Sakar Vision")

    # Create and show the main window
    ex = ImageAnnotationTool()

    # Load images from folder if specified on startup
    if hasattr(ex, 'image_folder_on_start') and ex.image_folder_on_start:
        ex.loadImagesFromFolder(ex.image_folder_on_start)

    ex.show()

    # Center the window on screen
    screen = app.primaryScreen()
    screen_geometry = screen.availableGeometry()
    window_geometry = ex.frameGeometry()
    center_point = screen_geometry.center()
    window_geometry.moveCenter(center_point)
    ex.move(window_geometry.topLeft())

    print("=" * 50)
    print("Features:")
    print("- Modern sidebar UI design")
    print("- Class management with visual tags")
    print("- Bounding box annotation")
    print("- Dataset splitting and training")
    print("- Undo/Redo functionality")
    print("- Image resizing options")
    print("- Auto annotation integration")
    print("=" * 50)

    sys.exit(app.exec_())
