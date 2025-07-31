#!/usr/bin/env python3
"""
auto_annotation.py - Comprehensive Auto Annotation and Model Training Interface

This module provides a sophisticated PyQt5-based GUI application for computer vision model training workflows,
specifically designed for YOLO object detection models with Azure cloud integration. The application serves as
a complete pipeline from data annotation to model deployment, featuring modern UI design and advanced functionality.

CORE FUNCTIONALITY OVERVIEW:
============================

1. AUTO ANNOTATION SYSTEM:
   - Automated object detection using pre-trained YOLO models with configurable confidence thresholds
   - Intelligent bounding box generation with class-specific color coding for visual distinction
   - Interactive annotation review system with manual editing capabilities through QGraphicsView/Scene
   - Support for multiple object classes with customizable names and automatic class extraction from models
   - Real-time preview and validation of annotations before final dataset creation

2. DATASET MANAGEMENT & SPLITTING:
   - Advanced dataset splitting with visual range slider for train/validation/test ratios (60%/20%/20% default)
   - Intelligent file pairing system that maintains image-label correspondence across splits
   - Automatic YAML configuration file generation with proper dataset structure for YOLO training
   - Support for multiple image formats (PNG, JPG, JPEG, BMP, TIFF) with automatic detection

3. AZURE CLOUD INTEGRATION:
   - Seamless upload/download capabilities with Azure Blob Storage for large-scale dataset management
   - Progress tracking with real-time file count and percentage indicators during cloud operations
   - Robust cancellation support and error handling for network interruptions
   - Automatic backup creation and overwrite protection for cloud-stored datasets

4. MODEL TRAINING WORKFLOWS:
   - Fast Training: Local dataset combination (2 datasets) for quick model iterations
   - Accurate Training: Comprehensive workflow (Azure download + 3 dataset combination + upload + training)
   - Configurable training parameters including epochs, batch size, learning rates, dropout, and device selection
   - Progress monitoring with persistent dialogs that prevent accidental cancellation during training
   - Automatic model versioning and latest model detection from training runs

5. ADVANCED UI COMPONENTS:
   - Custom RangeSlider widget with visual color-coded segments for dataset split visualization
   - Modern card-based layout with responsive design and professional styling
   - Interactive graphics editing with EditableRectItem for precise bounding box adjustment
   - Real-time validation and status feedback throughout all operations

TECHNICAL ARCHITECTURE:
=======================

The application follows a modular design pattern with specialized dialog classes for different workflows:
- TrainingProgressDialog: Non-closable progress tracking for model training operations
- AzureUploadProgressDialog/AzureDownloadProgressDialog: Cloud operation monitoring with cancellation support
- ManualAnnotationDialog: QGraphicsView-based annotation editor with drawing and editing tools
- AnnotationReviewDialog: Image-by-image review system with approve/reject/edit functionality
- CombineDatasetProgressDialog: Dataset merging operations with progress tracking
- FastTrainingOverlayDialog/AccurateTrainingOverlayDialog: Multi-tab configuration interfaces

The system integrates with external modules including YOLO (Ultralytics), Azure SDK, and custom utilities
for class management and database operations, providing a complete end-to-end solution for computer vision
model development and deployment in production environments.
"""


import datetime
import os
import random
import re
import shutil
import sys
import threading

import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QObject, QTimerEvent
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import QRectF
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTabWidget
from PyQt5.QtWidgets import QLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPolygon
from PyQt5.QtWidgets import QSpinBox
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QMenu
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QScrollArea

from ultralytics import YOLO
from deploy_ui import DeploymentUI
from utils import ClassManager
from utils import set_window_icon

# Ensure the Qt plugin path is set (if needed)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'

# -------------------------
# Custom RangeSlider Widget (improved design)
# -------------------------


class RangeSlider(QWidget):
    valueChanged = pyqtSignal(int, int)  # Emits (lowerValue, upperValue)

    def __init__(self, minimum=0, maximum=100, lowerValue=60, upperValue=80, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._lowerValue = lowerValue
        self._upperValue = upperValue
        self._handleRadius = 12  # Larger handle for better look
        self._movingHandle = None
        self.setMinimumHeight(50)
        self.setMinimumWidth(300)

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
        track_pen = QPen(QColor("#E0E4E7"), 8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(track_pen)
        painter.drawLine(start_x, line_y, end_x, line_y)

        # Draw colored segments
        lower_x = self._valueToPos(self._lowerValue)
        upper_x = self._valueToPos(self._upperValue)

        # Train segment (blue)
        train_pen = QPen(QColor("#4285F4"), 8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(train_pen)
        painter.drawLine(start_x, line_y, lower_x, line_y)

        # Valid segment (green)
        valid_pen = QPen(QColor("#34A853"), 8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(valid_pen)
        painter.drawLine(lower_x, line_y, upper_x, line_y)

        # Test segment (orange)
        test_pen = QPen(QColor("#FF9800"), 8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(test_pen)
        painter.drawLine(upper_x, line_y, end_x, line_y)

        # Draw handles: white fill with shadow
        handle_brush = QBrush(QColor("#FFFFFF"))
        handle_pen = QPen(QColor("#DADCE0"), 2)

        # Add shadow effect
        shadow_brush = QBrush(QColor(0, 0, 0, 30))
        painter.setBrush(shadow_brush)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(lower_x + 2, line_y + 2),
                            self._handleRadius, self._handleRadius)
        painter.drawEllipse(QPoint(upper_x + 2, line_y + 2),
                            self._handleRadius, self._handleRadius)

        # Draw actual handles
        painter.setBrush(handle_brush)
        painter.setPen(handle_pen)
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

# -------------------------
# Dialog to edit a bounding box's coordinates manually
# -------------------------


class EditBoxDialog(QDialog):
    def __init__(self, x, y, w, h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Bounding Box")
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QLabel {
                font-size: 14px;
                color: #202124;
                font-weight: 500;
            }
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus {
                border: 2px solid #4285F4;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        self.xEdit = QLineEdit(str(x))
        self.yEdit = QLineEdit(str(y))
        self.wEdit = QLineEdit(str(w))
        self.hEdit = QLineEdit(str(h))

        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.xEdit)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.yEdit)
        layout.addWidget(QLabel("Width:"))
        layout.addWidget(self.wEdit)
        layout.addWidget(QLabel("Height:"))
        layout.addWidget(self.hEdit)

        buttonLayout = QHBoxLayout()
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")
        cancelButton.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                color: #3C4043;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)
        buttonLayout.addWidget(cancelButton)
        buttonLayout.addWidget(okButton)
        layout.addLayout(buttonLayout)

    def getValues(self):
        try:
            return int(self.xEdit.text()), int(self.yEdit.text()), int(self.wEdit.text()), int(self.hEdit.text())
        except Exception as e:
            print("Invalid input:", e)
            return None

# -------------------------
# Custom Class Selection Dialog with modern styling
# -------------------------


class ClassSelectionDialog(QDialog):
    def __init__(self, parent=None, class_names=None, title="Select Class"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.selected_class = ""
        self.selected_index = 0

        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QComboBox {
                padding: 12px 16px;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                font-size: 14px;
                background-color: #FFFFFF;
                min-width: 200px;
            }
            QComboBox:focus {
                border: 2px solid #4285F4;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #202124;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                selection-background-color: #E8F0FE;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #F8F9FA;
                color: #4285F4;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Create a styled combo box
        self.comboBox = QComboBox()
        self.comboBox.addItems(class_names if class_names else ["Default"])
        layout.addWidget(self.comboBox)

        # Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def accept(self):
        self.selected_class = self.comboBox.currentText()
        self.selected_index = self.comboBox.currentIndex()
        super().accept()

# -------------------------
# Custom QGraphicsRectItem that is movable and editable on double-click
# -------------------------


class EditableRectItem(QGraphicsRectItem):
    def __init__(self, rect, detection, *args, **kwargs):
        super().__init__(rect, *args, **kwargs)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.detection = detection  # dictionary storing 'class_id', 'confidence'

        # Set color based on detection if available
        if 'color' in detection:
            color = detection['color']
            # Convert BGR (OpenCV) to RGB (Qt)
            if isinstance(color, tuple) and len(color) == 3:
                # Note: OpenCV uses BGR but we need RGB for Qt
                r, g, b = color[2], color[1], color[0]
                self.setPen(QPen(QColor(r, g, b), 2))
            else:
                self.setPen(QPen(Qt.green, 2))
        else:
            self.setPen(QPen(Qt.green, 2))

        self.setBrush(Qt.transparent)

    def mouseDoubleClickEvent(self, event):
        rect = self.rect()
        dialog = EditBoxDialog(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.getValues()
            if values:
                x, y, w, h = values
                self.setRect(x, y, w, h)
        super().mouseDoubleClickEvent(event)

# -------------------------
# Custom QGraphicsScene to support drawing new boxes in manual annotation mode
# -------------------------


class AnnotationScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.startPoint = None
        self.tempRect = None  # Temporary rectangle during drawing
        self.class_colors = {}  # Will be populated by ManualAnnotationDialog

    def mousePressEvent(self, event):
        if self.drawing:
            self.startPoint = event.scenePos()
            self.tempRect = self.addRect(QRectF(self.startPoint, self.startPoint), QPen(Qt.red, 2))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.tempRect:
            rect = QRectF(self.startPoint, event.scenePos()).normalized()
            self.tempRect.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and self.tempRect:
            rect = self.tempRect.rect()
            self.removeItem(self.tempRect)
            self.tempRect = None
            # When finished drawing, prompt user for the class label.

            class_names = self.parent().class_names if hasattr(self.parent(), 'class_names') else []
            if class_names:
                dialog = ClassSelectionDialog(self.parent(), class_names, "Select Class")
                if dialog.exec_() == QDialog.Accepted:
                    class_id = dialog.selected_index
                else:
                    class_id = 0
            else:
                class_id = 0

            # Get color for this class
            color = self.class_colors.get(class_id, (0, 255, 0))

            # Create detection dictionary with color
            detection = {
                'class_id': class_id,
                'confidence': 1.0,
                'color': color
            }

            # Create an editable rectangle item with the detection
            new_item = EditableRectItem(rect, detection)
            self.addItem(new_item)
            self.drawing = False  # Exit drawing mode after adding one box.
        else:
            super().mouseReleaseEvent(event)

    def showClassMenu(self, pos):
        """Show a menu for class selection at the given position"""
        class_names = self.parent().class_names if hasattr(self.parent(), 'class_names') else []
        if not class_names:
            return 0

        menu = QMenu(self.parent())
        menu.setStyleSheet("""
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 16px;
                color: #202124;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #E8F0FE;
                color: #4285F4;
            }
        """)

        actions = []
        for i, name in enumerate(class_names):
            action = menu.addAction(name)
            actions.append(action)

        selected_action = menu.exec_(self.parent().mapToGlobal(pos))

        if selected_action:
            return class_names.index(selected_action.text())
        return 0


# -------------------------
# Manual Annotation Dialog using QGraphicsView/Scene for editing bounding boxes
# -------------------------
class ManualAnnotationDialog(QDialog):
    def __init__(self, image, detections, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Annotation Editor")
        self.class_names = class_names  # list of class names (e.g., ["keyboard", "phone", "buds"])
        self.image = image  # original image (numpy array, BGR)

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
                min-height: 16px;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
            QPushButton:pressed {
                background-color: #2850A0;
            }
        """)

        # Generate class colors if not already in detections
        self.class_colors = {}
        self.generate_class_colors(class_names)

        # Convert image to QPixmap
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_img.shape
        bytesPerLine = 3 * width
        q_img = QImage(rgb_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_img)

        # Create a QGraphicsScene and QGraphicsView
        self.scene = AnnotationScene(self)
        self.scene.setSceneRect(0, 0, self.pixmap.width(), self.pixmap.height())
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMinimumSize(600, 400)
        self.view.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #DADCE0;
                border-radius: 8px;
                background-color: #FFFFFF;
            }
        """)

        # Add the image as background
        self.scene.addPixmap(self.pixmap)

        # Pass class colors to the scene
        self.scene.class_colors = self.class_colors

        # Add existing detections (if any)
        for det in detections:
            # Make sure each detection has a color
            if 'color' not in det:
                det['color'] = self.class_colors.get(det['class_id'], (0, 255, 0))

            # det['box'] is (x1, y1, x2, y2); convert to (x, y, width, height)
            x1, y1, x2, y2 = det['box']
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = EditableRectItem(rect, det)
            self.scene.addItem(item)

        # Buttons: Add Box, Delete Box, Save, Cancel
        self.addBoxButton = QPushButton("Add Box")
        self.deleteBoxButton = QPushButton("Delete Selected Box")
        self.saveButton = QPushButton("Save")
        self.cancelButton = QPushButton("Cancel")

        # Special styling for different button types
        self.addBoxButton.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)

        self.deleteBoxButton.setStyleSheet("""
            QPushButton {
                background-color: #EA4335;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)

        self.cancelButton.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #3C4043;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)

        self.addBoxButton.clicked.connect(self.activateDrawingMode)
        self.deleteBoxButton.clicked.connect(self.deleteSelectedBox)
        self.saveButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

        btnLayout = QHBoxLayout()
        btnLayout.setSpacing(12)
        btnLayout.addWidget(self.addBoxButton)
        btnLayout.addWidget(self.deleteBoxButton)
        btnLayout.addStretch()
        btnLayout.addWidget(self.cancelButton)
        btnLayout.addWidget(self.saveButton)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(self.view)
        layout.addLayout(btnLayout)
        self.setLayout(layout)

    def generate_class_colors(self, class_names):
        """Generates distinct colors for each class."""
        # Predefined colors for better distinction (BGR format for OpenCV)
        color_list = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Brown
            (128, 128, 0),  # Teal
            (0, 0, 128),    # Dark Red
        ]

        # If we have more classes than predefined colors, generate random ones
        for i in range(len(class_names)):
            if i < len(color_list):
                self.class_colors[i] = color_list[i]
            else:
                # Generate a random color
                while True:
                    color = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)
                    )
                    # Ensure the color is not too similar to existing ones
                    if color not in self.class_colors.values():
                        self.class_colors[i] = color
                        break

        return self.class_colors

    def activateDrawingMode(self):
        # Enable drawing mode in the scene.
        self.scene.drawing = True

    def deleteSelectedBox(self):
        # Remove any selected EditableRectItem from the scene.
        for item in self.scene.selectedItems():
            if isinstance(item, EditableRectItem):
                self.scene.removeItem(item)

    def getDetections(self):
        # Return a list of detection dictionaries from the items in the scene.
        detections = []
        for item in self.scene.items():
            if isinstance(item, EditableRectItem):
                rect = item.rect()
                # Convert rect (x, y, width, height) to (x1, y1, x2, y2)
                x1 = int(rect.x())
                y1 = int(rect.y())
                x2 = int(rect.x() + rect.width())
                y2 = int(rect.y() + rect.height())

                # Create detection with all properties from the original
                detection = dict(item.detection)  # Copy all original attributes

                # Update with new coordinates
                detection.update({
                    'box': (x1, y1, x2, y2)
                })

                # Make sure color is included
                if 'color' not in detection:
                    detection['color'] = self.class_colors.get(detection['class_id'], (0, 255, 0))

                detections.append(detection)
        return detections

# -------------------------
# Annotation Review Dialog with Navigation and Manual Edit Option
# -------------------------


class AnnotationReviewDialog(QDialog):
    def __init__(self, annotated_files, annotated_images_dict, detections_dict, class_names, input_dir, output_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Annotated Images")
        self.annotated_files = annotated_files  # List of filenames (e.g., "img1.jpg")
        # Dict: filename -> annotated image (numpy array, BGR)
        self.annotated_images_dict = annotated_images_dict
        # Dict: filename -> list of detections (each is a dict with keys 'class_id', 'box', 'confidence')
        self.detections_dict = detections_dict
        self.class_names = class_names
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.current_index = 0
        self.results = {}  # filename -> True (confirmed) or False (rejected)

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
            }
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                padding: 16px;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)

        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(600, 400)

        self.prevButton = QPushButton("Previous")
        self.nextButton = QPushButton("Next")
        self.yesButton = QPushButton("✓ Approve")
        self.noButton = QPushButton("✗ Delete")
        self.editButton = QPushButton("✏️ Edit")

        # Special styling for action buttons
        self.yesButton.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)

        self.noButton.setStyleSheet("""
            QPushButton {
                background-color: #EA4335;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)

        self.editButton.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)

        self.prevButton.clicked.connect(self.showPrevious)
        self.nextButton.clicked.connect(self.showNext)
        self.yesButton.clicked.connect(self.confirmYes)
        self.noButton.clicked.connect(self.confirmNo)
        self.editButton.clicked.connect(self.editAnnotation)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(self.imageLabel)

        navLayout = QHBoxLayout()
        navLayout.setSpacing(12)
        navLayout.addWidget(self.prevButton)
        navLayout.addWidget(self.nextButton)
        navLayout.addStretch()
        navLayout.addWidget(self.yesButton)
        navLayout.addWidget(self.noButton)
        navLayout.addWidget(self.editButton)

        layout.addLayout(navLayout)
        self.setLayout(layout)
        self.updateImage()

    def updateImage(self):
        if not self.annotated_files:
            self.imageLabel.setText("No images to review.")
            self.prevButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.yesButton.setEnabled(False)
            self.noButton.setEnabled(False)
            self.editButton.setEnabled(False)
            self.setWindowTitle("Review Annotated Images")
            return
        filename = self.annotated_files[self.current_index]
        # Get the annotated image (numpy array in BGR) from the dictionary and convert to QPixmap.
        annotated_img = self.annotated_images_dict.get(filename)
        if annotated_img is not None:
            rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_img.shape
            bytesPerLine = 3 * width
            q_img = QImage(rgb_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.imageLabel.setPixmap(pixmap.scaled(
                600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.imageLabel.setText("Image not found.")
        self.setWindowTitle(
            f"Review {self.current_index+1}/{len(self.annotated_files)}: {self.annotated_files[self.current_index]}")

    def showPrevious(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.updateImage()

    def showNext(self):
        if self.current_index < len(self.annotated_files) - 1:
            self.current_index += 1
            self.updateImage()

    def confirmYes(self):
        filename = self.annotated_files[self.current_index]
        self.results[filename] = True
        self.moveNextOrClose()

    def confirmNo(self):
        filename = self.annotated_files[self.current_index]
        self.results[filename] = False
        del self.annotated_files[self.current_index]
        if self.current_index >= len(self.annotated_files) and self.current_index > 0:
            self.current_index -= 1
        self.updateImage()

        # Delete the image and its corresponding label file from the output directory
        output_image_path = os.path.join(self.output_dir, filename)
        label_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.txt")
        for path in [output_image_path, label_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Deleted {path}")
                except Exception as e:
                    print(f"Error deleting {path}: {e}")

    def editAnnotation(self):
        # Open the manual annotation dialog for the current image.
        try:
            filename = self.annotated_files[self.current_index]

            # Ensure we have a valid original image
            original_img_path = os.path.join(self.input_dir, filename)
            if not os.path.exists(original_img_path):
                # Fallback to output dir if input image not found
                original_img_path = os.path.join(self.output_dir, filename)
                if not os.path.exists(original_img_path):
                    QMessageBox.warning(
                        self, "Error", f"Cannot find original image at: {original_img_path}")
                    return

            original_image = cv2.imread(original_img_path)
            if original_image is None:
                QMessageBox.warning(self, "Error", "Failed to load original image!")
                return

            # Ensure we have a valid detections list (even if empty)
            current_detections = self.detections_dict.get(filename, [])
            if current_detections is None:
                current_detections = []

            # Open the manual annotation dialog
            dialog = ManualAnnotationDialog(
                original_image, current_detections, self.class_names, self)
            if dialog.exec_() == QDialog.Accepted:
                # Get the updated detections from the dialog
                new_detections = dialog.getDetections()
                self.detections_dict[filename] = new_detections

                # Regenerate annotated image based on new detections
                annotated_image = original_image.copy()
                for det in new_detections:
                    x1, y1, x2, y2 = det['box']
                    class_id = det['class_id']

                    # Use class color if available, otherwise use green
                    color = det.get('color', (0, 255, 0))

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{self.class_names[class_id]} {det['confidence']:.2f}" if self.class_names else f"Class {class_id} {det['confidence']:.2f}"
                    cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                self.annotated_images_dict[filename] = annotated_image
                self.results[filename] = True  # Mark as confirmed after manual editing.
                self.updateImage()

                # --- Real-time update of label.txt ---
                output_image_path = os.path.join(self.output_dir, filename)
                try:
                    image = cv2.imread(output_image_path)
                    if image is not None:
                        label_file_path = os.path.join(
                            self.output_dir, f"{os.path.splitext(filename)[0]}.txt")
                        with open(label_file_path, 'w') as f:
                            for det in new_detections:
                                x1, y1, x2, y2 = det['box']
                                class_id = det['class_id']
                                x_center = ((x1 + x2) / 2) / image.shape[1]
                                y_center = ((y1 + y2) / 2) / image.shape[0]
                                box_width = (x2 - x1) / image.shape[1]
                                box_height = (y2 - y1) / image.shape[0]
                                f.write(
                                    f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
                except Exception as e:
                    print(f"Error updating label file: {e}")
                # -----------------------------------------
        except Exception as e:
            print(f"Error in editAnnotation: {e}")
            QMessageBox.warning(self, "Error", f"An error occurred during editing: {str(e)}")

    def moveNextOrClose(self):
        if self.current_index < len(self.annotated_files) - 1:
            self.current_index += 1
            self.updateImage()
        else:
            self.accept()

# -------------------------
# Training Progress Dialog to show persistent progress message
# -------------------------


class TrainingProgressDialog(QDialog):
    """A persistent dialog that shows training progress and cannot be closed until training completes."""

    def __init__(self, is_retrain=False, parent=None):
        super().__init__(parent)
        self.is_retrain = is_retrain
        self.training_type = "Retraining" if is_retrain else "Training"
        self.setWindowTitle(f"Model {self.training_type} Progress")
        self.setMinimumSize(450, 250)
        self.setModal(True)

        # Prevent closing with X button or Esc key
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QLabel {
                color: #202124;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(32, 32, 32, 32)
        self.layout.setSpacing(20)

        # Status message
        self.status_label = QLabel(
            f"Model {self.training_type.lower()} is in progress.\nThis may take a while depending on your dataset size and parameters.")
        self.status_label.setStyleSheet(
            "font-size: 16px; margin-bottom: 10px; text-align: center;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # Progress animation (just a text label that will be updated)
        self.progress_label = QLabel("Working...")
        self.progress_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4285F4;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.progress_label)

        # Add some instructions
        instruction_text = (
            "Please wait for the training to complete.\n"
            "This window will close automatically when finished.\n\n"
            "You can check the console for detailed progress information."
        )
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("color: #5F6368; font-size: 14px;")
        self.layout.addWidget(self.instruction_label)

        # Animation timer for updating the progress indicator
        self.animation_counter = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(500)  # Update every 500ms

        # This will be set to True when training completes
        self.finished = False

    def update_animation(self):
        """Updates the animation text to show ongoing activity."""
        self.animation_counter += 1
        dots = "." * (self.animation_counter % 6 + 1)
        self.progress_label.setText(f"{self.training_type} in progress{dots}")

    def set_complete(self, output_dir):
        """Called when training completes successfully."""
        self.animation_timer.stop()
        self.progress_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #34A853;")
        self.progress_label.setText(f"{self.training_type} Complete!")
        self.status_label.setText(
            f"Model {self.training_type.lower()} has completed successfully.\nResults saved in: {output_dir}")
        self.instruction_label.setText("You can close this window now.")

        # Add a close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.layout.addWidget(self.close_button)

        self.finished = True

    def set_error(self, error_message):
        """Called when training encounters an error."""
        self.animation_timer.stop()
        self.progress_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #EA4335;")
        self.progress_label.setText(f"{self.training_type} Failed!")
        self.status_label.setText(f"An error occurred during model {self.training_type.lower()}:")

        # Add error details in a scrollable text area
        self.error_text = QLabel(error_message)
        self.error_text.setStyleSheet(
            "color: #EA4335; background-color: #FFEBEE; padding: 12px; border-radius: 8px; border: 1px solid #FFCDD2;")
        self.error_text.setWordWrap(True)
        self.layout.addWidget(self.error_text)

        # Add a close button
        self.close_button = QPushButton("Close")
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

# -------------------------
# Azure Upload Progress Dialog
# -------------------------


class AzureUploadProgressDialog(QDialog):
    """Custom progress dialog for Azure upload process with real progress tracking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancelled = False
        self.cancel_requested = False
        self.setWindowTitle("Uploading to Azure")
        self.setMinimumSize(500, 300)
        self.setModal(True)

        # Initially prevent closing with X button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        # Apply modern Azure-themed styling
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QLabel {
                color: #202124;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:disabled {
                background-color: #6C757D;
                color: #ADB5BD;
            }
            QProgressBar {
                border: 2px solid #E1E4E8;
                border-radius: 8px;
                text-align: center;
                background-color: #F6F8FA;
                height: 30px;
                font-size: 14px;
                font-weight: bold;
                color: #24292E;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 6px;
            }
        """)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Title
        self.titleLabel = QLabel("Uploading to Azure Storage")
        self.titleLabel.setStyleSheet("color: #0078D4; font-size: 18px; font-weight: bold;")
        self.titleLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.titleLabel)

        # Status label
        self.statusLabel = QLabel("Preparing upload...")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setMinimumHeight(40)
        layout.addWidget(self.statusLabel)

        # Progress bar with percentage display
        self.progressBar = QProgressBar()
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, 100)  # Determinate mode for real progress
        self.progressBar.setValue(0)
        self.progressBar.setFormat("0%")  # Explicitly set initial format
        layout.addWidget(self.progressBar)

        # File counter label
        self.fileCounterLabel = QLabel("Files: 0 / 0")
        self.fileCounterLabel.setAlignment(Qt.AlignCenter)
        self.fileCounterLabel.setStyleSheet("color: #5F6368; font-size: 12px;")
        layout.addWidget(self.fileCounterLabel)

        # Animated spinner
        self.spinnerLabel = QLabel()
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        self.spinnerLabel.setMinimumHeight(40)

        # Create spinner animation frames
        self.spinnerPixmaps = []
        for i in range(8):  # 8-frame animation
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#0078D4"), 3))
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawArc(4, 4, 24, 24, i * 45 * 16, 180 * 16)
            painter.end()
            self.spinnerPixmaps.append(pixmap)

        self.spinnerLabel.setPixmap(self.spinnerPixmaps[0])
        layout.addWidget(self.spinnerLabel)

        # Cancel button
        self.cancelButton = QPushButton("Cancel Upload")
        self.cancelButton.clicked.connect(self.cancelUpload)

        # Button layout
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.cancelButton)
        buttonLayout.addStretch()
        layout.addLayout(buttonLayout)

        # Animation timer for spinner
        self.animationTimer = QTimer(self)
        self.animationTimer.timeout.connect(self.updateSpinner)
        self.animationTimer.start(150)  # Update every 150ms
        self.animationIndex = 0

        # Completion flag
        self.isComplete = False

    def updateSpinner(self):
        """Updates the spinner animation frame."""
        try:
            if not self.cancelled and not self.cancel_requested and not self.isComplete:
                self.animationIndex = (self.animationIndex + 1) % len(self.spinnerPixmaps)
                self.spinnerLabel.setPixmap(self.spinnerPixmaps[self.animationIndex])
        except Exception as e:
            print(f"Error in updateSpinner: {e}")

    def updateStatus(self, message, progress=None):
        """Updates the status message and progress if provided."""
        try:
            if not self.cancelled and not self.cancel_requested:
                self.statusLabel.setText(message)

                if progress is not None:
                    # Ensure progress is an integer between 0 and 100
                    if isinstance(progress, (int, float)):
                        progress_value = max(0, min(100, int(progress)))
                        self.progressBar.setValue(progress_value)
                        self.progressBar.setFormat(f"{progress_value}%")
                        print(f"Progress bar updated to: {progress_value}%")  # Debug output
                    else:
                        print(f"Invalid progress type: {type(progress)}, value: {progress}")

                # Force immediate UI update
                self.repaint()
                QApplication.processEvents()

        except Exception as e:
            print(f"Error in updateStatus: {e}")

    def updateFileProgress(self, current_file, total_files):
        """Updates the file counter display."""
        try:
            if not self.cancelled and not self.cancel_requested:
                self.fileCounterLabel.setText(f"Files: {current_file} / {total_files}")
                
                # Also update progress bar based on file count
                if total_files > 0:
                    file_progress = int((current_file / total_files) * 100)
                    self.updateStatus(f"Processing file {current_file} of {total_files}", file_progress)
                
                QApplication.processEvents()
        except Exception as e:
            print(f"Error in updateFileProgress: {e}")

    def setProgress(self, value):
        """Directly set progress bar value - useful for debugging."""
        try:
            progress_value = max(0, min(100, int(value)))
            self.progressBar.setValue(progress_value)
            self.progressBar.setFormat(f"{progress_value}%")
            self.repaint()
            QApplication.processEvents()
            print(f"Direct progress set to: {progress_value}%")
        except Exception as e:
            print(f"Error in setProgress: {e}")

    def setComplete(self, stats):
        """Updates the dialog to show completion status with statistics."""
        try:
            # Stop animations
            if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                self.animationTimer.stop()

            # Create a success icon
            successPixmap = QPixmap(32, 32)
            successPixmap.fill(Qt.transparent)
            painter = QPainter(successPixmap)
            painter.setPen(QPen(QColor("#34A853"), 4))
            painter.setBrush(QColor("#34A853"))
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw a checkmark
            points = [QPoint(8, 16), QPoint(14, 22), QPoint(24, 10)]
            painter.drawPolyline(QPolygon(points))
            painter.end()
            self.spinnerLabel.setPixmap(successPixmap)

            # Update status message with statistics
            stats_msg = "Upload completed successfully!\n\n"
            stats_msg += f"✓ Images: {stats.get('images', 0)}\n"
            stats_msg += f"✓ Text files: {stats.get('text_files', 0)}\n"
            stats_msg += f"✓ Documents: {stats.get('documents', 0)}\n"
            stats_msg += f"✓ Code files: {stats.get('code_files', 0)}\n"
            stats_msg += f"✓ Other files: {stats.get('other_files', 0)}"

            self.statusLabel.setText(stats_msg)
            self.statusLabel.setStyleSheet("color: #34A853; font-weight: bold;")
            self.titleLabel.setText("Upload Complete")

            # Update progress bar to 100%
            self.progressBar.setValue(100)
            self.progressBar.setFormat("100% - Complete!")
            self.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #34A853;
                    border-radius: 8px;
                    text-align: center;
                    background-color: #F6F8FA;
                    height: 30px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #34A853;
                    border-radius: 6px;
                }
            """)

            # Change cancel button to close
            self.cancelButton.setText("Close")
            self.cancelButton.clicked.disconnect()
            self.cancelButton.clicked.connect(self.accept)

            # Enable window close button
            self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
            self.show()

            self.isComplete = True

        except Exception as e:
            print(f"Error in setComplete: {e}")

    def setError(self, error_message):
        """Updates the dialog to show an error message."""
        try:
            # Stop animations
            if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                self.animationTimer.stop()

            # Create an error icon
            errorPixmap = QPixmap(32, 32)
            errorPixmap.fill(Qt.transparent)
            painter = QPainter(errorPixmap)
            painter.setPen(QPen(QColor("#EA4335"), 4))
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw an X
            painter.drawLine(8, 8, 24, 24)
            painter.drawLine(8, 24, 24, 8)
            painter.end()
            self.spinnerLabel.setPixmap(errorPixmap)

            # Update status message
            self.statusLabel.setText(f"Upload failed:\n{error_message}")
            self.statusLabel.setStyleSheet("color: #EA4335; font-weight: bold;")
            self.titleLabel.setText("Upload Failed")

            # Update progress bar to show error
            self.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #EA4335;
                    border-radius: 8px;
                    text-align: center;
                    background-color: #F6F8FA;
                    height: 30px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #EA4335;
                    border-radius: 6px;
                }
            """)
            self.progressBar.setFormat("Failed!")

            # Change cancel button to close
            self.cancelButton.setText("Close")
            self.cancelButton.clicked.disconnect()
            self.cancelButton.clicked.connect(self.accept)

            # Enable window close button
            self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
            self.show()

            self.isComplete = True

        except Exception as e:
            print(f"Error in setError: {e}")

    def cancelUpload(self):
        """Handle cancel button clicks based on current state."""
        try:
            if self.isComplete:
                # Just close if already complete
                self.accept()
            elif not self.cancel_requested:
                # Start cancellation process
                print("User clicked cancel upload button")
                self.cancel_requested = True
                self.cancelled = True

                # Immediate UI feedback
                self.cancelButton.setEnabled(False)
                self.cancelButton.setText("Cancelling...")
                self.statusLabel.setText("Cancelling upload...")
                self.statusLabel.setStyleSheet("color: #FF9800; font-weight: bold;")

                # Force UI update
                self.repaint()
                QApplication.processEvents()

        except Exception as e:
            print(f"Error in cancelUpload: {e}")

    def closeEvent(self, event):
        """Handle window close events."""
        try:
            if not self.isComplete and not self.cancelled:
                # Treat close as cancel
                print("Window close attempted - treating as cancel")
                self.cancelUpload()
                event.ignore()
            else:
                # Allow close and stop animations
                if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                    self.animationTimer.stop()
                event.accept()
        except Exception as e:
            print(f"Error in closeEvent: {e}")
            event.accept()

# -------------------------
# Combine Dataset Progress Dialog
# -------------------------
# ==================================
# STEP 1: ADD THIS NEW CLASS AFTER AzureUploadProgressDialog
# ==================================


class AzureDownloadProgressDialog(QDialog):
    """Custom progress dialog specifically for Azure download process with real progress tracking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancelled = False
        self.cancel_requested = False
        self.setWindowTitle("Downloading from Azure")
        self.setMinimumSize(500, 300)
        self.setModal(True)

        # Initially prevent closing with X button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        # Apply modern Azure-themed styling for download
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QLabel {
                color: #202124;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #6C757D;
                color: #ADB5BD;
            }
            QProgressBar {
                border: 2px solid #E1E4E8;
                border-radius: 8px;
                text-align: center;
                background-color: #F6F8FA;
                height: 30px;
                font-size: 14px;
                font-weight: bold;
                color: #24292E;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 6px;
            }
        """)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Title
        self.titleLabel = QLabel("Downloading from Azure Storage")
        self.titleLabel.setStyleSheet("color: #2196F3; font-size: 18px; font-weight: bold;")
        self.titleLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.titleLabel)

        # Status label
        self.statusLabel = QLabel("Preparing download...")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setMinimumHeight(40)
        layout.addWidget(self.statusLabel)

        # Progress bar with percentage display
        self.progressBar = QProgressBar()
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, 100)  # Determinate mode for real progress
        self.progressBar.setValue(0)
        layout.addWidget(self.progressBar)

        # File counter label
        self.fileCounterLabel = QLabel("Files: 0 / 0")
        self.fileCounterLabel.setAlignment(Qt.AlignCenter)
        self.fileCounterLabel.setStyleSheet("color: #5F6368; font-size: 12px;")
        layout.addWidget(self.fileCounterLabel)

        # Animated spinner for download
        self.spinnerLabel = QLabel()
        self.spinnerLabel.setAlignment(Qt.AlignCenter)
        self.spinnerLabel.setMinimumHeight(40)

        # Create download spinner animation frames (blue theme)
        self.spinnerPixmaps = []
        for i in range(8):  # 8-frame animation
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#2196F3"), 3))
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawArc(4, 4, 24, 24, i * 45 * 16, 180 * 16)
            painter.end()
            self.spinnerPixmaps.append(pixmap)

        self.spinnerLabel.setPixmap(self.spinnerPixmaps[0])
        layout.addWidget(self.spinnerLabel)

        # Cancel button
        self.cancelButton = QPushButton("Cancel Download")
        self.cancelButton.clicked.connect(self.cancelDownload)

        # Button layout
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.cancelButton)
        buttonLayout.addStretch()
        layout.addLayout(buttonLayout)

        # Animation timer for spinner
        self.animationTimer = QTimer(self)
        self.animationTimer.timeout.connect(self.updateSpinner)
        self.animationTimer.start(150)  # Update every 150ms
        self.animationIndex = 0

        # Completion flag
        self.isComplete = False

    def updateSpinner(self):
        """Updates the spinner animation frame."""
        try:
            if not self.cancelled and not self.cancel_requested and not self.isComplete:
                self.animationIndex = (self.animationIndex + 1) % len(self.spinnerPixmaps)
                self.spinnerLabel.setPixmap(self.spinnerPixmaps[self.animationIndex])
        except Exception as e:
            print(f"Error in updateSpinner: {e}")

    def updateStatus(self, message, progress=None):
        """Updates the status message and progress if provided."""
        try:
            if not self.cancelled and not self.cancel_requested:
                self.statusLabel.setText(message)

                if progress is not None and isinstance(progress, int):
                    # Real progress update
                    self.progressBar.setValue(progress)
                    self.progressBar.setFormat(f"{progress}%")

                QApplication.processEvents()  # Ensure UI updates immediately
        except Exception as e:
            print(f"Error in updateStatus: {e}")

    def updateFileProgress(self, current_file, total_files):
        """Updates the file counter display."""
        try:
            if not self.cancelled and not self.cancel_requested:
                self.fileCounterLabel.setText(f"Files: {current_file} / {total_files}")
                QApplication.processEvents()
        except Exception as e:
            print(f"Error in updateFileProgress: {e}")

    def setComplete(self, stats):
        """Updates the dialog to show completion status with statistics."""
        try:
            # Stop animations
            if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                self.animationTimer.stop()

            # Create a success icon (download arrow)
            successPixmap = QPixmap(32, 32)
            successPixmap.fill(Qt.transparent)
            painter = QPainter(successPixmap)
            painter.setPen(QPen(QColor("#34A853"), 4))
            painter.setBrush(QColor("#34A853"))
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw a checkmark
            points = [QPoint(8, 16), QPoint(14, 22), QPoint(24, 10)]
            painter.drawPolyline(QPolygon(points))
            painter.end()
            self.spinnerLabel.setPixmap(successPixmap)

            # Update status message with statistics
            if isinstance(stats, dict):
                stats_msg = "Download completed successfully!\n\n"
                stats_msg += f"✓ Images: {stats.get('images', 0)}\n"
                stats_msg += f"✓ Labels: {stats.get('labels', 0)}\n"
                stats_msg += f"✓ YAML: {stats.get('yaml', 0)}"
            else:
                stats_msg = "Download completed successfully!"

            self.statusLabel.setText(stats_msg)
            self.statusLabel.setStyleSheet("color: #34A853; font-weight: bold;")
            self.titleLabel.setText("Download Complete")

            # Update progress bar to 100%
            self.progressBar.setValue(100)
            self.progressBar.setFormat("100% - Complete!")
            self.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #34A853;
                    border-radius: 8px;
                    text-align: center;
                    background-color: #F6F8FA;
                    height: 30px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #34A853;
                    border-radius: 6px;
                }
            """)

            # Change cancel button to close
            self.cancelButton.setText("Close")
            self.cancelButton.clicked.disconnect()
            self.cancelButton.clicked.connect(self.accept)

            # Enable window close button
            self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
            self.show()

            self.isComplete = True

        except Exception as e:
            print(f"Error in setComplete: {e}")

    def setError(self, error_message):
        """Updates the dialog to show an error message."""
        try:
            # Stop animations
            if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                self.animationTimer.stop()

            # Create an error icon
            errorPixmap = QPixmap(32, 32)
            errorPixmap.fill(Qt.transparent)
            painter = QPainter(errorPixmap)
            painter.setPen(QPen(QColor("#EA4335"), 4))
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw an X
            painter.drawLine(8, 8, 24, 24)
            painter.drawLine(8, 24, 24, 8)
            painter.end()
            self.spinnerLabel.setPixmap(errorPixmap)

            # Update status message
            self.statusLabel.setText(f"Download failed:\n{error_message}")
            self.statusLabel.setStyleSheet("color: #EA4335; font-weight: bold;")
            self.titleLabel.setText("Download Failed")

            # Update progress bar to show error
            self.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #EA4335;
                    border-radius: 8px;
                    text-align: center;
                    background-color: #F6F8FA;
                    height: 30px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #EA4335;
                    border-radius: 6px;
                }
            """)
            self.progressBar.setFormat("Failed!")

            # Change cancel button to close
            self.cancelButton.setText("Close")
            self.cancelButton.clicked.disconnect()
            self.cancelButton.clicked.connect(self.accept)

            # Enable window close button
            self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
            self.show()

            self.isComplete = True

        except Exception as e:
            print(f"Error in setError: {e}")

    def cancelDownload(self):
        """Handle cancel button clicks based on current state."""
        try:
            if self.isComplete:
                # Just close if already complete
                self.accept()
            elif not self.cancel_requested:
                # Start cancellation process
                print("User clicked cancel download button")
                self.cancel_requested = True
                self.cancelled = True

                # Immediate UI feedback
                self.cancelButton.setEnabled(False)
                self.cancelButton.setText("Cancelling...")
                self.statusLabel.setText("Cancelling download...")
                self.statusLabel.setStyleSheet("color: #FF9800; font-weight: bold;")

                # Force UI update
                QApplication.processEvents()

        except Exception as e:
            print(f"Error in cancelDownload: {e}")

    def closeEvent(self, event):
        """Handle window close events."""
        try:
            if not self.isComplete and not self.cancelled:
                # Treat close as cancel
                print("Window close attempted - treating as cancel")
                self.cancelDownload()
                event.ignore()
            else:
                # Allow close and stop animations
                if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                    self.animationTimer.stop()
                event.accept()
        except Exception as e:
            print(f"Error in closeEvent: {e}")
            event.accept()


class CombineDatasetProgressDialog(QDialog):
    """Custom progress dialog for dataset combination process with guaranteed closability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combining Datasets")
        self.setMinimumSize(450, 250)
        self.setModal(True)

        # Prevent closing with X button initially
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QLabel {
                color: #202124;
                font-size: 14px;
            }
            QPushButton {
                background-color: #34A853;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
            QProgressBar {
                border: 1px solid #DADCE0;
                border-radius: 8px;
                text-align: center;
                background-color: #F8F9FA;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #34A853;
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Add title with styled text
        self.titleLabel = QLabel("Combining Datasets")
        self.titleLabel.setStyleSheet("color: #34A853; font-size: 18px; font-weight: bold;")
        self.titleLabel.setAlignment(Qt.AlignCenter)

        # Add a proper progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 0)  # Indeterminate mode
        self.progressBar.setTextVisible(True)

        self.statusLabel = QLabel("Combining datasets from different sources...")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setAlignment(Qt.AlignCenter)

        # Animation timer for updating the progress indicator text
        self.animationCounter = 0
        self.animationTimer = QTimer(self)
        self.animationTimer.timeout.connect(self.updateAnimation)
        self.animationTimer.start(300)  # Update every 300ms

        # Layout setup
        layout.addWidget(self.titleLabel)
        layout.addSpacing(10)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.statusLabel)
        layout.addSpacing(10)

        self.setLayout(layout)

        # Flags to track dialog state
        self.isComplete = False
        self.canForceClose = False
        self.close_button = None

    def updateAnimation(self):
        """Updates the animation text to show ongoing activity."""
        try:
            if not self.isComplete:
                self.animationCounter += 1
                dots = "." * (self.animationCounter % 5 + 1)
                self.statusLabel.setText(f"Combining datasets from different sources{dots}")
        except Exception as e:
            print(f"Error in updateAnimation: {e}")

    def setComplete(self, stats):
        """Called when combination completes successfully."""
        try:
            print("=== Dialog setComplete called ===")

            # Stop timer safely
            if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                self.animationTimer.stop()
                print("Animation timer stopped")

            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(100)

            # Format stats message
            if isinstance(stats, dict):
                message = "Dataset combination complete!\n"
                for source, data in stats.items():
                    img_count = data.get('images', 0)
                    label_count = data.get('labels', 0)
                    message += f"{source}: {img_count} images, {label_count} labels\n"
            else:
                message = "Dataset combination complete!"

            self.statusLabel.setText(message)
            self.statusLabel.setStyleSheet("color: #34A853;")  # Green for success
            self.titleLabel.setText("Combination Complete ✓")

            # Add a close button if not already present
            if not self.close_button:
                self.close_button = QPushButton("Close")
                self.close_button.clicked.connect(self.accept)
                self.layout().addWidget(self.close_button)
                print("Close button added")

            # Enable window close button and force close capability
            self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
            self.canForceClose = True
            self.isComplete = True
            self.show()  # Refresh to show close button

            print("=== Dialog setComplete finished ===")

        except Exception as e:
            print(f"Error in setComplete: {e}")
            # Mark as complete even if there's an error
            self.isComplete = True
            self.canForceClose = True

    def setError(self, error_message):
        """Called when combination encounters an error."""
        try:
            print(f"=== Dialog setError called: {error_message} ===")

            # Stop timer safely
            if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                self.animationTimer.stop()

            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
            self.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #DADCE0;
                    border-radius: 8px;
                    text-align: center;
                    background-color: #F8F9FA;
                }
                QProgressBar::chunk {
                    background-color: #EA4335;
                    border-radius: 6px;
                }
            """)

            self.statusLabel.setText(f"Error: {error_message}")
            self.statusLabel.setStyleSheet("color: #EA4335;")
            self.titleLabel.setText("Combination Failed ✗")

            # Add a close button if not already present
            if not self.close_button:
                self.close_button = QPushButton("Close")
                self.close_button.setStyleSheet("""
                    QPushButton {
                        background-color: #EA4335;
                        color: white;
                        font-weight: bold;
                        padding: 12px 24px;
                        border-radius: 8px;
                        border: none;
                        font-size: 14px;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
                self.close_button.clicked.connect(self.accept)
                self.layout().addWidget(self.close_button)

            # Enable window close button and force close capability
            self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
            self.canForceClose = True
            self.isComplete = True
            self.show()  # Refresh to show close button

            print("=== Dialog setError finished ===")

        except Exception as e:
            print(f"Error in setError: {e}")
            # Mark as complete even if there's an error
            self.isComplete = True
            self.canForceClose = True

    def closeEvent(self, event):
        """Override close event to handle closing properly."""
        try:
            print(
                f"=== Dialog closeEvent called. canForceClose: {self.canForceClose}, isComplete: {self.isComplete} ===")

            if not self.isComplete and not self.canForceClose:
                # If combination is still in progress and we can't force close, prevent closing
                print("Preventing close - combination in progress")
                event.ignore()
            else:
                # Allow closing
                print("Allowing close")
                # Stop timer before closing
                if hasattr(self, 'animationTimer') and self.animationTimer.isActive():
                    self.animationTimer.stop()
                event.accept()

        except Exception as e:
            print(f"Error in closeEvent: {e}")
            # Always allow closing if there's an error
            event.accept()

# -------------------------
# Folder Selection Dialog
# -------------------------


class FolderSelectionDialog(QDialog):
    """Dialog for selecting the 3 folders needed for dataset combination."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Folders for Dataset Combination")
        self.setMinimumSize(700, 400)
        self.setModal(True)

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #E8EAED;
                border-radius: 12px;
                padding-top: 20px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: 600;
                color: #202124;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                background-color: #F8F9FA;
            }
            QLabel {
                color: #5F6368;
                font-size: 14px;
                background: transparent;
                border: none;
            }
            QLineEdit {
                background: #F8F9FA;
                padding: 12px 16px;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)

        # Store folder paths
        self.azure_folder = ""
        self.manual_folder = ""
        self.auto_folder = ""
        self.output_folder = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Select the folders containing your datasets:")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #202124;")
        layout.addWidget(title_label)

        # Azure data folder selection
        azure_group = QGroupBox("Azure Downloaded Data")
        azure_layout = QVBoxLayout()
        azure_layout.setSpacing(12)

        azure_desc = QLabel(
            "Select the folder where Azure data was downloaded (should contain train/valid/test subfolders)")
        azure_desc.setWordWrap(True)
        azure_layout.addWidget(azure_desc)

        azure_path_layout = QHBoxLayout()
        self.azure_path_label = QLineEdit("No folder selected")
        self.azure_path_label.setReadOnly(True)
        azure_path_layout.addWidget(self.azure_path_label)

        self.azure_browse_btn = QPushButton("Browse")
        self.azure_browse_btn.clicked.connect(self.select_azure_folder)
        azure_path_layout.addWidget(self.azure_browse_btn)

        azure_layout.addLayout(azure_path_layout)
        azure_group.setLayout(azure_layout)
        layout.addWidget(azure_group)

        # Manual annotation folder selection
        manual_group = QGroupBox("Manual Annotations")
        manual_layout = QVBoxLayout()
        manual_layout.setSpacing(12)

        manual_desc = QLabel("Select the folder containing manually annotated images and labels")
        manual_desc.setWordWrap(True)
        manual_layout.addWidget(manual_desc)

        manual_path_layout = QHBoxLayout()
        self.manual_path_label = QLineEdit("No folder selected")
        self.manual_path_label.setReadOnly(True)
        manual_path_layout.addWidget(self.manual_path_label)

        self.manual_browse_btn = QPushButton("Browse")
        self.manual_browse_btn.clicked.connect(self.select_manual_folder)
        manual_path_layout.addWidget(self.manual_browse_btn)

        manual_layout.addLayout(manual_path_layout)
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

        # Auto annotation folder selection
        auto_group = QGroupBox("Auto Annotations")
        auto_layout = QVBoxLayout()
        auto_layout.setSpacing(12)

        auto_desc = QLabel("Select the folder containing auto-annotated images and labels")
        auto_desc.setWordWrap(True)
        auto_layout.addWidget(auto_desc)

        auto_path_layout = QHBoxLayout()
        self.auto_path_label = QLineEdit("No folder selected")
        self.auto_path_label.setReadOnly(True)
        auto_path_layout.addWidget(self.auto_path_label)

        self.auto_browse_btn = QPushButton("Browse")
        self.auto_browse_btn.clicked.connect(self.select_auto_folder)
        auto_path_layout.addWidget(self.auto_browse_btn)

        auto_layout.addLayout(auto_path_layout)
        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)

        # Output folder selection
        output_group = QGroupBox("Output Folder")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(12)

        output_desc = QLabel("Select where to save the combined dataset")
        output_desc.setWordWrap(True)
        output_layout.addWidget(output_desc)

        output_path_layout = QHBoxLayout()
        self.output_path_label = QLineEdit("No folder selected")
        self.output_path_label.setReadOnly(True)
        output_path_layout.addWidget(self.output_path_label)

        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.select_output_folder)
        output_path_layout.addWidget(self.output_browse_btn)

        output_layout.addLayout(output_path_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                color: #3C4043;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        self.combine_btn = QPushButton("Combine Datasets")
        self.combine_btn.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #E8EAED;
                color: #9AA0A6;
            }
        """)
        self.combine_btn.clicked.connect(self.accept)
        self.combine_btn.setEnabled(False)  # Disabled until all folders selected

        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.combine_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #EA4335; font-style: italic;")
        layout.addWidget(self.status_label)

    def select_azure_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Azure Downloaded Data Folder",
            self.azure_folder if self.azure_folder else ""
        )
        if folder:
            self.azure_folder = folder
            self.azure_path_label.setText(folder)
            self.validate_folders()

    def select_manual_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Manual Annotations Folder",
            self.manual_folder if self.manual_folder else ""
        )
        if folder:
            self.manual_folder = folder
            self.manual_path_label.setText(folder)
            self.validate_folders()

    def select_auto_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Auto Annotations Folder",
            self.auto_folder if self.auto_folder else ""
        )
        if folder:
            self.auto_folder = folder
            self.auto_path_label.setText(folder)
            self.validate_folders()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Combined Dataset",
            self.output_folder if self.output_folder else ""
        )
        if folder:
            self.output_folder = folder
            self.output_path_label.setText(folder)
            self.validate_folders()

    def validate_folders(self):
        """Validate selected folders and enable/disable combine button"""
        issues = []

        # Check if all folders are selected
        if not self.azure_folder:
            issues.append("Azure folder not selected")
        elif not os.path.exists(self.azure_folder):
            issues.append("Azure folder does not exist")

        if not self.manual_folder:
            issues.append("Manual annotations folder not selected")
        elif not os.path.exists(self.manual_folder):
            issues.append("Manual annotations folder does not exist")

        if not self.auto_folder:
            issues.append("Auto annotations folder not selected")
        elif not os.path.exists(self.auto_folder):
            issues.append("Auto annotations folder does not exist")

        if not self.output_folder:
            issues.append("Output folder not selected")

        # Check for folder contents (optional validation)
        if self.azure_folder and os.path.exists(self.azure_folder):
            # Check if it has train/valid/test structure
            expected_splits = ['train', 'valid', 'test']
            missing_splits = []
            for split in expected_splits:
                split_path = os.path.join(self.azure_folder, split)
                if not os.path.exists(split_path):
                    missing_splits.append(split)

            if missing_splits:
                issues.append(f"Azure folder missing: {', '.join(missing_splits)} subfolders")

        # Update status and button
        if issues:
            self.status_label.setText("Issues: " + "; ".join(issues))
            self.status_label.setStyleSheet("color: #EA4335; font-style: italic;")
            self.combine_btn.setEnabled(False)
        else:
            self.status_label.setText("✓ All folders selected and validated")
            self.status_label.setStyleSheet("color: #34A853; font-style: italic;")
            self.combine_btn.setEnabled(True)

    def get_selected_folders(self):
        """Return dictionary of selected folder paths"""
        return {
            'azure_folder': self.azure_folder,
            'manual_folder': self.manual_folder,
            'auto_folder': self.auto_folder,
            'output_folder': self.output_folder
        }

# -------------------------
# Azure Upload Selection Dialog
# -------------------------

# -------------------------
# Fast Training Overlay Dialog (2 datasets combine + upload + training)
# -------------------------


class FastTrainingOverlayDialog(QDialog):
    """Overlay dialog for Fast Training with combine datasets + upload + training config"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fast Training Configuration")
        self.setModal(True)
        self.setMinimumSize(900, 800)

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
            }
            QTabWidget::pane {
                border: 1px solid #DADCE0;
                border-radius: 8px;
                background-color: #FFFFFF;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                padding: 12px 24px;
                margin-right: 2px;
                border-radius: 8px 8px 0px 0px;
            }
            QTabBar::tab:selected {
                background-color: #4285F4;
                color: white;
                border-bottom: none;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #E8EAED;
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: 600;
                color: #202124;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #FFFFFF;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 8px 12px;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #4285F4;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
        """)

        # Store configurations
        self.dataset1_path = ""
        self.dataset2_path = ""
        self.upload_directory = ""
        self.azure_destination = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Fast Training Configuration")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4285F4;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Tab 1: Combine Datasets
        self.combine_tab = self.create_combine_datasets_tab()
        self.tab_widget.addTab(self.combine_tab, "📁 Combine Datasets")

        # Tab 2: Upload to Azure
        self.upload_tab = self.create_upload_to_azure_tab()
        self.tab_widget.addTab(self.upload_tab, "☁️ Upload to Azure")

        # Tab 3: Training Configuration
        self.training_tab = self.create_training_configuration_tab()
        self.tab_widget.addTab(self.training_tab, "⚙️ Training Configuration")

        layout.addWidget(self.tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                color: #3C4043;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        self.start_training_btn = QPushButton("🚀 Start Training")
        self.start_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        self.start_training_btn.clicked.connect(self.start_training)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.start_training_btn)

        layout.addLayout(button_layout)

    def create_combine_datasets_tab(self):
        """Create the combine datasets tab (2 datasets)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Instructions
        instruction_label = QLabel("Select two datasets to combine:")
        instruction_label.setStyleSheet("font-size: 14px; color: #5F6368; margin-bottom: 10px;")
        layout.addWidget(instruction_label)

        # Dataset 1
        dataset1_group = QGroupBox("Dataset 1")
        dataset1_layout = QVBoxLayout()

        dataset1_path_layout = QHBoxLayout()
        self.dataset1_input = QLineEdit("Choose first dataset")
        self.dataset1_input.setReadOnly(True)
        dataset1_browse_btn = QPushButton("Browse")
        dataset1_browse_btn.clicked.connect(self.browse_dataset1)

        dataset1_path_layout.addWidget(self.dataset1_input)
        dataset1_path_layout.addWidget(dataset1_browse_btn)
        dataset1_layout.addLayout(dataset1_path_layout)
        dataset1_group.setLayout(dataset1_layout)

        # Dataset 2
        dataset2_group = QGroupBox("Dataset 2")
        dataset2_layout = QVBoxLayout()

        dataset2_path_layout = QHBoxLayout()
        self.dataset2_input = QLineEdit("Choose second dataset")
        self.dataset2_input.setReadOnly(True)
        dataset2_browse_btn = QPushButton("Browse")
        dataset2_browse_btn.clicked.connect(self.browse_dataset2)

        dataset2_path_layout.addWidget(self.dataset2_input)
        dataset2_path_layout.addWidget(dataset2_browse_btn)
        dataset2_layout.addLayout(dataset2_path_layout)
        dataset2_group.setLayout(dataset2_layout)

        # Combine button
        combine_btn = QPushButton("🔄 Combine Datasets")
        combine_btn.clicked.connect(self.combine_datasets)

        layout.addWidget(dataset1_group)
        layout.addWidget(dataset2_group)
        layout.addWidget(combine_btn)
        layout.addStretch()

        return widget

    def create_upload_to_azure_tab(self):
        """Create the upload to azure tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Upload Directory Selection
        upload_group = QGroupBox("Upload Directory")
        upload_layout = QVBoxLayout()

        upload_desc = QLabel("Select the directory to upload to Azure:")
        upload_desc.setStyleSheet("color: #5F6368; margin-bottom: 10px;")
        upload_layout.addWidget(upload_desc)

        upload_path_layout = QHBoxLayout()
        self.upload_input = QLineEdit("Choose upload directory")
        self.upload_input.setReadOnly(True)
        upload_browse_btn = QPushButton("Browse")
        upload_browse_btn.clicked.connect(self.browse_upload_directory)

        upload_path_layout.addWidget(self.upload_input)
        upload_path_layout.addWidget(upload_browse_btn)
        upload_layout.addLayout(upload_path_layout)
        upload_group.setLayout(upload_layout)

        # Azure Destination
        azure_group = QGroupBox("Azure Destination")
        azure_layout = QVBoxLayout()

        azure_desc = QLabel("Specify Azure folder path:")
        azure_desc.setStyleSheet("color: #5F6368; margin-bottom: 10px;")
        azure_layout.addWidget(azure_desc)

        # Predefined options
        azure_buttons_layout = QHBoxLayout()
        savai_btn = QPushButton("SAVAI_METAL_INSPECTION_GEN")
        savai_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        savai_btn.clicked.connect(lambda: self.set_azure_destination("SAVAI_METAL_INSPECTION_GEN"))

        custom_btn = QPushButton("Custom Path")
        custom_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        custom_btn.clicked.connect(self.set_custom_azure_path)

        azure_buttons_layout.addWidget(savai_btn)
        azure_buttons_layout.addWidget(custom_btn)
        azure_buttons_layout.addStretch()

        self.azure_destination_input = QLineEdit()
        self.azure_destination_input.setPlaceholderText("Enter Azure folder path")
        self.azure_destination_input.textChanged.connect(self.on_azure_path_changed)

        azure_layout.addLayout(azure_buttons_layout)
        azure_layout.addWidget(self.azure_destination_input)
        azure_group.setLayout(azure_layout)

        # Upload button
        upload_btn = QPushButton("☁️ Upload to Azure")
        upload_btn.clicked.connect(self.upload_to_azure)

        layout.addWidget(upload_group)
        layout.addWidget(azure_group)
        layout.addWidget(upload_btn)
        layout.addStretch()

        return widget

    def create_training_configuration_tab(self):
        """Create the training configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Model and Data paths
        paths_group = QGroupBox("Model and Data Paths")
        paths_layout = QGridLayout()

        # Model Path
        paths_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setText("/home/sakar02/sakar-vision-ui/y11n.pt")
        paths_layout.addWidget(self.model_path_input, 0, 1)
        model_browse_btn = QPushButton("Choose Model Path")
        model_browse_btn.clicked.connect(self.browse_model_path)
        paths_layout.addWidget(model_browse_btn, 0, 2)

        # Data YAML Path
        paths_layout.addWidget(QLabel("Data YAML Path:"), 1, 0)
        self.yaml_path_input = QLineEdit()
        self.yaml_path_input.setText("ally_annotated/dataset.yaml")
        paths_layout.addWidget(self.yaml_path_input, 1, 1)
        yaml_browse_btn = QPushButton("Choose Data YAML Path")
        yaml_browse_btn.clicked.connect(self.browse_yaml_path)
        paths_layout.addWidget(yaml_browse_btn, 1, 2)

        paths_group.setLayout(paths_layout)

        # Training Parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()

        # Epochs
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        params_layout.addWidget(self.epochs_input, 0, 1)

        # Image Size
        params_layout.addWidget(QLabel("Image Size:"), 0, 2)
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(32, 2048)
        self.imgsz_input.setValue(640)
        params_layout.addWidget(self.imgsz_input, 0, 3)

        # Batch Size
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(16)
        params_layout.addWidget(self.batch_input, 1, 1)

        # Initial Learning Rate
        params_layout.addWidget(QLabel("Initial Learning Rate:"), 1, 2)
        self.lr0_input = QDoubleSpinBox()
        self.lr0_input.setRange(1e-6, 1.0)
        self.lr0_input.setDecimals(6)
        self.lr0_input.setValue(0.000100)
        params_layout.addWidget(self.lr0_input, 1, 3)

        # Final Learning Rate Multiplier
        params_layout.addWidget(QLabel("Final Learning Rate Multiplier:"), 2, 0)
        self.lrf_input = QDoubleSpinBox()
        self.lrf_input.setRange(1e-6, 1.0)
        self.lrf_input.setDecimals(6)
        self.lrf_input.setValue(0.010000)
        params_layout.addWidget(self.lrf_input, 2, 1)

        # Dropout
        params_layout.addWidget(QLabel("Dropout:"), 2, 2)
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0.0, 1.0)
        self.dropout_input.setDecimals(2)
        self.dropout_input.setValue(0.25)
        params_layout.addWidget(self.dropout_input, 2, 3)

        # Device
        params_layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_input = QComboBox()
        self.device_input.addItems(["0 (GPU)", "1 (GPU)", "cpu"])
        params_layout.addWidget(self.device_input, 3, 1)

        # Seed
        params_layout.addWidget(QLabel("Seed:"), 3, 2)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 10000)
        self.seed_input.setValue(42)
        params_layout.addWidget(self.seed_input, 3, 3)

        params_group.setLayout(params_layout)

        layout.addWidget(paths_group)
        layout.addWidget(params_group)
        layout.addStretch()

        return widget

    # Browse functions
    def browse_dataset1(self):
        folder = QFileDialog.getExistingDirectory(self, "Select First Dataset Directory")
        if folder:
            self.dataset1_path = folder
            self.dataset1_input.setText(folder)

    def browse_dataset2(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Second Dataset Directory")
        if folder:
            self.dataset2_path = folder
            self.dataset2_input.setText(folder)

    def browse_upload_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Upload")
        if folder:
            self.upload_directory = folder
            self.upload_input.setText(folder)

    def browse_model_path(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt)")
        if file:
            self.model_path_input.setText(file)

    def browse_yaml_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml)")
        if file:
            self.yaml_path_input.setText(file)

    def set_azure_destination(self, path):
        self.azure_destination = path
        self.azure_destination_input.setText(path)

    def set_custom_azure_path(self):
        text, ok = QInputDialog.getText(self, "Custom Azure Path", "Enter Azure folder path:")
        if ok and text.strip():
            self.azure_destination = text.strip()
            self.azure_destination_input.setText(text.strip())

    def on_azure_path_changed(self):
        self.azure_destination = self.azure_destination_input.text().strip()

    # Action functions
    def combine_datasets(self):
        if not self.dataset1_path or not self.dataset2_path:
            QMessageBox.warning(self, "Warning", "Please select all two datasets to combine.")
            return

        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Combined Dataset")
        if not output_folder:
            return

        # --- ADD THIS: Call your function directly ---
        from data_integration import combine_two_datasets_with_yaml
        success, stats = combine_two_datasets_with_yaml(
            self.dataset1_path,  # Azure
            self.dataset2_path,  # Manual # Auto
            output_folder
        )
        if success:
            QMessageBox.information(
                self, "Success", f"Datasets combined successfully!\nStats: {stats}")
            self.yaml_path_input.setText(os.path.join(output_folder, "data.yaml"))
        else:
            QMessageBox.critical(self, "Error", "Failed to combine datasets.")

    def upload_to_azure(self):
        if not self.upload_directory or not self.azure_destination:
            QMessageBox.warning(
                self, "Warning", "Please select upload directory and Azure destination.")
            return

        # Call parent's upload function
        if hasattr(self.parent(), 'uploadToAzure_FromDialog'):
            success = self.parent().uploadToAzure_FromDialog(
                self.upload_directory,
                self.azure_destination
            )
            if success:
                QMessageBox.information(self, "Success", "Upload to Azure completed!")

    def start_training(self):
        # Validate all required fields
        if not self.model_path_input.text() or not self.yaml_path_input.text():
            QMessageBox.warning(
                self, "Warning", "Please fill in all required training configuration fields.")
            return

        self.accept()

    def get_training_config(self):
        """Get the training configuration"""
        return {
            'model_path': self.model_path_input.text(),
            'yaml_path': self.yaml_path_input.text(),
            'epochs': self.epochs_input.value(),
            'imgsz': self.imgsz_input.value(),
            'batch': self.batch_input.value(),
            'lr0': self.lr0_input.value(),
            'lrf': self.lrf_input.value(),
            'dropout': self.dropout_input.value(),
            'device': self.device_input.currentText().split()[0],  # Extract device number/type
            'seed': self.seed_input.value()
        }


# -------------------------
# Accurate Training Overlay Dialog (download + 3 datasets combine + upload + training)
# -------------------------

class AccurateTrainingOverlayDialog(QDialog):
    """Overlay dialog for Accurate Training with download + combine + upload + training config"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Accurate Training Configuration")
        self.setModal(True)
        self.setMinimumSize(900, 800)

        # Apply same modern styling as FastTrainingOverlayDialog
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
            }
            QTabWidget::pane {
                border: 1px solid #DADCE0;
                border-radius: 8px;
                background-color: #FFFFFF;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                padding: 12px 24px;
                margin-right: 2px;
                border-radius: 8px 8px 0px 0px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
                border-bottom: none;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #E8EAED;
                border-radius: 8px;
                padding-top: 15px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: 600;
                color: #202124;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #FFFFFF;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 8px 12px;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
            QPushButton {
                background-color: #1976D2;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        # Store configurations
        self.azure_download_path = ""
        self.dataset1_path = ""
        self.dataset2_path = ""
        self.dataset3_path = ""
        self.upload_directory = ""
        self.azure_destination = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Accurate Training Configuration")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #9C27B0;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Tab 1: Download from Azure
        self.download_tab = self.create_download_from_azure_tab()
        self.tab_widget.addTab(self.download_tab, "⬇️ Download from Azure")

        # Tab 2: Combine Datasets (3 datasets)
        self.combine_tab = self.create_combine_datasets_tab()
        self.tab_widget.addTab(self.combine_tab, "📁 Combine Datasets")

        # Tab 3: Upload to Azure
        self.upload_tab = self.create_upload_to_azure_tab()
        self.tab_widget.addTab(self.upload_tab, "☁️ Upload to Azure")

        # Tab 4: Training Configuration
        self.training_tab = self.create_training_configuration_tab()
        self.tab_widget.addTab(self.training_tab, "⚙️ Training Configuration")

        layout.addWidget(self.tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                color: #3C4043;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        self.start_training_btn = QPushButton("🚀 Start Training")
        self.start_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
        """)
        self.start_training_btn.clicked.connect(self.start_training)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.start_training_btn)

        layout.addLayout(button_layout)

    def create_download_from_azure_tab(self):
        """Create the download from azure tab with separate download button"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Instructions
        instruction_label = QLabel("Download training data from Azure Storage:")
        instruction_label.setStyleSheet("font-size: 14px; color: #5F6368; margin-bottom: 10px;")
        layout.addWidget(instruction_label)

        # Azure Directory Selection
        azure_group = QGroupBox("Azure Directory")
        azure_layout = QVBoxLayout()

        azure_desc = QLabel("Select Azure directory to download from:")
        azure_desc.setStyleSheet("color: #5F6368; margin-bottom: 10px;")
        azure_layout.addWidget(azure_desc)

        # Predefined Azure directories
        azure_buttons_layout = QHBoxLayout()
        savai_btn = QPushButton("SAVAI_METAL_INSPECTION_GEN")
        savai_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        savai_btn.clicked.connect(
            lambda: self.set_azure_download_path("SAVAI_METAL_INSPECTION_GEN"))

        custom_btn = QPushButton("Custom Path")
        custom_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        custom_btn.clicked.connect(self.set_custom_download_path)

        azure_buttons_layout.addWidget(savai_btn)
        azure_buttons_layout.addWidget(custom_btn)
        azure_buttons_layout.addStretch()

        self.azure_download_input = QLineEdit()
        self.azure_download_input.setPlaceholderText("Enter Azure folder path to download")
        self.azure_download_input.textChanged.connect(self.on_download_path_changed)

        azure_layout.addLayout(azure_buttons_layout)
        azure_layout.addWidget(self.azure_download_input)
        azure_group.setLayout(azure_layout)

        # Download destination selection
        local_group = QGroupBox("Local Download Destination")
        local_layout = QVBoxLayout()

        local_desc = QLabel("Select where to save the downloaded data locally:")
        local_desc.setStyleSheet("color: #5F6368; margin-bottom: 10px;")
        local_layout.addWidget(local_desc)

        local_path_layout = QHBoxLayout()
        self.local_download_input = QLineEdit()
        self.local_download_input.setPlaceholderText("Choose local folder for download")
        self.local_download_input.setReadOnly(True)
        local_path_layout.addWidget(self.local_download_input)

        local_browse_btn = QPushButton("Browse")
        local_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        local_browse_btn.clicked.connect(self.browse_local_download_folder)
        local_path_layout.addWidget(local_browse_btn)

        local_layout.addLayout(local_path_layout)
        local_group.setLayout(local_layout)

        # SEPARATE Download button - clearly distinct from upload
        download_btn = QPushButton("⬇️ Download from Azure")
        download_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #E8EAED;
                color: #9AA0A6;
            }
        """)
        download_btn.clicked.connect(self.download_from_azure)

        # Status label for download
        self.download_status_label = QLabel("")
        self.download_status_label.setStyleSheet(
            "color: #5F6368; font-style: italic; margin-top: 5px;")

        layout.addWidget(azure_group)
        layout.addWidget(local_group)
        layout.addWidget(download_btn)
        layout.addWidget(self.download_status_label)
        layout.addStretch()

        return widget

    def create_combine_datasets_tab(self):
        """Create the combine datasets tab (3 datasets)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Instructions
        instruction_label = QLabel("Select three datasets to combine:")
        instruction_label.setStyleSheet("font-size: 14px; color: #5F6368; margin-bottom: 10px;")
        layout.addWidget(instruction_label)

        # Dataset 1 (Azure)
        dataset1_group = QGroupBox("Dataset 1 (Azure Data)")
        dataset1_layout = QVBoxLayout()

        dataset1_path_layout = QHBoxLayout()
        self.dataset1_input = QLineEdit("Choose Azure dataset")
        self.dataset1_input.setReadOnly(True)
        dataset1_browse_btn = QPushButton("Browse")
        dataset1_browse_btn.clicked.connect(self.browse_dataset1)

        dataset1_path_layout.addWidget(self.dataset1_input)
        dataset1_path_layout.addWidget(dataset1_browse_btn)
        dataset1_layout.addLayout(dataset1_path_layout)
        dataset1_group.setLayout(dataset1_layout)

        # Dataset 2 (Manual)
        dataset2_group = QGroupBox("Dataset 2 (Manual Annotations)")
        dataset2_layout = QVBoxLayout()

        dataset2_path_layout = QHBoxLayout()
        self.dataset2_input = QLineEdit("Choose manual dataset")
        self.dataset2_input.setReadOnly(True)
        dataset2_browse_btn = QPushButton("Browse")
        dataset2_browse_btn.clicked.connect(self.browse_dataset2)

        dataset2_path_layout.addWidget(self.dataset2_input)
        dataset2_path_layout.addWidget(dataset2_browse_btn)
        dataset2_layout.addLayout(dataset2_path_layout)
        dataset2_group.setLayout(dataset2_layout)

        # Dataset 3 (Auto)
        dataset3_group = QGroupBox("Dataset 3 (Auto Annotations)")
        dataset3_layout = QVBoxLayout()

        dataset3_path_layout = QHBoxLayout()
        self.dataset3_input = QLineEdit("Choose auto dataset")
        self.dataset3_input.setReadOnly(True)
        dataset3_browse_btn = QPushButton("Browse")
        dataset3_browse_btn.clicked.connect(self.browse_dataset3)

        dataset3_path_layout.addWidget(self.dataset3_input)
        dataset3_path_layout.addWidget(dataset3_browse_btn)
        dataset3_layout.addLayout(dataset3_path_layout)
        dataset3_group.setLayout(dataset3_layout)

        # Combine button
        combine_btn = QPushButton("🔄 Combine Datasets")
        combine_btn.clicked.connect(self.combine_datasets)

        layout.addWidget(dataset1_group)
        layout.addWidget(dataset2_group)
        layout.addWidget(dataset3_group)
        layout.addWidget(combine_btn)
        layout.addStretch()

        return widget

    def create_upload_to_azure_tab(self):
        """Create the upload to azure tab (same as fast training)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Upload Directory Selection
        upload_group = QGroupBox("Upload Directory")
        upload_layout = QVBoxLayout()

        upload_desc = QLabel("Select the directory to upload to Azure:")
        upload_desc.setStyleSheet("color: #5F6368; margin-bottom: 10px;")
        upload_layout.addWidget(upload_desc)

        upload_path_layout = QHBoxLayout()
        self.upload_input = QLineEdit("Choose upload directory")
        self.upload_input.setReadOnly(True)
        upload_browse_btn = QPushButton("Browse")
        upload_browse_btn.clicked.connect(self.browse_upload_directory)

        upload_path_layout.addWidget(self.upload_input)
        upload_path_layout.addWidget(upload_browse_btn)
        upload_layout.addLayout(upload_path_layout)
        upload_group.setLayout(upload_layout)

        # Azure Destination
        azure_group = QGroupBox("Azure Destination")
        azure_layout = QVBoxLayout()

        azure_desc = QLabel("Specify Azure folder path:")
        azure_desc.setStyleSheet("color: #5F6368; margin-bottom: 10px;")
        azure_layout.addWidget(azure_desc)

        # Predefined options
        azure_buttons_layout = QHBoxLayout()
        savai_btn = QPushButton("SAVAI_METAL_INSPECTION_GEN")
        savai_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        savai_btn.clicked.connect(lambda: self.set_azure_destination("SAVAI_METAL_INSPECTION_GEN"))

        custom_btn = QPushButton("Custom Path")
        custom_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        custom_btn.clicked.connect(self.set_custom_azure_path)

        azure_buttons_layout.addWidget(savai_btn)
        azure_buttons_layout.addWidget(custom_btn)
        azure_buttons_layout.addStretch()

        self.azure_destination_input = QLineEdit()
        self.azure_destination_input.setPlaceholderText("Enter Azure folder path")
        self.azure_destination_input.textChanged.connect(self.on_azure_path_changed)

        azure_layout.addLayout(azure_buttons_layout)
        azure_layout.addWidget(self.azure_destination_input)
        azure_group.setLayout(azure_layout)

        # Upload button - clearly distinct from download
        upload_btn = QPushButton("☁️ Upload to Azure")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        upload_btn.clicked.connect(self.upload_to_azure)

        layout.addWidget(upload_group)
        layout.addWidget(azure_group)
        layout.addWidget(upload_btn)
        layout.addStretch()

        return widget

    def create_training_configuration_tab(self):
        """Create the training configuration tab (same as fast training)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)

        # Model and Data paths
        paths_group = QGroupBox("Model and Data Paths")
        paths_layout = QGridLayout()

        # Model Path
        paths_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setText("/home/sakar02/sakar-vision-ui/y11n.pt")
        paths_layout.addWidget(self.model_path_input, 0, 1)
        model_browse_btn = QPushButton("Choose Model Path")
        model_browse_btn.clicked.connect(self.browse_model_path)
        paths_layout.addWidget(model_browse_btn, 0, 2)

        # Data YAML Path
        paths_layout.addWidget(QLabel("Data YAML Path:"), 1, 0)
        self.yaml_path_input = QLineEdit()
        self.yaml_path_input.setText("ally_annotated/dataset.yaml")
        paths_layout.addWidget(self.yaml_path_input, 1, 1)
        yaml_browse_btn = QPushButton("Choose Data YAML Path")
        yaml_browse_btn.clicked.connect(self.browse_yaml_path)
        paths_layout.addWidget(yaml_browse_btn, 1, 2)

        paths_group.setLayout(paths_layout)

        # Training Parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()

        # Epochs
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        params_layout.addWidget(self.epochs_input, 0, 1)

        # Image Size
        params_layout.addWidget(QLabel("Image Size:"), 0, 2)
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(32, 2048)
        self.imgsz_input.setValue(640)
        params_layout.addWidget(self.imgsz_input, 0, 3)

        # Batch Size
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(16)
        params_layout.addWidget(self.batch_input, 1, 1)

        # Initial Learning Rate
        params_layout.addWidget(QLabel("Initial Learning Rate:"), 1, 2)
        self.lr0_input = QDoubleSpinBox()
        self.lr0_input.setRange(1e-6, 1.0)
        self.lr0_input.setDecimals(6)
        self.lr0_input.setValue(0.000100)
        params_layout.addWidget(self.lr0_input, 1, 3)

        # Final Learning Rate Multiplier
        params_layout.addWidget(QLabel("Final Learning Rate Multiplier:"), 2, 0)
        self.lrf_input = QDoubleSpinBox()
        self.lrf_input.setRange(1e-6, 1.0)
        self.lrf_input.setDecimals(6)
        self.lrf_input.setValue(0.010000)
        params_layout.addWidget(self.lrf_input, 2, 1)

        # Dropout
        params_layout.addWidget(QLabel("Dropout:"), 2, 2)
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0.0, 1.0)
        self.dropout_input.setDecimals(2)
        self.dropout_input.setValue(0.25)
        params_layout.addWidget(self.dropout_input, 2, 3)

        # Device
        params_layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_input = QComboBox()
        self.device_input.addItems(["0 (GPU)", "1 (GPU)", "cpu"])
        params_layout.addWidget(self.device_input, 3, 1)

        # Seed
        params_layout.addWidget(QLabel("Seed:"), 3, 2)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 10000)
        self.seed_input.setValue(42)
        params_layout.addWidget(self.seed_input, 3, 3)

        params_group.setLayout(params_layout)

        layout.addWidget(paths_group)
        layout.addWidget(params_group)
        layout.addStretch()

        return widget

    # Browse and action functions
    def browse_local_download_folder(self):
        """Browse for local download destination folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Local Folder to Download To")
        if folder:
            self.local_download_input.setText(folder)
            self.validate_download_inputs()

    def validate_download_inputs(self):
        """Validate download inputs and update status"""
        azure_path = self.azure_download_input.text().strip()
        local_path = self.local_download_input.text().strip()

        if azure_path and local_path:
            self.download_status_label.setText("✓ Ready to download")
            self.download_status_label.setStyleSheet("color: #34A853; font-style: italic;")
        elif not azure_path:
            self.download_status_label.setText("Please specify Azure path")
            self.download_status_label.setStyleSheet("color: #EA4335; font-style: italic;")
        elif not local_path:
            self.download_status_label.setText("Please select local download folder")
            self.download_status_label.setStyleSheet("color: #EA4335; font-style: italic;")

    def browse_dataset1(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Azure Dataset Directory")
        if folder:
            self.dataset1_path = folder
            self.dataset1_input.setText(folder)

    def browse_dataset2(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Manual Dataset Directory")
        if folder:
            self.dataset2_path = folder
            self.dataset2_input.setText(folder)

    def browse_dataset3(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Auto Dataset Directory")
        if folder:
            self.dataset3_path = folder
            self.dataset3_input.setText(folder)

    def browse_upload_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Upload")
        if folder:
            self.upload_directory = folder
            self.upload_input.setText(folder)

    def browse_model_path(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt)")
        if file:
            self.model_path_input.setText(file)

    def browse_yaml_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml)")
        if file:
            self.yaml_path_input.setText(file)

    def set_azure_download_path(self, path):
        self.azure_download_path = path
        self.azure_download_input.setText(path)
        self.validate_download_inputs()

    def set_custom_download_path(self):
        text, ok = QInputDialog.getText(
            self, "Custom Azure Download Path", "Enter Azure folder path:")
        if ok and text.strip():
            self.azure_download_path = text.strip()
            self.azure_download_input.setText(text.strip())
            self.validate_download_inputs()

    def on_download_path_changed(self):
        self.azure_download_path = self.azure_download_input.text().strip()
        self.validate_download_inputs()

    def set_azure_destination(self, path):
        self.azure_destination = path
        self.azure_destination_input.setText(path)

    def set_custom_azure_path(self):
        text, ok = QInputDialog.getText(self, "Custom Azure Path", "Enter Azure folder path:")
        if ok and text.strip():
            self.azure_destination = text.strip()
            self.azure_destination_input.setText(text.strip())

    def on_azure_path_changed(self):
        self.azure_destination = self.azure_destination_input.text().strip()

    # Action functions
    def download_from_azure(self):
        """Handle download from Azure with proper validation"""
        print("=== DOWNLOAD FUNCTION CALLED ===")

        azure_path = self.azure_download_input.text().strip()
        local_path = self.local_download_input.text().strip()

        if not azure_path or not local_path:
            QMessageBox.warning(
                self, "Warning", "Please fill in both Azure path and local folder.")
            return

        try:
            from data_integration import download_azure_training_data

            print("Starting download...")
            self.download_status_label.setText("🔄 Downloading...")
            QApplication.processEvents()

            # Try different parameter combinations
            try:
                result = download_azure_training_data(local_path, azure_folder=azure_path)
            except TypeError:
                # If that fails, try with just local_path
                result = download_azure_training_data(local_path)

            print(f"Download result: {result}")

            if result:
                QMessageBox.information(self, "Success", "Download completed!")
                self.download_status_label.setText("✓ Download completed")
            else:
                QMessageBox.warning(self, "Failed", "Download failed")
                self.download_status_label.setText("✗ Download failed")

        except Exception as e:
            print(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Download error: {e}")

    def combine_datasets(self):
        if not self.dataset1_path or not self.dataset2_path or not self.dataset3_path:
            QMessageBox.warning(self, "Warning", "Please select all three datasets to combine.")
            return

        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Combined Dataset")
        if not output_folder:
            return

        # --- ADD THIS: Call your function directly ---
        from data_integration import combine_three_datasets_with_yaml
        success, stats = combine_three_datasets_with_yaml(
            self.dataset1_path,  # Azure
            self.dataset2_path,  # Manual
            self.dataset3_path,  # Auto
            output_folder
        )
        if success:
            QMessageBox.information(
                self, "Success", f"Datasets combined successfully!\nStats: {stats}")
            self.yaml_path_input.setText(os.path.join(output_folder, "data.yaml"))
        else:
            QMessageBox.critical(self, "Error", "Failed to combine datasets.")

    def upload_to_azure(self):
        if not self.upload_directory or not self.azure_destination:
            QMessageBox.warning(
                self, "Warning", "Please select upload directory and Azure destination.")
            return

        # Call parent's upload function
        if hasattr(self.parent(), 'uploadToAzure_FromDialog'):
            success = self.parent().uploadToAzure_FromDialog(
                self.upload_directory,
                self.azure_destination
            )
            if success:
                QMessageBox.information(self, "Success", "Upload to Azure completed!")

    def start_training(self):
        # Validate all required fields
        if not self.model_path_input.text() or not self.yaml_path_input.text():
            QMessageBox.warning(
                self, "Warning", "Please fill in all required training configuration fields.")
            return

        self.accept()

    def get_training_config(self):
        """Get the training configuration"""
        return {
            'model_path': self.model_path_input.text(),
            'yaml_path': self.yaml_path_input.text(),
            'epochs': self.epochs_input.value(),
            'imgsz': self.imgsz_input.value(),
            'batch': self.batch_input.value(),
            'lr0': self.lr0_input.value(),
            'lrf': self.lrf_input.value(),
            'dropout': self.dropout_input.value(),
            'device': self.device_input.currentText().split()[0],  # Extract device number/type
            'seed': self.seed_input.value()
        }


class AzureUploadSelectionDialog(QDialog):
    """Dialog for selecting local folder to upload and Azure destination path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload to Azure - Select Source and Destination")
        self.setMinimumSize(750, 500)
        self.setModal(True)

        # Store paths
        self.local_folder = ""
        self.azure_destination = ""

        # Apply modern Azure-themed styling
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
            }
            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #E8EAED;
                border-radius: 12px;
                padding-top: 20px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: 600;
                color: #202124;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                background-color: #F8F9FA;
            }
            QLabel {
                color: #5F6368;
                font-size: 14px;
                background: transparent;
                border: none;
            }
            QLineEdit {
                background: #FFFFFF;
                padding: 12px 16px;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #0078D4;
            }
            QPushButton {
                background-color: #0078D4;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QCheckBox {
                font-size: 14px;
                color: #202124;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Upload Dataset to Azure Storage")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #0078D4; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "Select the local folder containing your dataset and specify where to upload it in Azure.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #5F6368; margin-bottom: 15px;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Local folder selection
        local_group = QGroupBox("Local Dataset Folder")
        local_layout = QVBoxLayout()
        local_layout.setSpacing(12)

        local_desc = QLabel(
            "Select the folder containing your dataset to upload.\nThis should contain train/valid/test subfolders with images and labels, plus data.yaml")
        local_desc.setWordWrap(True)
        local_layout.addWidget(local_desc)

        local_path_layout = QHBoxLayout()
        self.local_path_label = QLineEdit("No folder selected")
        self.local_path_label.setReadOnly(True)
        local_path_layout.addWidget(self.local_path_label)

        self.local_browse_btn = QPushButton("Browse")
        self.local_browse_btn.clicked.connect(self.select_local_folder)
        local_path_layout.addWidget(self.local_browse_btn)

        local_layout.addLayout(local_path_layout)

        # Show folder contents preview
        self.folder_contents_label = QLabel("")
        self.folder_contents_label.setStyleSheet(
            "color: #34A853; font-size: 12px; margin-top: 5px;")
        local_layout.addWidget(self.folder_contents_label)

        local_group.setLayout(local_layout)
        layout.addWidget(local_group)

        # Azure destination path
        azure_group = QGroupBox("Azure Destination Path")
        azure_layout = QVBoxLayout()
        azure_layout.setSpacing(12)

        azure_desc = QLabel(
            "Specify the folder path in Azure where your dataset will be uploaded.\nExample: SAVAI_METAL_INSPECTION_GEN or MyProject/Dataset_v2")
        azure_desc.setWordWrap(True)
        azure_layout.addWidget(azure_desc)

        # Predefined options
        predefined_layout = QHBoxLayout()

        self.savai_btn = QPushButton("SAVAI_METAL_INSPECTION_GEN")
        self.savai_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        self.savai_btn.clicked.connect(lambda: self.set_azure_path("SAVAI_METAL_INSPECTION_GEN"))

        self.custom_project_btn = QPushButton("Custom Project...")
        self.custom_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                color: #3C4043;
                border: 1px solid #DADCE0;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        self.custom_project_btn.clicked.connect(self.set_custom_path)

        predefined_layout.addWidget(self.savai_btn)
        predefined_layout.addWidget(self.custom_project_btn)
        predefined_layout.addStretch()

        azure_layout.addLayout(predefined_layout)

        # Custom path input
        self.azure_path_input = QLineEdit()
        self.azure_path_input.setPlaceholderText(
            "Enter Azure folder path (e.g., MyProject/Dataset_v1)")
        self.azure_path_input.textChanged.connect(self.on_azure_path_changed)
        azure_layout.addWidget(self.azure_path_input)

        # Current destination display
        self.azure_dest_label = QLabel("Destination: Not set")
        self.azure_dest_label.setStyleSheet(
            "color: #EA4335; margin-top: 5px;")
        azure_layout.addWidget(self.azure_dest_label)

        azure_group.setLayout(azure_layout)
        layout.addWidget(azure_group)

        # Upload options
        options_group = QGroupBox("Upload Options")
        options_layout = QVBoxLayout()
        options_layout.setSpacing(8)

        self.overwrite_checkbox = QCheckBox("Overwrite existing files")
        self.overwrite_checkbox.setChecked(True)
        options_layout.addWidget(self.overwrite_checkbox)

        self.backup_checkbox = QCheckBox("Create backup before overwriting")
        self.backup_checkbox.setEnabled(False)  # Only enabled when overwrite is checked
        self.overwrite_checkbox.toggled.connect(self.backup_checkbox.setEnabled)
        options_layout.addWidget(self.backup_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
                color: #3C4043;
            }
            QPushButton:hover {
                background-color: #F1F3F4;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        self.upload_btn = QPushButton("Start Upload")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #34A853;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #E8EAED;
                color: #9AA0A6;
            }
        """)
        self.upload_btn.clicked.connect(self.accept)
        self.upload_btn.setEnabled(False)  # Disabled until both fields are filled

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.upload_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #EA4335; font-style: italic; margin-top: 5px;")
        layout.addWidget(self.status_label)

    def select_local_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder to Upload",
            self.local_folder if self.local_folder else ""
        )
        if folder:
            self.local_folder = folder
            self.local_path_label.setText(folder)
            self.analyze_folder_contents()
            self.validate_inputs()

    def analyze_folder_contents(self):
        """Analyze and display folder contents"""
        if not self.local_folder or not os.path.exists(self.local_folder):
            return

        contents = []

        # Check for data.yaml
        yaml_path = os.path.join(self.local_folder, "data.yaml")
        if os.path.exists(yaml_path):
            contents.append("✓ data.yaml")
        else:
            contents.append("✗ data.yaml missing")

        # Check for train/valid/test folders
        splits = ['train', 'valid', 'test']
        split_info = []
        for split in splits:
            split_path = os.path.join(self.local_folder, split)
            if os.path.exists(split_path):
                images_path = os.path.join(split_path, 'images')
                labels_path = os.path.join(split_path, 'labels')

                img_count = len([f for f in os.listdir(images_path)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(images_path) else 0
                label_count = len([f for f in os.listdir(labels_path)
                                   if f.endswith('.txt')]) if os.path.exists(labels_path) else 0

                split_info.append(f"✓ {split}: {img_count} images, {label_count} labels")
            else:
                split_info.append(f"✗ {split} folder missing")

        contents.extend(split_info)
        self.folder_contents_label.setText("\n".join(contents))

        # Update style based on completeness
        if all("✓" in item for item in contents):
            self.folder_contents_label.setStyleSheet(
                "color: #34A853; font-size: 12px; margin-top: 5px;")
        else:
            self.folder_contents_label.setStyleSheet(
                "color: #FF9800; font-size: 12px; margin-top: 5px;")

    def set_azure_path(self, path):
        """Set predefined Azure path"""
        self.azure_path_input.setText(path)
        self.azure_destination = path
        self.azure_dest_label.setText(f"Destination: {path}")
        self.azure_dest_label.setStyleSheet(
            "color: #34A853; margin-top: 5px;")
        self.validate_inputs()

    def set_custom_path(self):
        """Open dialog for custom path input"""
        text, ok = QInputDialog.getText(
            self,
            "Custom Azure Path",
            "Enter the Azure folder path:\n(e.g., MyProject/Dataset_v1)",
            text=self.azure_path_input.text()
        )
        if ok and text.strip():
            self.azure_path_input.setText(text.strip())

    def on_azure_path_changed(self):
        """Handle changes to Azure path input"""
        path = self.azure_path_input.text().strip()
        if path:
            self.azure_destination = path
            self.azure_dest_label.setText(f"Destination: {path}")
            self.azure_dest_label.setStyleSheet(
                "color: #34A853; margin-top: 5px;")
        else:
            self.azure_destination = ""
            self.azure_dest_label.setText("Destination: Not set")
            self.azure_dest_label.setStyleSheet(
                "color: #EA4335; margin-top: 5px;")

        self.validate_inputs()

    def validate_inputs(self):
        """Validate inputs and enable/disable upload button"""
        issues = []

        if not self.local_folder:
            issues.append("Local folder not selected")
        elif not os.path.exists(self.local_folder):
            issues.append("Local folder does not exist")

        if not self.azure_destination:
            issues.append("Azure destination path not specified")

        # Update status and button
        if issues:
            self.status_label.setText("Required: " + "; ".join(issues))
            self.status_label.setStyleSheet("color: #EA4335; font-style: italic; margin-top: 5px;")
            self.upload_btn.setEnabled(False)
        else:
            self.status_label.setText("✓ Ready to upload")
            self.status_label.setStyleSheet("color: #34A853; font-style: italic; margin-top: 5px;")
            self.upload_btn.setEnabled(True)

    def get_upload_config(self):
        """Return dictionary of upload configuration"""
        return {
            'local_folder': self.local_folder,
            'azure_destination': self.azure_destination,
            'overwrite': self.overwrite_checkbox.isChecked(),
            'backup': self.backup_checkbox.isChecked()
        }

# -------------------------
# TrainDialog for model training options
# -------------------------


class TrainDialog(QDialog):
    def __init__(self, parent=None, is_retrain=False):
        super().__init__(parent)
        self.setWindowTitle("Train Model" if not is_retrain else "Retrain Model")
        self.setMinimumSize(500, 600)
        self.is_retrain = is_retrain

        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            QLabel {
                color: #202124;
                font-size: 14px;
                font-weight: 500;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                padding: 8px 12px;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #4285F4;
            }
            QPushButton {
                background-color: #4285F4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                color: #FFFFFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3367D6;
            }
            QDialogButtonBox QPushButton {
                min-width: 80px;
            }
        """)

        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.layout.setSpacing(16)

        self.model_path_input = QLineEdit(self)
        self.model_path_button = QPushButton("Choose Model Path", self)
        self.model_path_button.clicked.connect(self.choose_model_path)
        self.layout.addRow("Model Path:", self.model_path_input)
        self.layout.addWidget(self.model_path_button)

        self.yaml_path_input = QLineEdit(self)
        self.yaml_path_button = QPushButton("Choose Data YAML Path", self)
        self.yaml_path_button.clicked.connect(self.choose_yaml_path)
        self.layout.addRow("Data YAML Path:", self.yaml_path_input)
        self.layout.addWidget(self.yaml_path_button)

        self.epochs_input = QSpinBox(self)
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        self.layout.addRow("Epochs:", self.epochs_input)

        self.imgsz_input = QSpinBox(self)
        self.imgsz_input.setRange(32, 2048)
        self.imgsz_input.setValue(640)
        self.layout.addRow("Image Size:", self.imgsz_input)

        self.batch_input = QSpinBox(self)
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(16)
        self.layout.addRow("Batch Size:", self.batch_input)

        self.lr0_input = QDoubleSpinBox(self)
        self.lr0_input.setRange(1e-6, 1.0)
        self.lr0_input.setDecimals(6)
        self.lr0_input.setValue(0.0001)
        self.layout.addRow("Initial Learning Rate:", self.lr0_input)

        self.lrf_input = QDoubleSpinBox(self)
        self.lrf_input.setRange(1e-6, 1.0)
        self.lrf_input.setDecimals(6)
        self.lrf_input.setValue(0.01)
        self.layout.addRow("Final Learning Rate Multiplier:", self.lrf_input)

        self.dropout_input = QDoubleSpinBox(self)
        self.dropout_input.setRange(0.0, 1.0)
        self.dropout_input.setDecimals(2)
        self.dropout_input.setValue(0.25)
        self.layout.addRow("Dropout:", self.dropout_input)

        self.device_input = QSpinBox(self)
        self.device_input.setRange(0, 8)
        self.device_input.setValue(0)
        self.layout.addRow("Device:", self.device_input)

        self.seed_input = QSpinBox(self)
        self.seed_input.setRange(0, 10000)
        self.seed_input.setValue(42)
        self.layout.addRow("Seed:", self.seed_input)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def choose_model_path(self):
        # Default model path for initial directory
        default_model_path = "/home/sakar02/sakar-vision-ui/yolo11n.pt"
        initial_dir = os.path.dirname(default_model_path) if os.path.exists(
            default_model_path) else ""

        model_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Path", initial_dir, "Model Files (*.pt *.pth)")
        if model_path:
            self.model_path_input.setText(model_path)

    def choose_yaml_path(self):
        # Try to get a sensible initial directory from the parent
        initial_dir = ""
        parent = self.parent()
        if parent and hasattr(parent, 'outputDirLineEdit') and parent.outputDirLineEdit.text():
            initial_dir = parent.outputDirLineEdit.text()

        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data YAML Path", initial_dir, "YAML Files (*.yaml)")
        if yaml_path:
            self.yaml_path_input.setText(yaml_path)

    def get_params(self):
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

# -------------------------
# Modern Card Widget
# -------------------------


class ModernCard(QFrame):
    def __init__(self, title, subtitle, icon_text, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E8EAED;
                border-radius: 12px;
                padding: 0px;
            }
        """)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 24)
        main_layout.setSpacing(16)

        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        # Icon
        icon_label = QLabel(icon_text)
        icon_label.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border-radius: 20px;
                padding: 8px;
                font-size: 16px;
                color: #5F6368;
                min-width: 24px;
                max-width: 40px;
                min-height: 24px;
                max-height: 40px;
            }
        """)
        icon_label.setAlignment(Qt.AlignCenter)

        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #202124;
                border: none;
                background: transparent;
            }
        """)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #5F6368;
                border: none;
                background: transparent;
            }
        """)

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        header_layout.addWidget(icon_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Content area
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(16)
        main_layout.addLayout(self.content_layout)

    def addContent(self, item):
        """Add content to the card - can be a widget or a layout"""
        if isinstance(item, QWidget):
            self.content_layout.addWidget(item)
        elif isinstance(item, QLayout):
            self.content_layout.addLayout(item)
        else:
            raise TypeError(f"Expected QWidget or QLayout, got {type(item)}")

# -------------------------
# Modern Input Field
# -------------------------


class ModernInputField(QWidget):
    def __init__(self, label, placeholder="", button_text="Browse", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Label
        label_widget = QLabel(label)
        label_widget.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #202124;
                border: none;
                background: transparent;
            }
        """)

        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.setStyleSheet("""
            QLineEdit {
                padding: 12px 16px;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                font-size: 14px;
                background-color: #FFFFFF;
                color: #202124;
            }
            QLineEdit:focus {
                border: 2px solid #4285F4;
                outline: none;
            }
            QLineEdit:hover {
                border: 1px solid #9AA0A6;
            }
        """)

        input_layout.addWidget(self.line_edit)

        if button_text:
            self.browse_button = QPushButton(button_text)
            self.browse_button.setStyleSheet("""
                QPushButton {
                    background-color: #F8F9FA;
                    border: 1px solid #DADCE0;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: 500;
                    color: #3C4043;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #F1F3F4;
                    border: 1px solid #C4C7C5;
                }
                QPushButton:pressed {
                    background-color: #E8EAED;
                }
            """)
            input_layout.addWidget(self.browse_button)

        layout.addWidget(label_widget)
        layout.addLayout(input_layout)

# -------------------------
# Modern Action Button
# -------------------------


class ModernActionButton(QPushButton):
    def __init__(self, text, style="primary", parent=None):
        super().__init__(text, parent)

        if style == "primary":
            self.setStyleSheet("""
                QPushButton {
                    background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
                    border: none;
                    border-radius: 12px;
                    padding: 16px 32px;
                    font-size: 16px;
                    font-weight: 600;
                    color: #ff914d;
                    background-color: #F1F3F4;
                    border: 1px solid #F1F3F4;          
                    min-height: 20px;
                }
                QPushButton:hover {
                    background: linear-gradient(135deg, #3367D6 0%, #2E7D32 100%);
                    margin-top: -2px;
                }
                QPushButton:pressed {
                    background: linear-gradient(135deg, #2850A0 0%, #1B5E20 100%);
                }
            """)
        elif style == "secondary":
            self.setStyleSheet("""
                QPushButton {
                    background-color: #F1F3F4;
                    border: 1px solid #F1F3F4;
                    border-radius: 12px;
                    padding: 16px 32px;
                    font-size: 16px;
                    font-weight: 600;
                    color: #3C4043;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #F1F3F4;
                    border: 1px solid #C4C7C5;
                }
                QPushButton:pressed {
                    background-color: #E8EAED;
                }
            """)

# -------------------------
# Training Option Card
# -------------------------


class TrainingOptionCard(QFrame):
    def __init__(self, title, subtitle, speed, accuracy, dataset, button_text, icon_color, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E8EAED;
                border-radius: 12px;
                padding: 0px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        # Icon
        icon_label = QLabel("⚡" if "Fast" in title else "🎯")
        icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: {icon_color};
                border-radius: 20px;
                padding: 8px;
                font-size: 16px;
                min-width: 24px;
                max-width: 40px;
                min-height: 24px;
                max-height: 40px;
            }}
        """)
        icon_label.setAlignment(Qt.AlignCenter)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #202124;
                border: none;
                background: transparent;
            }
        """)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #5F6368;
                border: none;
                background: transparent;
            }
        """)

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        header_layout.addWidget(icon_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Properties
        props_layout = QVBoxLayout()
        props_layout.setSpacing(8)

        # Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        speed_label = QLabel(speed)
        speed_color = "#34A853" if speed == "Fast" else "#EA4335"
        speed_label.setStyleSheet(f"color: {speed_color}; font-weight: 600;")
        speed_layout.addWidget(speed_label)
        speed_layout.addStretch()

        # Accuracy
        accuracy_layout = QHBoxLayout()
        accuracy_layout.addWidget(QLabel("Accuracy:"))
        accuracy_label = QLabel(accuracy)
        accuracy_color = "#FF9800" if accuracy == "Moderate" else "#34A853"
        accuracy_label.setStyleSheet(f"color: {accuracy_color}; font-weight: 600;")
        accuracy_layout.addWidget(accuracy_label)
        accuracy_layout.addStretch()

        # Dataset
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset:"))
        dataset_label = QLabel(dataset)
        dataset_color = "#4285F4" if "Local" in dataset else "#9C27B0"
        dataset_label.setStyleSheet(f"color: {dataset_color}; font-weight: 600;")
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addStretch()

        props_layout.addLayout(speed_layout)
        props_layout.addLayout(accuracy_layout)
        props_layout.addLayout(dataset_layout)

        layout.addLayout(props_layout)

        # Button
        self.button = QPushButton(button_text)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #5F6368;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #3C4043;
            }
            QPushButton:pressed {
                background-color: #202124;
            }
        """)

        layout.addWidget(self.button)

# -------------------------
# Main UI: YOLO Auto Annotation and Dataset Splitter with Modern Design
# -------------------------


class AutoAnnotateUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window icon directly
        self.setWindowIcon(QIcon("sakar.png"))

        self.setWindowTitle("Auto Annotation and Dataset Splitter")
        self.setGeometry(200, 200, 1200, 800)

        # Set modern styling
        self.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
        """)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #F8F9FA;
            }
            QScrollBar:vertical {
                background-color: #F1F3F4;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #DADCE0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #BDC1C6;
            }
        """)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(420, 40, 420, 40)
        content_layout.setSpacing(32)

        # Auto Annotation Settings Card
        auto_card = ModernCard(
            "Auto Annotation Settings",
            "Configure automated object detection parameters",
            "⚙️"
        )

        # Model input
        self.model_field = ModernInputField("Model (.pt)", "Select your .pt file")
        self.model_field.browse_button.clicked.connect(self.chooseModelFile)
        auto_card.addContent(self.model_field)

        # Confidence and class names row
        conf_class_layout = QHBoxLayout()
        conf_class_layout.setSpacing(24)

        # Confidence threshold
        conf_widget = QWidget()
        conf_layout = QVBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.setSpacing(8)

        conf_label = QLabel("Confidence Threshold")
        conf_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #202124;
            }
        """)

        self.confidenceSpinBox = QDoubleSpinBox()
        self.confidenceSpinBox.setRange(0.0, 1.0)
        self.confidenceSpinBox.setSingleStep(0.1)
        self.confidenceSpinBox.setValue(0.5)
        self.confidenceSpinBox.setStyleSheet("""
            QDoubleSpinBox {
                padding: 12px 16px;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                font-size: 14px;
                background-color: #FFFFFF;
                min-width: 100px;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #4285F4;
            }
        """)

        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.confidenceSpinBox)

        # Class names
        class_widget = QWidget()
        class_layout = QVBoxLayout(class_widget)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(8)

        class_label = QLabel("Class Names (comma separated)")
        class_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #202124;
            }
        """)

        self.classNamesLineEdit = QLineEdit()
        self.classNamesLineEdit.setPlaceholderText("Damage, Hole, Rolling, Crazing, Scratch")
        self.classNamesLineEdit.setText("Damage, Hole, Rolling, Crazing, Scratch")
        self.classNamesLineEdit.setStyleSheet("""
            QLineEdit {
                padding: 12px 16px;
                border: 1px solid #DADCE0;
                border-radius: 8px;
                font-size: 14px;
                background-color: #FFFFFF;
            }
            QLineEdit:focus {
                border: 2px solid #4285F4;
            }
        """)

        class_layout.addWidget(class_label)
        class_layout.addWidget(self.classNamesLineEdit)

        conf_class_layout.addWidget(conf_widget)
        conf_class_layout.addWidget(class_widget, 2)
        auto_card.addContent(conf_class_layout)

        # Input and output directories
        self.input_field = ModernInputField("Input Image Directory", "Select input directory")
        self.input_field.browse_button.clicked.connect(self.chooseInputDir)
        auto_card.addContent(self.input_field)

        self.output_field = ModernInputField("Output Directory", "Select output directory")
        self.output_field.browse_button.clicked.connect(self.chooseOutputDir)
        auto_card.addContent(self.output_field)

        # Run button
        run_button = ModernActionButton("▶ Run Auto Annotation")
        run_button.clicked.connect(self.runAutoAnnotation)
        auto_card.addContent(run_button)

        content_layout.addWidget(auto_card)

        # Dataset Split Settings Card
        split_card = ModernCard(
            "Dataset Split Settings",
            "Configure train/validation/test data splits",
            "📊"
        )

        # Dataset directory to split
        self.split_field = ModernInputField(
            "Dataset Directory to Split", "Select dataset directory")
        self.split_field.browse_button.clicked.connect(self.chooseSplitDir)
        split_card.addContent(self.split_field)

        # Split configuration
        split_config_widget = QWidget()
        split_config_layout = QVBoxLayout(split_config_widget)
        split_config_layout.setSpacing(16)

        split_label = QLabel("Split Configuration")
        split_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #202124;
            }
        """)
        split_config_layout.addWidget(split_label)

        # Range slider
        self.rangeSlider = RangeSlider(0, 100, 60, 80, self)
        self.rangeSlider.valueChanged.connect(self.updateRangeSliderLabel)
        split_config_layout.addWidget(self.rangeSlider)

        # Split ratio labels
        labels_layout = QHBoxLayout()
        train_label = QLabel("🔵 Train: 60%")
        train_label.setStyleSheet("color: #4285F4; font-weight: 600;")
        valid_label = QLabel("🟢 Valid: 20%")
        valid_label.setStyleSheet("color: #34A853; font-weight: 600;")
        test_label = QLabel("🟠 Test: 20%")
        test_label.setStyleSheet("color: #FF9800; font-weight: 600;")

        labels_layout.addWidget(train_label)
        labels_layout.addWidget(valid_label)
        labels_layout.addWidget(test_label)
        labels_layout.addStretch()

        self.train_label = train_label
        self.valid_label = valid_label
        self.test_label = test_label

        split_config_layout.addLayout(labels_layout)
        split_card.addContent(split_config_widget)

        # Split button
        split_button = ModernActionButton("🔀 Split Dataset", "secondary")
        split_button.clicked.connect(self.splitDataset)
        split_card.addContent(split_button)

        content_layout.addWidget(split_card)

        # Final Model Training Card
        training_card = ModernCard(
            "Final Model Training",
            "Train the final model using combined data from both local annotations and Azure.",
            "🎓"
        )

        # Training options
        training_options_layout = QHBoxLayout()
        training_options_layout.setSpacing(24)

        # Fast Training Option
        self.fast_training_card = TrainingOptionCard(
            "Fast Training",
            "Current images only",
            "Fast",
            "Moderate",
            "Local Only",
            "Train Fast Model",
            "#E3F2FD"
        )
        self.fast_training_card.button.clicked.connect(self.trainFastModel)

        # Accurate Training Option
        self.accurate_training_card = TrainingOptionCard(
            "Accurate Training",
            "Large dataset + current images",
            "Slow",
            "High",
            "Azure + Local",
            "Train Accurate Model",
            "#F3E5F5"
        )
        self.accurate_training_card.button.clicked.connect(self.trainAccurateModel)

        training_options_layout.addWidget(self.fast_training_card)
        training_options_layout.addWidget(self.accurate_training_card)

        training_card.addContent(training_options_layout)
        content_layout.addWidget(training_card)

        # Azure Data Integration

        # Deploy model button
        deploy_button_layout = QHBoxLayout()
        deploy_button_layout.addStretch()

        self.deployModelButton = ModernActionButton("🚀 Deploy Model")
        self.deployModelButton.clicked.connect(self.deployModel)

        deploy_button_layout.addWidget(self.deployModelButton)
        deploy_button_layout.addStretch()

        training_card.addContent(deploy_button_layout)
        content_layout.addWidget(training_card)

        # Set scroll area content
        scroll.setWidget(content_widget)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # Initialize YOLO model variables
        self.MODEL_PATH = ""
        self.det_model = None
        self.detection_results = {}
        self.detections_dict = {}

        # Load class manager
        class_manager = ClassManager()
        if class_manager.initialized or class_manager.load_from_file():
            classes = class_manager.get_classes()
            self.classNamesLineEdit.setText(", ".join(classes))

        self.modelComboBox = QComboBox()
        self.modelComboBox.setStyleSheet("""
            QComboBox {
                padding: 6px;
                border: 1px solid #DADCE0;
                border-radius: 4px;
                background-color: #FFFFFF;
                color: #333;
            }
            QComboBox::drop-down {
                background-color: #F8F9FA;
                border: 1px solid #DADCE0;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #333;
                border: 1px solid #DADCE0;
                selection-background-color: #E8F0FE;
                selection-color: #4285F4;
            }
            QComboBox QAbstractItemView::item {
                background-color: #FFFFFF;
                color: #333;
                padding: 8px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #F8F9FA;
                color: #4285F4;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #E8F0FE;
                color: #4285F4;
            }
        """)

    def updateRangeSliderLabel(self, lower, upper):
        train_percent = lower
        valid_percent = upper - lower
        test_percent = 100 - upper

        self.train_label.setText(f"🔵 Train: {train_percent}%")
        self.valid_label.setText(f"🟢 Valid: {valid_percent}%")
        self.test_label.setText(f"🟠 Test: {test_percent}%")

    # Add this to the AutoAnnotateUI class in auto_annotation.py
    def updateClassNames(self, class_names):
        """
        Updates the class names in the Auto Annotation UI.
        This can be called from the Manual Annotation UI.

        Args:
            class_names: List of class names or comma-separated string of class names
        """
        if isinstance(class_names, list):
            class_names_str = ", ".join(class_names)
        else:
            class_names_str = class_names

        self.classNamesLineEdit.setText(class_names_str)
        print(f"Auto Annotation UI classes updated to: {class_names_str}")

    # ----- Helper methods for browsing directories and files -----

    def chooseModelFile(self):
        """
        Opens a file dialog to select a YOLO model file and updates classes from model.
        Always extracts classes from the model file in class ID order, overriding any manually set classes.
        """
        # Default model path as specified
        default_model_path = "/home/sakar02/sakar-vision-ui/y11n.pt"
        initial_dir = os.path.dirname(default_model_path) if os.path.exists(
            default_model_path) else ""

        file, _ = QFileDialog.getOpenFileName(
            self, "Select Model (.pt)", initial_dir, "PyTorch Model Files (*.pt)")
        if file:
            self.model_field.line_edit.setText(file)
            self.MODEL_PATH = file
            try:
                # Try to load the model - suppress model structure printing
                import sys
                import io
                # Temporarily redirect stdout to suppress verbose model output
                original_stdout = sys.stdout
                sys.stdout = io.StringIO()

                print(f"Loading YOLO model from: {self.MODEL_PATH}")
                self.det_model = YOLO(self.MODEL_PATH)

                # Restore stdout
                sys.stdout = original_stdout
                print(f"Successfully loaded model from: {self.MODEL_PATH}")

                # Use the extract_classes_from_model function from utils
                from utils import extract_classes_from_model
                print(f"Attempting to extract classes from model: {self.MODEL_PATH}")
                success, class_dict = extract_classes_from_model(self.MODEL_PATH)

                if success and class_dict:
                    # Make sure classes are in numeric ID order
                    ordered_classes = []

                    # Ensure all keys are properly interpreted as integers for sorting
                    try:
                        sorted_keys = sorted([int(k) if isinstance(k, (int, str)) and str(k).isdigit() else k
                                              for k in class_dict.keys()])
                        print(f"Sorted class IDs: {sorted_keys}")

                        for class_id in sorted_keys:
                            ordered_classes.append(class_dict[class_id])

                        print(f"Successfully extracted classes in order: {ordered_classes}")

                        # Update the UI and ClassManager with ordered classes
                        class_manager = ClassManager()
                        class_manager.update_classes(ordered_classes, f"model:{self.MODEL_PATH}")

                        # Update UI with classes from model in class ID order
                        self.classNamesLineEdit.setText(", ".join(ordered_classes))

                        QMessageBox.information(self, "Classes Loaded",
                                                f"Successfully loaded {len(ordered_classes)} classes from model in class ID order.")
                    except Exception as sorting_err:
                        print(f"Error sorting classes: {sorting_err}")
                        QMessageBox.warning(self, "Warning",
                                            f"Error sorting extracted classes: {sorting_err}\nUsing classes as-is.")
                        # Use classes without sorting if sorting fails
                        ordered_classes = [class_dict[k] for k in class_dict.keys()]
                        self.classNamesLineEdit.setText(", ".join(ordered_classes))
                else:
                    # Print model attributes to help diagnose the issue
                    print(f"Model has names attribute: {hasattr(self.det_model, 'names')}")
                    if hasattr(self.det_model, 'names'):
                        print(f"Model names attribute type: {type(self.det_model.names)}")
                        print(f"Model names attribute content: {self.det_model.names}")
                    else:
                        print("Model does not have 'names' attribute")

                    QMessageBox.warning(self, "Warning",
                                        "Could not extract classes from model. Using existing classes.")

                QMessageBox.information(self, "Model Loaded",
                                        f"Successfully loaded model from:\n{self.MODEL_PATH}")
            except Exception as e:
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def chooseInputDir(self):
        """
        Opens a directory dialog to select input images, automatically using Folder B when available.
        """
        # Check if we have a parent with capture_folder_b set
        parent = self.parent()
        folder_b_path = None

        while parent is not None:
            if hasattr(parent, 'capture_folder_b') and parent.capture_folder_b:
                folder_b_path = parent.capture_folder_b
                break
            parent = parent.parent()

        # Use folder dialog with initial path if available
        if folder_b_path and os.path.exists(folder_b_path):
            initial_dir = folder_b_path
        else:
            initial_dir = ""

        dir = QFileDialog.getExistingDirectory(self, "Select Input Image Directory", initial_dir)
        if dir:
            self.input_field.line_edit.setText(dir)

            # Auto-configure output directory based on input
            if not self.output_field.line_edit.text().strip():
                # Check if we're using Folder B and can find the parent directory
                parent_dir = os.path.dirname(dir)
                base_name = os.path.basename(dir)
                if (base_name == "Folder_B" and os.path.exists(parent_dir)):
                    # Set output to auto_annotated in parent directory
                    auto_annotated_path = os.path.join(parent_dir, "auto_annotated")
                    os.makedirs(auto_annotated_path, exist_ok=True)
                    self.output_field.line_edit.setText(auto_annotated_path)

                    # Also set the split directory to the same path
                    if not self.split_field.line_edit.text().strip():
                        self.split_field.line_edit.setText(auto_annotated_path)
                else:
                    # Default behavior - use input dir for split dir
                    if not self.split_field.line_edit.text().strip():
                        self.split_field.line_edit.setText(dir)

    def combineDatasets_Accurate(self, azure_path, manual_path, auto_path, output_path):
        """Combine 3 datasets for accurate training"""
        try:
            # Use your existing combineDatasets method but with 3 specific paths
            # This calls your existing combine function that handles 3 folders

            # Create temporary folder selection to pass to existing function
            folders = {
                'azure_folder': azure_path,
                'manual_folder': manual_path,
                'auto_folder': auto_path,
                'output_folder': output_path
            }

            # Call your existing combine logic
            from data_integration import combine_local_and_azure_data, initialize_combined_data

            # Create progress dialog
            progress_dialog = CombineDatasetProgressDialog(self)
            progress_dialog.show()
            QApplication.processEvents()

            def combine_thread_func():
                try:
                    # Extract class names (you may need to modify this based on your setup)
                    class_names_text = self.classNamesLineEdit.text().strip()
                    CLASS_NAMES = [name.strip()
                                   for name in class_names_text.split(",") if name.strip()]

                    initialize_combined_data(CLASS_NAMES)

                    success, stats = combine_local_and_azure_data(
                        manual_path,
                        auto_path,
                        azure_path,
                        output_path,
                        CLASS_NAMES,
                        update_config=False
                    )

                    if success:
                        QTimer.singleShot(0, lambda: progress_dialog.setComplete(stats))
                    else:
                        QTimer.singleShot(0, lambda: progress_dialog.setError(
                            "Failed to combine datasets"))

                except Exception as e:
                    QTimer.singleShot(0, lambda: progress_dialog.setError(str(e)))

            # Start combination in separate thread
            import threading
            combine_thread = threading.Thread(target=combine_thread_func)
            combine_thread.daemon = True
            combine_thread.start()

            # Show dialog
            progress_dialog.exec_()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to combine datasets: {str(e)}")
            return False

    def uploadToAzure_FromDialog(self, local_folder, azure_destination):
        """Upload to Azure from dialog"""
        try:
            # Use your existing upload logic
            from data_integration import upload_combined_dataset_to_azure

            # Create progress dialog
            progress_dialog = AzureUploadProgressDialog(self)
            progress_dialog.show()
            QApplication.processEvents()

            # Cancellation handling
            import threading
            upload_cancel_event = threading.Event()

            def cancel_check_func():
                return upload_cancel_event.is_set() or progress_dialog.cancelled

            def upload_thread_func():
                try:
                    def progress_callback(msg, current_file=None, total_files=None):
                        if cancel_check_func():
                            return False

                        if current_file is not None and total_files is not None:
                            progress_percent = int((current_file / total_files) * 100)
                            QTimer.singleShot(0, lambda: progress_dialog.updateStatus(
                                f"{msg} ({current_file}/{total_files})", progress_percent))
                        else:
                            QTimer.singleShot(0, lambda: progress_dialog.updateStatus(msg, None))

                        return True

                    success, stats = upload_combined_dataset_to_azure(
                        local_folder,
                        azure_folder=azure_destination,
                        progress_callback=progress_callback
                    )

                    if cancel_check_func():
                        QTimer.singleShot(0, lambda: progress_dialog.setCancelled())
                        return

                    if success:
                        QTimer.singleShot(0, lambda: progress_dialog.setComplete(stats))
                    else:
                        QTimer.singleShot(0, lambda: progress_dialog.setError("Upload failed"))

                except Exception as e:
                    QTimer.singleShot(0, lambda: progress_dialog.setError(str(e)))

            # Handle cancellation
            def handle_cancel():
                upload_cancel_event.set()
                progress_dialog.cancelled = True

            progress_dialog.cancelButton.clicked.connect(handle_cancel)

            # Start upload thread
            upload_thread = threading.Thread(target=upload_thread_func)
            upload_thread.daemon = True
            upload_thread.start()

            # Show dialog
            progress_dialog.exec_()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to upload to Azure: {str(e)}")
            return False

    def downloadFromAzure_FromDialog(self, azure_path, local_folder):
        """Download from Azure from dialog using the dedicated download dialog"""
        try:
            # Use your existing download logic
            from data_integration import download_azure_training_data

            # Create DOWNLOAD progress dialog (not upload) - THIS IS THE KEY CHANGE
            progress_dialog = AzureDownloadProgressDialog(self)
            progress_dialog.show()
            QApplication.processEvents()

            # Cancellation handling
            import threading
            download_cancel_event = threading.Event()

            def cancel_check_func():
                return download_cancel_event.is_set() or progress_dialog.cancelled

            def download_thread_func():
                try:
                    def progress_callback(msg, current_file=None, total_files=None):
                        if cancel_check_func():
                            return False

                        if current_file is not None and total_files is not None:
                            progress_percent = int((current_file / total_files) * 100)
                            QTimer.singleShot(0, lambda: progress_dialog.updateStatus(
                                f"{msg} ({current_file}/{total_files})", progress_percent))
                            QTimer.singleShot(0, lambda: progress_dialog.updateFileProgress(
                                current_file, total_files))
                        else:
                            QTimer.singleShot(0, lambda: progress_dialog.updateStatus(msg, None))

                        return True

                    result = download_azure_training_data(
                        local_folder,
                        azure_folder=azure_path,
                        cancel_check_func=cancel_check_func,
                        progress_callback=progress_callback
                    )

                    if cancel_check_func():
                        QTimer.singleShot(0, lambda: progress_dialog.setError(
                            "Download was cancelled"))
                        return

                    if result:
                        QTimer.singleShot(0, lambda: progress_dialog.setComplete(result))
                    else:
                        QTimer.singleShot(0, lambda: progress_dialog.setError("Download failed"))

                except Exception as e:
                    QTimer.singleShot(0, lambda: progress_dialog.setError(str(e)))

            # Handle cancellation
            def handle_cancel():
                download_cancel_event.set()
                progress_dialog.cancelled = True

            progress_dialog.cancelButton.clicked.connect(handle_cancel)

            # Start download thread
            download_thread = threading.Thread(target=download_thread_func)
            download_thread.daemon = True
            download_thread.start()

            # Show dialog
            progress_dialog.exec_()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to download from Azure: {str(e)}")
            return False

    def combineDatasets_Fast(self, dataset1_path, dataset2_path, output_path):
        """Combine 2 datasets for fast training"""
        try:
            from data_integration import combine_two_datasets

            # Create progress dialog
            progress_dialog = CombineDatasetProgressDialog(self)
            progress_dialog.show()
            QApplication.processEvents()

            def combine_thread_func():
                try:
                    success, stats = combine_two_datasets(
                        dataset1_path,
                        dataset2_path,
                        output_path
                    )

                    if success:
                        # Use QTimer.singleShot to call setComplete in the main thread
                        QTimer.singleShot(0, lambda: progress_dialog.setComplete(stats))
                        print(f"Dataset combination successful. Stats: {stats}")
                        QTimer.singleShot(0, lambda: QMessageBox.information(
                            self, "Success", f"Datasets combined successfully!\n\nStats:\n{stats}"))
                    else:
                        QTimer.singleShot(0, lambda: progress_dialog.setError(
                            "Failed to combine datasets"))
                        print("Dataset combination failed.")
                        QTimer.singleShot(0, lambda: QMessageBox.critical(
                            self, "Error", "Failed to combine datasets. Please check the logs."))

                except Exception as e:
                    QTimer.singleShot(0, lambda: progress_dialog.setError(str(e)))
                    print(f"Error during dataset combination: {e}")
                    QTimer.singleShot(0, lambda: QMessageBox.critical(
                        self, "Error", f"An error occurred during dataset combination:\n{e}"))

            # Start combination in a separate thread
            combine_thread = threading.Thread(target=combine_thread_func)
            combine_thread.daemon = True
            combine_thread.start()

            # Show dialog
            progress_dialog.exec_()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to combine datasets: {str(e)}")
            print(f"Failed to combine datasets: {e}")
            return False

    def chooseOutputDir(self):
        """
        Opens a directory dialog to select output directory, with auto_annotated folder preferred.
        """
        # Check if we have a parent with capture_storage_folder set
        parent = self.parent()
        root_folder = None

        while parent is not None:
            if hasattr(parent, 'capture_storage_folder') and parent.capture_storage_folder:
                root_folder = parent.capture_storage_folder
                break
            parent = parent.parent()

        # Set default output dir to auto_annotated in root folder if available
        default_dir = ""
        if root_folder:
            default_dir = os.path.join(root_folder, "auto_annotated")
            os.makedirs(default_dir, exist_ok=True)

        dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", default_dir)
        if dir:
            self.output_field.line_edit.setText(dir)

            # If split directory is not yet set, use the same output directory
            if not self.split_field.line_edit.text().strip():
                self.split_field.line_edit.setText(dir)

    def chooseSplitDir(self):
        """
        Opens a directory dialog to select the dataset directory to split.
        """
        # Default to output directory if already set
        initial_dir = self.output_field.line_edit.text().strip()
        # If no output dir set, check for auto_annotated in root folder
        if not initial_dir:
            parent = self.parent()
            root_folder = None

            while parent is not None:
                if hasattr(parent, 'capture_storage_folder') and parent.capture_storage_folder:
                    root_folder = parent.capture_storage_folder
                    break
                parent = parent.parent()
            if root_folder:
                auto_annotated = os.path.join(root_folder, "auto_annotated")
                if os.path.exists(auto_annotated):
                    initial_dir = auto_annotated

        dir = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory to Split (Output Directory)", initial_dir)
        if dir:
            self.split_field.line_edit.setText(dir)

    # ----- Auto Annotation Process -----
    def generate_class_colors(self, class_names):
        """
        Generates distinct colors for each class.
        Returns a dictionary mapping class_id -> (B,G,R) color tuple.
        """
        class_colors = {}
        # Predefined colors for better distinction (BGR format for OpenCV)
        color_list = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Brown
            (128, 128, 0),  # Teal
            (0, 0, 128),    # Dark Red
        ]

        # If we have more classes than predefined colors, generate random ones
        for i in range(len(class_names)):
            if i < len(color_list):
                class_colors[i] = color_list[i]
            else:
                # Generate a random color
                while True:
                    color = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)
                    )
                    # Ensure the color is not too similar to existing ones
                    if color not in class_colors.values():
                        class_colors[i] = color
                        break

        return class_colors

    def runAutoAnnotation(self):
        if not self.det_model:
            QMessageBox.warning(self, "Warning", "Please select a Model (.pt) file first.")
            return

        image_dir = self.input_field.line_edit.text().strip()
        output_dir = self.output_field.line_edit.text().strip()
        conf_threshold = self.confidenceSpinBox.value()
        class_names_text = self.classNamesLineEdit.text().strip()
        if not image_dir or not output_dir or not class_names_text:
            QMessageBox.warning(
                self, "Warning", "Please provide input directory, output directory, and class names.")
            return
        CLASS_NAMES = [name.strip() for name in class_names_text.split(",") if name.strip()]

        class_names_text = self.classNamesLineEdit.text().strip()
        if not image_dir or not output_dir or not class_names_text:
            QMessageBox.warning(
                self, "Warning", "Please provide input directory, output directory, and class names.")
            return
        CLASS_NAMES = [name.strip() for name in class_names_text.split(",") if name.strip()]

        # Update ClassManager with these classes
        class_manager = ClassManager()
        class_manager.update_classes(CLASS_NAMES, "auto_annotation_run")

        # Generate a color for each class
        class_colors = self.generate_class_colors(CLASS_NAMES)

        # Use the output directory for storing the original images and label.txt side by side
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.detection_results = {}  # Reset detection results (annotated images)
        self.detections_dict = {}      # Reset detection details

        # Process each image in the input directory
        for image_name in os.listdir(image_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load image {image_name}. Skipping.")
                continue
            print(f"Original image shape for {image_name}: {image.shape}")

            # Run YOLO detection
            results = self.det_model(image)
            if isinstance(results, list):
                results = results[0]

            # Prepare annotated image for verification (do not save this one)
            annotated_image = image.copy()
            detections = []
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()
                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    if confidence < conf_threshold:
                        continue
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    class_id_int = int(class_id)
                    # Use class-specific color instead of fixed green
                    # Default to green if not found
                    color = class_colors.get(class_id_int, (0, 255, 0))
                    # Draw rectangle with class-specific color
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    # Draw label with the same color
                    label_text = f"{CLASS_NAMES[class_id_int]} {confidence:.2f}" if CLASS_NAMES else f"Class {class_id_int} {confidence:.2f}"
                    cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detections.append({
                        'class_id': class_id_int,
                        'box': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'color': color  # Store the color with the detection
                    })
                self.detections_dict[image_name] = detections
            else:
                self.detections_dict[image_name] = []
            self.detection_results[image_name] = annotated_image

            # Copy the original image to output_dir (unchanged)
            output_image_path = os.path.join(output_dir, image_name)
            shutil.copy2(image_path, output_image_path)
            # Write label file (even if empty)
            label_file_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
            with open(label_file_path, 'w') as label_file:
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    # Convert to YOLO format: x_center, y_center, width, height (normalized)
                    x_center = ((x1 + x2) / 2) / image.shape[1]
                    y_center = ((y1 + y2) / 2) / image.shape[0]
                    box_width = (x2 - x1) / image.shape[1]
                    box_height = (y2 - y1) / image.shape[0]
                    label_file.write(
                        f"{det['class_id']} {x_center} {y_center} {box_width} {box_height}\n")

        # Check if we found any images to process
        if not self.detection_results:
            QMessageBox.warning(self, "Warning",
                                f"No valid images found in {image_dir}\n\nPlease check the input directory.")
            return

        # Launch the annotation review dialog
        annotated_files = list(self.detection_results.keys())
        dialog = AnnotationReviewDialog(
            annotated_files, self.detection_results, self.detections_dict, CLASS_NAMES, image_dir, output_dir, self)
        dialog.exec_()
        # After review, process results:
        for filename, confirmed in dialog.results.items():
            output_image_path = os.path.join(output_dir, filename)
            label_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            if confirmed is True:
                # For manually edited images, update label file with new detections.
                image = cv2.imread(output_image_path)
                if image is None:
                    continue
                with open(label_path, 'w') as f:
                    for det in self.detections_dict.get(filename, []):
                        x1, y1, x2, y2 = det['box']
                        x_center = ((x1 + x2) / 2) / image.shape[1]
                        y_center = ((y1 + y2) / 2) / image.shape[0]
                        box_width = (x2 - x1) / image.shape[1]
                        box_height = (y2 - y1) / image.shape[0]
                        f.write(
                            f"{det['class_id']} {x_center} {y_center} {box_width} {box_height}\n")
            elif confirmed is False:
                # Delete rejected images and their label files from output_dir
                for path in [output_image_path, label_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            print(f"Deleted {path}")
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")
        # Update split directory to point to output
        if not self.split_field.line_edit.text().strip():
            self.split_field.line_edit.setText(output_dir)
        QMessageBox.information(
            self, "Done", f"Auto-annotation completed. Results saved to {output_dir}")

    # ----- Dataset Splitting Process -----
    def splitDataset(self):
        folder = self.split_field.line_edit.text().strip()
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Select Folder to Split Datasets")
        if not folder:
            QMessageBox.warning(self, "Warning", "Please select a folder to split.")
            return

        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        file_list = os.listdir(folder)
        basenames = []
        for file in file_list:
            name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions and (name + '.txt') in file_list:
                basenames.append(name)
        basenames = list(set(basenames))
        if not basenames:
            QMessageBox.warning(
                self, "Warning", "No image and label pairs found in the selected folder.")
            return
        random.shuffle(basenames)
        n = len(basenames)
        lower = self.rangeSlider._lowerValue
        upper = self.rangeSlider._upperValue
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
        class_names = [name.strip()
                       for name in self.classNamesLineEdit.text().strip().split(",") if name.strip()]
        if not class_names:
            class_names = ["class_0"]  # Default if no class names provided

        yaml_content = f"""
train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(class_names)}
names: {class_names}
"""
        yaml_path = os.path.join(folder, 'data.yaml')
        with open(yaml_path, 'w') as yaml_file:
            yaml_file.write(yaml_content)

        QMessageBox.information(self, "Split Completed",
                                f"Datasets split into Train: {train_count}, Valid: {valid_count}, Test: {test_count}\n\n"
                                f"YAML configuration file created at: {yaml_path}")

    # ----- Model Training Functions -----
    def find_latest_best_pt(self):
        """
        Finds the latest best.pt file from training runs.
        Returns the path to the file or None if not found.
        """
        latest_model = None
        highest_train_num = 0

        print("DEBUG: Starting search for latest best.pt file...")

        # Check for runs directories in various possible locations
        search_dirs = []

        # First, check in output_folder_path if available
        output_dir = self.output_field.line_edit.text().strip()
        if output_dir and os.path.exists(output_dir):
            runs_dir = os.path.join(output_dir, "runs")
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
                else:
                    print(f"DEBUG: Directory doesn't exist: {manual_runs}")

                    # Try creating the directory structure if it doesn't exist
                    try:
                        os.makedirs(manual_runs, exist_ok=True)
                        print(f"DEBUG: Created directory: {manual_runs}")
                    except Exception as e:
                        print(f"DEBUG: Failed to create directory: {e}")

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
                # List all directories starting with "train"
                try:
                    dir_contents = os.listdir(runs_dir)
                    print(f"DEBUG: Directory contents: {dir_contents}")

                    train_dirs = [d for d in dir_contents
                                  if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("train")]

                    # Sort train directories in descending order to find the latest first
                    train_dirs.sort(reverse=True)
                    print(f"DEBUG: Found train directories (sorted): {train_dirs}")

                    for train_dir in train_dirs:
                        # Extract the train number (trainX where X is a number)
                        match = re.search(r'train(\d+)', train_dir)
                        if match:
                            train_num = int(match.group(1))
                            print(f"DEBUG: Processing train{train_num}")

                            # Check if best.pt exists in this directory
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

        print("DEBUG: No best.pt models found in standard search, trying recursive search...")

        # Fallback to looking specifically for best.pt files in all directories
        best_pt_files = []
        for runs_dir in search_dirs:
            if os.path.exists(runs_dir):
                print(f"DEBUG: Recursive search for best.pt in {runs_dir}")
                for root, dirs, files in os.walk(runs_dir):
                    for file in files:
                        if file == 'best.pt':  # Only look for files named exactly "best.pt"
                            model_path = os.path.join(root, file)
                            print(f"DEBUG: Found best.pt file: {model_path}")
                            # Store the file along with its modification time
                            mod_time = os.path.getmtime(model_path)
                            best_pt_files.append((model_path, mod_time))

        # Return the most recently modified best.pt file if any were found
        if best_pt_files:
            # Sort by modification time (newest first)
            best_pt_files.sort(key=lambda x: x[1], reverse=True)
            newest_best_pt = best_pt_files[0][0]
            print(f"DEBUG: Selected most recent best.pt: {newest_best_pt}")
            return newest_best_pt

        # If no best.pt found, look for other .pt files but NOT last.pt
        print("DEBUG: No best.pt found in any directory. Looking for other model files...")
        for runs_dir in search_dirs:
            if os.path.exists(runs_dir):
                for root, dirs, files in os.walk(runs_dir):
                    for file in files:
                        # Look for .pt files that are NOT last.pt
                        if file.endswith('.pt') and file != 'last.pt':
                            model_path = os.path.join(root, file)
                            print(f"DEBUG: Found alternative model file: {model_path}")
                            return model_path

        print("DEBUG: No suitable model files found")
        return None

    def trainFastModel(self):
        """Show Fast Training overlay with combine + upload + training config"""
        dialog = FastTrainingOverlayDialog(self)

        if dialog.exec_() == QDialog.Accepted:
            # Get the training configuration
            training_config = dialog.get_training_config()

            # Start the actual training with the configuration
            self.runTraining(training_config)

            QMessageBox.information(
                self,
                "Fast Training Started",
                "Fast training has been initiated with your selected configuration."
            )

    def trainAccurateModel(self):
        """Show Accurate Training overlay with download + combine + upload + training config"""
        dialog = AccurateTrainingOverlayDialog(self)

        if dialog.exec_() == QDialog.Accepted:
            # Get the training configuration
            training_config = dialog.get_training_config()

            # Start the actual training with the configuration
            self.runTraining(training_config)

            QMessageBox.information(
                self,
                "Accurate Training Started",
                "Accurate training has been initiated with your selected configuration."
            )

    def autoCloseMessage(self, title, message, timeout=2000):
        """
        Displays an auto-closing message box with an "OK" button.
        """
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        QTimer.singleShot(timeout, msg.close)
        msg.exec_()

    def runTraining(self, params):
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
                    project=run_folder  # Specify the run folder
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
                # Update the dialog with error
                print(f"Error during training: {str(e)}")
                error_msg = str(e)
                progress_dialog.set_error(error_msg)

        # Start the training in a separate thread so dialog remains responsive
        training_thread = threading.Thread(target=train_thread_func)
        training_thread.daemon = True  # Allow application to exit if thread is still running
        training_thread.start()

        # Show the dialog (this will block until dialog is closed)
        progress_dialog.exec_()

    def deployModel(self):
        """
        Navigates to the deployment UI page.
        """
        parent = self.parent()
        stacked_widget = None

        # Traverse up the parent hierarchy to find the QStackedWidget
        while parent is not None:
            if isinstance(parent, QWidget) and hasattr(parent, 'stacked_widget') and isinstance(getattr(parent, 'stacked_widget'), QStackedWidget):
                stacked_widget = parent.stacked_widget
                break
            parent = parent.parent()

        if stacked_widget is not None:
            # Find and hide ALL title bars in ALL widgets
            for i in range(stacked_widget.count()):
                widget = stacked_widget.widget(i)
                if hasattr(widget, 'title_bar') and widget.title_bar:
                    widget.title_bar.hide()

                # Also check for nested camera_feed_ui that might have a title bar
                if hasattr(widget, 'camera_feed_ui') and hasattr(widget.camera_feed_ui, 'title_bar'):
                    widget.camera_feed_ui.title_bar.hide()

            # Switch to the DeploymentUI page
            for i in range(stacked_widget.count()):
                if isinstance(stacked_widget.widget(i), DeploymentUI):
                    deployment_ui = stacked_widget.widget(i)
                    # Show only the deployment UI title bar
                    if hasattr(deployment_ui, 'title_bar'):
                        deployment_ui.title_bar.show()
                    stacked_widget.setCurrentWidget(deployment_ui)
                    break
        else:
            QMessageBox.warning(
                self,
                "Navigation Error",
                "Could not navigate to the deployment page. Please check the application setup."
            )

    def downloadAzureData(self):
        """
        Downloads training data from Azure Storage with robust cancellation support using dedicated download dialog.
        """
        # Import the data integration module
        from data_integration import download_azure_training_data

        # Create a local folder to store the downloaded data
        root_folder = None
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'capture_storage_folder') and parent.capture_storage_folder:
                root_folder = parent.capture_storage_folder
                break
            parent = parent.parent()

        if not root_folder:
            root_folder = os.path.expanduser("~/sakar_vision_ai")
            os.makedirs(root_folder, exist_ok=True)

        # Create an azure_data folder for the downloaded content
        azure_data_folder = os.path.join(root_folder, "azure_data")
        os.makedirs(azure_data_folder, exist_ok=True)

        # Create and show the DOWNLOAD progress dialog (not upload)
        progress_dialog = AzureDownloadProgressDialog(self)
        progress_dialog.show()
        QApplication.processEvents()

        # Shared cancellation flag that both threads can access
        import threading
        self.download_cancel_event = threading.Event()
        self.download_thread_ref = None

        def cancel_check_func():
            """Function to check if download should be cancelled"""
            return self.download_cancel_event.is_set() or progress_dialog.cancelled

        def download_thread_func():
            try:
                print("Starting download thread...")

                def progress_callback(msg, current_file=None, total_files=None):
                    if cancel_check_func():
                        return False

                    if current_file is not None and total_files is not None:
                        progress_percent = int((current_file / total_files) * 100)
                        QTimer.singleShot(0, lambda: progress_dialog.updateStatus(
                            f"{msg} ({current_file}/{total_files})", progress_percent))
                        QTimer.singleShot(0, lambda: progress_dialog.updateFileProgress(
                            current_file, total_files))
                    else:
                        QTimer.singleShot(0, lambda: progress_dialog.updateStatus(msg, None))

                    return True

                # Call the download function with cancel check
                result = download_azure_training_data(
                    azure_data_folder,
                    cancel_check_func=cancel_check_func,
                    progress_callback=progress_callback
                )

                print(f"Download thread completed. Result: {result}")

                # Check if cancelled
                if cancel_check_func():
                    print("Download was cancelled")
                    QTimer.singleShot(0, lambda: progress_dialog.setError(
                        "Download was cancelled"))
                    QTimer.singleShot(1000, lambda: progress_dialog.accept()
                                      )  # Auto-close after 1 second
                    return

                # Success case
                if result:
                    print("Download completed successfully")
                    QTimer.singleShot(0, lambda: progress_dialog.setComplete(result))
                    QTimer.singleShot(2000, lambda: progress_dialog.accept()
                                      )  # Auto-close after 2 seconds
                else:
                    print("Download failed")
                    QTimer.singleShot(0, lambda: progress_dialog.setError("Download failed"))

            except Exception as e:
                print(f"Exception in download thread: {e}")
                error_msg = str(e)
                QTimer.singleShot(0, lambda: progress_dialog.setError(error_msg))

        def handle_cancel():
            """Handle cancellation with immediate force"""
            print("Cancel button clicked - initiating cancellation")

            # Set the cancellation event
            self.download_cancel_event.set()
            progress_dialog.cancelled = True

            # Update UI immediately
            progress_dialog.statusLabel.setText("Cancelling download...")
            progress_dialog.statusLabel.setStyleSheet("color: #ff6600;")
            progress_dialog.cancelButton.setEnabled(False)
            progress_dialog.cancelButton.setText("Cancelling...")
            QApplication.processEvents()

            # Force close after 3 seconds if thread doesn't respond
            def force_close():
                print("Force closing dialog due to timeout")
                if not progress_dialog.isComplete:
                    progress_dialog.setError("Download was cancelled")
                    progress_dialog.accept()

            QTimer.singleShot(3000, force_close)

        try:
            # Start the download thread
            download_thread = threading.Thread(target=download_thread_func)
            download_thread.daemon = True
            self.download_thread_ref = download_thread
            download_thread.start()

            # Connect cancel button
            progress_dialog.cancelButton.clicked.disconnect() if progress_dialog.cancelButton.receivers(
                progress_dialog.cancelButton.clicked) > 0 else None
            progress_dialog.cancelButton.clicked.connect(handle_cancel)

            # Enhanced close event handling
            original_close_event = progress_dialog.closeEvent

            def enhanced_close_event(event):
                if not progress_dialog.isComplete and not progress_dialog.cancelled:
                    print("Window close button clicked - treating as cancel")
                    handle_cancel()
                    event.ignore()  # Don't close yet
                else:
                    original_close_event(event)

            progress_dialog.closeEvent = enhanced_close_event

            # Show dialog and wait for completion
            result = progress_dialog.exec_()

            # Ensure cancellation is set when dialog closes
            if not self.download_cancel_event.is_set():
                print("Dialog closed - setting cancellation flag")
                self.download_cancel_event.set()

            print(f"Download dialog completed with result: {result}")

        except Exception as e:
            print(f"Error in downloadAzureData: {e}")
            QMessageBox.critical(
                self,
                "Download Error",
                f"Failed to start download: {str(e)}"
            )

    def combineDatasets(self):
        """
        Combines datasets from user-selected folders, merges class names from data.yaml, removes duplicates,
        normalizes semantically similar classes, and creates a unified data.yaml for the final dataset.
        """
        import yaml
        from collections import defaultdict
        from data_integration import combine_local_and_azure_data, initialize_combined_data

        def extract_classes_from_yaml(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get('names', [])
            except Exception as e:
                print(f"Failed to read {yaml_path}: {e}")
                return []

        def normalize_label(label):
            label = label.strip().lower().replace(" ", "_")
            if label in ["casting_with_burr", "polished_casting", "unpolished_casting"]:
                return "casting"
            if label == "stain":
                return "spot"
            if label == "rust":
                return "corrosion"
            if "scratch" in label:
                return "scratch"
            elif "burr" in label:
                return "burr"
            elif "dent" in label:
                return "dent"
            elif "patch" in label:
                return "patch"
            elif "pits" in label:
                return "pits"
            elif "hole" in label:
                return "hole"
            elif "spot" in label:
                return "spot"
            elif "punch" in label:
                return "punch"
            elif "cut" in label:
                return "cut"
            elif "damage" in label:
                return "damage"
            elif "crack" in label:
                return "crack"
            else:
                return label

        folder_dialog = FolderSelectionDialog(self)
        if folder_dialog.exec_() != QDialog.Accepted:
            return

        folders = folder_dialog.get_selected_folders()
        azure_folder = folders['azure_folder']
        manual_folder = folders['manual_folder']
        auto_folder = folders['auto_folder']
        output_folder = folders['output_folder']

        for folder_name, folder_path in folders.items():
            if not os.path.exists(folder_path):
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Selected {folder_name.replace('_', ' ')} does not exist:\n{folder_path}"
                )
                return

        os.makedirs(output_folder, exist_ok=True)
        self.combined_data_path = output_folder

        yaml_paths = [
            os.path.join(manual_folder, "data.yaml"),
            os.path.join(auto_folder, "data.yaml"),
            os.path.join(azure_folder, "data.yaml")
        ]

        original_to_normalized = defaultdict(set)

        for path in yaml_paths:
            classes = extract_classes_from_yaml(path)
            for c in classes:
                norm = normalize_label(c)
                original_to_normalized[norm].add(c)

        selected_classes = sorted(original_to_normalized.keys())

        final_yaml_path = os.path.join(output_folder, "data.yaml")
        if os.path.exists(final_yaml_path):
            reply = QMessageBox.warning(
                self,
                "Overwrite Warning",
                f"A 'data.yaml' file already exists at the output location:\n{final_yaml_path}\n\n"
                "Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        try:
            initialize_combined_data(selected_classes)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while initializing combined dataset: {str(e)}"
            )
            return

        progress_dialog = CombineDatasetProgressDialog(self)
        progress_dialog.show()
        QApplication.processEvents()

        self.combine_progress_dialog = progress_dialog
        self.combine_completed = False
        self.combine_auto_close_timer = None

        def force_close_dialog():
            try:
                if hasattr(self, 'combine_progress_dialog') and self.combine_progress_dialog:
                    print("Force closing combine dialog")
                    if not self.combine_progress_dialog.isComplete:
                        self.combine_progress_dialog.isComplete = True
                    self.combine_progress_dialog.accept()
                    self.combine_progress_dialog = None
            except Exception as e:
                print(f"Error force closing dialog: {e}")

        def schedule_auto_close(delay_ms=2000):
            try:
                print(f"Scheduling auto-close in {delay_ms}ms")
                if hasattr(self, 'combine_auto_close_timer') and self.combine_auto_close_timer:
                    self.combine_auto_close_timer.stop()

                self.combine_auto_close_timer = QTimer()
                self.combine_auto_close_timer.setSingleShot(True)
                self.combine_auto_close_timer.timeout.connect(force_close_dialog)
                self.combine_auto_close_timer.start(delay_ms)
            except Exception as e:
                print(f"Error scheduling auto-close: {e}")
                QTimer.singleShot(100, force_close_dialog)

        def combine_thread_func():
            try:
                print(f"Starting dataset combination...")

                success, stats = combine_local_and_azure_data(
                    manual_folder,
                    auto_folder,
                    azure_folder,
                    output_folder,
                    selected_classes,
                    update_config=False
                )

                print(f"Combination completed. Success: {success}")
                if success:
                    self.combine_success = True
                    self.combine_stats = stats
                    QTimer.singleShot(0, handle_combine_completion)
                else:
                    self.combine_error_msg = "Failed to combine datasets. Check the logs for details."
                    QTimer.singleShot(0, handle_combine_error)

            except Exception as e:
                print(f"Exception in combine thread: {e}")
                self.combine_error_msg = str(e)
                QTimer.singleShot(0, handle_combine_error)

        def handle_combine_completion():
            try:
                print("=== Handling combine completion in main thread ===")

                if self.combine_completed:
                    print("Already handled completion, skipping")
                    return

                self.combine_completed = True

                self.combine_progress_dialog.setComplete(self.combine_stats)

                if hasattr(self, 'combined_data_path') and os.path.exists(self.combined_data_path):
                    self.split_field.line_edit.setText(self.combined_data_path)

                try:
                    from data_integration import update_defects_config
                    update_defects_config(selected_classes, selected_classes)
                    print("Updated defects config")
                except Exception as e:
                    print(f"Error updating defects config: {e}")

                final_yaml_path = os.path.join(output_folder, "data.yaml")
                with open(final_yaml_path, 'w') as f:
                    yaml.dump({"names": selected_classes, "nc": len(selected_classes)}, f)
                print("Final data.yaml generated")

                schedule_auto_close(2000)

                total_images = sum(data.get('images', 0) for data in self.combine_stats.values())
                total_labels = sum(data.get('labels', 0) for data in self.combine_stats.values())

                def show_success_message():
                    try:
                        QMessageBox.information(
                            self,
                            "Dataset Combination Complete",
                            f"Successfully combined datasets!\n\n"
                            f"Total: {total_images} images, {total_labels} labels\n"
                            f"Output saved to: {self.combined_data_path}\n\n"
                            f"Details:\n" +
                            "\n".join([f"• {source}: {data.get('images', 0)} images, {data.get('labels', 0)} labels"
                                       for source, data in self.combine_stats.items()])
                        )
                    except Exception as e:
                        print(f"Error showing success message: {e}")

                QTimer.singleShot(2500, show_success_message)

            except Exception as e:
                print(f"Error in handle_combine_completion: {e}")
                schedule_auto_close(1000)

        def handle_combine_error():
            try:
                print(f"=== Handling combine error in main thread: {self.combine_error_msg} ===")

                if self.combine_completed:
                    print("Already handled completion, skipping error")
                    return

                self.combine_completed = True
                self.combine_progress_dialog.setError(self.combine_error_msg)
                schedule_auto_close(3000)

                def show_error_message():
                    try:
                        QMessageBox.critical(
                            self,
                            "Combination Failed",
                            f"Failed to combine datasets:\n{self.combine_error_msg}"
                        )
                    except Exception as e:
                        print(f"Error showing error message: {e}")

                QTimer.singleShot(3500, show_error_message)

            except Exception as e:
                print(f"Error in handle_combine_error: {e}")
                schedule_auto_close(1000)

        try:
            combine_thread = threading.Thread(target=combine_thread_func)
            combine_thread.daemon = True
            combine_thread.start()
            self.combine_thread = combine_thread

            backup_timer = QTimer()
            backup_timer.setSingleShot(True)
            backup_timer.timeout.connect(lambda: (
                print("Backup timer triggered - force closing dialog"),
                force_close_dialog()
            ))
            backup_timer.start(30000)

            print("Starting dialog execution")
            result = progress_dialog.exec_()
            print(f"Dialog execution finished with result: {result}")

            if backup_timer.isActive():
                backup_timer.stop()

            if hasattr(self, 'combine_auto_close_timer') and self.combine_auto_close_timer:
                self.combine_auto_close_timer.stop()

        except Exception as e:
            print(f"Error in combineDatasets: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while combining datasets: {str(e)}"
            )

    def uploadToAzure(self):
        """
        Uploads a selected dataset to Azure Storage with real progress tracking and cancellation.
        """
        # Import the data integration module
        from data_integration import upload_combined_dataset_to_azure

        # Show upload selection dialog
        upload_dialog = AzureUploadSelectionDialog(self)
        if upload_dialog.exec_() != QDialog.Accepted:
            return  # User cancelled

        # Get upload configuration
        config = upload_dialog.get_upload_config()
        local_folder = config['local_folder']
        azure_destination = config['azure_destination']
        overwrite = config['overwrite']
        backup = config['backup']

        print(f"Upload configuration:")
        print(f"  Local folder: {local_folder}")
        print(f"  Azure destination: {azure_destination}")
        print(f"  Overwrite: {overwrite}")
        print(f"  Backup: {backup}")

        # Validate that the local folder exists and has content
        if not os.path.exists(local_folder):
            QMessageBox.critical(
                self,
                "Error",
                f"Selected folder does not exist:\n{local_folder}"
            )
            return

        # Check for essential files and count total files
        essential_checks = []
        total_files_to_upload = 0

        # Check for data.yaml
        yaml_path = os.path.join(local_folder, "data.yaml")
        if os.path.exists(yaml_path):
            essential_checks.append("✓ data.yaml found")
            total_files_to_upload += 1
        else:
            essential_checks.append("⚠ data.yaml missing")

        # Check for split folders and count files
        splits_found = []
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(local_folder, split)
            if os.path.exists(split_path):
                splits_found.append(split)
                essential_checks.append(f"✓ {split} folder found")

                # Count images and labels
                images_dir = os.path.join(split_path, 'images')
                labels_dir = os.path.join(split_path, 'labels')

                if os.path.exists(images_dir):
                    img_count = len([f for f in os.listdir(images_dir)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    total_files_to_upload += img_count

                if os.path.exists(labels_dir):
                    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                    total_files_to_upload += label_count
            else:
                essential_checks.append(f"⚠ {split} folder missing")

        # Show confirmation dialog with file count
        if not splits_found:
            QMessageBox.critical(
                self,
                "Invalid Dataset Folder",
                f"The selected folder does not appear to contain a valid dataset.\n\n"
                f"Expected structure:\n"
                f"• data.yaml\n"
                f"• train/images/ and train/labels/\n"
                f"• valid/images/ and valid/labels/\n"
                f"• test/images/ and test/labels/\n\n"
                f"Found:\n" + "\n".join(essential_checks)
            )
            return

        # Count files for confirmation
        total_images = 0
        total_labels = 0
        for split in splits_found:
            images_dir = os.path.join(local_folder, split, 'images')
            labels_dir = os.path.join(local_folder, split, 'labels')

            if os.path.exists(images_dir):
                total_images += len([f for f in os.listdir(images_dir)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            if os.path.exists(labels_dir):
                total_labels += len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        # Final confirmation with file count
        confirm_msg = (
            f"Ready to upload dataset to Azure!\n\n"
            f"📁 Local folder: {local_folder}\n"
            f"☁️ Azure destination: {azure_destination}\n\n"
            f"📊 Dataset summary:\n"
            f"• Splits: {', '.join(splits_found)}\n"
            f"• Images: {total_images}\n"
            f"• Labels: {total_labels}\n"
            f"• YAML: {'Yes' if os.path.exists(yaml_path) else 'No'}\n"
            f"• Total files to upload: {total_files_to_upload}\n\n"
            f"⚙️ Options:\n"
            f"• Overwrite existing files: {'Yes' if overwrite else 'No'}\n"
            f"• Create backup: {'Yes' if backup else 'No'}\n\n"
            f"This upload may take some time and consume bandwidth.\n"
            f"Continue with upload?"
        )

        confirm = QMessageBox.question(
            self,
            "Confirm Upload to Azure",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        # Create and show the progress dialog
        progress_dialog = AzureUploadProgressDialog(self)
        progress_dialog.show()
        QApplication.processEvents()

        # Shared cancellation flag using threading.Event for thread safety
        import threading
        self.upload_cancel_event = threading.Event()
        self.upload_cancelled = False

        def cancel_check_func():
            """Function to check if upload should be cancelled"""
            return self.upload_cancel_event.is_set() or progress_dialog.cancelled

        def upload_thread_func():
            try:
                print(f"Starting upload to Azure destination: {azure_destination}")
                print(f"Total files to upload: {total_files_to_upload}")

                # Enhanced progress callback with real progress tracking
                uploaded_files = 0
                last_progress_update = 0

                def progress_callback(msg, current_file=None, total_files=None):
                    nonlocal uploaded_files, last_progress_update

                    # Check for cancellation
                    if cancel_check_func():
                        print(f"Upload cancelled during progress callback")
                        return False  # Signal to stop

                    # Calculate progress percentage
                    if current_file is not None and total_files is not None:
                        uploaded_files = current_file
                        progress_percent = int((current_file / total_files) * 100)

                        # Only update UI if progress has changed significantly (reduces UI update overhead)
                        if progress_percent != last_progress_update:
                            last_progress_update = progress_percent
                            
                            print(f"Upload progress: {current_file}/{total_files} ({progress_percent}%)")
                            
                            # Update progress in main thread - use direct method call instead of lambda
                            def update_ui():
                                try:
                                    if not progress_dialog.cancelled:
                                        progress_dialog.updateStatus(
                                            f"{msg} ({current_file}/{total_files})",
                                            progress_percent
                                        )
                                        progress_dialog.updateFileProgress(current_file, total_files)
                                except Exception as e:
                                    print(f"Error updating UI: {e}")
                            
                            # Use QTimer.singleShot for thread-safe UI updates
                            QTimer.singleShot(0, update_ui)
                    else:
                        # Indeterminate progress
                        def update_status():
                            try:
                                if not progress_dialog.cancelled:
                                    progress_dialog.updateStatus(msg, None)
                            except Exception as e:
                                print(f"Error updating status: {e}")
                        
                        QTimer.singleShot(0, update_status)

                    return True  # Continue

                # Test progress bar update
                QTimer.singleShot(0, lambda: progress_dialog.setProgress(5))
                
                success, stats = upload_combined_dataset_to_azure(
                    local_folder,
                    azure_folder=azure_destination,
                    progress_callback=progress_callback
                )

                # Check if cancelled during upload
                if cancel_check_func():
                    print("Upload was cancelled")
                    def set_cancelled():
                        progress_dialog.setError("Upload was cancelled by user")
                    QTimer.singleShot(0, set_cancelled)
                    return

                if success:
                    # Show success result with statistics
                    print(f"Upload completed successfully to {azure_destination}")
                    def set_complete():
                        progress_dialog.setComplete(stats)
                    QTimer.singleShot(0, set_complete)
                    
                    # Auto-close after 3 seconds
                    QTimer.singleShot(3000, lambda: progress_dialog.accept())
                else:
                    # Show error if upload failed
                    def set_error():
                        progress_dialog.setError("Upload failed. Check logs for details.")
                    QTimer.singleShot(0, set_error)

            except Exception as e:
                # Handle any exceptions during upload
                error_msg = f"Error during upload: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                
                def set_error():
                    progress_dialog.setError(error_msg)
                QTimer.singleShot(0, set_error)

        def handle_cancel():
            """Handle cancellation with immediate response"""
            if not self.upload_cancelled:
                print("User clicked cancel - initiating upload cancellation")
                self.upload_cancelled = True
                self.upload_cancel_event.set()
                progress_dialog.cancelled = True

                # Update UI immediately
                progress_dialog.statusLabel.setText("Cancelling upload...")
                progress_dialog.statusLabel.setStyleSheet("color: #ffc107;")
                progress_dialog.cancelButton.setEnabled(False)
                progress_dialog.cancelButton.setText("Cancelling...")
                progress_dialog.repaint()
                QApplication.processEvents()

                # Force close after 5 seconds if upload doesn't respond
                def force_close():
                    print("Force closing upload dialog due to timeout")
                    if not progress_dialog.isComplete:
                        progress_dialog.setError("Upload was cancelled")
                        progress_dialog.accept()

                QTimer.singleShot(5000, force_close)

        try:
            # Create and start the upload thread
            upload_thread = threading.Thread(target=upload_thread_func)
            upload_thread.daemon = True
            upload_thread.start()

            # Store thread reference
            self.upload_thread = upload_thread

            # Connect cancel functionality with debouncing
            progress_dialog.cancelButton.clicked.disconnect() if progress_dialog.cancelButton.receivers(
                progress_dialog.cancelButton.clicked) > 0 else None
            progress_dialog.cancelButton.clicked.connect(handle_cancel)

            # Enhanced close event handling
            original_close_event = progress_dialog.closeEvent

            def enhanced_close_event(event):
                if not progress_dialog.isComplete and not progress_dialog.cancelled:
                    print("Window close button clicked - treating as cancel")
                    handle_cancel()
                    event.ignore()
                else:
                    original_close_event(event)

            progress_dialog.closeEvent = enhanced_close_event

            # Show dialog and wait for completion
            progress_dialog.exec_()

            # Ensure cancellation is set when dialog closes
            if not self.upload_cancel_event.is_set():
                print("Dialog closed - setting upload cancellation flag")
                self.upload_cancel_event.set()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Upload Error",
                f"Failed to start upload: {str(e)}"
            )

def trainFinalModel(self):
    """
    Trains a model using user-selected dataset - works independently.
    """
    # Open the training dialog
    dialog = TrainDialog(self, is_retrain=False)
    dialog.setWindowTitle("Train Final Model")

    # Pre-fill model path
    default_model_path = "/home/sakar02/sakar-vision-ui/y11n.pt"
    if os.path.exists(default_model_path):
        dialog.model_path_input.setText(default_model_path)

    # Always prompt user to select YAML file manually
    yaml_path, _ = QFileDialog.getOpenFileName(
        self,
        "Select Data YAML File for Final Training",
        "",
        "YAML Files (*.yaml)"
    )
    if yaml_path and os.path.exists(yaml_path):
        dialog.yaml_path_input.setText(yaml_path)
    else:
        QMessageBox.warning(
            self,
            "YAML File Required",
            "Please select a data.yaml file to proceed with training."
        )
        return

    # Show the dialog and run training if accepted
    if dialog.exec_() == QDialog.Accepted:
        params = dialog.get_params()

        # Set a specific model name for the final model
        original_model_path = params['model_path']
        base_name = os.path.basename(original_model_path)
        model_name = f"sakar_vision_final_{base_name}"
        params['name'] = model_name

        # Run the training
        self.runTraining(params)

        # Show a message about where the final model will be saved
        QMessageBox.information(
            self,
            "Final Model Training Started",
            f"Training for the final model has started.\n\nThe trained model will be saved as '{model_name}' in the runs/detect/train folder.\n\nYou can deploy this model using the 'Deploy Model' button when training completes."
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application style for modern look
    app.setStyle('Fusion')

    # Set modern color palette
    palette = app.palette()
    palette.setColor(palette.Window, QColor(248, 249, 250))
    palette.setColor(palette.WindowText, QColor(32, 33, 36))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    palette.setColor(palette.AlternateBase, QColor(241, 243, 244))
    palette.setColor(palette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ToolTipText, QColor(32, 33, 36))
    palette.setColor(palette.Text, QColor(32, 33, 36))
    palette.setColor(palette.Button, QColor(248, 249, 250))
    palette.setColor(palette.ButtonText, QColor(32, 33, 36))
    palette.setColor(palette.BrightText, QColor(255, 255, 255))
    palette.setColor(palette.Link, QColor(66, 133, 244))
    palette.setColor(palette.Highlight, QColor(232, 240, 254))
    palette.setColor(palette.HighlightedText, QColor(32, 33, 36))
    app.setPalette(palette)

    # Create the main stacked widget
    stacked_widget = QStackedWidget()

    # Store stacked_widget as an attribute so it can be found by deployModel
    stacked_widget.stacked_widget = stacked_widget

    # Create AutoAnnotateUI and DeploymentUI
    auto_annotate_ui = AutoAnnotateUI()
    deployment_ui = DeploymentUI()

    # Add both UIs to the stacked widget
    stacked_widget.addWidget(auto_annotate_ui)
    stacked_widget.addWidget(deployment_ui)

    # Set the stacked widget as the parent for both UIs
    auto_annotate_ui.setParent(stacked_widget)
    deployment_ui.setParent(stacked_widget)

    # Show the AutoAnnotateUI initially
    stacked_widget.setCurrentWidget(auto_annotate_ui)

    # Show the stacked widget
    stacked_widget.setWindowTitle("Sakar Vision AI - Modern Interface")
    stacked_widget.setGeometry(100, 100, 1400, 900)
    stacked_widget.show()

    sys.exit(app.exec_())
