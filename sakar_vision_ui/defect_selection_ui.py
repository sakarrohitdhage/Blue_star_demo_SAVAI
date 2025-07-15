#!/usr/bin/env python3
"""
SAKAR VISION AI - Defect Selection UI Module

OVERVIEW:
This module implements the defect selection interface for the Sakar Vision AI platform, serving as a critical 
intermediary step between user authentication and the main application workflow. It provides an intuitive, 
card-based interface for users to select specific defect types that the AI system should detect during inspection 
processes, supporting both fabric and metal sheet inspection workflows with dynamic UI adaptation based on the 
inspection type determined during login.

KEY FUNCTIONALITY:
The system features a responsive grid-based defect selection interface with custom-styled checkboxes that provide 
visual feedback and card-like interactions, comprehensive user session management that maintains consistency across 
the application lifecycle, and seamless integration with Azure MySQL database for persistent defect configuration 
storage. It includes intelligent defect configuration management with automatic merging of existing and newly selected 
defects, robust navigation capabilities that support multiple UI views (camera feed, demo feed, deployment interfaces) 
through a QStackedWidget architecture, and comprehensive error handling with user-friendly messaging for database 
operations and UI transitions.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with custom widget styling and a sophisticated event-driven architecture, the module employs 
a QStackedWidget-based navigation system for seamless transitions between different application modules, comprehensive 
Azure database integration with transaction management, retry logic, and proper connection handling following Azure 
best practices. The architecture features modular UI components (CustomTitleBar, DefectCheckBox, DefectGroupBox) 
with consistent styling and shadow effects, intelligent user session state management across multiple JSON configuration 
files, and dynamic inspection type handling that adapts the interface and backend connections based on fabric vs. 
metal sheet inspection requirements. The system also includes real-time internet connectivity monitoring and status 
indicators for enhanced user experience.
"""

from utils import set_window_icon
from functools import partial
import json
import os
import sys
import logging
# Fix the logging import to use the correct module
from utils.logging import get_logger
from utils.logging.ui_event_logger import UIEventTracker
from utils.logging import get_ui_logger, track_ui_method

from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QLinearGradient
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'
# Database function import for defects has been removed

# Path to save selected defects
DEFECTS_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "defects_config.json")

# Config paths for different inspection types
FABRIC_LOCATION_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "location_config.json")
METAL_SHEET_LOCATION_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "metal_sheet_location_config.json")


# Create a logger instance
logger = get_logger("defect_selection_ui")
# Create a UI event tracker for tracking interactions
ui_tracker = UIEventTracker()

# Initialize logger for defect selection UI and UI event tracker
ui_logger = get_ui_logger("defect_selection")


class CustomTitleBar(QFrame):
    def __init__(self, parent=None, demo_button=None):
        super().__init__(parent)
        self.setFixedHeight(70)
        self.setStyleSheet("""
            background-color: white; 
            border: none;
        """)

        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        # Logo image label setup
        self.logo_label = QLabel()
        logo_path = "logo.jpeg"
        logo_pixmap = QPixmap(logo_path)
        logo_pixmap = logo_pixmap.scaledToHeight(45, Qt.SmoothTransformation)
        self.logo_label.setPixmap(logo_pixmap)

        # Title label (adjustable positioning)
        self.title_label = QLabel("SHEET METAL AI - DEFECT SELECTION")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Adjust these margin values to move the text left or right
        left_margin = 0     # Pixels to adjust from left
        right_margin = 195  # Pixels to adjust from right

        self.title_label.setStyleSheet(f"""
            color: white;
            font-weight: bold;
            font-size: 28px;
            margin-left: {left_margin}px;
            margin-right: {right_margin}px;
        """)

        # Add logo and TM symbol to layout
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch(0)
        logo_layout.setSpacing(2)

        # Add widgets to main layout
        layout.addLayout(logo_layout)
        layout.addWidget(self.title_label, 1)  # 1 is the stretch factor

        # Add system status indicator to the right side of title bar (same as camera feed UI)
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

        # Create right section for status indicators (matching camera feed UI layout)
        right_section = QHBoxLayout()
        right_section.setSpacing(10)
        right_section.addWidget(self.status_indicator)
        right_section.addWidget(self.status_label)

        layout.addLayout(right_section)

        # Add any additional button if provided
        if demo_button:
            layout.addWidget(demo_button)

        self.setLayout(layout)

        # Timer to check internet connectivity (same as camera feed UI)
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
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.update_status(True)
            return True
        except OSError:
            try:
                # Fallback: Try HTTP request to Google
                import requests
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


class DefectCheckBox(QCheckBox):
    """Custom styled checkbox for defect selection."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                font-size: 14pt;
                color: #333333;
                padding: 15px 20px;
                background-color: white;
                border: 1px solid #E0E3E7;
                border-radius: 8px;
                min-width: 200px;
                min-height: 60px;
                spacing: 12px;
            }
            QCheckBox:hover {
                border: 1px solid #FF914D;
                background-color: #F5F9FF;
            }
            QCheckBox:checked {
                border: 2px solid #FF914D;
                background-color: #F5F9FF;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
                border: 2px solid #919EAB;
                border-radius: 4px;
                background-color: white;
            }
            QCheckBox::indicator:hover {
                border-color: #FF914D;
            }
            QCheckBox::indicator:checked {
                background-color: #FF914D;
                border-color: #FF914D;
                image: url('checkmark.png');
            }
        """)

        # Set cursor using Qt's setCursor method instead of stylesheet
        self.setCursor(Qt.PointingHandCursor)

        # Add shadow effect for card-like appearance
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 15))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def mousePressEvent(self, event):
        """Override mouse press event to toggle checkbox state when clicking anywhere on the card"""
        if event.button() == Qt.LeftButton:
            self.setChecked(not self.isChecked())
        super().mousePressEvent(event)


class DefectGroupBox(QGroupBox):
    """Group box for categorizing defects."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14pt;
                font-weight: bold;
                color: #333333;
                border: 1px solid #dddddd;
                border-radius: 8px;
                margin-top: 20px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background-color: white;
            }
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

        # Set layout
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(15, 25, 15, 15)


class NewDefectButton(QPushButton):
    """Custom styled button for adding new defects."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)


class CustomButton(QPushButton):
    """Custom styled button."""

    def __init__(self, text, parent=None, is_primary=True):
        super().__init__(text, parent)
        self.is_primary = is_primary
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)

        if is_primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #ff914d;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14pt;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff7730;
                }
                QPushButton:pressed {
                    background-color: #e86d25;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    color: #333333;
                    border: 1px solid #d0d0d0;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14pt;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QPushButton:pressed {
                    background-color: #c0c0c0;
                }
            """)


class DefectSelectionUI(QWidget):
    """
    Interface for selecting defect types before proceeding to the main application.
    Users can select multiple defects from categorized lists.
    """

    def __init__(self, user_info=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAKAR VISION AI - Defect Selection")
        set_window_icon(self)
        self.setMinimumSize(900, 700)

        # Store user info from login
        self.user_info = user_info or {"username": "User", "full_name": "Default User"}

        # Store user ID in session state for consistency across the application
        self.store_user_session_state()

        # Determine inspection type from user_info
        self.inspection_type = "fabric"  # Default
        if isinstance(user_info, dict) and "inspection_type" in user_info:
            self.inspection_type = user_info["inspection_type"]

        # Set appropriate configuration path based on inspection type
        if self.inspection_type == "metal_sheet":
            self.location_config_path = METAL_SHEET_LOCATION_CONFIG_PATH
            self.title_bar_text = "SHEET METAL AI - DEFECT SELECTION"
        else:  # Default to fabric
            self.location_config_path = FABRIC_LOCATION_CONFIG_PATH
            self.title_bar_text = "FABRIC INSPECTION AI - DEFECT SELECTION"

        # Dict to store checkboxes by their text/defect name
        self.defect_checkboxes = {}

        # Create stacked widget for tab-like navigation
        self.stacked_widget = QStackedWidget(self)

        # Create main defect selection page
        self.defect_selection_page = QWidget()

        # Camera feed UI will be initialized later
        self.camera_feed_ui = None

        # Initialize UI
        self.init_ui()

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QScrollArea {
                border: none;
                background-color: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            DefectSelectionUI {
                background-color: #F5F8FA;
            }
            QWidget#whiteContainer {
                background-color: white;
                border-radius: 12px;
                margin: 20px 40px;
                padding: 30px;
            }
        """)

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout for the entire application
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Set the background color for the main window
        self.setStyleSheet("""
            DefectSelectionUI {
                background-color: #F5F8FA;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QScrollArea {
                border: none;
                background-color: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
        """)

        # Replace HeaderPanel with CustomTitleBar
        self.title_bar = CustomTitleBar(self)
        # Set the title based on inspection type
        if hasattr(self.title_bar, 'title_label'):
            self.title_bar.title_label.setText(self.title_bar_text)
        main_layout.addWidget(self.title_bar)

        # Add stacked widget to main layout
        main_layout.addWidget(self.stacked_widget)

        # Setup defect selection page
        self.setup_defect_selection_page()

        # Add the defect selection page to the stacked widget
        self.stacked_widget.addWidget(self.defect_selection_page)

        # Initially show the defect selection page
        self.stacked_widget.setCurrentWidget(self.defect_selection_page)

    def setup_defect_selection_page(self):
        """Set up the defect selection page."""
        # Create a white background container widget with shadow
        container = QWidget()
        container.setObjectName("whiteContainer")

        # Add shadow effect to the container
        shadow = QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 2)
        container.setGraphicsEffect(shadow)

        # Content layout for defect selection page with proper padding
        main_layout = QVBoxLayout(self.defect_selection_page)
        main_layout.setContentsMargins(40, 20, 40, 40)  # Padding from screen edges
        main_layout.setSpacing(0)

        # Container layout
        content_layout = QVBoxLayout(container)
        content_layout.setContentsMargins(30, 30, 30, 30)  # Padding inside white container
        content_layout.setSpacing(20)

        # Add welcome message with user's name
        welcome_label = QLabel(f"Welcome, Integrator")
        welcome_label.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            color: #333333;
        """)

        # Add instruction
        instruction_label = QLabel(
            "Choose the types of defects you want the AI system to detect.")
        instruction_label.setStyleSheet("""
            font-size: 14pt;
            color: #555555;
            margin-bottom: 20px;
        """)

        content_layout.addWidget(welcome_label)
        content_layout.addWidget(instruction_label)

        # Create scroll area for defect categories
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
        """)

        # Create widget to hold all defect groups
        defect_container = QWidget()
        defect_layout = QVBoxLayout(defect_container)
        defect_layout.setSpacing(20)
        defect_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins inside scroll area

        # Add defect categories and checkboxes
        self.add_defect_categories(defect_layout)

        scroll_area.setWidget(defect_container)
        content_layout.addWidget(scroll_area)

        # Button layout with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        # Select All and Deselect All buttons with updated styling
        select_all_btn = QPushButton("Select All")
        select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #00C853;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00B04A;
            }
        """)
        select_all_btn.setCursor(Qt.PointingHandCursor)
        select_all_btn.clicked.connect(self.select_all_defects)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5252;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF1744;
            }
        """)
        deselect_all_btn.setCursor(Qt.PointingHandCursor)
        deselect_all_btn.clicked.connect(self.deselect_all_defects)

        # Add buttons to layout
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)

        # First, add stretches before and after the button to center it
        button_layout.addStretch(1)  # Add stretch before the button
        self.defective_check_btn = QPushButton("Defective Check")
        self.defective_check_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 24px;
                font-size: 18pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.defective_check_btn.setCursor(Qt.PointingHandCursor)
        self.defective_check_btn.clicked.connect(self.goto_defective_check)
        # Set a fixed width to make the button more centered (optional)
        self.defective_check_btn.setFixedWidth(300)
        button_layout.addWidget(self.defective_check_btn)
        button_layout.addStretch(1)  # Add stretch after the button

        # Proceed and Cancel buttons
        self.cancel_btn = CustomButton("Cancel", is_primary=False)
        self.proceed_btn = CustomButton("Proceed", is_primary=True)

        self.cancel_btn.clicked.connect(self.close)
        self.proceed_btn.clicked.connect(self.proceed_to_main_app)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.proceed_btn)

        # Add button layout to content layout
        content_layout.addLayout(button_layout)

        # Add the white container to the main layout
        main_layout.addWidget(container)

    def add_defect_categories(self, layout):
        """Add defect categories and checkboxes to the layout."""
        # Load defects from configuration file
        defects = [
            'Wrinkle', 'Pitting', 'Punch', 'Burr', 'Hole', 'Spot', 'Inclusion',
            'Corrosion', 'Damage', 'Crack', 'Dent', 'Patch', 'Scratch', 'Rolling',
            'Cut', 'Casting', 'Crazing'
        ]

        # Create content widget to hold the grid
        content_widget = QWidget()
        grid_layout = QGridLayout(content_widget)
        grid_layout.setSpacing(15)  # Increased spacing between cards
        grid_layout.setContentsMargins(20, 20, 20, 20)

        # Add checkboxes in a grid (4 columns)
        for i, defect in enumerate(defects):
            row, col = i // 4, i % 4
            checkbox = DefectCheckBox(defect)
            self.defect_checkboxes[defect] = checkbox
            grid_layout.addWidget(checkbox, row, col)

        # Add "Add New Defect" button in the next available grid position
        new_defect_btn = QPushButton("+ Add New Defect")
        new_defect_btn.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
                color: #4DBBFF;
                padding: 15px 20px;
                background-color: #F5F9FF;
                border: 2px dashed #4DBBFF;
                border-radius: 8px;
                min-width: 200px;
                min-height: 60px;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
                border-color: #1565C0;
            }
        """)
        new_defect_btn.setCursor(Qt.PointingHandCursor)
        new_defect_btn.clicked.connect(self.show_camera_feed)

        # Calculate next available grid position
        next_row = (len(defects) // 4) + (1 if len(defects) % 4 else 0)
        next_col = 0
        grid_layout.addWidget(new_defect_btn, next_row, next_col)

        # Create scroll area and add the content widget
        scroll_area = QScrollArea()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # Add scroll area to main layout
        layout.addWidget(scroll_area)

    def select_all_defects(self):
        """Select all defect checkboxes."""
        for checkbox in self.defect_checkboxes.values():
            checkbox.setChecked(True)

    def deselect_all_defects(self):
        """Deselect all defect checkboxes."""
        for checkbox in self.defect_checkboxes.values():
            checkbox.setChecked(False)

    def get_selected_defects(self):
        """Get list of selected defects."""
        return [defect for defect, checkbox in self.defect_checkboxes.items()
                if checkbox.isChecked()]

    def save_defect_configuration(self):
        """Save the selected defects to a configuration file."""
        selected_defects = self.get_selected_defects()

        config = {
            "selected_defects": selected_defects,
            "user": self.user_info.get("username", "unknown"),
            "timestamp": import_datetime().now().isoformat()
        }

        try:
            with open(DEFECTS_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Defect configuration saved to {DEFECTS_CONFIG_PATH}")
            return True
        except Exception as e:
            print(f"Error saving defect configuration: {e}")
            return False

    def proceed_to_main_app(self):
        """Save selections and proceed to main application."""
        selected_defects = self.get_selected_defects()

        if not selected_defects:
            reply = QMessageBox.question(
                self,
                "No Defects Selected",
                "You haven't selected any defects. Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

        # Save the configuration to file
        self.save_defect_configuration()

        # Store defect selections in Azure MySQL database
        self.store_defects_in_database(selected_defects)

        try:
            print(f"Proceeding with inspection type: {self.inspection_type}")

            # Close any existing instance before creating a new one
            if hasattr(self, 'main_window') and self.main_window is not None:
                self.main_window.close()
                self.main_window.deleteLater()

            # Get user_id to pass to the deployment UI
            user_id = None
            if isinstance(self.user_info, dict) and "user_id" in self.user_info:
                user_id = self.user_info.get("user_id")

            # Choose appropriate deployment UI based on inspection type
            if self.inspection_type == "metal_sheet":
                # For metal sheet inspection, use the standard DeploymentUI
                from deploy_ui import DeploymentUI
                print("Loading DeploymentUI for metal sheet inspection")
                self.main_window = DeploymentUI()

                # Set the user ID in the deployment UI
                if user_id and hasattr(self.main_window, 'set_user_info'):
                    self.main_window.set_user_info(
                        {"id": user_id, "username": self.user_info.get("username")})
                    print(f"Set user ID in DeploymentUI to: {user_id}")
            else:
                # Default to fabric inspection with FabricDeploymentUI
                from fabric_deploy_ui import FabricDeploymentUI
                print("Loading FabricDeploymentUI for fabric inspection")
                self.main_window = FabricDeploymentUI(self.user_info)

            # Connect destroyed signal to show this window again if deployment screen is closed
            self.main_window.destroyed.connect(self.show)

            # Set maximized state first, then show
            self.main_window.showMaximized()

            # Hide this window - don't use any delays or timers
            self.hide()

        except Exception as e:
            # Reset button style in case of error
            if hasattr(self, 'original_style'):
                self.proceed_btn.setStyleSheet(self.original_style)

            QMessageBox.critical(
                self,
                "Error",
                f"Failed to launch main application: {e}"
            )
            print(f"Error launching main application: {e}")
            import traceback
            traceback.print_exc()

    def store_defects_in_database(self, selected_defects):
        """
        Store selected defects in the Azure MySQL database.

        Args:
            selected_defects (list): List of selected defect names

        Returns:
            bool: True if successful, False otherwise
        """
        # Import needed modules here to avoid circular imports
        from azure_database import get_connection, get_user_id_by_username, get_latest_location_id, update_defect_count
        from datetime import datetime
        import logging
        import traceback
        from mysql.connector.errors import ProgrammingError
        from PyQt5.QtWidgets import QMessageBox

        # Set up logging for defect storage operations
        logger = logging.getLogger('defect_storage')

        if not selected_defects:
            logger.info("No defects selected, skipping database storage")
            return True

        # Enhanced user ID retrieval with strict session consistency - PRIORITIZE CURRENT SESSION
        user_id = None
        username = None

        # FIRST PRIORITY: Get from user_info dictionary (current login session - most reliable)
        if isinstance(self.user_info, dict):
            # Try to get user_id directly if present (most reliable)
            if "user_id" in self.user_info and self.user_info["user_id"]:
                user_id = self.user_info["user_id"]
                username = self.user_info.get("username", "unknown")
                logger.info(f"Using user_id {user_id} directly from current session user_info")
            # Try to get id field as alternative
            elif "id" in self.user_info and self.user_info["id"]:
                user_id = self.user_info["id"]
                username = self.user_info.get("username", "unknown")
                logger.info(f"Using id {user_id} from current session user_info")
            # REMOVED: No longer fallback to username lookup from user_info to avoid SR025/SR027 confusion

        # SECOND PRIORITY: Try to extract from session state file (fallback for current session)
        if not user_id:
            try:
                session_state_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "session_state.json")
                if os.path.exists(session_state_path):
                    with open(session_state_path, 'r') as f:
                        session_data = json.load(f)
                        if "current_user_id" in session_data and session_data["current_user_id"]:
                            user_id = session_data["current_user_id"]
                            username = session_data.get("current_username", "session_user")
                            logger.info(f"Using user_id {user_id} from session state")
                        elif "user_id" in session_data and session_data["user_id"]:
                            user_id = session_data["user_id"]
                            username = session_data.get("username", "session_user")
                            logger.info(f"Using user_id {user_id} from session data")
            except Exception as session_err:
                logger.warning(f"Error reading session state: {str(session_err)}")

        # THIRD PRIORITY: Try location config file ONLY if it has a direct user_id (not username lookup)
        if not user_id and hasattr(self, 'location_config_path') and os.path.exists(self.location_config_path):
            try:
                with open(self.location_config_path, 'r') as f:
                    config = json.load(f)
                    # ONLY use user_id if directly specified in config - avoid username lookups
                    if "user_id" in config and config["user_id"]:
                        user_id = config["user_id"]
                        username = config.get("username", config.get("user", "config_user"))
                        logger.warning(
                            f"FALLBACK: Using user_id {user_id} directly from location config")
                    # REMOVED: No longer fallback to username lookup from config to avoid SR025/SR027 confusion
            except Exception as config_err:
                logger.warning(f"Error reading location config: {str(config_err)}")

        # ERROR: Cannot determine user ID - fail securely with better error message
        if not user_id:
            error_msg = f"Cannot store defects: Unable to determine user ID from current session. user_info: {self.user_info}"
            logger.error(error_msg)
            logger.error(
                "This typically happens when the login session has expired or user_info doesn't contain user_id/id fields")

            QMessageBox.warning(
                self,
                "Session Error",
                "Your login session appears to have expired or is incomplete. Please log out and log in again to ensure your data is saved correctly."
            )
            return False

        # Log the final user_id being used for transparency
        logger.info(
            f"Final user_id being used for defect storage: {user_id} (username: {username})")

        # Get the latest location_id for this user
        location_id = get_latest_location_id(user_id)
        logger.info(f"Retrieved location_id={location_id} for user_id={user_id}")

        # Azure Best Practice: Connection acquisition with retry logic
        connection = get_connection()
        if not connection:
            logger.error("Cannot store defects: No database connection")
            QMessageBox.warning(
                self,
                "Database Connection Error",
                "Could not connect to the database. Defect selections will not be saved."
            )
            return False

        cursor = None
        try:
            # Azure Best Practice: Verify connection is active
            if not connection.is_connected():
                logger.warning("Connection lost, attempting to reconnect")
                connection.reconnect(attempts=2, delay=1)
                if not connection.is_connected():
                    logger.error("Failed to reconnect to database")
                    return False

            cursor = connection.cursor()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Azure Best Practice: Verify table exists first
            cursor.execute("SHOW TABLES LIKE 'defects'")
            if cursor.fetchone() is None:
                logger.error("Defects table does not exist in the database")
                QMessageBox.warning(
                    self,
                    "Database Schema Error",
                    "The defects table does not exist. Defect selections will not be saved."
                )
                return False

            # Ensure no transaction is already in progress
            if connection.in_transaction:
                logger.warning(
                    "Transaction already in progress. Rolling back before starting a new one.")
                connection.rollback()

            # Start transaction for ACID compliance
            connection.start_transaction()

            # Process each selected defect
            for defect in selected_defects:
                # Check if this defect already exists for this user, DATE, AND LOCATION
                if location_id:
                    # Add location_id to uniqueness criteria
                    cursor.execute(
                        "SELECT structural_id, location_id FROM defects WHERE id = %s AND defect = %s AND location_id = %s AND DATE(timestamp) = DATE(%s)",
                        (user_id, defect, location_id, current_time)
                    )
                else:
                    cursor.execute(
                        "SELECT structural_id, location_id FROM defects WHERE id = %s AND defect = %s AND DATE(timestamp) = DATE(%s)",
                        (user_id, defect, current_time)
                    )

                existing_record = cursor.fetchone()

                if existing_record:
                    # If it already exists for today and this location, update any missing fields
                    structural_id = existing_record[0]
                    existing_location_id = existing_record[1] if len(existing_record) > 1 else None

                    logger.info(
                        f"Found existing record for user {user_id}, defect {defect}, same date and location - structural_id={structural_id}, location_id={existing_location_id}")

                    # If we have a location_id but the record doesn't, update it
                    if location_id and (existing_location_id is None or existing_location_id == 0):
                        update_query = """
                        UPDATE defects SET location_id = %s WHERE structural_id = %s
                        """
                        cursor.execute(update_query, (location_id, structural_id))
                        logger.info(f"Updated existing record with location_id={location_id}")
                else:
                    # No record exists for today with this location - create a new one
                    logger.info(
                        f"Creating new record for user {user_id}, defect {defect}, date {current_time}, location_id={location_id}")

                    # Include location_id in the insert if available
                    if location_id:
                        insert_query = """
                        INSERT INTO defects (id, defect, count, threshold, timestamp, location_id) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_query, (user_id, defect,
                                       0, 0, current_time, location_id))
                        logger.info(
                            f"Inserted defect {defect} with initial count 0 and location_id={location_id}")
                    else:
                        insert_query = """
                        INSERT INTO defects (id, defect, count, threshold, timestamp) 
                        VALUES (%s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_query, (user_id, defect, 0, 0, current_time))
                        logger.info(
                            f"Inserted defect {defect} with initial count 0 (no location_id available)")

            # Commit transaction
            connection.commit()
            logger.info(
                f"Successfully stored {len(selected_defects)} defects for user ID {user_id} with location_id={location_id}")

            # Call database initialization to ensure the dashboard's pre-populated defects are also initialized
            try:
                threshold = 50  # Default threshold
                for defect_name in selected_defects:
                    update_defect_count(
                        defect_name,
                        0,  # Initial count of 0
                        threshold,
                        is_initialization=True,  # Mark as initialization
                        user_id=user_id,
                        location_id=location_id
                    )
                    logger.info(
                        f"Initialized defect in database: {defect_name} for user {user_id} with location_id {location_id}")
            except Exception as init_error:
                logger.warning(f"Error during defect initialization: {str(init_error)}")
                # Continue even if initialization fails

            return True

        except Exception as e:
            # Azure Best Practice: Proper transaction error handling
            if connection and connection.is_connected():
                try:
                    connection.rollback()
                    logger.info("Transaction rolled back successfully")
                except Exception as rollback_err:
                    logger.error(f"Error during rollback: {str(rollback_err)}")

            error_type = type(e).__name__
            logger.error(f"Error storing defects in database ({error_type}): {str(e)}")
            logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")

            # Azure Best Practice: User-friendly error messages
            QMessageBox.warning(
                self,
                "Database Error",
                f"An error occurred while saving your defect selections: {error_type}. Please try again."
            )
            return False

        finally:
            # Azure Best Practice: Proper resource cleanup in reverse order of acquisition
            if cursor:
                try:
                    cursor.close()
                except Exception as cursor_err:
                    logger.warning(f"Error closing cursor: {str(cursor_err)}")

            if connection and hasattr(connection, 'is_connected') and connection.is_connected():
                try:
                    connection.close()
                    logger.debug("Database connection closed properly")
                except Exception as conn_err:
                    logger.warning(f"Error closing connection: {str(conn_err)}")

    def show_camera_feed(self):
        """Show the camera feed UI."""
        try:
            # Import necessary modules
            from camera_feed_cloud_ui import CameraFeedUI
            from PyQt5.QtCore import QTimer

            # Save currently selected defects before navigating away
            selected_defects = self.get_selected_defects()

            # Read any existing saved defects to merge with current selection
            existing_defects = []
            if os.path.exists(DEFECTS_CONFIG_PATH):
                try:
                    with open(DEFECTS_CONFIG_PATH, 'r') as f:
                        existing_config = json.load(f)
                        existing_defects = existing_config.get("selected_defects", [])
                except Exception as e:
                    print(f"Error reading existing defect configuration: {e}")

            # Combine existing defects with newly selected ones (avoid duplicates)
            combined_defects = list(set(existing_defects + selected_defects))

            # Save the combined defect list
            config = {
                "selected_defects": combined_defects,
                "user": self.user_info.get("username", "unknown"),
                "timestamp": import_datetime().now().isoformat()
            }

            try:
                with open(DEFECTS_CONFIG_PATH, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"Combined defect configuration saved to {DEFECTS_CONFIG_PATH}")

                # Show a brief notification about saved defects
                if selected_defects:
                    notification = QMessageBox(self)
                    notification.setWindowTitle("Defects Saved")
                    notification.setText(
                        f"Selected defects have been saved.\nSelected defects: {', '.join(selected_defects)}")
                    notification.setIcon(QMessageBox.Information)
                    notification.setStandardButtons(QMessageBox.Ok)

                    # Set a timeout to auto-dismiss after 2 seconds
                    QTimer.singleShot(2000, notification.accept)
                    notification.exec_()
            except Exception as e:
                print(f"Error saving combined defect configuration: {e}")

            # Create the camera feed instance if it doesn't exist yet
            if not self.camera_feed_ui:
                self.camera_feed_ui = CameraFeedUI()

                # Hide the defect selection title bar when showing camera feed
                if hasattr(self, 'title_bar'):
                    self.title_bar.hide()

                # Make sure the Camera Feed UI has its own correct title
                if hasattr(self.camera_feed_ui, 'title_bar'):
                    # Ensure the camera feed title bar is visible
                    self.camera_feed_ui.title_bar.show()
                    # Set the correct title text
                    if hasattr(self.camera_feed_ui.title_bar, 'title_label'):
                        self.camera_feed_ui.title_bar.title_label.setText("SHEET METAL AI")

                # Find and hide any duplicate/incorrect title elements
                for label in self.camera_feed_ui.findChildren(QLabel):
                    if "DEFECT SELECTION" in label.text():
                        label.hide()

                self.stacked_widget.addWidget(self.camera_feed_ui)

            self.stacked_widget.setCurrentWidget(self.camera_feed_ui)
            self.camera_feed_ui.setWindowTitle("SAKAR VISION AI - Camera Feed")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load Camera Feed UI: {e}"
            )
            print(f"Error loading Camera Feed UI: {e}")
            return

    def goto_defective_check(self):
        """Navigate directly to the Demo Feed UI for defect checking."""
        try:
            # Import the demo feed UI
            from demo_feed_ui import DemoFeedUI

            # Save current defect selections before navigating
            self.save_defect_configuration()

            # Create instance of DemoFeedUI if it doesn't exist
            if not hasattr(self, 'demo_feed_ui') or not self.demo_feed_ui:
                self.demo_feed_ui = DemoFeedUI(self)  # Pass self as parent

                # Add the demo feed UI to the stacked widget if needed
                if isinstance(self.stacked_widget, QStackedWidget):
                    demo_feed_index = -1
                    # Check if it's already in the stacked widget
                    for i in range(self.stacked_widget.count()):
                        if self.stacked_widget.widget(i) == self.demo_feed_ui:
                            demo_feed_index = i
                            break

                    # If not found, add it
                    if demo_feed_index == -1:
                        demo_feed_index = self.stacked_widget.addWidget(self.demo_feed_ui)
                        print(f"Added Demo Feed UI to stacked widget at index {demo_feed_index}")

            # Activate the demo feed before showing it
            if hasattr(self.demo_feed_ui, 'is_active'):
                self.demo_feed_ui.is_active = True

            # Show the demo feed UI
            if isinstance(self.stacked_widget, QStackedWidget):
                # Find the index of the demo feed UI
                demo_feed_index = -1
                for i in range(self.stacked_widget.count()):
                    if self.stacked_widget.widget(i) == self.demo_feed_ui:
                        demo_feed_index = i
                        break

                if demo_feed_index != -1:
                    print(f"Switching to Demo Feed UI at index {demo_feed_index}")
                    self.stacked_widget.setCurrentIndex(demo_feed_index)
                else:
                    print("Demo Feed UI not found in stacked widget")

            # Start the camera if the demo feed has that method
            if hasattr(self.demo_feed_ui, 'resume_feed'):
                # Start camera after a small delay
                QTimer.singleShot(500, self.demo_feed_ui.resume_feed)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load Demo Feed UI: {e}"
            )
            print(f"Error loading Demo Feed UI: {e}")
            import traceback
            traceback.print_exc()

    def show_defect_selection(self):
        """Switch back to the defect selection view."""
        if hasattr(self, 'title_bar'):
            self.title_bar.show()

        self.stacked_widget.setCurrentWidget(self.defect_selection_page)

        if self.camera_feed_ui and hasattr(self.camera_feed_ui, 'title_bar'):
            self.camera_feed_ui.title_bar.hide()

        self.setWindowTitle("SAKAR VISION AI - Defect Selection")

    def closeEvent(self, event):
        """Handle the window close event properly to ensure application terminates."""
        # This ensures when this window is closed, any parent application knows about it
        print("DefectSelectionUI closing, stopping all processes")
        # Gracefully close any running processes
        if hasattr(self, 'camera_feed_ui') and self.camera_feed_ui:
            if hasattr(self.camera_feed_ui, 'stop_camera_feed'):
                self.camera_feed_ui.stop_camera_feed()

        # Call original close event
        super().closeEvent(event)

    def showEvent(self, event):
        """Called when the widget is shown."""
        super().showEvent(event)
        # Immediately maximize without any delay
        self.showMaximized()

    def store_user_session_state(self):
        """Store user ID in session state for consistency."""
        try:
            session_state_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "session_state.json")

            # Load existing session state if file exists
            if os.path.exists(session_state_path):
                with open(session_state_path, 'r') as f:
                    session_data = json.load(f)
            else:
                session_data = {}

            # Update or set the user ID and username in the session state
            if isinstance(self.user_info, dict):
                if "user_id" in self.user_info and self.user_info["user_id"]:
                    session_data["current_user_id"] = self.user_info["user_id"]
                    session_data["current_username"] = self.user_info.get("username", "")
                elif "id" in self.user_info and self.user_info["id"]:
                    session_data["current_user_id"] = self.user_info["id"]
                    session_data["current_username"] = self.user_info.get("username", "")
                else:
                    session_data["current_username"] = self.user_info.get("username", "")

            # Save the updated session state back to the file
            with open(session_state_path, 'w') as f:
                json.dump(session_data, f, indent=4)

            print(f"Session state updated: {session_data}")
        except Exception as e:
            print(f"Error managing session state: {e}")


# Helper function to import datetime only when needed


def import_datetime():
    from datetime import datetime
    return datetime


def isMaximized(self):
    """Check if window is maximized."""
    return bool(self.windowState() & Qt.WindowMaximized)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Sample user info
    user_info = {
        "username": "admin",
        "full_name": "Administrator",
        "inspection_type": "metal_sheet"  # Set inspection type to metal_sheet for standalone execution
    }

    window = DefectSelectionUI(user_info)
    # Set maximized state first, then show
    window.setWindowState(Qt.WindowMaximized)
    window.show()

    sys.exit(app.exec_())
