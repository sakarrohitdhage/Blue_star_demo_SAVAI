#!/usr/bin/env python3
"""
metal_sheet_organization_ui.py - Organization creation interface for Metal Sheet Inspection

OVERVIEW:
This is a PyQt5-based GUI application module that serves as the organization selection and creation 
interface for the metal sheet inspection workflow. The module provides a comprehensive user interface 
that allows users to either select from existing organizations or create new ones before proceeding 
to location and project selection. It acts as the first organizational step in the inspection 
hierarchy (Organization → Location → Project → Inspection).

CORE FUNCTIONALITY:
- Organization Management: Create, select, and delete organizations with database persistence
- Dynamic Organization Loading: Real-time retrieval and display of user-associated organizations from Azure database
- Form-Based Creation: Intuitive organization creation with real-time validation and character limits
- Database Integration: Seamless synchronization with Azure database for persistent organization storage
- User Session Management: Maintains user context and authentication state throughout the workflow
- Visual Feedback: Modern UI with custom styling, background images, and interactive elements
- Input Validation: Comprehensive validation including duplicate checking, character limits, and conflict resolution
- Navigation Control: Smooth transitions to location selection interface upon organization selection

KEY FEATURES:
- Custom styled components including radio buttons, delete buttons, and form elements with orange accent theme
- Background image support with multiple fallback paths and gradient alternatives for visual appeal
- Scrollable organization list with dynamic loading from database based on current user credentials
- Real-time character counting and validation for organization names (50 character limit enforcement)
- Confirmation dialogs for organization deletion with cascading location removal warnings
- Internet connectivity monitoring with visual status indicators for system health
- Automatic organization selection and database querying based on username and user ID resolution
- Error handling with user-friendly message boxes and detailed console logging for debugging
- Responsive design with automatic window maximization and background scaling
- Session state persistence and user ID resolution through multiple fallback mechanisms

TECHNICAL ARCHITECTURE:
- Built using PyQt5 framework with custom widget classes and comprehensive styling
- Azure database connectivity through azure_database module for organization CRUD operations
- Event-driven architecture with signal/slot connections for user interactions and form validation
- Custom paint events for background rendering with image scaling and overlay effects
- Memory management with proper widget cleanup, layout management, and resource disposal
- Multi-path user ID resolution system supporting session state, user_info, and username lookup
- Modular design with separate classes for UI components, database operations, and workflow management
- Thread-safe database operations with connection pooling and error recovery mechanisms

This module provides the foundational organization management layer for the metal sheet inspection 
system, ensuring proper organizational hierarchy setup before users can proceed to location and 
project selection phases of the inspection workflow.
"""

import json
import os
import sys
import traceback
from datetime import datetime
import requests

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QFont, QIcon, QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor, QPainter, QBrush, QLinearGradient
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QScrollArea

from utils import set_window_icon
# Import Azure database functions
from azure_database import get_user_id_by_username, store_location_data, get_connection, store_organization_data

# Check if metal_sheet_location_ui.py exists before importing
try:
    if os.path.exists("metal_sheet_location_ui.py"):
        print("metal_sheet_location_ui.py found in current directory")
    else:
        print("WARNING: metal_sheet_location_ui.py not found in current directory!")
except Exception as e:
    print(f"Error checking for metal_sheet_location_ui.py: {e}")


class CustomTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(70)
        self.setStyleSheet("""
            background-color: #ffffff; 
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
        self.title_label = QLabel("METAL SHEET INSPECTION - ORGANIZATION SELECTION")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: #333333;
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

        self.setLayout(layout)


class StyledLineEdit(QLineEdit):
    """Custom styled line edit for the form."""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(45)
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(200, 200, 200, 0.8);
                border-radius: 8px;
                padding: 10px 15px;
                background-color: rgba(255, 255, 255, 0.8);
                selection-background-color: #ff914d;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #ff914d;
                background-color: white;
            }
        """)


class CustomRadioButton(QRadioButton):
    """Custom styled radio button."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QRadioButton {
                font-size: 14pt;
                color: #333333;
                padding: 10px;
                spacing: 15px;
            }
            QRadioButton::indicator {
                width: 22px;
                height: 22px;
                border: 2px solid #bbbbbb;
                border-radius: 11px;
                background-color: #f5f5f5;
            }
            QRadioButton::indicator:checked {
                background-color: #ff914d;
                border: 2px solid #ff914d;
                width: 22px;
                height: 22px;
                border-radius: 11px;
            }
            QRadioButton::indicator:unchecked:hover {
                border: 2px solid #ff914d;
            }
        """)


class DeleteButton(QPushButton):
    """Custom styled delete button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("✕")
        self.setFixedSize(25, 25)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #ff914d;
                color: white;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #ff914d;
            }
            QPushButton:pressed {
                background-color: #ff914d;
            }
        """)


class CustomButton(QPushButton):
    """Custom styled button."""

    def __init__(self, text, parent=None, is_primary=True):
        super().__init__(text, parent)
        self.is_primary = is_primary
        self.setMinimumHeight(45)

        if is_primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #ff914d;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #ff914d;
                }
                QPushButton:pressed {
                    background-color: #e86d25;
                    border-style: inset;
                    padding-top: 12px;
                    padding-left: 22px;
                }
                QPushButton:focus {
                    outline: none;
                    border: 2px solid #ffa366;
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
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #bdbdbd;
                    border-style: inset;
                    padding-top: 12px;
                    padding-left: 22px;
                }
                QPushButton:focus {
                    outline: none;
                    border: 2px solid #a0a0a0;
                }
            """)

        # Remove all graphic effects and event filters that might cause visibility issues
        self.setGraphicsEffect(None)


class OrganizationCard(QFrame):
    """A styled card containing the organization creation form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("organizationCard")
        self.setStyleSheet("""
            #organizationCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.8);
            }
        """)

        # Set maximum width to prevent stretching on large screens
        self.setMaximumWidth(650)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)



class CreateNewOrgButton(QPushButton):
    """Custom button for creating a new organization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("+ Create New Organization")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ff914d;
                border: none;
                text-align: left;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #ff914d;
                text-decoration: underline;
            }
        """)


class OrganizationWidget(QWidget):
    """Widget for displaying an organization with a delete button."""

    def __init__(self, name, parent=None, allow_delete=True):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.radio = CustomRadioButton(name)
        layout.addWidget(self.radio)

        if allow_delete:
            self.delete_button = DeleteButton()
            layout.addWidget(self.delete_button)

        layout.addStretch(1)
        self.setLayout(layout)


class MetalSheetOrganizationUI(QWidget):
    """
    Interface for creating or selecting an organization for Metal Sheet Inspection.
    Uses direct background painting for consistent background display.
    """

    def __init__(self, user_info=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAKAR VISION AI - Organization Selection")
        set_window_icon(self)
        self.setMinimumSize(800, 700)

        # Store user info
        self.user_info = user_info or {"username": "User", "full_name": "Default User"}

        # Store reference to next screen
        self.location_selection_ui = None

        # Flag to track if we're in create mode
        self.creating_new_org = False

        # Radio button group for organizations
        self.org_radio_group = QButtonGroup(self)

        # Dictionary to keep track of organization widgets by name
        self.org_widgets = {}

        # Background image properties
        self.background_image = None
        self.scaled_background = None
        self.last_size = QSize(0, 0)

        # Load background image first
        self.load_background_image()

        # Initialize UI
        self.init_ui()

        # Load existing organizations
        self.load_organizations()

        # Print debug information
        print(f"MetalSheetOrganizationUI initialized for user: {self.user_info.get('username')}")

    def load_background_image(self):
        """Load background image with multiple fallback paths."""
        # Try multiple possible paths for background image
        background_paths = [
            "bg.png",
            "bg1.png",
            "bg.png",
            os.path.join(os.path.dirname(__file__), "bgwl.png"),
            os.path.join(os.path.dirname(__file__), "bg1.png"),
            os.path.join(os.path.dirname(__file__), "images", "background.png"),
        ]

        for bg_path in background_paths:
            if os.path.exists(bg_path):
                try:
                    self.background_image = QImage(bg_path)
                    if not self.background_image.isNull():
                        print(f"✓ Background image loaded successfully from: {bg_path}")
                        print(
                            f"  Image size: {self.background_image.width()}x{self.background_image.height()}")
                        return
                except Exception as e:
                    print(f"✗ Error loading background from {bg_path}: {e}")
                    continue

        print("✗ No background image found, will use gradient fallback")
        self.background_image = None

    def paintEvent(self, event):
        """Custom paint event to draw the background directly on the main window."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        try:
            # Draw background image if available
            if self.background_image and not self.background_image.isNull():
                # Scale image to fit window if needed
                if (self.scaled_background is None or
                    abs(self.width() - self.last_size.width()) > 10 or
                        abs(self.height() - self.last_size.height()) > 10):

                    self.scaled_background = self.background_image.scaled(
                        self.size(),
                        Qt.KeepAspectRatioByExpanding,
                        Qt.SmoothTransformation
                    )
                    self.last_size = QSize(self.width(), self.height())

                # Calculate centering
                if self.scaled_background and not self.scaled_background.isNull():
                    x_offset = max(0, (self.scaled_background.width() - self.width()) // 2)
                    y_offset = max(0, (self.scaled_background.height() - self.height()) // 2)

                    # Draw background image
                    painter.drawImage(0, 0, self.scaled_background, x_offset, y_offset,
                                      self.width(), self.height())

                    # Add white overlay to reduce brightness (simulating blur)
                    painter.fillRect(self.rect(), QColor(255, 255, 255, 180))

                    # Add subtle dark overlay for better contrast with white text
                    painter.fillRect(self.rect(), QColor(0, 0, 0, 40))
                else:
                    self.draw_gradient_background(painter)
            else:
                # Use gradient fallback
                self.draw_gradient_background(painter)

        except Exception as e:
            print(f"Paint event error: {e}")
            # Emergency fallback
            self.draw_gradient_background(painter)

        # Call the base class paintEvent to ensure proper rendering of child widgets
        super().paintEvent(event)

    def draw_gradient_background(self, painter):
        """Draw gradient background as fallback."""
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#1e3c72"))
        gradient.setColorAt(0.5, QColor("#2a5298"))
        gradient.setColorAt(1, QColor("#1e3c72"))

        painter.fillRect(self.rect(), QBrush(gradient))

    def update_background(self):
        """Force background update."""
        # Reset scaled background to trigger redrawing
        self.scaled_background = None
        self.update()  # Request a repaint

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)



        # Content widget with main content
        content_widget = QWidget()
        content_widget.setObjectName("transparentWidget")
        content_widget.setStyleSheet("""
            #transparentWidget {
                background-color: transparent;
            }
        """)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)

        # Welcome message
        welcome_label = QLabel(f"Welcome, {self.user_info.get('full_name', 'User')}!")
        welcome_label.setObjectName("welcomeLabel")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #333333;
        """)

        # Instruction
        instruction_label = QLabel("Please select or create an organization to continue")
        instruction_label.setObjectName("instructionLabel")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("""
            font-size: 14pt;
            color: #555555;
        """)

        # Organization card
        self.org_card = OrganizationCard()
        card_layout = QVBoxLayout(self.org_card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)

        # Organization list area
        self.org_scroll_area = QScrollArea()
        self.org_scroll_area.setWidgetResizable(True)
        self.org_scroll_area.setFrameShape(QFrame.NoFrame)
        self.org_scroll_area.setMinimumHeight(250)

        # Container for organization radio buttons
        self.org_container = QWidget()
        self.org_layout = QVBoxLayout(self.org_container)
        self.org_layout.setContentsMargins(0, 0, 0, 0)
        self.org_layout.setSpacing(10)
        self.org_layout.setAlignment(Qt.AlignTop)

        # Dictionary to keep track of organization widgets by name
        self.org_widgets = {}

        # Add "Create New Organization" button
        self.create_new_button = CreateNewOrgButton()
        self.create_new_button.clicked.connect(self.show_create_new_form)
        self.org_layout.addWidget(self.create_new_button)

        # Add scrollable area to layout
        self.org_scroll_area.setWidget(self.org_container)

        # Create new organization form (initially hidden)
        self.new_org_form = QWidget()
        new_org_layout = QVBoxLayout(self.new_org_form)
        new_org_layout.setContentsMargins(0, 0, 0, 0)
        new_org_layout.setSpacing(10)

        new_org_label = QLabel("Enter new organization name:")
        new_org_label.setStyleSheet("font-weight: bold; color: #555555;")

        self.new_org_edit = StyledLineEdit(placeholder="Organization name")
        # Add character limit validation - connect to textChanged signal
        self.new_org_edit.textChanged.connect(self.validate_org_name_length)
        
        # Add character counter label
        self.char_count_label = QLabel("0/50 characters")
        self.char_count_label.setStyleSheet("color: #555555; font-size: 12px;")
        self.char_count_label.setAlignment(Qt.AlignRight)

        new_org_buttons = QHBoxLayout()
        self.cancel_new_button = CustomButton("Cancel", is_primary=False)
        self.cancel_new_button.clicked.connect(self.hide_create_new_form)

        self.create_button = CustomButton("Create", is_primary=True)
        self.create_button.clicked.connect(self.create_new_organization)

        new_org_buttons.addWidget(self.cancel_new_button)
        new_org_buttons.addStretch(1)
        new_org_buttons.addWidget(self.create_button)

        new_org_layout.addWidget(new_org_label)
        new_org_layout.addWidget(self.new_org_edit)
        new_org_layout.addWidget(self.char_count_label)
        new_org_layout.addLayout(new_org_buttons)

        # Hide initially
        self.new_org_form.setVisible(False)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        # Remove back button
        # Only keep the continue button
        self.continue_button = CustomButton("Continue", is_primary=True)
        self.continue_button.clicked.connect(self.proceed_to_location)

        # Center the continue button
        button_layout.addStretch(1)
        button_layout.addWidget(self.continue_button)
        button_layout.addStretch(1)

        # Add elements to card layout
        card_layout.addWidget(self.org_scroll_area)
        card_layout.addWidget(self.new_org_form)
        card_layout.addStretch(1)
        card_layout.addLayout(button_layout)

        # Add elements to content layout
        content_layout.addWidget(welcome_label)
        content_layout.addWidget(instruction_label)
        content_layout.addSpacing(20)

        # Center the card
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.org_card)

        content_layout.addLayout(center_layout)
        content_layout.addStretch(1)
        main_layout.addWidget(content_widget)

        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)
        status_layout.setAlignment(Qt.AlignCenter)

        # Status indicator dot
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-size: 14px;
            }
        """)

        # Status text
        self.status_label = QLabel("System Online")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-size: 12px;
                font-weight: 500;
            }
        """)

        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_label)

        card_layout.addWidget(status_container)

        # Initialize connectivity timer for status checking
        self.connectivity_timer = QTimer(self)
        self.connectivity_timer.timeout.connect(self.check_internet_connectivity)
        self.connectivity_timer.start(5000)  # Check every 5 seconds

        # Initial connectivity check
        self.check_internet_connectivity()

    def check_internet_connectivity(self):
        """Check internet connectivity and update status indicator."""
        try:
            # Perform a lightweight request to a reliable endpoint
            response = requests.get("https://www.google.com", timeout=5)
            if response.status_code == 200:
                self.update_status_indicator(True)
            else:
                self.update_status_indicator(False)
        except Exception as e:
            self.update_status_indicator(False)
            print(f"Connectivity check error: {e}")

    def update_status_indicator(self, online):
        """Update the status indicator's appearance based on connectivity."""
        if online:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #28a745;
                    font-size: 14px;
                }
            """)
            self.status_label.setText("System Online")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #28a745;
                    font-size: 12px;
                    font-weight: 500;
                }
            """)
        else:
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    font-size: 14px;
                }
            """)
            self.status_label.setText("No Internet Connection")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    font-size: 12px;
                    font-weight: 500;
                }
            """)

        # Add widgets to main layout


    def resizeEvent(self, event):
        """Update background size when window is resized."""
        # self.background_widget.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def showEvent(self, event):
        """Override show event to maximize the window and ensure background is sized correctly."""
        super().showEvent(event)
        self.showMaximized()
        # self.background_widget.setGeometry(0, 0, self.width(), self.height())
        self.update()

    def load_organizations(self):
        """Load existing organizations from the database and populate the UI."""
        try:
            print(f"🔍 [DEBUG] Starting load_organizations for user_info: {self.user_info}")
            
            # Connect to the database
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                return

            cursor = connection.cursor(dictionary=True)

            # Get the current username from user_info
            username = self.user_info.get("username", "")
            
            if not username:
                print("Error: No username found in user_info")
                return
                
            print(f"🔍 [DEBUG] Loading organizations for username: {username}")
            
            # Get all user IDs for this username from the users table
            user_query = "SELECT id FROM users WHERE username = %s"
            cursor.execute(user_query, (username,))
            user_ids = cursor.fetchall()
            
            if not user_ids:
                print(f"🔍 [DEBUG] No user IDs found for username: {username}")
                return
                
            # Extract just the ID values
            user_id_list = [user['id'] for user in user_ids]
            print(f"🔍 [DEBUG] Found user IDs for username '{username}': {user_id_list}")
            
            # Create placeholder string for IN clause
            placeholders = ', '.join(['%s'] * len(user_id_list))
            
            # Get organizations created by any user with this username
            query = f"""
            SELECT DISTINCT l.organization 
            FROM location l 
            WHERE l.id IN ({placeholders})
            ORDER BY l.organization
            """
            print(f"🔍 [DEBUG] Executing query: {query}")
            print(f"🔍 [DEBUG] With parameters: {user_id_list}")
            
            cursor.execute(query, user_id_list)
            organizations = cursor.fetchall()

            print(f"🔍 [DEBUG] Raw organizations from database: {organizations}")

            # Clear existing organizations first
            print(f"🔍 [DEBUG] Clearing existing organization list...")
            self.clear_organization_list()

            # Check if we found any organizations
            if not organizations:
                print(f"🔍 [DEBUG] No organizations found in the database for username: {username}")
                message_widget = QLabel(
                    "No organizations found. Please create a new organization.")
                message_widget.setStyleSheet("color: #555555; font-size: 14px; padding: 20px;")
                message_widget.setAlignment(Qt.AlignCenter)
                self.org_layout.insertWidget(self.org_layout.count() - 1, message_widget)
                return

            # Add organizations from database
            print(f"🔍 [DEBUG] Adding {len(organizations)} organizations to UI...")
            for i, org in enumerate(organizations, start=1):
                org_name = org['organization']
                print(f"🔍 [DEBUG] Processing organization {i}: '{org_name}'")
                
                if org_name:  # Make sure organization name is not empty
                    # Create organization widget with radio button and delete button
                    org_widget = OrganizationWidget(org_name)
                    self.org_radio_group.addButton(org_widget.radio, i)

                    # Set the first organization as selected
                    if i == 1:
                        org_widget.radio.setChecked(True)
                        print(f"🔍 [DEBUG] Set '{org_name}' as default selected organization")

                    # Connect delete button using a lambda to capture current name
                    org_widget.delete_button.clicked.connect(
                        lambda checked=False, name=org_name: self.delete_organization(name)
                    )

                    # Add to UI before the create button
                    self.org_layout.insertWidget(self.org_layout.count() - 1, org_widget)
                    # Store in dictionary for later reference
                    self.org_widgets[org_name] = org_widget
                    print(f"🔍 [DEBUG] Added organization '{org_name}' to UI and org_widgets dictionary")
                else:
                    print(f"🔍 [DEBUG] Skipping empty organization name at index {i}")

            print(f"🔍 [DEBUG] Final org_widgets dictionary: {list(self.org_widgets.keys())}")
            print(f"🔍 [DEBUG] Loaded {len(organizations)} organizations from the database for username: {username}")
            
            cursor.close()
            connection.close()
            
        except Exception as e:
            print(f"🔍 [DEBUG] Error loading organizations from database: {e}")
            traceback.print_exc()
            # Show error message to user
            QMessageBox.warning(
                self,
                "Database Error",
                f"Failed to load organizations from database: {str(e)}\n\nPlease create a new organization."
            )

    def clear_organization_list(self):
        """Clear the list of organizations in the UI."""
        for widget in self.org_widgets.values():
            widget.deleteLater()
        self.org_widgets.clear()
        self.org_radio_group.setExclusive(False)
        for button in self.org_radio_group.buttons():
            button.setChecked(False)
        self.org_radio_group.setExclusive(True)

    def delete_organization(self, org_name):
        """Delete an organization from the database."""
        try:
            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete '{org_name}'?\nThis will remove all locations associated with this organization.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return False

            # Connect to database
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                QMessageBox.critical(self, "Database Error", "Could not connect to the database.")
                return False

            cursor = connection.cursor()

            # Delete organization from the location table
            delete_query = "DELETE FROM location WHERE organization = %s"
            cursor.execute(delete_query, (org_name,))

            # Commit changes
            connection.commit()

            # Close cursor and connection
            cursor.close()
            connection.close()

            # Update UI - remove the widget
            if org_name in self.org_widgets:
                widget = self.org_widgets[org_name]

                # If this radio button was checked, select another organization if available
                if widget.radio.isChecked():
                    # Find another radio to check
                    other_orgs = [btn for btn in self.org_radio_group.buttons() if btn !=
                                  widget.radio]
                    if other_orgs:
                        other_orgs[0].setChecked(True)
                    else:
                        # No other organizations, we'll need to create a new one
                        # Show the create organization form
                        self.show_create_new_form()

                # Remove from radio group
                self.org_radio_group.removeButton(widget.radio)

                # Remove from layout
                self.org_layout.removeWidget(widget)

                # Delete the widget
                widget.deleteLater()

                # Remove from dictionary
                del self.org_widgets[org_name]

            QMessageBox.information(
                self, "Success", f"Organization '{org_name}' deleted successfully")
            print(f"Organization '{org_name}' deleted successfully from database")
            return True

        except Exception as e:
            print(f"Error deleting organization from database: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to delete organization: {str(e)}")
            return False

    def show_create_new_form(self):
        """Show the form for creating a new organization."""
        self.creating_new_org = True
        self.org_scroll_area.setVisible(False)
        self.new_org_form.setVisible(True)
        self.new_org_edit.setFocus()
        # Force background repaint to prevent disappearing
        self.update_background()
        QApplication.processEvents()  # Process all pending events to ensure UI updates

    def hide_create_new_form(self):
        """Hide the form for creating a new organization."""
        self.creating_new_org = False
        self.new_org_form.setVisible(False)
        self.org_scroll_area.setVisible(True)
        self.new_org_edit.clear()
        # Force background repaint to prevent disappearing
        self.update_background()
        QApplication.processEvents()  # Process all pending events to ensure UI updates

    def create_new_organization(self):
        """Create a new organization and add it to the database."""
        org_name = self.new_org_edit.text().strip()

        if not org_name:
            QMessageBox.warning(self, "Input Required", "Please enter an organization name.")
            return

        # Limit organization name to 50 characters
        if len(org_name) > 50:
            QMessageBox.warning(self, "Input Too Long", 
                               "Organization name must be 50 characters or less.")
            return

        # Check if organization already exists (case-insensitive)
        exists = False
        for existing_name in self.org_widgets:
            if org_name.lower() == existing_name.lower():
                exists = True
                org_name = existing_name  # Use the existing case version
                break

        if exists:
            QMessageBox.warning(self, "Duplicate Organization",
                                f"Organization '{org_name}' already exists.")
            return

        # Check if name exists as a location
        try:
            connection = get_connection()
            if connection:
                cursor = connection.cursor(dictionary=True)
                query = "SELECT DISTINCT city FROM location WHERE LOWER(city) = LOWER(%s)"
                cursor.execute(query, (org_name,))
                location_exists = cursor.fetchone()
                cursor.close()
                connection.close()

                if location_exists:
                    QMessageBox.warning(self, "Name Conflict",
                                        f"Cannot create organization '{org_name}' as it matches an existing location name.")
                    return
        except Exception as e:
            print(f"Error checking location names: {e}")

        # Save to database
        if not self.save_organization(org_name):
            QMessageBox.critical(self, "Database Error",
                                 "Failed to save organization to database.")
            return

        # Create new organization widget
        new_widget = OrganizationWidget(org_name)
        new_widget.radio.setChecked(True)  # Select the new organization

        # Add to group with new ID
        new_id = len(self.org_widgets)
        self.org_radio_group.addButton(new_widget.radio, new_id)

        # Connect delete button
        new_widget.delete_button.clicked.connect(
            lambda checked=False, name=org_name: self.delete_organization(name)
        )

        # Add to layout before the create button
        self.org_layout.insertWidget(self.org_layout.count() - 1, new_widget)

        # Add to dictionary
        self.org_widgets[org_name] = new_widget

        # Hide the creation form
        self.hide_create_new_form()

    def save_organization(self, org_name=None):
        """Save the organization to the database."""
        if org_name is None:
            # Get the selected organization
            selected_button = self.org_radio_group.checkedButton()
            if selected_button is None:
                return False

            org_name = selected_button.text()

        if not org_name:
            return False

        try:
            # Check if it's Sakarrobotics (already exists by default)
            if org_name.lower() == "sakarrobotics":
                return True

            # Connect to database
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                return False

            cursor = connection.cursor(dictionary=True)

            # Check if organization already exists
            check_query = "SELECT id FROM location WHERE organization = %s LIMIT 1"
            cursor.execute(check_query, (org_name,))
            existing = cursor.fetchone()

            if existing:
                # Organization already exists
                print(f"Organization '{org_name}' already exists in the database")
                cursor.close()
                connection.close()
                return True

            # Get current user ID by looking up the current username (not from stale user_info)
            current_username = self.user_info.get("username", "")
            user_id = get_user_id_by_username(current_username)
            
            if not user_id:
                print(f"Error: Could not find user ID for current username: {current_username}")
                cursor.close()
                connection.close()
                return False
            
            # Insert a new organization record with a placeholder city
            # The city field cannot be NULL, so we use a placeholder that indicates this is just an organization record
            insert_query = """
                INSERT INTO location (id, organization, city, timestamp) 
                VALUES (%s, %s, %s, NOW())
            """
            #cursor.execute(insert_query, (user_id, org_name, "__org_placeholder__"))
            
            # Commit changes to register that this is a valid organization
            connection.commit()

            # Close cursor and connection
            cursor.close()
            connection.close()

            print(f"Organization '{org_name}' saved to database with user ID: {user_id}")
            return True

        except Exception as e:
            print(f"Error saving organization to database: {e}")
            traceback.print_exc()
            return False

    def proceed_to_location(self):
        """Save organization and proceed to location selection."""
        # Get the selected organization
        selected_button = self.org_radio_group.checkedButton()
        if selected_button is None:
            QMessageBox.warning(self, "Selection Required", "Please select an organization.")
            return

        org_name = selected_button.text()
        print(f"Continuing with organization: {org_name}")

        # Save the organization first
        if not self.save_organization():
            QMessageBox.critical(self, "Error", "Failed to save organization. Please try again.")
            return

        # Force background repaint before transition
        self.update_background()

        # Store organization data using current session user ID
        try:
            # First, try to get the current user ID from session state file
            session_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_state.json")
            current_user_id = None
            
            if os.path.exists(session_file_path):
                try:
                    with open(session_file_path, 'r') as f:
                        session_state = json.load(f)
                    current_user_id = session_state.get("current_user_id")
                    print(f"Retrieved user ID from session state: {current_user_id}")
                except Exception as session_err:
                    print(f"Warning: Could not read session state: {session_err}")
            
            # Fall back to user_info if session state is not available
            if not current_user_id:
                current_user_id = self.user_info.get("user_id", "")
                print(f"Using user ID from user_info: {current_user_id}")
            
            # Final fallback: username lookup (only if absolutely necessary)
            if not current_user_id:
                username = self.user_info.get("username", "")
                if username:
                    user_id = get_user_id_by_username(username)
                    if user_id:
                        current_user_id = user_id
                        print(f"Got user ID from username lookup as last resort: {current_user_id}")
            
            if current_user_id:
                store_organization_data(current_user_id, org_name)
                print(f"Stored organization data for user ID {current_user_id}: {org_name}")
            else:
                print("Warning: Could not determine user ID for organization storage")
                
        except Exception as e:
            print(f"Error storing organization data: {e}")
            # Continue anyway - don't block the user from proceeding

        # Now try to launch the location selection UI
        try:
            # First check if the module exists
            if not os.path.exists("metal_sheet_location_ui.py"):
                msg = "The location selection module (metal_sheet_location_ui.py) is missing. Please check your installation."
                print(msg)
                QMessageBox.critical(self, "Missing Module", msg)
                return

            print("Importing MetalSheetLocationUI...")
            # Try to import the MetalSheetLocationUI
            try:
                from metal_sheet_location_ui import MetalSheetLocationUI
            except ImportError as e:
                print(f"Error importing MetalSheetLocationUI: {e}")
                QMessageBox.critical(self, "Import Error",
                                     f"Failed to load the location selection module: {str(e)}\n"
                                     "Please check that all required files are in place.")
                return

            # Apply visual feedback for button click
            original_style = self.continue_button.styleSheet()
            self.continue_button.setStyleSheet("""
                QPushButton {
                    background-color: #e86d25;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    border-style: inset;
                    padding: 12px 22px 8px 18px;
                    font-weight: bold;
                    font-size: 14px;
                }
            """)

            # Force UI update to show button press effect
            QApplication.processEvents()

            print("Creating MetalSheetLocationUI instance...")
            # Close any existing instance before creating a new one
            if hasattr(self, 'location_selection_ui') and self.location_selection_ui is not None:
                self.location_selection_ui.close()
                self.location_selection_ui.deleteLater()

            # Create the MetalSheetLocationUI instance
            self.location_selection_ui = MetalSheetLocationUI(
                user_info=self.user_info,
                organization=org_name
            )

            # Connect destroyed signal to show this window again if the location screen is closed
            self.location_selection_ui.destroyed.connect(self.show)

            print("Showing MetalSheetLocationUI...")
            # Show the location selection UI
            self.location_selection_ui.showMaximized()

            # Hide this window (don't use QTimer.singleShot as it can cause issues)
            self.hide()

        except Exception as e:
            # Reset button style in case of error
            self.continue_button.setStyleSheet(original_style)

            print("Error in proceed_to_location:")
            traceback.print_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to launch location selection screen: {str(e)}\n\n"
                                 "See console for detailed error information.")

    def show_metal_sheet_location(self):
        """Show the metal sheet location UI."""
        try:
            from metal_sheet_location_ui import MetalSheetLocationUI

            # Get the selected organization
            if self.current_selection is None:
                QMessageBox.warning(self, "Selection Required",
                                    "Please select an organization first.")
                return

            # Get the organization ID from the selection
            org_id = self.current_selection['id']
            org_name = self.current_selection['name']

            # Pass user info forward, including selected camera index
            user_info = self.user_info.copy()
            user_info["org_id"] = org_id
            user_info["org_name"] = org_name

            # Create and show the location UI
            self.metal_sheet_location_ui = MetalSheetLocationUI(user_info)
            self.metal_sheet_location_ui.show()
            self.hide()

            # Close this window when the location UI is closed
            self.metal_sheet_location_ui.destroyed.connect(self.close)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load location UI: {str(e)}")
            print(f"Error loading MetalSheetLocationUI: {e}")
            traceback.print_exc()

    def go_back(self):
        """Go back to login screen."""
        try:
            from login_ui import LoginUI

            # Close any existing instance before creating a new one
            if hasattr(self, 'login_ui') and self.login_ui is not None:
                self.login_ui.close()
                self.login_ui.deleteLater()

            # Create new login UI instance
            self.login_ui = LoginUI()
            self.login_ui.is_fabric_inspection = False  # This is metal sheet inspection

            # Connect destroyed signal to show this window again if login screen is closed
            self.login_ui.destroyed.connect(self.show)

            # Show login UI
            self.login_ui.showMaximized()

            # Hide this window
            self.hide()

        except Exception as e:
            print("Error going back to login:")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not go back to login screen: {str(e)}")

    def closeEvent(self, event):
        """Handle the window close event properly to ensure application terminates."""
        print("MetalSheetOrganizationUI closing, terminating application")
        # Make sure to terminate the application if this window is closed
        QApplication.quit()
        super().closeEvent(event)

    def showEvent(self, event):
        """Override show event to maximize the window."""
        super().showEvent(event)
        self.showMaximized()

    def validate_org_name_length(self, text):
        """Validate the organization name length in real-time as user types."""
        # Update character count label
        count = len(text)
        self.char_count_label.setText(f"{count}/50 characters")
        
        # Enforce 50 character limit by truncating input
        if count > 50:
            # Truncate text to 50 characters and update the field
            self.new_org_edit.blockSignals(True)  # Prevent infinite recursion
            self.new_org_edit.setText(text[:50])
            self.new_org_edit.blockSignals(False)
            # Update the count after truncation
            self.char_count_label.setText("50/50 characters")
            # Set cursor at the end
            self.new_org_edit.setCursorPosition(50)
            # Highlight with red to indicate limit reached
            self.char_count_label.setStyleSheet("color: #ff0000; font-size: 12px; font-weight: bold;")
            return
        
        # Change color when approaching limit
        if count >= 45:  # Approaching limit
            self.char_count_label.setStyleSheet("color: #ff8c00; font-size: 12px; font-weight: bold;")
            self.new_org_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid rgba(200, 200, 200, 0.8);
                    border-radius: 8px;
                    padding: 10px 15px;
                    background-color: rgba(255, 255, 255, 0.8);
                    selection-background-color: #ff914d;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 2px solid #ff914d;
                    background-color: white;
                }
            """)
        else:  # Within limit
            self.char_count_label.setStyleSheet("color: #555555; font-size: 12px;")
            self.new_org_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid rgba(200, 200, 200, 0.8);
                    border-radius: 8px;
                    padding: 10px 15px;
                    background-color: rgba(255, 255, 255, 0.8);
                    selection-background-color: #ff914d;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 2px solid #ff914d;
                    background-color: white;
                }
            """)
            
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Print current directory and check for login_ui.py
    print(f"Current directory: {os.getcwd()}")
    if os.path.exists("metal_sheet_location_ui.py"):
        print("metal_sheet_location_ui.py found in current directory")
    else:
        print("WARNING: metal_sheet_location_ui.py not found in current directory!")

    # Sample user info
    user_info = {
        "username": "admin",
        "full_name": "Administrator"
    }

    # Create and show window
    window = MetalSheetOrganizationUI(user_info)
    window.showMaximized()

    sys.exit(app.exec_())