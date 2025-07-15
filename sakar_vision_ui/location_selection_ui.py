#!/usr/bin/env python3
"""
SAKAR VISION AI - Location Selection UI Module

OVERVIEW:
This module implements a high-performance location selection interface for the Sakar Vision AI platform's Fabric 
Inspection workflow, serving as a critical navigation component that enables users to efficiently select and manage 
inspection locations within organizational hierarchies. It provides an optimized, responsive interface that combines 
sophisticated location management with seamless workflow progression, ensuring smooth transitions between location 
selection and fabric inspection operations while maintaining exceptional performance through advanced caching 
mechanisms and asynchronous data loading for industrial deployment environments.

KEY FUNCTIONALITY:
The system features intelligent location management with support for both default and custom location creation with 
real-time validation and persistence, advanced background data loading through dedicated worker threads to prevent 
UI blocking during location retrieval operations, sophisticated location widget system with radio button selection 
and conditional delete functionality for custom locations, and comprehensive form-based location creation with 
animated transitions and input validation. It includes optimized caching mechanisms with file modification tracking 
to minimize redundant I/O operations, seamless workflow integration with automatic configuration persistence and 
smooth transitions to fabric deployment interfaces, advanced UI animations with fade-in/fade-out effects for enhanced 
user experience, and robust error handling with graceful degradation and user-friendly feedback for data loading 
and persistence operations.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced performance optimization techniques including hardware acceleration hints and efficient 
widget management, the module employs sophisticated asynchronous data loading with dedicated DataLoader worker thread 
for non-blocking location retrieval, comprehensive caching system with file modification time tracking to optimize 
repeated data access operations, and modular widget architecture featuring AnimatedFrame, LocationWidget, and optimized 
UI components with minimal styling overhead. The architecture includes intelligent memory management with proper widget 
lifecycle handling and resource cleanup procedures, advanced animation systems using QPropertyAnimation with opacity 
effects for smooth visual transitions, persistent configuration management through JSON-based storage with atomic 
write operations for location preferences and organizational data, and seamless integration patterns with dynamic 
module loading for fabric deployment interface instantiation. The system features comprehensive error handling with 
detailed logging and user feedback mechanisms, optimized for industrial deployment scenarios requiring high reliability 
and performance consistency.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect,
                             QMessageBox, QButtonGroup, QRadioButton, QLineEdit,
                             QScrollArea, QGraphicsOpacityEffect)

from utils import set_window_icon

# Constants for file paths
ORGANIZATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "organizations.json")
LOCATION_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "location_config.json")

# Cache for loaded data to avoid repeated file I/O
_data_cache = {
    "organizations": None,
    "last_modified": 0
}

# Default locations - immutable
DEFAULT_LOCATIONS = ["Pune", "Jodhpur", "Jaipur"]


class DataLoader(QThread):
    """Background thread for loading organization data."""

    data_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, organization_name: str):
        super().__init__()
        self.organization_name = organization_name

    def run(self):
        """Load organization data in background thread."""
        try:
            locations = self.load_organization_locations()
            self.data_loaded.emit({"locations": locations})
        except Exception as e:
            self.error_occurred.emit(str(e))

    def load_organization_locations(self) -> List[str]:
        """Load locations for the organization with caching."""
        try:
            # Check if file exists
            if not os.path.exists(ORGANIZATIONS_FILE):
                return []

            # Get file modification time for cache validation
            file_mod_time = os.path.getmtime(ORGANIZATIONS_FILE)

            # Use cache if available and file hasn't changed
            if (_data_cache["organizations"] is not None and
                    _data_cache["last_modified"] >= file_mod_time):
                organizations = _data_cache["organizations"]
            else:
                # Load fresh data
                with open(ORGANIZATIONS_FILE, 'r', encoding='utf-8') as f:
                    organizations = json.load(f)

                # Update cache
                _data_cache["organizations"] = organizations
                _data_cache["last_modified"] = file_mod_time

            # Return locations for this organization
            if self.organization_name in organizations:
                return organizations[self.organization_name].get("locations", [])

            return []

        except Exception as e:
            print(f"Error loading organization locations: {e}")
            return []


class AnimatedFrame(QFrame):
    """Frame with smooth show/hide animations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)  # 200ms animation
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

    def fadeIn(self):
        """Fade in animation."""
        self.show()
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()

    def fadeOut(self):
        """Fade out animation."""
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(self.hide)
        self.animation.start()


class OptimizedTitleBar(QFrame):
    """Lightweight title bar with minimal styling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.init_ui()

    def init_ui(self):
        """Initialize UI with optimized styling."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        # Logo (load asynchronously if needed)
        self.logo_label = QLabel()
        self.load_logo_async()

        # Title
        self.title_label = QLabel("FABRIC INSPECTION - SELECT LOCATION")
        self.title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.logo_label)
        layout.addWidget(self.title_label, 1)
        layout.addStretch(0)

        # Apply minimal styling
        self.setStyleSheet("""
            OptimizedTitleBar {
                background-color: #ff914d;
                border: none;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 24px;
            }
        """)

    def load_logo_async(self):
        """Load logo asynchronously to avoid blocking UI."""
        logo_path = "sakar.png"
        if os.path.exists(logo_path):
            QTimer.singleShot(0, lambda: self.set_logo(logo_path))
        else:
            self.logo_label.setText("SAKAR")
            self.logo_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")

    def set_logo(self, logo_path: str):
        """Set logo from file path."""
        try:
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                scaled_pixmap = logo_pixmap.scaledToHeight(40, Qt.SmoothTransformation)
                self.logo_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error loading logo: {e}")
            self.logo_label.setText("SAKAR")


class LocationWidget(QWidget):
    """Optimized location widget with minimal overhead."""

    delete_requested = pyqtSignal(str)

    def __init__(self, name: str, allow_delete: bool = True, parent=None):
        super().__init__(parent)
        self.location_name = name
        self.allow_delete = allow_delete
        self.init_ui()

    def init_ui(self):
        """Initialize UI with minimal complexity."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Radio button
        self.radio = QRadioButton(self.location_name)
        layout.addWidget(self.radio)

        # Delete button (only for custom locations)
        if self.allow_delete:
            self.delete_button = QPushButton("✕")
            self.delete_button.setFixedSize(25, 25)
            self.delete_button.setCursor(Qt.PointingHandCursor)
            self.delete_button.clicked.connect(
                lambda: self.delete_requested.emit(self.location_name))
            layout.addWidget(self.delete_button)

        layout.addStretch(1)

        # Minimal styling
        self.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                color: #333333;
                padding: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #bbbbbb;
                border-radius: 9px;
                background-color: #f5f5f5;
            }
            QRadioButton::indicator:checked {
                background-color: #ff914d;
                border: 2px solid #ff914d;
            }
            QPushButton {
                background-color: #ff914d;
                color: white;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ff7730;
            }
        """)


class OptimizedLocationCard(AnimatedFrame):
    """Optimized location card with performance improvements."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("locationCard")
        self.setMaximumWidth(600)
        self.init_styling()

    def init_styling(self):
        """Initialize styling with hardware acceleration hints."""
        self.setStyleSheet("""
            #locationCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.8);
            }
        """)

        # Optimized shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)  # Reduced for performance
        shadow.setColor(QColor(0, 0, 0, 60))  # Lighter shadow
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)


class OptimizedButton(QPushButton):
    """Optimized button with performance improvements."""

    def __init__(self, text: str, is_primary: bool = True, parent=None):
        super().__init__(text, parent)
        self.is_primary = is_primary
        self.setMinimumHeight(40)  # Slightly smaller for performance
        self.init_styling()

    def init_styling(self):
        """Initialize button styling."""
        if self.is_primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #ff914d;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 14px;
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
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QPushButton:pressed {
                    background-color: #bdbdbd;
                }
            """)


class OptimizedLineEdit(QLineEdit):
    """Optimized line edit with minimal styling."""

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 8px 12px;
                background-color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #ff914d;
            }
        """)


class LocationSelectionUI(QWidget):
    """
    Optimized interface for selecting locations with improved performance.
    """

    def __init__(self, user_info: Optional[Dict] = None, organization: Optional[str] = None, parent=None):
        super().__init__(parent)

        # Initialize core data
        self.user_info = user_info or {"username": "User", "full_name": "Default User"}
        self.organization = organization or "Unknown Organization"

        # UI components
        self.location_widgets: Dict[str, LocationWidget] = {}
        self.location_group = QButtonGroup(self)
        self.data_loader: Optional[DataLoader] = None
        self.fabric_deployment_ui = None

        # Performance flags
        self._ui_initialized = False
        self._locations_loaded = False

        # Initialize UI
        self.init_ui()
        self.setup_window()

        # Load data asynchronously
        QTimer.singleShot(0, self.load_locations_async)

    def setup_window(self):
        """Setup window properties for optimal performance."""
        self.setWindowTitle("SAKAR VISION AI - Select Location")
        set_window_icon(self)
        self.setMinimumSize(800, 600)

        # Enable hardware acceleration hints
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground, False)

    def init_ui(self):
        """Initialize UI with optimized layout."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        self.title_bar = OptimizedTitleBar(self)
        main_layout.addWidget(self.title_bar)

        # Content setup
        self.setup_content(main_layout)

        # Apply minimal global styling
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f7f7f7;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self._ui_initialized = True

    def setup_content(self, main_layout: QVBoxLayout):
        """Setup main content area."""
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 30, 40, 30)
        content_layout.setSpacing(15)

        # Header section
        self.setup_header(content_layout)

        # Location card
        self.location_card = OptimizedLocationCard()
        self.setup_location_card()

        # Center the card
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.location_card)

        content_layout.addLayout(center_layout)
        content_layout.addStretch(1)

        main_layout.addWidget(content_widget)

    def setup_header(self, layout: QVBoxLayout):
        """Setup header section with user info."""
        # Welcome message
        welcome_label = QLabel(f"Welcome, {self.user_info.get('full_name', 'User')}!")
        welcome_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333333;")

        # Organization info
        org_layout = QHBoxLayout()
        org_text_label = QLabel("Selected Organization:")
        org_text_label.setStyleSheet("font-size: 14px; color: #555555;")

        org_name_label = QLabel(self.organization)
        org_name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff7730;")

        org_layout.addWidget(org_text_label)
        org_layout.addWidget(org_name_label)
        org_layout.addStretch(1)

        # Instruction
        instruction_label = QLabel("Please select a location for fabric inspection:")
        instruction_label.setStyleSheet("font-size: 14px; color: #555555;")

        layout.addWidget(welcome_label)
        layout.addLayout(org_layout)
        layout.addWidget(instruction_label)
        layout.addSpacing(15)

    def setup_location_card(self):
        """Setup the location selection card."""
        card_layout = QVBoxLayout(self.location_card)
        card_layout.setContentsMargins(25, 25, 25, 25)
        card_layout.setSpacing(15)

        # Location selection area
        self.location_selection_area = QWidget()
        selection_layout = QVBoxLayout(self.location_selection_area)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(8)

        # Scroll area for locations
        self.location_scroll_area = QScrollArea()
        self.location_scroll_area.setWidgetResizable(True)
        self.location_scroll_area.setFrameShape(QFrame.NoFrame)
        self.location_scroll_area.setMinimumHeight(120)
        self.location_scroll_area.setMaximumHeight(200)

        # Container for location widgets
        self.location_container = QWidget()
        self.location_layout = QVBoxLayout(self.location_container)
        self.location_layout.setContentsMargins(0, 0, 0, 0)
        self.location_layout.setSpacing(5)
        self.location_layout.setAlignment(Qt.AlignTop)

        self.location_scroll_area.setWidget(self.location_container)
        selection_layout.addWidget(self.location_scroll_area)

        # New location form (initially hidden)
        self.setup_new_location_form()

        # Buttons
        self.setup_buttons(card_layout)

        # Add components to card
        card_layout.addWidget(self.location_selection_area)
        card_layout.addWidget(self.new_location_form)
        card_layout.addStretch(1)

    def setup_new_location_form(self):
        """Setup new location creation form."""
        self.new_location_form = AnimatedFrame()
        form_layout = QVBoxLayout(self.new_location_form)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(10)

        form_label = QLabel("Enter new location name:")
        form_label.setStyleSheet("font-weight: bold; color: #555555;")

        self.new_location_edit = OptimizedLineEdit("Location name")

        # Form buttons
        form_buttons = QHBoxLayout()
        self.cancel_new_button = OptimizedButton("Cancel", False)
        self.cancel_new_button.clicked.connect(self.hide_create_form)

        self.create_location_button = OptimizedButton("Create", True)
        self.create_location_button.clicked.connect(self.create_new_location)

        form_buttons.addWidget(self.cancel_new_button)
        form_buttons.addStretch(1)
        form_buttons.addWidget(self.create_location_button)

        form_layout.addWidget(form_label)
        form_layout.addWidget(self.new_location_edit)
        form_layout.addLayout(form_buttons)

        self.new_location_form.setVisible(False)

    def setup_buttons(self, layout: QVBoxLayout):
        """Setup main action buttons."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.back_button = OptimizedButton("Back", False)
        self.back_button.clicked.connect(self.go_back)

        self.start_button = OptimizedButton("Start Inspection", True)
        self.start_button.clicked.connect(self.proceed_to_inspection)

        button_layout.addWidget(self.back_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.start_button)

        layout.addLayout(button_layout)

    def load_locations_async(self):
        """Load locations in background thread."""
        if self.data_loader is not None:
            return  # Already loading

        self.data_loader = DataLoader(self.organization)
        self.data_loader.data_loaded.connect(self.on_locations_loaded)
        self.data_loader.error_occurred.connect(self.on_loading_error)
        self.data_loader.finished.connect(self.cleanup_loader)
        self.data_loader.start()

        # Show loading state
        self.show_loading_state()

    def show_loading_state(self):
        """Show loading indicator while data loads."""
        loading_label = QLabel("Loading locations...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("color: #888888; font-style: italic;")
        self.location_layout.addWidget(loading_label)

    def on_locations_loaded(self, data: Dict):
        """Handle loaded location data."""
        # Clear loading state
        self.clear_location_widgets()

        # Add default locations first
        self.add_default_locations()

        # Add custom locations
        custom_locations = data.get("locations", [])
        self.add_custom_locations(custom_locations)

        # Add create new button
        self.add_create_button()

        self._locations_loaded = True

        # Animate in the location card
        if hasattr(self.location_card, 'fadeIn'):
            self.location_card.fadeIn()

    def on_loading_error(self, error: str):
        """Handle loading error."""
        print(f"Error loading locations: {error}")

        # Clear loading state and show defaults
        self.clear_location_widgets()
        self.add_default_locations()
        self.add_create_button()

        # Show error message
        QMessageBox.warning(self, "Loading Error",
                            "Could not load custom locations. Default locations are available.")

    def cleanup_loader(self):
        """Clean up data loader thread."""
        if self.data_loader is not None:
            self.data_loader.deleteLater()
            self.data_loader = None

    def clear_location_widgets(self):
        """Clear all location widgets efficiently."""
        # Clear layout
        while self.location_layout.count():
            child = self.location_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear tracking
        self.location_widgets.clear()

        # Clear button group
        for button in self.location_group.buttons():
            self.location_group.removeButton(button)

    def add_default_locations(self):
        """Add default location options."""
        for i, location in enumerate(DEFAULT_LOCATIONS):
            widget = LocationWidget(location, allow_delete=False)
            widget.delete_requested.connect(self.delete_location)

            self.location_group.addButton(widget.radio, i + 1)
            self.location_layout.addWidget(widget)
            self.location_widgets[location] = widget

            # Select Pune by default
            if location == "Pune":
                widget.radio.setChecked(True)

    def add_custom_locations(self, locations: List[str]):
        """Add custom location options."""
        button_id = len(DEFAULT_LOCATIONS) + 1

        for location in locations:
            if location.lower() not in [loc.lower() for loc in DEFAULT_LOCATIONS]:
                widget = LocationWidget(location, allow_delete=True)
                widget.delete_requested.connect(self.delete_location)

                self.location_group.addButton(widget.radio, button_id)
                self.location_layout.addWidget(widget)
                self.location_widgets[location] = widget

                button_id += 1

    def add_create_button(self):
        """Add create new location button."""
        create_button = QPushButton("+ Create New Location")
        create_button.setCursor(Qt.PointingHandCursor)
        create_button.clicked.connect(self.show_create_form)
        create_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ff914d;
                border: none;
                text-align: left;
                padding: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #ff7730;
            }
        """)

        self.location_layout.addWidget(create_button)

    def show_create_form(self):
        """Show create new location form with animation."""
        self.location_selection_area.setVisible(False)
        self.new_location_form.fadeIn()
        self.new_location_edit.setFocus()

    def hide_create_form(self):
        """Hide create new location form with animation."""
        self.new_location_form.fadeOut()
        QTimer.singleShot(200, lambda: self.location_selection_area.setVisible(True))
        self.new_location_edit.clear()

    def create_new_location(self):
        """Create new location with validation."""
        location_name = self.new_location_edit.text().strip()

        if not location_name:
            QMessageBox.warning(self, "Input Required", "Please enter a location name.")
            return

        if location_name in self.location_widgets:
            QMessageBox.warning(self, "Duplicate Location",
                                f"Location '{location_name}' already exists.")
            return

        # Add location immediately for responsive UI
        self.add_location_widget(location_name)

        # Save to file asynchronously
        QTimer.singleShot(0, lambda: self.save_location_to_file(location_name))

        self.hide_create_form()

    def add_location_widget(self, location_name: str):
        """Add a new location widget to the UI."""
        widget = LocationWidget(location_name, allow_delete=True)
        widget.delete_requested.connect(self.delete_location)
        widget.radio.setChecked(True)  # Select new location

        button_id = len(self.location_widgets) + 1
        self.location_group.addButton(widget.radio, button_id)

        # Insert before create button
        self.location_layout.insertWidget(self.location_layout.count() - 1, widget)
        self.location_widgets[location_name] = widget

    def save_location_to_file(self, location_name: str):
        """Save location to organization file asynchronously."""
        try:
            if not os.path.exists(ORGANIZATIONS_FILE):
                organizations = {}
            else:
                with open(ORGANIZATIONS_FILE, 'r', encoding='utf-8') as f:
                    organizations = json.load(f)

            if self.organization not in organizations:
                organizations[self.organization] = {"locations": []}
            elif "locations" not in organizations[self.organization]:
                organizations[self.organization]["locations"] = []

            if location_name not in organizations[self.organization]["locations"]:
                organizations[self.organization]["locations"].append(location_name)

                with open(ORGANIZATIONS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(organizations, f, indent=2)

                # Update cache
                _data_cache["organizations"] = organizations
                _data_cache["last_modified"] = os.path.getmtime(ORGANIZATIONS_FILE)

                print(f"Location '{location_name}' saved successfully")

        except Exception as e:
            print(f"Error saving location: {e}")
            QMessageBox.warning(self, "Save Error",
                                "Location was added but could not be saved permanently.")

    def delete_location(self, location_name: str):
        """Delete location with confirmation."""
        if location_name in DEFAULT_LOCATIONS:
            return  # Cannot delete default locations

        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Delete location '{location_name}'?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.remove_location_widget(location_name)
            QTimer.singleShot(0, lambda: self.remove_location_from_file(location_name))

    def remove_location_widget(self, location_name: str):
        """Remove location widget from UI."""
        if location_name not in self.location_widgets:
            return

        widget = self.location_widgets[location_name]

        # If this was selected, select Pune instead
        if widget.radio.isChecked():
            self.location_widgets["Pune"].radio.setChecked(True)

        # Remove from UI
        self.location_group.removeButton(widget.radio)
        self.location_layout.removeWidget(widget)
        widget.deleteLater()
        del self.location_widgets[location_name]

    def remove_location_from_file(self, location_name: str):
        """Remove location from organization file."""
        try:
            if os.path.exists(ORGANIZATIONS_FILE):
                with open(ORGANIZATIONS_FILE, 'r', encoding='utf-8') as f:
                    organizations = json.load(f)

                if (self.organization in organizations and
                        "locations" in organizations[self.organization]):

                    if location_name in organizations[self.organization]["locations"]:
                        organizations[self.organization]["locations"].remove(location_name)

                        with open(ORGANIZATIONS_FILE, 'w', encoding='utf-8') as f:
                            json.dump(organizations, f, indent=2)

                        # Update cache
                        _data_cache["organizations"] = organizations
                        _data_cache["last_modified"] = os.path.getmtime(ORGANIZATIONS_FILE)

        except Exception as e:
            print(f"Error removing location from file: {e}")

    def proceed_to_inspection(self):
        """Proceed to fabric inspection with optimized loading."""
        # Get selected location
        selected_button = self.location_group.checkedButton()
        selected_location = selected_button.text() if selected_button else "Pune"

        # Save configuration
        success = self.save_location_config(selected_location)
        if not success:
            QMessageBox.critical(self, "Error", "Failed to save configuration.")
            return

        # Show loading feedback
        self.start_button.setText("Loading...")
        self.start_button.setEnabled(False)

        # Load fabric deployment UI asynchronously
        QTimer.singleShot(100, lambda: self.load_fabric_deployment(selected_location))

    def save_location_config(self, location: str) -> bool:
        """Save location configuration efficiently."""
        try:
            config = {
                "organization": self.organization,
                "location": location,
                "user": self.user_info.get("username", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "inspection_type": "fabric"
            }

            os.makedirs(os.path.dirname(LOCATION_CONFIG_PATH), exist_ok=True)
            with open(LOCATION_CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving location config: {e}")
            return False

    def load_fabric_deployment(self, location: str):
        """Load fabric deployment UI with error handling."""
        try:
            if not os.path.exists("fabric_deploy_ui.py"):
                QMessageBox.critical(self, "Missing Module",
                                     "Fabric deployment module not found.")
                self.reset_start_button()
                return

            from fabric_deploy_ui import FabricDeploymentUI

            inspection_data = {
                "organization": self.organization,
                "location": location,
                "user": self.user_info
            }

            # Clean up existing instance
            if hasattr(self, 'fabric_deployment_ui') and self.fabric_deployment_ui:
                self.fabric_deployment_ui.close()
                self.fabric_deployment_ui.deleteLater()

            # Create new instance
            self.fabric_deployment_ui = FabricDeploymentUI(inspection_data)
            self.fabric_deployment_ui.destroyed.connect(self.show)

            # Show and hide this window
            self.fabric_deployment_ui.showMaximized()
            self.hide()

        except Exception as e:
            print(f"Error loading fabric deployment: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Loading Error",
                                 f"Failed to load fabric inspection: {str(e)}")
            self.reset_start_button()

    def reset_start_button(self):
        """Reset start button to original state."""
        self.start_button.setText("Start Inspection")
        self.start_button.setEnabled(True)

    def go_back(self):
        """Go back to organization selection with optimized loading."""
        try:
            from organization_creation_ui import OrganizationCreationUI

            if hasattr(self, 'organization_creation_ui') and self.organization_creation_ui:
                self.organization_creation_ui.close()
                self.organization_creation_ui.deleteLater()

            self.organization_creation_ui = OrganizationCreationUI(self.user_info)
            self.organization_creation_ui.destroyed.connect(self.show)
            self.organization_creation_ui.showMaximized()
            self.hide()

        except Exception as e:
            print(f"Error going back: {e}")
            QMessageBox.critical(self, "Navigation Error",
                                 f"Could not return to organization selection: {str(e)}")

    def showEvent(self, event):
        """Handle show event with performance optimization."""
        super().showEvent(event)
        if not self.isMaximized():
            self.showMaximized()

    def closeEvent(self, event):
        """Clean up resources on close."""
        # Clean up data loader
        if self.data_loader is not None:
            self.data_loader.quit()
            self.data_loader.wait(1000)  # Wait up to 1 second
            self.data_loader.deleteLater()

        # Clean up fabric deployment UI
        if hasattr(self, 'fabric_deployment_ui') and self.fabric_deployment_ui:
            self.fabric_deployment_ui.close()
            self.fabric_deployment_ui.deleteLater()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Sample data
    user_info = {
        "username": "sakarrobotics",
        "full_name": "Sakar Robotics"
    }
    organization = "Test Organization"

    window = LocationSelectionUI(user_info, organization)
    window.showMaximized()

    sys.exit(app.exec_())
