#!/usr/bin/env python3
"""
organization_creation_ui.py - Organization creation interface for Fabric Inspection

This module provides a screen for creating or selecting an organization
before proceeding with fabric inspection.
"""

import json
import os
import sys
import traceback
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor
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

# Path to save organization data
ORGANIZATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "organizations.json")

# Check if location_selection_ui.py exists before importing
try:
    if os.path.exists("location_selection_ui.py"):
        print("location_selection_ui.py found in current directory")
    else:
        print("WARNING: location_selection_ui.py not found in current directory!")
except Exception as e:
    print(f"Error checking for location_selection_ui.py: {e}")


class CustomTitleBar(QFrame):
    def __init__(self, parent=None):
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
        logo_path = "logo.jpeg"  # Update with actual logo path
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_pixmap = logo_pixmap.scaledToHeight(45, Qt.SmoothTransformation)
            self.logo_label.setPixmap(logo_pixmap)
        else:
            # Fallback if logo not found
            self.logo_label.setText("SAKAR")
            self.logo_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")

        # Title label
        self.title_label = QLabel("FABRIC INSPECTION - ORGANIZATION SELECTION")
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

        self.setLayout(layout)


class SearchLineEdit(QLineEdit):
    """Custom styled search line edit."""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(45)
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid rgba(200, 200, 200, 0.8);
                border-radius: 20px;
                padding: 10px 15px 10px 40px;
                background-color: rgba(255, 255, 255, 0.8);
                selection-background-color: #ff914d;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #ff914d;
                background-color: white;
            }
        """)

        # Add search icon if available
        search_icon_path = "search_icon.png"  # Replace with actual path
        if os.path.exists(search_icon_path):
            self.addAction(QIcon(search_icon_path), QLineEdit.LeadingPosition)
        else:
            # Add text margin for search icon placeholder
            self.setTextMargins(25, 0, 0, 0)


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
                background-color: #f44336;
                color: white;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
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
                color: #4CAF50;
                border: none;
                text-align: left;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #45a049;
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


class OrganizationCreationUI(QWidget):
    """
    Interface for creating or selecting an organization for Fabric Inspection.
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

        # Initialize UI
        self.init_ui()

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12pt;
                background-color: #f7f7f7;
            }
            QLabel#welcomeLabel {
                font-size: 18pt;
                font-weight: bold;
                color: #333333;
            }
            QLabel#instructionLabel {
                font-size: 14pt;
                color: #555555;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
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

        # Load existing organizations
        self.load_organizations()

        # Print debug information
        print(f"OrganizationCreationUI initialized for user: {self.user_info.get('username')}")

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)

        # Content layout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)

        # Welcome message
        welcome_label = QLabel(f"Welcome, {self.user_info.get('full_name', 'User')}!")
        welcome_label.setObjectName("welcomeLabel")

        # Instruction
        instruction_label = QLabel("Please select or create an organization to continue")
        instruction_label.setObjectName("instructionLabel")

        # Organization card
        self.org_card = OrganizationCard()
        card_layout = QVBoxLayout(self.org_card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)

        # Search bar
        search_layout = QHBoxLayout()
        self.search_edit = SearchLineEdit(placeholder="Search organizations...")
        self.search_edit.textChanged.connect(self.filter_organizations)
        search_layout.addWidget(self.search_edit)

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

        # Add default "Railway" option
        railway_widget = OrganizationWidget("Railway", allow_delete=False)
        railway_widget.radio.setChecked(True)  # Default selection
        self.org_radio_group.addButton(railway_widget.radio, 0)
        self.org_layout.addWidget(railway_widget)
        self.org_widgets["Railway"] = railway_widget

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
        card_layout.addLayout(search_layout)
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

        # Add content widget to main layout
        main_layout.addWidget(content_widget)

    def load_organizations(self):
        """Load existing organizations from file and populate the UI."""
        try:
            if os.path.exists(ORGANIZATIONS_FILE):
                with open(ORGANIZATIONS_FILE, 'r') as f:
                    organizations = json.load(f)

                # Add organizations (skip Railway as it's already added by default)
                for i, org_name in enumerate(organizations.keys(), start=1):
                    if org_name.lower() != "railway":
                        # Create organization widget with radio button and delete button
                        org_widget = OrganizationWidget(org_name)
                        self.org_radio_group.addButton(org_widget.radio, i)

                        # Connect delete button
                        org_widget.delete_button.clicked.connect(
                            lambda checked=False, name=org_name: self.delete_organization(name)
                        )

                        # Add to UI
                        self.org_layout.insertWidget(self.org_layout.count() - 1, org_widget)
                        self.org_widgets[org_name] = org_widget

                print(f"Loaded {len(organizations)} organizations")
            else:
                print("No organizations file found, only default options shown")

        except Exception as e:
            print(f"Error loading organizations: {e}")
            traceback.print_exc()

    def filter_organizations(self, text):
        """Filter organization list based on search text."""
        search_text = text.lower()

        # Show/hide organization widgets based on search
        for name, widget in self.org_widgets.items():
            if search_text in name.lower():
                widget.setVisible(True)
            else:
                widget.setVisible(False)

    def delete_organization(self, org_name):
        """Delete an organization from the list and file."""
        try:
            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete '{org_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return False

            # Load organizations
            if os.path.exists(ORGANIZATIONS_FILE):
                with open(ORGANIZATIONS_FILE, 'r') as f:
                    organizations = json.load(f)

                # Remove the organization
                if org_name in organizations:
                    del organizations[org_name]

                    # Save updated file
                    with open(ORGANIZATIONS_FILE, 'w') as f:
                        json.dump(organizations, f, indent=4)

                    # Update UI - remove the widget
                    if org_name in self.org_widgets:
                        widget = self.org_widgets[org_name]

                        # If this radio button was checked, select Railway instead
                        if widget.radio.isChecked():
                            self.org_widgets["Railway"].radio.setChecked(True)

                        # Remove from radio group
                        self.org_radio_group.removeButton(widget.radio)

                        # Remove from layout
                        self.org_layout.removeWidget(widget)

                        # Delete the widget
                        widget.deleteLater()

                        # Remove from dictionary
                        del self.org_widgets[org_name]

                    QMessageBox.information(self, "Success", f"Organization '{org_name}' deleted successfully")
                    print(f"Organization '{org_name}' deleted successfully")
                    return True

            return False
        except Exception as e:
            print(f"Error deleting organization: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to delete organization: {str(e)}")
            return False

    def show_create_new_form(self):
        """Show the form for creating a new organization."""
        self.creating_new_org = True
        self.org_scroll_area.setVisible(False)
        self.new_org_form.setVisible(True)
        self.new_org_edit.setFocus()

    def hide_create_new_form(self):
        """Hide the form for creating a new organization."""
        self.creating_new_org = False
        self.new_org_form.setVisible(False)
        self.org_scroll_area.setVisible(True)
        self.new_org_edit.clear()

    def create_new_organization(self):
        """Create a new organization and add it to the list."""
        org_name = self.new_org_edit.text().strip()

        if not org_name:
            QMessageBox.warning(self, "Input Required", "Please enter an organization name.")
            return

        # Check if organization already exists
        if org_name in self.org_widgets:
            QMessageBox.warning(self, "Duplicate Organization",
                                f"Organization '{org_name}' already exists.")
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

        # Save to file
        self.save_organization(org_name)

        # Hide the creation form
        self.hide_create_new_form()

    def save_organization(self, org_name=None):
        """Save the selected organization to configuration file."""
        if org_name is None:
            # Get the selected organization
            selected_button = self.org_radio_group.checkedButton()
            if selected_button is None:
                return False

            org_name = selected_button.text()

        if not org_name:
            return False

        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(ORGANIZATIONS_FILE)
            if not os.path.exists(directory) and directory:
                os.makedirs(directory)

            # Load existing organizations or create new file
            organizations = {}
            if os.path.exists(ORGANIZATIONS_FILE):
                try:
                    with open(ORGANIZATIONS_FILE, 'r') as f:
                        organizations = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error loading {ORGANIZATIONS_FILE}, creating new file")
                    organizations = {}

            # Add new organization if it doesn't exist
            if org_name not in organizations:
                organizations[org_name] = {
                    "created_by": self.user_info.get("username", "unknown"),
                    "creation_date": datetime.now().isoformat(),
                    "locations": []
                }

            # Save to file
            with open(ORGANIZATIONS_FILE, 'w') as f:
                json.dump(organizations, f, indent=4)

            print(f"Organization '{org_name}' saved to {ORGANIZATIONS_FILE}")
            return True
        except Exception as e:
            print(f"Error saving organization: {e}")
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

        # Now try to launch the location selection UI
        try:
            # First check if the module exists
            if not os.path.exists("location_selection_ui.py"):
                msg = "The location selection module (location_selection_ui.py) is missing. Please check your installation."
                print(msg)
                QMessageBox.critical(self, "Missing Module", msg)
                return

            print("Importing LocationSelectionUI...")
            # Try to import the LocationSelectionUI
            try:
                from location_selection_ui import LocationSelectionUI
            except ImportError as e:
                print(f"Error importing LocationSelectionUI: {e}")
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

            print("Creating LocationSelectionUI instance...")
            # Close any existing instance before creating a new one
            if hasattr(self, 'location_selection_ui') and self.location_selection_ui is not None:
                self.location_selection_ui.close()
                self.location_selection_ui.deleteLater()

            # Create the LocationSelectionUI instance
            self.location_selection_ui = LocationSelectionUI(
                user_info=self.user_info,
                organization=org_name
            )

            # Connect destroyed signal to show this window again if the location screen is closed
            self.location_selection_ui.destroyed.connect(self.show)

            print("Showing LocationSelectionUI...")
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
            self.login_ui.is_fabric_inspection = True

            # Connect destroyed signal to show this window again if login screen is closed
            self.login_ui.destroyed.connect(self.show)

            # Show login UI
            self.login_ui.showMaximized()

            # Hide this window
            self.hide()

        except Exception as e:
            print("Error going back to login:")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not go back to login screen: {str(e)})")

    def closeEvent(self, event):
        """Handle the window close event properly to ensure application terminates."""
        print("OrganizationCreationUI closing, terminating application")
        # Make sure to terminate the application if this window is closed
        QApplication.quit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Print current directory and check for login_ui.py
    print(f"Current directory: {os.getcwd()}")
    if os.path.exists("location_selection_ui.py"):
        print("location_selection_ui.py found in current directory")
    else:
        print("WARNING: location_selection_ui.py not found in current directory!")

    # Sample user info
    user_info = {
        "username": "sakarrobotics",
        "full_name": "Sakar Robotics"
    }

    # Create and show window
    window = OrganizationCreationUI(user_info)
    window.showMaximized()

    sys.exit(app.exec_())
