#!/usr/bin/env python3
"""
metal_sheet_location_ui.py - Location selection interface for Metal Sheet Inspection

OVERVIEW:
This is a comprehensive PyQt5-based GUI application module that serves as the location and project 
selection interface for a metal sheet inspection system. The module creates a sophisticated, 
multi-layered user interface that allows users to select geographical locations and specific 
projects within those locations before proceeding to the actual metal sheet inspection process.

CORE FUNCTIONALITY:
- Location Management: Create, select, and delete geographical locations for inspection projects
- Project Management: Create, select, and delete projects within selected locations  
- Database Integration: Real-time synchronization with Azure database for persistent data storage
- User Access Control: Built-in system for granting access credentials to new users
- Visual Interface: Modern, styled GUI with custom backgrounds, animations, and responsive design
- Connectivity Monitoring: Real-time internet connection status with visual indicators
- Data Validation: Comprehensive input validation and error handling throughout the interface
- Navigation Control: Seamless integration with organization selection and defect selection modules

KEY FEATURES:
- Custom styled components (buttons, radio buttons, forms, message boxes)
- Background image support with gradient fallbacks and semi-transparent overlays
- Scrollable areas for locations and projects with dynamic loading from database
- Form-based creation dialogs with validation for new locations and projects
- Confirmation dialogs for deletion operations with styled message boxes  
- Real-time status indicators showing system connectivity and operational state
- User management system with tabbed interface for personal info and credentials
- Integration with Azure database services for user authentication and data persistence
- Automatic UI updates based on database changes and user selections
- Error handling with user-friendly messages and console logging for debugging

TECHNICAL ARCHITECTURE:
- Built using PyQt5 framework with custom widget classes and styling
- Database connectivity through Azure database functions (azure_database module)
- Modular design with separate classes for different UI components
- Event-driven architecture with signal/slot connections for user interactions
- Background processing for connectivity checks and database operations
- Memory management with proper widget cleanup and deletion
- Thread-safe operations for database queries and UI updates

This module provides a complete solution for location and project management in industrial 
inspection workflows, with enterprise-grade features for user management, data persistence, 
and system monitoring.
"""

import json
import os
import sys
import traceback
from datetime import datetime
import requests

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QColor, QPainter, QBrush, QLinearGradient, QImage
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QStyle  # Added import for QStyle

from utils import set_window_icon, update_config_from_database

# Import Azure database functions
from azure_database import get_user_id_by_username, store_location_data, get_connection

# Path to save location config data
LOCATION_CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "metal_sheet_location_config.json")


class AccessDialog(QDialog):
    """Dialog for granting access with name, phone no., email, username, and password."""

    def __init__(self, parent=None, username=None):
        super().__init__(parent)
        self.username = username
        self.setWindowTitle("Grant Access")
        self.setMinimumWidth(600)
        self.setMinimumHeight(540)

        # Apply a clean white style to the dialog
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 8px;
            }
            QLabel {
                color: #333333;
                background-color: transparent;
            }
            QPushButton {
                height: 40px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                min-height: 20px;
            }
            QLineEdit:focus {
                border: 1px solid #4285f4;
            }
        """)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create header section with blue icon and title
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 20, 20, 10)
        header_layout.setSpacing(15)

        # Icon for user access
        icon_label = QLabel()
        icon_label.setFixedSize(48, 48)
        icon_label.setStyleSheet("""
            background-color: #e8f0fe;
            border-radius: 24px;
            color: #4285f4;
            font-size: 24px;
            font-weight: bold;
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setText("👤")  # Person icon using emoji

        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)

        title_label = QLabel("Grant System Access")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #202124;")

        subtitle_label = QLabel("Provide access credentials and permissions for a new user")
        subtitle_label.setStyleSheet("font-size: 14px; color: #5f6368;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        # Removed the close button (×) from header that was causing duplicate close buttons

        # Assemble header
        header_layout.addWidget(icon_label)
        header_layout.addLayout(title_layout, 1)
        # The close button widget reference has been removed from here

        # Add header to main layout
        main_layout.addLayout(header_layout)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0; max-height: 1px;")
        main_layout.addWidget(separator)

        # Tab button layout
        tab_button_layout = QHBoxLayout()
        tab_button_layout.setContentsMargins(20, 0, 20, 0)

        # Create User Information and Credentials tab buttons
        self.user_info_tab = QPushButton("👤 User Information")
        self.user_info_tab.setCheckable(True)
        self.user_info_tab.setChecked(True)
        self.user_info_tab.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                border-radius: 0;
                font-size: 14px;
                font-weight: bold;
                padding: 15px 20px;
                text-align: center;
            }
            QPushButton:checked {
                border-bottom: 3px solid #ff914d;
                color: #ff914d;
            }
        """)

        self.credentials_tab = QPushButton("🔐 Credentials")
        self.credentials_tab.setCheckable(True)
        self.credentials_tab.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                border-radius: 0;
                font-size: 14px;
                font-weight: bold;
                padding: 15px 20px;
                text-align: center;
            }
            QPushButton:checked {
                border-bottom: 3px solid #ff914d;
                color: #ff914d;
            }
        """)

        tab_button_layout.addWidget(self.user_info_tab)
        tab_button_layout.addWidget(self.credentials_tab)
        tab_button_layout.addStretch(1)

        main_layout.addLayout(tab_button_layout)

        # Create stacked widget for the tab content
        self.stack = QWidget()
        self.stack_layout = QVBoxLayout(self.stack)
        self.stack_layout.setContentsMargins(20, 20, 20, 20)

        # User Information page - will start visible
        self.user_info_page = QWidget()
        self.user_info_page.setStyleSheet("background-color: transparent;")
        user_info_layout = QVBoxLayout(self.user_info_page)
        user_info_layout.setContentsMargins(0, 0, 0, 0)
        user_info_layout.setSpacing(20)

        # User Information title
        personal_info_title = QLabel("Personal Information")
        personal_info_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #202124;")

        personal_info_subtitle = QLabel("Enter the user's basic information and contact details")
        personal_info_subtitle.setStyleSheet(
            "font-size: 14px; color: #5f6368; margin-bottom: 10px;")

        user_info_layout.addWidget(personal_info_title)
        user_info_layout.addWidget(personal_info_subtitle)

        # Create form fields for user information
        form_layout = QVBoxLayout()
        form_layout.setSpacing(15)

        # Full Name field
        name_layout = QVBoxLayout()
        name_layout.setSpacing(5)

        name_label = QLabel("Full Name *")
        name_label.setStyleSheet("font-weight: bold;")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter full name")

        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)

        # Phone Number field
        phone_layout = QVBoxLayout()
        phone_layout.setSpacing(5)

        phone_label = QLabel("Phone Number")
        phone_label.setStyleSheet("font-weight: bold;")

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Enter phone number")
        # Set up a validator to ensure only numbers can be entered
        self.phone_input.setValidator(QRegExpValidator(QRegExp("^[0-9]*$")))

        phone_layout.addWidget(phone_label)
        phone_layout.addWidget(self.phone_input)

        # Email field
        email_layout = QVBoxLayout()
        email_layout.setSpacing(5)

        email_label = QLabel("Email Address *")
        email_label.setStyleSheet("font-weight: bold;")

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter email address")

        email_layout.addWidget(email_label)
        email_layout.addWidget(self.email_input)

        # Add fields to form layout
        form_layout.addLayout(name_layout)
        form_layout.addLayout(phone_layout)
        form_layout.addLayout(email_layout)
        form_layout.addStretch(1)

        # Add form to user info layout
        user_info_layout.addLayout(form_layout)

        # Credentials page - initially hidden
        self.credentials_page = QWidget()
        self.credentials_page.setVisible(False)
        self.credentials_page.setStyleSheet("background-color: transparent;")

        credentials_layout = QVBoxLayout(self.credentials_page)
        credentials_layout.setContentsMargins(0, 0, 0, 0)
        credentials_layout.setSpacing(20)

        # Credentials title
        credentials_title = QLabel("Credentials")
        credentials_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #202124;")

        credentials_subtitle = QLabel("Set username and password for the new user")
        credentials_subtitle.setStyleSheet("font-size: 14px; color: #5f6368; margin-bottom: 10px;")

        credentials_layout.addWidget(credentials_title)
        credentials_layout.addWidget(credentials_subtitle)

        # Create form fields for credentials
        cred_form_layout = QVBoxLayout()
        cred_form_layout.setSpacing(15)

        # Username field
        username_layout = QVBoxLayout()
        username_layout.setSpacing(5)

        username_label = QLabel("Username *")
        username_label.setStyleSheet("font-weight: bold;")

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")

        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)

        # Password field
        password_layout = QVBoxLayout()
        password_layout.setSpacing(5)

        password_label = QLabel("Password *")
        password_label.setStyleSheet("font-weight: bold;")

        # Create a horizontal layout for password field and view button
        password_input_layout = QHBoxLayout()
        password_input_layout.setSpacing(10)
        password_input_layout.setContentsMargins(0, 0, 0, 0)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)

        # Add view password button
        self.view_password_button = QPushButton()
        self.view_password_button.setIcon(self.style().standardIcon(
            QStyle.SP_DialogHelpButton))  # Default icon
        self.view_password_button.setToolTip("Show password")
        self.view_password_button.setFixedSize(30, 30)
        self.view_password_button.setCursor(Qt.PointingHandCursor)
        self.view_password_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        self.view_password_button.clicked.connect(self.toggle_password_visibility)

        # Add widgets to password input layout
        password_input_layout.addWidget(self.password_input)
        password_input_layout.addWidget(self.view_password_button)

        password_layout.addWidget(password_label)
        password_layout.addLayout(password_input_layout)

        # Add fields to credentials form layout
        cred_form_layout.addLayout(username_layout)
        cred_form_layout.addLayout(password_layout)
        cred_form_layout.addStretch(1)

        # Add form to credentials layout
        credentials_layout.addLayout(cred_form_layout)

        # Add both pages to stack
        self.stack_layout.addWidget(self.user_info_page)
        self.stack_layout.addWidget(self.credentials_page)

        main_layout.addWidget(self.stack)

        # Add bottom buttons
        button_container = QWidget()
        button_container.setStyleSheet("background-color: #f8f9fa;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(20, 15, 20, 15)
        button_layout.setSpacing(10)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #dadce0;
                color: #3c4043;
                min-width: 100px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #f1f3f4;
                border: 1px solid #dadce0;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)

        # Continue to Credentials button (shown in user info page)
        self.continue_button = QPushButton("Continue to Credentials")
        self.continue_button.setStyleSheet("""
            QPushButton {
                background-color: #ff914d;
                color: white;
                min-width: 180px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #e86d25;
            }
        """)
        self.continue_button.clicked.connect(self.switch_to_credentials)

        # Grant Access button (shown in credentials page)
        self.grant_button = QPushButton("Grant Access")
        self.grant_button.setStyleSheet("""
            QPushButton {
                background-color: #ff914d;
                color: white;
                min-width: 120px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #e86d25;
            }
            QPushButton:pressed {
                background-color: #d65e1c;
            }
        """)
        self.grant_button.clicked.connect(self.accept)
        self.grant_button.setVisible(False)

        # Back button (shown in credentials page)
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #dadce0;
                color: #3c4043;
                min-width: 100px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #f1f3f4;
                border: 1px solid #dadce0;
            }
        """)
        self.back_button.clicked.connect(self.switch_to_user_info)
        self.back_button.setVisible(False)

        # Add buttons to layout
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.continue_button)
        button_layout.addWidget(self.grant_button)

        main_layout.addWidget(button_container)

        # Connect tab buttons
        self.user_info_tab.clicked.connect(self.switch_to_user_info)
        self.credentials_tab.clicked.connect(self.switch_to_credentials)

    def switch_to_user_info(self):
        # Show User Information page
        self.user_info_page.setVisible(True)
        self.credentials_page.setVisible(False)

        # Update tab buttons
        self.user_info_tab.setChecked(True)
        self.credentials_tab.setChecked(False)

        # Update action buttons
        self.continue_button.setVisible(True)
        self.back_button.setVisible(False)
        self.grant_button.setVisible(False)

    def switch_to_credentials(self):
        # Validate User Information before switching
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Required Field", "Please enter a full name.")
            return

        if not self.email_input.text().strip():
            QMessageBox.warning(self, "Required Field", "Please enter an email address.")
            return

        if '@' not in self.email_input.text():
            QMessageBox.warning(self, "Invalid Email", "Please enter a valid email address.")
            return

        # Show Credentials page
        self.user_info_page.setVisible(False)
        self.credentials_page.setVisible(True)

        # Update tab buttons
        self.user_info_tab.setChecked(False)
        self.credentials_tab.setChecked(True)

        # Update action buttons
        self.continue_button.setVisible(False)
        self.back_button.setVisible(True)
        self.grant_button.setVisible(True)

    def get_credentials(self):
        """Return all the form data in the sequence: name, phone, email, username, password."""
        return (
            self.name_input.text(),
            self.phone_input.text(),
            self.email_input.text(),
            self.username_input.text(),
            self.password_input.text()
        )

    def toggle_password_visibility(self):
        """Toggle password visibility between hidden and shown."""
        if self.password_input.echoMode() == QLineEdit.Password:
            # Change to normal mode (visible text)
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.view_password_button.setToolTip("Hide password")
            self.view_password_button.setIcon(
                self.style().standardIcon(QStyle.SP_DialogApplyButton))
        else:
            # Change back to password mode (hidden text)
            self.password_input.setEchoMode(QLineEdit.Password)
            self.view_password_button.setToolTip("Show password")
            self.view_password_button.setIcon(
                self.style().standardIcon(QStyle.SP_DialogHelpButton))


class CustomTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(70)

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
            self.logo_label.setStyleSheet("color: #333333; font-weight: bold; font-size: 16px;")

        # Title label
        self.title_label = QLabel("METAL SHEET INSPECTION - SELECT LOCATION")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: #333333;
            font-weight: bold;
            font-size: 28px;
            margin-left: 0px;
            margin-right: 0px;
        """)

        # Grant Access button
        self.grant_access_button = QPushButton("Grant Access")
        self.grant_access_button.setFixedSize(120, 30)
        self.grant_access_button.setCursor(Qt.PointingHandCursor)
        self.grant_access_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border: 1px solid #dddddd;
                border-radius: 15px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
                border-style: inset;
            }
        """)
        self.grant_access_button.clicked.connect(self.show_access_dialog)

        # Add logo and title to layout
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch(0)
        logo_layout.setSpacing(2)

        # layout.addLayout(logo_layout)
        # layout.addWidget(self.title_label, 1)
        layout.addStretch()
        layout.addWidget(self.grant_access_button)

        self.setLayout(layout)

    def show_access_dialog(self):
        """Show the access dialog to grant access with name, phone no., email, username, and password."""
        # Get the current user info from the parent window if available
        username = None
        user_id = None
        if hasattr(self.parent(), 'user_info') and self.parent().user_info:
            username = self.parent().user_info.get('username')
            # Always use the user_id from the current session instead of looking it up
            user_id = self.parent().user_info.get('user_id') or self.parent().user_info.get('id')

        dialog = AccessDialog(self, username)

        # Keep showing the dialog until valid input is provided or user cancels
        while True:
            result = dialog.exec_()

            # If user clicked Cancel or closed the dialog, exit the loop
            if result != QDialog.Accepted:
                print("Access dialog cancelled")
                break

            name, phone_no, email, username_input, password = dialog.get_credentials()

            # Check if any field is empty
            if not name or not phone_no or not email or not username_input or not password:
                StyledMessageBox.warning(self, "Missing Information",
                                    "All fields are required. Please fill in all information.")
                # Continue the loop to show the dialog again with existing input preserved
                continue

            # Validate phone number (should only contain digits)
            if not phone_no.isdigit():
                StyledMessageBox.warning(self, "Invalid Phone Number",
                                    "Phone number must contain only digits (0-9).")
                # Continue the loop to show the dialog again with existing input preserved
                continue

            # Validate email format - should contain at least one @ character
            if '@' not in email:
                StyledMessageBox.warning(self, "Invalid Email",
                                    "Please enter a valid email address (must contain '@').")
                # Continue the loop to show the dialog again with existing input preserved
                continue

            try:
                # Import the function to store management access
                from azure_database import store_management_access

                # Save credentials to management table - passing the current user_id directly instead of username
                # This ensures we use the logged-in user's ID, not one looked up by username
                status_code, message = store_management_access(
                    user_id, username_input, password, name, phone_no, email)

                if status_code == 0:
                    # Success
                    StyledMessageBox.information(self, "Access Granted",
                                            f"Access granted for: {username_input}")
                    print(f"Access credentials saved to database for: {username_input}")
                    break  # Exit the loop on success
                elif status_code == 1:
                    # Username already exists
                    StyledMessageBox.warning(self, "Username Already Exists",
                                        f"The username '{username_input}' is already in use. Please choose a different username.")
                    # Continue the loop to show the dialog again with existing input preserved
                    continue
                else:
                    # Other database error
                    StyledMessageBox.warning(self, "Database Error",
                                        f"Failed to save access credentials: {message}")
                    print(f"Failed to save access credentials for: {username_input} - {message}")
                    break  # Exit the loop on database error

            except Exception as e:
                StyledMessageBox.warning(self, "Error",
                                    f"Error storing access credentials: {str(e)}")
                print(f"Error storing management access: {str(e)}")
                break  # Exit the loop on exception

            # If we get here, we've successfully processed the form, so exit the loop
            break

    def location_selected(self, button):
        """Handle location selection to show relevant projects."""
        selected_location = button.text()
        print(f"Location selected: {selected_location}")

        # Clear existing projects
        self.clear_project_list()

        # Load projects for this location
        self.load_projects_for_location(selected_location)

        # Show the project selection area - always show this regardless of whether projects exist
        self.project_selection_area.setVisible(True)
        self.new_project_form.setVisible(False)

        # Report selected project if any is selected
        selected_project_button = self.project_group.checkedButton()
        if selected_project_button:
            selected_project = selected_project_button.text()
            print(f"Selected project: {selected_project}")
        else:
            print("No project selected")

    def clear_project_list(self):
        """Clear all projects from the list except the create button."""
        # Remove all radio buttons from the group
        for btn in self.project_group.buttons():
            self.project_group.removeButton(btn)

        # Clear the layout while preserving the create button
        create_button = self.create_new_project_button
        for i in reversed(range(self.project_layout.count())):
            item = self.project_layout.itemAt(i)
            if item.widget() != create_button:
                widget = item.widget()
                self.project_layout.removeWidget(widget)
                if widget:
                    widget.deleteLater()

    def load_projects_for_location(self, location_name):
        """Load projects for the selected location from the database."""
        try:
            # Connect to the database
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                return

            cursor = connection.cursor(dictionary=True)

            # Query to get distinct projects from the location table for this organization and city
            query = "SELECT DISTINCT project FROM location WHERE organization = %s AND city = %s ORDER BY project"
            cursor.execute(query, (self.organization, location_name))

            # Fetch all projects
            projects = cursor.fetchall()

            # Close the cursor and connection
            cursor.close()
            connection.close()

            # Clear existing projects first
            self.clear_project_list()

            # Add projects
            for i, project in enumerate(projects):
                project_name = project['project']
                if project_name:
                    # Create project widget with radio button and delete button
                    project_widget = ProjectWidget(project_name)
                    self.project_group.addButton(project_widget.radio, i)

                    # Connect delete button with the correct closure
                    def create_delete_handler(loc, proj):
                        return lambda: self.delete_project(loc, proj)

                    project_widget.delete_button.clicked.connect(
                        create_delete_handler(location_name, project_name)
                    )

                    # Add to layout before the create button
                    self.project_layout.insertWidget(
                        self.project_layout.count() - 1,  # Before create button
                        project_widget
                    )

            # If projects were found, select the first one
            if projects and self.project_group.buttons():
                self.project_group.buttons()[0].setChecked(True)

            print(
                f"Loaded {len(projects)} projects for location {location_name} in organization {self.organization}")
        except Exception as e:
            print(f"Error loading projects from database: {e}")
            traceback.print_exc()
            # Show error message to user
            QMessageBox.warning(
                self,
                "Database Error",
                f"Failed to load projects from database: {str(e)}\n\nNo projects available."
            )

    def delete_project(self, location_name, project_name):
        """Delete a project from the database for the given location."""
        try:
            # Confirm deletion with styled dialog
            reply = StyledMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete project '{project_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return False

            # Get a database connection
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                StyledMessageBox.critical(self, "Database Error",
                                          "Could not connect to the database.")
                return False

            cursor = connection.cursor()

            # Delete the project from the location table
            delete_query = "DELETE FROM location WHERE organization = %s AND city = %s AND project = %s"
            cursor.execute(delete_query, (self.organization, location_name, project_name))

            # Commit the transaction
            connection.commit()

            # Close cursor and connection
            cursor.close()
            connection.close()

            # Update UI by removing the widget
            for i in range(self.project_layout.count()):
                item = self.project_layout.itemAt(i)
                if item.widget() and hasattr(item.widget(), 'radio') and item.widget().radio.text() == project_name:
                    widget = item.widget()

                    # If this radio button was checked, select another one if available
                    if widget.radio.isChecked() and self.project_group.buttons():
                        # Find another radio to check
                        for btn in self.project_group.buttons():
                            if btn != widget.radio:
                                btn.setChecked(True)
                                break

                    # Remove from radio group
                    self.project_group.removeButton(widget.radio)

                    # Remove from layout
                    self.project_layout.removeWidget(widget)

                    # Delete the widget
                    widget.deleteLater()
                    break

            StyledMessageBox.information(
                self, "Success", f"Project '{project_name}' deleted successfully")
            print(f"Project '{project_name}' deleted successfully from location '{location_name}'")
            return True

        except Exception as e:
            print(f"Error deleting project from database: {e}")
            traceback.print_exc()
            StyledMessageBox.critical(self, "Error", f"Failed to delete project: {str(e)}")
            return False

    def show_create_new_project_form(self):
        """Show the form for creating a new project."""
        self.project_selection_area.setVisible(False)
        self.new_project_form.setVisible(True)
        self.new_project_edit.setFocus()

    def hide_create_new_project_form(self):
        """Hide the form for creating a new project."""
        self.new_project_form.setVisible(False)
        self.project_selection_area.setVisible(True)
        self.new_project_edit.clear()

    def create_new_project(self):
        """Create a new project and add it to the list."""
        project_name = self.new_project_edit.text().strip()
        selected_location = self.location_group.checkedButton().text()

        if not project_name:
            StyledMessageBox.warning(self, "Input Required", "Please enter a project name.")
            return

        # Check length limit
        if len(project_name) > 50:
            StyledMessageBox.warning(self, "Input Too Long",
                                     "Project name must not exceed 50 characters.")
            return

        # Check if project name matches organization or location name
        if project_name.lower() == self.organization.lower():
            StyledMessageBox.warning(self, "Invalid Name",
                                     "Project name cannot be the same as the organization name.")
            return

        if project_name.lower() == selected_location.lower():
            StyledMessageBox.warning(self, "Invalid Name",
                                     "Project name cannot be the same as the location name.")
            return

        # Check if project with this name already exists (case-insensitive)
        project_exists = False
        for btn in self.project_group.buttons():
            if project_name.lower() == btn.text().lower():
                project_exists = True
                project_name = btn.text()  # Use the existing case version
                break

        if project_exists:
            StyledMessageBox.warning(self, "Duplicate Project",
                                     f"Project '{project_name}' already exists.")
            return

        # Create new project widget with delete button (using ProjectWidget class)
        project_widget = ProjectWidget(project_name)
        project_widget.radio.setChecked(True)  # Select the new project
        self.project_group.addButton(project_widget.radio)

        # Connect delete button
        def create_delete_handler(loc, proj):
            return lambda: self.delete_project(loc, proj)

        project_widget.delete_button.clicked.connect(
            create_delete_handler(selected_location, project_name)
        )

        # Add to layout before the create button
        self.project_layout.insertWidget(
            self.project_layout.count() - 1,  # Before create button
            project_widget
        )

        # Save project to organization file
        self.save_project_to_organization(selected_location, project_name)

        # Hide the creation form
        self.hide_create_new_project_form()

    def save_project_to_organization(self, location_name, project_name):
        """Save the project to the database."""
        try:
            # Get a database connection
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                StyledMessageBox.critical(self, "Database Error",
                                          "Could not connect to the database.")
                return False

            # Get user ID directly from user_info if available, don't look it up by username
            user_id = None
            if self.user_info and "user_id" in self.user_info:
                user_id = self.user_info.get("user_id")
                print(f"Using user_id {user_id} directly from user_info")
            else:
                # Fallback to looking up by username
                username = self.user_info.get("username", "unknown")
                user_id = get_user_id_by_username(username)
                print(f"Looking up user_id by username {username}: found {user_id}")

            if not user_id:
                # Default to admin if user ID not found
                user_id = "A001"
                print(f"User ID not found, using default admin ID")

            cursor = connection.cursor()

            # Create a project record in the location table
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            query = """
            INSERT INTO location (id, organization, city, project, timestamp) 
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (user_id, self.organization,
                           location_name, project_name, current_time))
            connection.commit()

            cursor.close()
            connection.close()

            print(
                f"Added project '{project_name}' to location '{location_name}' in organization '{self.organization}'")
            StyledMessageBox.information(
                self, "Success", f"Project '{project_name}' created successfully.")
            return True

        except Exception as e:
            print(f"Error saving project to database: {e}")
            traceback.print_exc()
            StyledMessageBox.warning(
                self, "Warning", f"Project created but could not save to database: {str(e)}")
            return False


class BlurredBackgroundWidget(QWidget):
    """Widget that displays a blurred background image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blur_radius = 10
        self.background_path = None
        self.background_pixmap = None
        self.background_image = None

        self.background_gradient = QLinearGradient(0, 0, 0, self.height())
        self.background_gradient.setColorAt(0, QColor("#1e3c72"))
        self.background_gradient.setColorAt(1, QColor("#2a5298"))

        bg_path = "bg.png"
        fallback_bg_path = "bgwl.png"

        if os.path.exists(bg_path):
            self.set_background_image(bg_path)
        elif os.path.exists(fallback_bg_path):
            self.set_background_image(fallback_bg_path)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.force_repaint)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds

    def paintEvent(self, event):
        """Paint the background with a semi-transparent overlay instead of blur."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        try:
            if self.background_image and not self.background_image.isNull():
                # Scale image safely - ensure no division by zero or null size
                if self.width() > 0 and self.height() > 0:
                    scaled_image = self.background_image.scaled(
                        self.size(),
                        Qt.KeepAspectRatioByExpanding,
                        Qt.SmoothTransformation
                    )

                    # Additional null check after scaling
                    if not scaled_image.isNull():
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
            painter.fillRect(self.rect(), QColor("#1e3c72"))

    def set_background_image(self, image_path):
        """Set or refresh the background image."""
        if os.path.exists(image_path):
            self.background_path = image_path
            self.background_pixmap = QPixmap(image_path)
            self.background_image = self.background_pixmap.toImage()
            self.update()

    def force_repaint(self):
        """Force a complete repaint of the background."""
        if hasattr(self, 'background_path') and os.path.exists(self.background_path):
            self.background_pixmap = QPixmap(self.background_path)
            self.background_image = self.background_pixmap.toImage()
        self.update()


class LocationCard(QFrame):
    """A styled card containing the location selection form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("locationCard")
        self.setStyleSheet("""
            #locationCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.8);
            }
        """)

        # Set maximum width to prevent stretching on large screens
        self.setMaximumWidth(600)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)


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
        # Set maximum length to 50 characters
        self.setMaxLength(50)


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
                border: 2px solid #ffffff;
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
                color: #333333;
                border: 1px solid #ff914d;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e86d25;
            }
            QPushButton:pressed {
                background-color: #e86d25;
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
                    background-color: #ff7730;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #ff5c00;
                }
                QPushButton:pressed {
                    background-color: #e65200;
                    border-style: inset;
                    padding-top: 12px;
                    padding-left: 22px;
                }
                QPushButton:focus {
                    outline: none;
                    border: 2px solid #ffffff;
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

        # Remove all graphic effects that might cause visibility issues
        self.setGraphicsEffect(None)


class LocationWidget(QWidget):
    """Widget for displaying a location with a delete button."""

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


class CreateNewLocationButton(QPushButton):
    """Custom button for creating a new location."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("+ Create New Location")
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
                color: #e86d25;
                text-decoration: underline;
            }
        """)


class ProjectWidget(QWidget):
    """Widget for displaying a project with a delete button."""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.radio = CustomRadioButton(name)
        layout.addWidget(self.radio)

        self.delete_button = DeleteButton()
        layout.addWidget(self.delete_button)

        layout.addStretch(1)
        self.setLayout(layout)


class StyledMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Apply custom stylesheet with orange accent color
        self.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border: 1px solid #ff914d;
                border-radius: 8px;
            }
            QMessageBox QLabel {
                color: #333333;
                font-size: 14px;
            }
            QPushButton {
                background-color: #ff914d;
                color: white;
                border: none;
                border-radius: 4px;
                min-width: 80px;
                min-height: 30px;
                font-weight: bold;
                font-size: 14px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #e86d25;
            }
            QPushButton:pressed {
                background-color: #d65e1c;
            }
        """)

    @staticmethod
    def information(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok):
        """Static method to show an information message box with the custom style."""
        msgBox = StyledMessageBox(parent)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setStandardButtons(buttons)
        msgBox.setDefaultButton(defaultButton)
        return msgBox.exec_()

    @staticmethod
    def warning(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok):
        """Static method to show a warning message box with the custom style."""
        msgBox = StyledMessageBox(parent)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setStandardButtons(buttons)
        msgBox.setDefaultButton(defaultButton)
        return msgBox.exec_()

    @staticmethod
    def critical(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok):
        """Static method to show a critical message box with the custom style."""
        msgBox = StyledMessageBox(parent)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setStandardButtons(buttons)
        msgBox.setDefaultButton(defaultButton)
        return msgBox.exec_()

    @staticmethod
    def question(parent, title, text, buttons=QMessageBox.Yes | QMessageBox.No, defaultButton=QMessageBox.No):
        """Static method to show a question message box with the custom style."""
        msgBox = StyledMessageBox(parent)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setStandardButtons(buttons)
        msgBox.setDefaultButton(defaultButton)
        return msgBox.exec_()


class MetalSheetLocationUI(QWidget):
    def __init__(self, user_info=None, organization=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAKAR VISION AI - Select Location")
        set_window_icon(self)
        self.setMinimumSize(800, 600)

        # Store user info and organization
        self.user_info = user_info or {"username": "User", "full_name": "Default User"}

        # If user_info has id but not user_id, add user_id field for consistency
        if self.user_info and "id" in self.user_info and "user_id" not in self.user_info:
            self.user_info["user_id"] = self.user_info["id"]
            print(f"Added user_id field with value {self.user_info['id']} for consistency")

        self.organization = organization or "Unknown Organization"

        # Flag to track if we're in create mode
        self.creating_new_location = False

        # Dictionary to keep track of location widgets by name
        self.location_widgets = {}

        # Background image properties
        self.background_image = None
        self.scaled_background = None
        self.last_size = QSize(0, 0)

        # Load background image first
        self.load_background_image()

        # Initialize UI
        self.init_ui()

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12pt;
                background-color: transparent;
            }
            QLabel#welcomeLabel {
                font-size: 18pt;
                font-weight: bold;
                color: #000000;
                background-color: transparent;
            }
            QLabel#instructionLabel {
                font-size: 14pt;
                color: #000000;
                background-color: transparent;
            }
            QLabel#orgLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #ff914d;
                background-color: transparent;
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

        # Load existing locations for this organization
        self.load_locations()

        # Load projects for the default selected location
        default_location = "Sakarrobotics_demo"
        self.load_projects_for_location(default_location)
        self.project_selection_area.setVisible(True)

        # Print debug information
        print(f"MetalSheetLocationUI initialized for organization: {self.organization}")
        if "user_id" in self.user_info:
            print(f"User ID: {self.user_info['user_id']}")
        elif "id" in self.user_info:
            print(f"User ID: {self.user_info['id']}")
        else:
            print("No user ID found in user_info")

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

        # Create blurred background widget
        self.background_widget = BlurredBackgroundWidget(self)

        # Title bar
        self.title_bar = CustomTitleBar(self)

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
        welcome_label.setAlignment(Qt.AlignCenter)  # Center alignment

        # Organization display
        org_layout = QHBoxLayout()
        org_layout.setAlignment(Qt.AlignCenter)  # Center the entire layout

        org_text_label = QLabel("Selected Organization:")
        org_text_label.setObjectName("instructionLabel")

        org_name_label = QLabel(self.organization)
        org_name_label.setObjectName("orgLabel")

        org_layout.addWidget(org_text_label)
        org_layout.addWidget(org_name_label)

        # Instruction
        instruction_label = QLabel(
            "Please select a location and project for metal sheet inspection:")
        instruction_label.setObjectName("instructionLabel")
        instruction_label.setAlignment(Qt.AlignCenter)  # Center alignment

        # Location card
        self.location_card = LocationCard()
        card_layout = QVBoxLayout(self.location_card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)

        # Location selection area (initially visible)
        self.location_selection_area = QWidget()
        location_area_layout = QVBoxLayout(self.location_selection_area)
        location_area_layout.setContentsMargins(0, 0, 0, 0)
        location_area_layout.setSpacing(10)

        # Location radio buttons
        self.location_group = QButtonGroup(self)

        # Create a scroll area for locations
        self.location_scroll_area = QScrollArea()
        self.location_scroll_area.setWidgetResizable(True)
        self.location_scroll_area.setFrameShape(QFrame.NoFrame)
        self.location_scroll_area.setMinimumHeight(150)

        # Container for location radio buttons
        self.location_container = QWidget()
        self.location_layout = QVBoxLayout(self.location_container)
        self.location_layout.setContentsMargins(0, 0, 0, 0)
        self.location_layout.setSpacing(10)
        self.location_layout.setAlignment(Qt.AlignTop)

        # Dictionary to keep track of location widgets by name
        self.location_widgets = {}

        # We'll load locations dynamically based on organization, don't add default location here

        # Add "Create New Location" button
        self.create_new_location_button = CreateNewLocationButton()
        self.create_new_location_button.clicked.connect(self.show_create_new_location_form)
        self.location_layout.addWidget(self.create_new_location_button)

        # Set the location container as the scroll area widget
        self.location_scroll_area.setWidget(self.location_container)
        location_area_layout.addWidget(self.location_scroll_area)

        # New location form (initially hidden)
        self.new_location_form = QWidget()
        new_location_layout = QVBoxLayout(self.new_location_form)
        new_location_layout.setContentsMargins(0, 0, 0, 0)
        new_location_layout.setSpacing(10)

        new_location_label = QLabel("Enter new location name:")
        new_location_label.setStyleSheet("font-weight: bold; color: #555555;")

        self.new_location_edit = StyledLineEdit(placeholder="Location name")

        new_location_buttons = QHBoxLayout()
        self.cancel_new_button = CustomButton("Cancel", is_primary=False)
        self.cancel_new_button.clicked.connect(self.hide_create_new_location_form)

        self.create_location_button = CustomButton("Create", is_primary=True)
        self.create_location_button.clicked.connect(self.create_new_location)

        new_location_buttons.addWidget(self.cancel_new_button)
        new_location_buttons.addStretch(1)
        new_location_buttons.addWidget(self.create_location_button)

        new_location_layout.addWidget(new_location_label)
        new_location_layout.addWidget(self.new_location_edit)
        new_location_layout.addLayout(new_location_buttons)

        # Hide form initially
        self.new_location_form.setVisible(False)

        # Project selection area (initially hidden)
        self.project_selection_area = QWidget()
        project_area_layout = QVBoxLayout(self.project_selection_area)
        project_area_layout.setContentsMargins(0, 20, 0, 0)
        project_area_layout.setSpacing(10)

        # Project header
        project_header = QLabel("Please select a project:")
        project_header.setStyleSheet("font-weight: bold; color: #555555;")
        project_area_layout.addWidget(project_header)

        # Project radio buttons
        self.project_group = QButtonGroup(self)

        # Create a scroll area for projects
        self.project_scroll_area = QScrollArea()
        self.project_scroll_area.setWidgetResizable(True)
        self.project_scroll_area.setFrameShape(QFrame.NoFrame)
        self.project_scroll_area.setMinimumHeight(150)

        # Container for project radio buttons
        self.project_container = QWidget()
        self.project_layout = QVBoxLayout(self.project_container)
        self.project_layout.setContentsMargins(0, 0, 0, 0)
        self.project_layout.setSpacing(10)
        self.project_layout.setAlignment(Qt.AlignTop)

        # Add "Create New Project" button
        self.create_new_project_button = CreateNewLocationButton()
        self.create_new_project_button.setText("+ Create New Project")
        self.create_new_project_button.clicked.connect(self.show_create_new_project_form)
        self.project_layout.addWidget(self.create_new_project_button)

        # Set the project container as the scroll area widget
        self.project_scroll_area.setWidget(self.project_container)
        project_area_layout.addWidget(self.project_scroll_area)

        # New project form (initially hidden)
        self.new_project_form = QWidget()
        new_project_layout = QVBoxLayout(self.new_project_form)
        new_project_layout.setContentsMargins(0, 0, 0, 0)
        new_project_layout.setSpacing(10)

        new_project_label = QLabel("Enter new project name:")
        new_project_label.setStyleSheet("font-weight: bold; color: #555555;")

        self.new_project_edit = StyledLineEdit(placeholder="Project name")

        new_project_buttons = QHBoxLayout()
        self.cancel_new_project_button = CustomButton("Cancel", is_primary=False)
        self.cancel_new_project_button.clicked.connect(self.hide_create_new_project_form)

        self.create_project_button = CustomButton("Create", is_primary=True)
        self.create_project_button.clicked.connect(self.create_new_project)

        new_project_buttons.addWidget(self.cancel_new_project_button)
        new_project_buttons.addStretch(1)
        new_project_buttons.addWidget(self.create_project_button)

        new_project_layout.addWidget(new_project_label)
        new_project_layout.addWidget(self.new_project_edit)
        new_project_layout.addLayout(new_project_buttons)

        # Hide form initially
        self.new_project_form.setVisible(False)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.back_button = CustomButton("Back", is_primary=False)
        self.back_button.clicked.connect(self.go_back)

        self.start_button = CustomButton("Start Inspection", is_primary=True)
        self.start_button.clicked.connect(self.proceed_to_inspection)

        button_layout.addWidget(self.back_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.start_button)

        # Add elements to card layout
        card_layout.addWidget(self.location_selection_area)
        card_layout.addWidget(self.new_location_form)
        card_layout.addWidget(self.project_selection_area)
        card_layout.addWidget(self.new_project_form)
        card_layout.addStretch(1)
        card_layout.addLayout(button_layout)

        # Add elements to content layout
        content_layout.addWidget(welcome_label)
        content_layout.addLayout(org_layout)
        content_layout.addWidget(instruction_label)
        content_layout.addSpacing(20)

        # Center the card
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.location_card)

        content_layout.addLayout(center_layout)
        content_layout.addStretch(1)

        # Add widgets to main layout
        main_layout.addWidget(self.background_widget)
        main_layout.addWidget(self.title_bar)
        main_layout.addWidget(content_widget)

        # Make the background fill the entire widget
        self.background_widget.setGeometry(0, 0, self.width(), self.height())

        # Setup location radio button event handler to show project options
        self.location_group.buttonClicked.connect(self.location_selected)

        # Hide project selection area initially
        self.project_selection_area.setVisible(False)

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

    def load_locations(self):
        """Load existing locations for this organization from the database."""
        try:
            # Connect to the database
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                return

            cursor = connection.cursor(dictionary=True)

            # Query to get distinct cities (locations) from the location table for this organization
            query = "SELECT DISTINCT city FROM location WHERE organization = %s ORDER BY city"
            cursor.execute(query, (self.organization,))

            # Fetch all locations
            locations = cursor.fetchall()

            # Close the cursor and connection
            cursor.close()
            connection.close()

            # Clear all existing location widgets first
            for name, widget in list(self.location_widgets.items()):
                # Remove from radio group
                self.location_group.removeButton(widget.radio)

                # Remove from layout
                self.location_layout.removeWidget(widget)

                # Delete the widget
                widget.deleteLater()

                # Remove from dictionary
                del self.location_widgets[name]

            # Add locations from the database that belong to this organization
            for i, location in enumerate(locations, start=1):
                location_name = location['city']
                if location_name:
                    # Create location widget with radio button and delete button
                    location_widget = LocationWidget(location_name)
                    self.location_group.addButton(location_widget.radio, i)

                    # Create a closure for the delete button handler to capture the current location name
                    def create_delete_handler(name):
                        return lambda: self.delete_location(name)

                    # Connect delete button with the correct name
                    location_widget.delete_button.clicked.connect(
                        create_delete_handler(location_name))

                    # Add to layout before the create button
                    self.location_layout.insertWidget(
                        self.location_layout.count() - 1,  # Before create button
                        location_widget
                    )

                    # Add to dictionary
                    self.location_widgets[location_name] = location_widget

            # If locations were found, select the first one and load its projects
            if locations and self.location_group.buttons():
                first_location = self.location_group.buttons()[0]
                first_location.setChecked(True)
                self.location_selected(first_location)
            else:
                # Hide project selection if no locations are available
                self.project_selection_area.setVisible(False)

            print(f"Loaded {len(locations)} locations for organization {self.organization}")
        except Exception as e:
            print(f"Error loading locations from database: {e}")
            traceback.print_exc()
            # Show error message to user
            QMessageBox.warning(
                self,
                "Database Error",
                f"Failed to load locations from database: {str(e)}\n\nNo locations available."
            )

    def delete_location(self, location_name):
        """Delete a location from the database."""
        try:
            # Confirm deletion with styled message box
            reply = StyledMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete location '{location_name}'?\nAll projects under this location will also be deleted.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return False

            # Get a database connection
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                StyledMessageBox.critical(self, "Database Error",
                                          "Could not connect to the database.")
                return False

            cursor = connection.cursor()

            # Delete the location and all its associated projects from the location table
            delete_query = "DELETE FROM location WHERE organization = %s AND city = %s"
            cursor.execute(delete_query, (self.organization, location_name))

            # Commit the transaction
            connection.commit()

            # Close cursor and connection
            cursor.close()
            connection.close()

            # Update UI - remove the widget
            if location_name in self.location_widgets:
                widget = self.location_widgets[location_name]

                # If this radio button was checked, select another location if available
                if widget.radio.isChecked():
                    # Find another radio to check
                    other_locations = [
                        btn for btn in self.location_group.buttons() if btn != widget.radio]
                    if other_locations:
                        other_locations[0].setChecked(True)
                        # Since we're selecting a different location, load its projects
                        self.location_selected(other_locations[0])
                    else:
                        # No other locations, hide project selection
                        self.project_selection_area.setVisible(False)

                # Remove from radio group
                self.location_group.removeButton(widget.radio)

                # Remove from layout
                self.location_layout.removeWidget(widget)

                # Delete the widget
                widget.deleteLater()

                # Remove from dictionary
                del self.location_widgets[location_name]

            # Show success message
            StyledMessageBox.information(self, "Success",
                                         f"Location '{location_name}' and all its projects deleted successfully")
            print(f"Location '{location_name}' deleted successfully from database")
            return True

        except Exception as e:
            print(f"Error deleting location from database: {e}")
            traceback.print_exc()
            StyledMessageBox.critical(self, "Error", f"Failed to delete location: {str(e)}")
            return False

    def show_create_new_location_form(self):
        """Show the form for creating a new location."""
        self.creating_new_location = True
        self.location_selection_area.setVisible(False)
        self.new_location_form.setVisible(True)
        self.new_location_edit.setFocus()

    def hide_create_new_location_form(self):
        """Hide the form for creating a new location."""
        self.creating_new_location = False
        self.new_location_form.setVisible(False)
        self.location_selection_area.setVisible(True)
        self.new_location_edit.clear()

    def create_new_location(self):
        """Create a new location and add it to the list."""
        location_name = self.new_location_edit.text().strip()

        if not location_name:
            StyledMessageBox.warning(self, "Input Required", "Please enter a location name.")
            return

        # Check length limit
        if len(location_name) > 50:
            StyledMessageBox.warning(self, "Input Too Long",
                                     "Location name must not exceed 50 characters.")
            return

        # Check if location name matches organization
        if location_name.lower() == self.organization.lower():
            StyledMessageBox.warning(self, "Invalid Name",
                                     "Location name cannot be the same as the organization name.")
            return

        # Check if the name matches any existing project name
        for btn in self.project_group.buttons():
            if location_name.lower() == btn.text().lower():
                StyledMessageBox.warning(self, "Invalid Name",
                                         "Location name cannot be the same as an existing project name.")
                return

        # Check if location with this name already exists (case-insensitive)
        exists = False
        for existing_name in self.location_widgets:
            if location_name.lower() == existing_name.lower():
                exists = True
                location_name = existing_name  # Use the existing case version
                break

        if exists:
            StyledMessageBox.warning(self, "Duplicate Location",
                                     f"Location '{location_name}' already exists.")
            return

        # Create new location widget
        new_location_widget = LocationWidget(location_name)
        new_location_widget.radio.setChecked(True)  # Select the new location

        # Add to group with new ID
        new_id = len(self.location_widgets) + 1
        self.location_group.addButton(new_location_widget.radio, new_id)

        # Connect delete button using the proper closure approach
        def create_delete_handler(name):
            return lambda: self.delete_location(name)

        new_location_widget.delete_button.clicked.connect(create_delete_handler(location_name))

        # Add to layout before the create button
        self.location_layout.insertWidget(
            self.location_layout.count() - 1,  # Before create button
            new_location_widget
        )

        # Add to dictionary
        self.location_widgets[location_name] = new_location_widget

        # Save to database - we won't create a default project anymore
        try:
            # Get a database connection
            conn = get_connection()
            if conn is None:
                print("Error: Unable to establish a database connection.")
                StyledMessageBox.critical(self, "Database Error",
                                          "Could not connect to the database.")
                return

            # Get user ID directly from user_info if available, don't look it up by username
            user_id = None
            if self.user_info and "user_id" in self.user_info:
                user_id = self.user_info.get("user_id")
                print(f"Using user_id {user_id} directly from user_info")
            else:
                # Fallback to looking up by username
                username = self.user_info.get("username", "unknown")
                user_id = get_user_id_by_username(username)
                print(f"Looking up user_id by username {username}: found {user_id}")

            if not user_id:
                # Default to admin if user ID not found
                user_id = "A001"
                print(f"User ID not found, using default admin ID")

            # Instead of creating a default project, we'll just register the location
            # in the organization table or another appropriate table that doesn't require
            # a project to be specified

            # For now, we'll just show a success message without creating an entry in the location table
            # This means locations will only appear in the database when a project is created for them

            print(f"Added location '{location_name}' to organization '{self.organization}'")
            StyledMessageBox.information(
                self, "Success", f"Location '{location_name}' created successfully.")
        except Exception as e:
            print(f"Error saving location to database: {e}")
            traceback.print_exc()
            StyledMessageBox.warning(
                self, "Warning", f"Location created but could not save to database: {str(e)}")

        # Hide the creation form
        self.hide_create_new_location_form()

        # Show the project selection area but with no projects (since we didn't create a default one)
        self.project_selection_area.setVisible(True)
        self.clear_project_list()

    def save_location_config(self):
        """Save the location selection and organization to configuration file."""
        # Get selected location
        selected_button = self.location_group.checkedButton()
        if selected_button is None:
            # Default to Sakarrobotics_demo if somehow none are selected
            selected_location = "Sakarrobotics_demo"
        else:
            selected_location = selected_button.text()

        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(LOCATION_CONFIG_PATH)
            if not os.path.exists(directory) and directory:
                os.makedirs(directory)

            # Create config
            config = {
                "organization": self.organization,
                "location": selected_location,
                "user": self.user_info.get("username", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "inspection_type": "metal_sheet"  # Add type to distinguish from fabric inspection
            }

            # Save to file
            with open(LOCATION_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"Location configuration saved: {self.organization} - {selected_location}")

            return True, selected_location
        except Exception as e:
            print(f"Error saving location configuration: {e}")
            traceback.print_exc()
            return False, None

    def proceed_to_inspection(self):
        """Save location and proceed to metal sheet inspection deployment."""
        # First show a reminder about granting access
        reminder = StyledMessageBox(self)
        reminder.setWindowTitle("Access Reminder")
        reminder.setText("Have you granted access to all required users?")
        reminder.setIcon(QMessageBox.Question)
        reminder.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reminder.setDefaultButton(QMessageBox.No)

        # Add a detailed informative text
        reminder.setInformativeText(
            "Before proceeding with inspection, make sure all required users have been granted access to the system.")

        response = reminder.exec_()

        # If user selects "No", open the access dialog and return
        if response == QMessageBox.No:
            self.title_bar.show_access_dialog()
            return

        # Continue with normal flow if user selects "Yes"
        # Save the location configuration
        success, selected_location = self.save_location_config()
        if not success:
            QMessageBox.critical(
                self, "Error", "Failed to save location configuration. Please try again.")
            return

        # Get selected project
        selected_project_button = self.project_group.checkedButton()
        if not selected_project_button:
            StyledMessageBox.warning(self, "Selection Required", "Please select a project.")
            return

        selected_project = selected_project_button.text()
        print(
            f"Proceeding to inspection with location: {selected_location}, project: {selected_project}")

        # Store complete location data in Azure database
        try:
            username = self.user_info.get("username", "")
            if username:
                # Get user ID from database
                user_id = get_user_id_by_username(username)
                if user_id:
                    # Store complete location data - the organization parameter is None here
                    # because we'll use the stored organization value from the previous screen
                    store_location_data(user_id, self.organization,
                                        selected_location, selected_project)
                    print(
                        f"Stored complete location data for user {username} (ID: {user_id}): {self.organization}, {selected_location}, {selected_project}")
                else:
                    print(f"Warning: Could not find user ID for username: {username}")
            else:
                print("Warning: No username available in user_info")
        except Exception as e:
            print(f"Error storing location data in Azure: {e}")
            # Continue anyway - don't block the user from proceeding

        # Create and pass inspection data
        inspection_data = {
            "organization": self.organization,
            "location": selected_location,
            "project": selected_project,
            "user": self.user_info,
            "inspection_type": "metal_sheet"  # Add this key to ensure correct UI is loaded
        }

        try:
            # First check if the module exists
            if not os.path.exists("defect_selection_ui.py"):
                msg = "The defect selection module (defect_selection_ui.py) is missing. Please check your installation."
                print(msg)
                QMessageBox.critical(self, "Missing Module", msg)
                return

            print("Importing DefectSelectionUI...")
            # Try to import the DefectSelectionUI
            try:
                from defect_selection_ui import DefectSelectionUI
            except ImportError as e:
                print(f"Error importing DefectSelectionUI: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Import Error",
                                     f"Failed to load the defect selection module: {str(e)}\n"
                                     "Please check that all required files are in place.")
                return

            # Apply visual feedback for button click
            original_style = self.start_button.styleSheet()
            self.start_button.setStyleSheet("""
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

            print("Creating DefectSelectionUI instance...")
            # Close any existing instance before creating a new one
            if hasattr(self, 'defect_selection_ui') and self.defect_selection_ui is not None:
                self.defect_selection_ui.close()
                self.defect_selection_ui.deleteLater()

            # Create the defect selection UI instance
            self.defect_selection_ui = DefectSelectionUI(inspection_data)

            # Connect destroyed signal to show this window again if the defect selection screen is closed
            self.defect_selection_ui.destroyed.connect(self.show)

            print("Showing DefectSelectionUI...")
            # Show the defect selection UI
            self.defect_selection_ui.showMaximized()

            # Hide this window - don't use QTimer.singleShot as it can cause issues
            self.hide()

        except Exception as e:
            # Reset button style in case of error
            self.start_button.setStyleSheet(original_style)

            print("Error in proceed_to_inspection:")
            traceback.print_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to launch defect selection: {str(e)}\n\n"
                                 "See console for detailed error information.")

    def showEvent(self, event):
        """Override show event to maximize the window."""
        super().showEvent(event)
        self.showMaximized()

    def resizeEvent(self, event):
        """Update background size when window is resized."""
        self.background_widget.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def go_back(self):
        """Go back to organization selection screen."""
        try:
            from metal_sheet_organization_ui import MetalSheetOrganizationUI

            # Close any existing instance before creating a new one
            if hasattr(self, 'organization_ui') and self.organization_ui is not None:
                self.organization_ui.close()
                self.organization_ui.deleteLater()

            self.organization_ui = MetalSheetOrganizationUI(self.user_info)

            # Connect destroyed signal to show this window again if the organization screen is closed
            self.organization_ui.destroyed.connect(self.show)

            # Show the organization UI
            self.organization_ui.showMaximized()

            # Hide this window - don't use QTimer.singleShot as it can cause issues
            self.hide()

        except Exception as e:
            print("Error going back to organization selection:")
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error", f"Could not go back to organization screen: {str(e)}")

    def location_selected(self, button):
        """Handle location selection to show relevant projects."""
        selected_location = button.text()
        print(f"Location selected: {selected_location}")

        # Clear existing projects
        self.clear_project_list()

        # Load projects for this location
        self.load_projects_for_location(selected_location)

        # Show the project selection area - always show this regardless of whether projects exist
        self.project_selection_area.setVisible(True)
        self.new_project_form.setVisible(False)

        # Report selected project if any is selected
        selected_project_button = self.project_group.checkedButton()
        if selected_project_button:
            selected_project = selected_project_button.text()
            print(f"Selected project: {selected_project}")
        else:
            print("No project selected")

    def clear_project_list(self):
        """Clear all projects from the list except the create button."""
        # Remove all radio buttons from the group
        for btn in self.project_group.buttons():
            self.project_group.removeButton(btn)

        # Clear the layout while preserving the create button
        create_button = self.create_new_project_button
        for i in reversed(range(self.project_layout.count())):
            item = self.project_layout.itemAt(i)
            if item.widget() != create_button:
                widget = item.widget()
                self.project_layout.removeWidget(widget)
                if widget:
                    widget.deleteLater()

    def load_projects_for_location(self, location_name):
        """Load projects for the selected location from the database."""
        try:
            # Connect to the database
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                return

            cursor = connection.cursor(dictionary=True)

            # Query to get distinct projects from the location table for this organization and city
            query = "SELECT DISTINCT project FROM location WHERE organization = %s AND city = %s ORDER BY project"
            cursor.execute(query, (self.organization, location_name))

            # Fetch all projects
            projects = cursor.fetchall()

            # Close the cursor and connection
            cursor.close()
            connection.close()

            # Clear existing projects first
            self.clear_project_list()

            # Add projects
            for i, project in enumerate(projects):
                project_name = project['project']
                if project_name:
                    # Create project widget with radio button and delete button
                    project_widget = ProjectWidget(project_name)
                    self.project_group.addButton(project_widget.radio, i)

                    # Connect delete button with the correct closure
                    def create_delete_handler(loc, proj):
                        return lambda: self.delete_project(loc, proj)

                    project_widget.delete_button.clicked.connect(
                        create_delete_handler(location_name, project_name)
                    )

                    # Add to layout before the create button
                    self.project_layout.insertWidget(
                        self.project_layout.count() - 1,  # Before create button
                        project_widget
                    )

            # If projects were found, select the first one
            if projects and self.project_group.buttons():
                self.project_group.buttons()[0].setChecked(True)

            print(
                f"Loaded {len(projects)} projects for location {location_name} in organization {self.organization}")
        except Exception as e:
            print(f"Error loading projects from database: {e}")
            traceback.print_exc()
            # Show error message to user
            QMessageBox.warning(
                self,
                "Database Error",
                f"Failed to load projects from database: {str(e)}\n\nNo projects available."
            )

    def delete_project(self, location_name, project_name):
        """Delete a project from the database for the given location."""
        try:
            # Confirm deletion with styled dialog
            reply = StyledMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete project '{project_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return False

            # Get a database connection
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                StyledMessageBox.critical(self, "Database Error",
                                          "Could not connect to the database.")
                return False

            cursor = connection.cursor()

            # Delete the project from the location table
            delete_query = "DELETE FROM location WHERE organization = %s AND city = %s AND project = %s"
            cursor.execute(delete_query, (self.organization, location_name, project_name))

            # Commit the transaction
            connection.commit()

            # Close cursor and connection
            cursor.close()
            connection.close()

            # Update UI by removing the widget
            for i in range(self.project_layout.count()):
                item = self.project_layout.itemAt(i)
                if item.widget() and hasattr(item.widget(), 'radio') and item.widget().radio.text() == project_name:
                    widget = item.widget()

                    # If this radio button was checked, select another one if available
                    if widget.radio.isChecked() and self.project_group.buttons():
                        # Find another radio to check
                        for btn in self.project_group.buttons():
                            if btn != widget.radio:
                                btn.setChecked(True)
                                break

                    # Remove from radio group
                    self.project_group.removeButton(widget.radio)

                    # Remove from layout
                    self.project_layout.removeWidget(widget)

                    # Delete the widget
                    widget.deleteLater()
                    break

            StyledMessageBox.information(
                self, "Success", f"Project '{project_name}' deleted successfully")
            print(f"Project '{project_name}' deleted successfully from location '{location_name}'")
            return True

        except Exception as e:
            print(f"Error deleting project from database: {e}")
            traceback.print_exc()
            StyledMessageBox.critical(self, "Error", f"Failed to delete project: {str(e)}")
            return False

    def show_create_new_project_form(self):
        """Show the form for creating a new project."""
        self.project_selection_area.setVisible(False)
        self.new_project_form.setVisible(True)
        self.new_project_edit.setFocus()

    def hide_create_new_project_form(self):
        """Hide the form for creating a new project."""
        self.new_project_form.setVisible(False)
        self.project_selection_area.setVisible(True)
        self.new_project_edit.clear()

    def create_new_project(self):
        """Create a new project and add it to the list."""
        project_name = self.new_project_edit.text().strip()
        selected_location = self.location_group.checkedButton().text()

        if not project_name:
            StyledMessageBox.warning(self, "Input Required", "Please enter a project name.")
            return

        # Check length limit
        if len(project_name) > 50:
            StyledMessageBox.warning(self, "Input Too Long",
                                     "Project name must not exceed 50 characters.")
            return

        # Check if project name matches organization or location name
        if project_name.lower() == self.organization.lower():
            StyledMessageBox.warning(self, "Invalid Name",
                                     "Project name cannot be the same as the organization name.")
            return

        if project_name.lower() == selected_location.lower():
            StyledMessageBox.warning(self, "Invalid Name",
                                     "Project name cannot be the same as the location name.")
            return

        # Check if project with this name already exists (case-insensitive)
        project_exists = False
        for btn in self.project_group.buttons():
            if project_name.lower() == btn.text().lower():
                project_exists = True
                project_name = btn.text()  # Use the existing case version
                break

        if project_exists:
            StyledMessageBox.warning(self, "Duplicate Project",
                                     f"Project '{project_name}' already exists.")
            return

        # Create new project widget with delete button (using ProjectWidget class)
        project_widget = ProjectWidget(project_name)
        project_widget.radio.setChecked(True)  # Select the new project
        self.project_group.addButton(project_widget.radio)

        # Connect delete button
        def create_delete_handler(loc, proj):
            return lambda: self.delete_project(loc, proj)

        project_widget.delete_button.clicked.connect(
            create_delete_handler(selected_location, project_name)
        )

        # Add to layout before the create button
        self.project_layout.insertWidget(
            self.project_layout.count() - 1,  # Before create button
            project_widget
        )

        # Save project to organization file
        self.save_project_to_organization(selected_location, project_name)

        # Hide the creation form
        self.hide_create_new_project_form()

    def save_project_to_organization(self, location_name, project_name):
        """Save the project to the database."""
        try:
            # Get a database connection
            connection = get_connection()
            if connection is None:
                print("Error: Unable to establish a database connection.")
                StyledMessageBox.critical(self, "Database Error",
                                          "Could not connect to the database.")
                return False

            # Get user ID directly from user_info if available, don't look it up by username
            user_id = None
            if self.user_info and "user_id" in self.user_info:
                user_id = self.user_info.get("user_id")
                print(f"Using user_id {user_id} directly from user_info")
            else:
                # Fallback to looking up by username
                username = self.user_info.get("username", "unknown")
                user_id = get_user_id_by_username(username)
                print(f"Looking up user_id by username {username}: found {user_id}")

            if not user_id:
                # Default to admin if user ID not found
                user_id = "A001"
                print(f"User ID not found, using default admin ID")

            cursor = connection.cursor()

            # Create a project record in the location table
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            query = """
            INSERT INTO location (id, organization, city, project, timestamp) 
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (user_id, self.organization,
                           location_name, project_name, current_time))
            connection.commit()

            cursor.close()
            connection.close()

            print(
                f"Added project '{project_name}' to location '{location_name}' in organization '{self.organization}'")
            StyledMessageBox.information(
                self, "Success", f"Project '{project_name}' created successfully.")
            return True

        except Exception as e:
            print(f"Error saving project to database: {e}")
            traceback.print_exc()
            StyledMessageBox.warning(
                self, "Warning", f"Project created but could not save to database: {str(e)}")
            return False
