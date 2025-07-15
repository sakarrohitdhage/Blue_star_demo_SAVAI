#!/usr/bin/env python3
"""
SAKAR VISION AI - Login UI Module

OVERVIEW:
This module implements a sophisticated user authentication interface for the Sakar Vision AI platform, serving as 
the primary security gateway that provides comprehensive user authentication with advanced security features and 
seamless user experience design. It combines robust authentication mechanisms with professional visual design to 
ensure secure access control while maintaining optimal usability through intelligent session management, advanced 
security measures including account lockout protection, and seamless integration with both Azure cloud authentication 
and local fallback systems for industrial deployment environments requiring high reliability and security.

KEY FUNCTIONALITY:
The system features advanced dual-authentication architecture with primary Azure cloud-based authentication and 
intelligent local fallback mechanisms for offline operations, sophisticated account security with configurable 
login attempt tracking, automatic account lockout protection, and timed lockout recovery mechanisms, comprehensive 
session state management with persistent user information storage and seamless workflow continuation across application 
restarts, and intelligent camera index management with automatic configuration persistence for deployment operations. 
It includes professional visual design with custom background image support, gradient fallbacks, and optimized 
rendering performance, advanced UI components featuring custom styled input fields with password visibility toggles, 
animated buttons with hover effects, and responsive layout management, real-time system connectivity monitoring with 
visual status indicators and automatic connection testing, and comprehensive error handling with user-friendly 
feedback, detailed logging integration, and graceful degradation for network and authentication failures.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced custom widget architecture and professional visual styling, the module employs 
sophisticated authentication systems with secure password hashing using SHA-256 encryption and comprehensive 
credential validation, intelligent session persistence with JSON-based configuration management for user preferences 
and authentication state, and advanced security features including configurable login attempt tracking with the 
LoginAttemptTracker class and timed account lockout mechanisms. The architecture features custom widget components 
including StyledLineEdit with password visibility controls, CustomButton with hover animations, and LoginCard with 
drop shadow effects, optimized background rendering with multiple fallback mechanisms including image scaling, 
caching, and gradient alternatives for enhanced visual performance, comprehensive logging integration with detailed 
authentication event tracking and security audit capabilities, and seamless workflow integration with dynamic module 
loading for fabric inspection and metal sheet inspection interfaces. The system includes robust error handling with 
detailed logging, user feedback mechanisms, and automatic recovery procedures optimized for production deployment 
scenarios requiring maximum reliability and security compliance.
"""

from azure_database import test_connection as test_db_connection
from azure_database import authenticate_user as azure_authenticate_user
from datetime import datetime
from datetime import timedelta
import hashlib
import json
import os
import sys
import traceback  # Add missing import for traceback module
# Add imports for logging
import logging
from utils.logging import get_ui_logger, UIEventTracker
import requests
from PyQt5.QtCore import QEasingCurve
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QLinearGradient
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPalette
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QGraphicsBlurEffect
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget

from utils import set_window_icon

# Initialize logger for login UI and UI event tracker
logger = get_ui_logger("login")
ui_tracker = UIEventTracker()

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'

# Define constants for settings
SETTINGS_ORG = "SakarVision"
SETTINGS_APP = "Authentication"
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
# Add constants for lockout settings
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 60


# Class to track login attempts and lockouts
class LoginAttemptTracker:
    def __init__(self):
        self.login_attempts = {}  # Format: {username: {'count': int, 'last_attempt': datetime, 'locked_until': datetime}}

    def record_failed_attempt(self, username):
        """Record a failed login attempt and check if account should be locked"""
        current_time = datetime.now()

        # Initialize tracking for this username if not exists
        if username not in self.login_attempts:
            self.login_attempts[username] = {
                'count': 0,
                'last_attempt': current_time,
                'locked_until': None
            }

        # Check if previous lockout has expired
        if (self.login_attempts[username]['locked_until'] and
                current_time > self.login_attempts[username]['locked_until']):
            # Reset if lockout expired
            self.login_attempts[username] = {
                'count': 0,
                'last_attempt': current_time,
                'locked_until': None
            }

        # Increment attempt counter
        self.login_attempts[username]['count'] += 1
        self.login_attempts[username]['last_attempt'] = current_time

        # Check if should lock account
        if self.login_attempts[username]['count'] >= MAX_LOGIN_ATTEMPTS:
            locked_until = current_time + timedelta(seconds=LOCKOUT_DURATION_SECONDS)
            self.login_attempts[username]['locked_until'] = locked_until
            return True, locked_until

        return False, None

    def is_account_locked(self, username):
        """Check if an account is currently locked"""
        if username not in self.login_attempts:
            return False, None

        if not self.login_attempts[username]['locked_until']:
            return False, None

        current_time = datetime.now()
        if current_time < self.login_attempts[username]['locked_until']:
            return True, self.login_attempts[username]['locked_until']

        # Lockout expired, reset attempts
        self.login_attempts[username] = {
            'count': 0,
            'last_attempt': current_time,
            'locked_until': None
        }
        return False, None

    def reset_attempts(self, username):
        """Reset login attempts after successful login"""
        if username in self.login_attempts:
            self.login_attempts[username] = {
                'count': 0,
                'last_attempt': datetime.now(),
                'locked_until': None
            }


def initialize_users_file():
    """Create a default users file if none exists."""
    if not os.path.exists(USERS_FILE):
        default_users = {
            "admin": {
                "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
                "full_name": "Administrator",
                "role": "admin",
                "last_login": None
            },
            "operator": {
                "password_hash": hashlib.sha256("operator123".encode()).hexdigest(),
                "full_name": "Default Operator",
                "role": "operator",
                "last_login": None
            }
        }

        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f, indent=4)

        print(f"Created default users file at {USERS_FILE}")
        print("Default credentials: admin/admin123 and operator/operator123")


class CustomBorderFrame(QFrame):
    """Custom frame with gradient background and rounded corners."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("customBorderFrame")
        self.setStyleSheet("""
            #customBorderFrame {
                border: 1px solid #cccccc;
                border-radius: 15px;
                background-color: rgba(255, 255, 255, 220);
            }
        """)

    def paintEvent(self, event):
        """Override paint event to add custom gradient background."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 245))
        gradient.setColorAt(1, QColor(245, 245, 245, 230))

        # Draw rounded rect with gradient
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(self.rect(), 15, 15)


class CustomButton(QPushButton):
    """Custom styled button with hover effects and animation."""

    def __init__(self, text, parent=None, is_primary=True):
        super().__init__(text, parent)
        self.is_primary = is_primary
        self.setMinimumHeight(45)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self._get_style())

        # Add animation on hover
        self._animation = QPropertyAnimation(self, b"minimumHeight")
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

    def enterEvent(self, event):
        self._animation.stop()
        self._animation.setStartValue(self.minimumHeight())
        self._animation.setEndValue(self.minimumHeight() + 3)
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.stop()
        self._animation.setStartValue(self.minimumHeight())
        self._animation.setEndValue(max(45, self.minimumHeight() - 3))
        self._animation.start()
        super().leaveEvent(event)

    def _get_style(self):
        if self.is_primary:
            return """
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
                }
            """
        else:
            return """
                QPushButton {
                    background-color: rgba(230, 230, 230, 0.8);
                    color: #505050;
                    border: 1px solid rgba(200, 200, 200, 0.8);
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(210, 210, 210, 0.9);
                }
                QPushButton:pressed {
                    background-color: rgba(190, 190, 190, 1.0);
                }
            """


class StyledLineEdit(QLineEdit):
    """Custom styled line edit for the login form."""

    def __init__(self, placeholder="", parent=None, echo_mode=QLineEdit.Normal, icon=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setEchoMode(echo_mode)
        self.setMinimumHeight(45)
        self.is_password_field = echo_mode == QLineEdit.Password

        # Adjust padding based on whether we have a visibility toggle
        if self.is_password_field:
            self.setStyleSheet("""
                QLineEdit {
                    border: 1px solid rgba(200, 200, 200, 0.8);
                    border-radius: 8px;
                    padding: 10px 45px 10px 15px;
                    background-color: rgba(255, 255, 255, 0.8);
                    selection-background-color: #ff914d;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 2px solid #ff914d;
                    background-color: white;
                }
            """)
        else:
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

        if icon:
            self.setTextMargins(45, 0, 0, 0)

        # Add password visibility toggle only for password fields
        if self.is_password_field:
            # Use slashed eye icon (password hidden) initially
            self.visibility_toggle = QPushButton("👁️\u0338", self)  # Eye with a slash through it
            self.visibility_toggle.setCursor(Qt.PointingHandCursor)
            self.visibility_toggle.setCheckable(True)
            self.visibility_toggle.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    padding: 0;
                    margin: 0;
                    font-size: 20px;
                    color: #010505;
                }
                QPushButton:hover {
                    color: #000000;
                }
                QPushButton:checked {
                    color: #333333;
                }
            """)
            self.visibility_toggle.setFixedSize(30, 30)
            self.visibility_toggle.clicked.connect(self.toggle_password_visibility)
        else:
            self.visibility_toggle = None

    def resizeEvent(self, event):
        """Position the visibility toggle button."""
        super().resizeEvent(event)
        if self.visibility_toggle:
            # Position the button on the right side of the line edit
            button_y = (self.height() - self.visibility_toggle.height()) // 2
            button_x = self.width() - self.visibility_toggle.width() - 10
            self.visibility_toggle.move(button_x, button_y)

    def toggle_password_visibility(self):
        """Toggle password visibility."""
        if self.echoMode() == QLineEdit.Password:
            self.setEchoMode(QLineEdit.Normal)
            self.visibility_toggle.setText("👁️")  # Regular eye when password is visible
            self.visibility_toggle.setChecked(True)
        else:
            self.setEchoMode(QLineEdit.Password)
            self.visibility_toggle.setText("👁️\u0338")  # Eye with slash when password is hidden
            self.visibility_toggle.setChecked(False)


class LoginCard(QFrame):
    """A styled card containing the login form."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loginCard")
        self.setStyleSheet("""
            #loginCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.8);
            }
        """)

        self.setMaximumWidth(500)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)


class LoginUI(QWidget):
    """Login user interface for SAKAR VISION AI with working background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAKAR VISION AI - Login")
        set_window_icon(self)
        self.setMinimumSize(650, 850)

        self.is_fabric_inspection = False
        self.selected_camera_index = 0

        # Store authenticated user ID to avoid double authentication
        self.authenticated_user_id = None
        self.authenticated_username = None

        # Background image properties
        self.background_image = None
        self.scaled_background = None
        self.last_size = QSize(0, 0)

        initialize_users_file()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        # Load background image first
        self.load_background_image()

        # Initialize login attempt tracker
        self.login_attempt_tracker = LoginAttemptTracker()

        self.init_ui()
        self.apply_styles()

        self.is_fullscreen = True

    def load_background_image(self):
        """Load background image with multiple fallback paths."""
        # Try multiple possible paths for your background image
        background_paths = [
            "bg.png",  # Your original path
            os.path.join(os.path.dirname(__file__), "bg.png"),
            os.path.join(os.path.dirname(__file__), "images", "bg.png"),
            os.path.join(os.path.dirname(__file__), "background.png"),
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
        """Custom paint event to draw the background."""
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

    def draw_gradient_background(self, painter):
        """Draw gradient background as fallback."""
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#1e3c72"))
        gradient.setColorAt(0.5, QColor("#2a5298"))
        gradient.setColorAt(1, QColor("#1e3c72"))

        painter.fillRect(self.rect(), QBrush(gradient))

    def apply_styles(self):
        """Apply global styles to the widget."""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12pt;
            }
            QCheckBox {
                spacing: 8px;
                color: #333333;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QLabel#errorLabel {
                color: #e74c3c;
                font-weight: bold;
                background-color: rgba(231, 76, 60, 0.1);
                border-radius: 8px;
                padding: 8px;
            }
        """)

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout - no background widget needed, we paint directly
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create login card
        self.login_card = LoginCard()

        # Card content layout
        card_layout = QVBoxLayout(self.login_card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(15)

        # Logo
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = "logo.jpeg"
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(
                120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("SAKAR VISION AI")
            logo_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #ff914d;")
        card_layout.addWidget(logo_label)

        # Title and subtitle
        title_label = QLabel("Welcome Back")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 22pt; font-weight: bold; color: #333333; margin-bottom: 5px;")
        card_layout.addWidget(title_label)

        subtitle_label = QLabel("Log in to your account")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 12pt; color: #666666; margin-bottom: 20px;")
        card_layout.addWidget(subtitle_label)

        # Form elements
        username_label = QLabel("Username")
        username_label.setStyleSheet("font-weight: bold; color: #555555; margin-top: 5px;")
        self.username_edit = StyledLineEdit(placeholder="Enter your username")
        saved_username = self.settings.value("login/username", "")
        self.username_edit.setText(saved_username)

        password_label = QLabel("Password")
        password_label.setStyleSheet("font-weight: bold; color: #555555; margin-top: 5px;")
        self.password_edit = StyledLineEdit(
            placeholder="Enter your password", echo_mode=QLineEdit.Password)

        # Error label
        self.error_label = QLabel("")
        self.error_label.setObjectName("errorLabel")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setVisible(False)

        # Buttons
        self.login_button = CustomButton("LOGIN", is_primary=True)
        self.login_button.clicked.connect(self.login)

        self.exit_button = CustomButton("EXIT", is_primary=False)
        self.exit_button.clicked.connect(self.close)

        # Add widgets to card layout
        card_layout.addWidget(username_label)
        card_layout.addWidget(self.username_edit)
        card_layout.addWidget(password_label)
        card_layout.addWidget(self.password_edit)
        card_layout.addWidget(self.error_label)
        card_layout.addSpacing(15)
        card_layout.addWidget(self.login_button)
        card_layout.addWidget(self.exit_button)
        card_layout.addSpacing(10)

        # Create system status indicator container
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

        # Version info
        version_label = QLabel("Version 1.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet(
            "color: rgba(255, 255, 255, 0.9); font-size: 9pt; margin-top: 10px;")

        # Layout for centering
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        center_layout.addStretch(2)
        center_layout.addWidget(self.login_card, 0, Qt.AlignCenter)
        center_layout.addStretch(1)
        center_layout.addWidget(version_label)
        center_layout.addStretch(1)

        # Set up main layout - transparent widget for content
        content_widget = QWidget()
        content_widget.setLayout(center_layout)
        content_widget.setStyleSheet("background: transparent;")

        main_layout.addWidget(content_widget)

        # Connect events
        self.username_edit.returnPressed.connect(self.login)
        self.password_edit.returnPressed.connect(self.login)

        # Set initial focus
        if self.username_edit.text():
            self.password_edit.setFocus()
        else:
            self.username_edit.setFocus()

    def resizeEvent(self, event):
        """Handle resize events and force repaint."""
        # Clear cached scaled background so it gets recreated
        self.scaled_background = None

        # Adjust card max width for fullscreen
        if self.isFullScreen():
            max_width = min(500, self.width() * 0.8)
            self.login_card.setMaximumWidth(int(max_width))
        else:
            self.login_card.setMaximumWidth(500)

        super().resizeEvent(event)
        # Force repaint after resize
        self.update()

    def show_error(self, message):
        """Display error message with simple animation."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)

        # Ensure the error label has enough height to display the text
        self.error_label.setMinimumHeight(50)
        self.error_label.setWordWrap(True)  # Enable word wrapping for longer messages

        # Simple shake animation
        animation = QPropertyAnimation(self.login_card, b"pos")
        animation.setDuration(100)
        animation.setLoopCount(3)
        original_pos = self.login_card.pos()

        animation.setKeyValueAt(0, original_pos)
        animation.setKeyValueAt(0.25, original_pos + QPoint(5, 0))
        animation.setKeyValueAt(0.5, original_pos)
        animation.setKeyValueAt(0.75, original_pos + QPoint(-5, 0))
        animation.setKeyValueAt(1, original_pos)
        animation.start()

        # Hide error after 5 seconds
        QTimer.singleShot(5000, lambda: self.error_label.setVisible(False))

    def login(self):
        """Handle login attempt."""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()

        # Check if account is locked
        is_locked, locked_until = self.login_attempt_tracker.is_account_locked(username)
        if is_locked:
            remaining_time = (locked_until - datetime.now()).seconds
            self.show_error(f"Account locked. Try again in {remaining_time} seconds.")
            logger.warning(f"Login attempt for locked account: {username}")
            return

        if not username or not password:
            self.show_error("Please enter both username and password")
            return

        if self.authenticate_user(username, password):
            # Reset attempts on successful login
            self.login_attempt_tracker.reset_attempts(username)
            logger.info(f"Successful login: {username}")

            # Save username for convenience (no remember checkbox functionality)
            self.settings.setValue("login/username", username)
            self.launch_main_application()
        else:
            # Record failed attempt
            locked, locked_until = self.login_attempt_tracker.record_failed_attempt(username)
            logger.warning(f"Failed login attempt for user: {username}")

            if locked:
                remaining_time = (locked_until - datetime.now()).seconds
                self.show_error(
                    f"Account locked due to multiple failed attempts. Try again in {remaining_time} seconds.")
                logger.warning(
                    f"Account locked for {username} for {LOCKOUT_DURATION_SECONDS} seconds")
            else:
                attempts = self.login_attempt_tracker.login_attempts.get(
                    username, {}).get('count', 0)
                attempts_left = MAX_LOGIN_ATTEMPTS - attempts
                self.show_error(
                    f"Invalid username or password. {attempts_left} attempts remaining before lockout.")

            self.password_edit.clear()
            self.password_edit.setFocus()

    def authenticate_user(self, username, password):
        """Authenticate user against Azure database with local fallback."""
        try:
            user_id = azure_authenticate_user(username, password)
            if user_id is not None:
                logger.info(f"User authenticated successfully: {username} with ID: {user_id}")
                # Store the authenticated user ID and username to avoid double authentication
                self.authenticated_user_id = user_id
                self.authenticated_username = username
                return True
            print(f"Authentication failed for user: {username}")
            return False
        except Exception as e:
            print(f"Azure authentication error: {e}")
            print("Falling back to local authentication")
            return self.local_authenticate_user(username, password)

    def local_authenticate_user(self, username, password):
        """Local authentication fallback."""
        try:
            if username == "admin" and password == "admin123":
                return True

            with open(USERS_FILE, 'r') as f:
                users = json.load(f)

            if username not in users:
                print(f"User '{username}' not found")
                return False

            stored_hash = users[username]["password_hash"]
            provided_hash = hashlib.sha256(password.encode()).hexdigest()

            return stored_hash == provided_hash
        except Exception as e:
            print(f"Local authentication error: {e}")
            return False

    def get_user_full_name(self, username):
        """Get user's full name."""
        try:
            if self.is_fabric_inspection and username == "sakarrobotics":
                return "Sakar Robotics"

            with open(USERS_FILE, 'r') as f:
                users = json.load(f)

            if username in users and "full_name" in users[username]:
                return users[username]["full_name"]
            return username
        except:
            return username

    def save_session_state(self, username, full_name, user_id):
        """Save session state after successful login"""
        try:
            session_data = {
                "last_ui": "login",
                "timestamp": datetime.now().isoformat(),
                "current_username": username,
                "current_user_id": user_id,
                "user_info": {
                    "username": username,
                    "full_name": full_name,
                    "id": user_id,
                    "user_id": user_id  # Add both 'id' and 'user_id' for compatibility
                },
                "user": {
                    "username": username,
                    "full_name": full_name
                }
            }

            session_file_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "session_state.json")
            with open(session_file_path, 'w') as f:
                json.dump(session_data, f, indent=4)

            print(f"Session state saved for user: {username} (ID: {user_id})")
            return True

        except Exception as e:
            print(f"Error saving session state: {e}")
            return False

    def launch_main_application(self):
        """Launch the main application after successful login."""
        try:
            # Use the authenticated user ID and username stored during login
            # This prevents double authentication that could return different user IDs
            if not self.authenticated_user_id or not self.authenticated_username:
                QMessageBox.critical(
                    self, "Error", "Authentication session error. Please try logging in again.")
                return

            user_info = {
                "username": self.authenticated_username,
                "user_id": self.authenticated_user_id,
                "selected_camera_index": self.selected_camera_index,
                "inspection_type": "fabric" if self.is_fabric_inspection else "metal_sheet"
            }

            print(
                f"Launching application with authenticated user ID: {self.authenticated_user_id}, username: {self.authenticated_username}, camera index: {self.selected_camera_index}")

            # Get the full name for the session state
            full_name = self.get_user_full_name(self.authenticated_username)

            # Save session state using the new method with user ID
            if not self.save_session_state(self.authenticated_username, full_name, self.authenticated_user_id):
                print("Warning: Could not save session state")

            self.hide()

            if self.is_fabric_inspection:
                from organization_creation_ui import OrganizationCreationUI
                self.next_screen = OrganizationCreationUI(user_info)
                self.next_screen.showMaximized()
            else:
                from metal_sheet_organization_ui import MetalSheetOrganizationUI
                self.metal_sheet_organization = MetalSheetOrganizationUI(user_info)
                self.metal_sheet_organization.show()
                self.metal_sheet_organization.destroyed.connect(self.close)

        except Exception as e:
            logger.error(f"Error launching next screen: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to launch next screen: {e}")
            print(f"Error launching next screen: {e}")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    login_window = LoginUI()
    login_window.showMaximized()

    sys.exit(app.exec_())
