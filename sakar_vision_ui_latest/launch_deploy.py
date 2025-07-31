#!/usr/bin/env python3
"""
SAKAR VISION AI - Launch Deploy Module

OVERVIEW:
This module implements a sophisticated smart launcher and session management system for the Sakar Vision AI platform, 
serving as the primary application entry point that provides intelligent UI restoration and seamless user experience 
continuity across application restarts. It combines advanced session persistence with comprehensive system validation 
to ensure optimal deployment readiness while maintaining user workflow context and preferences through intelligent 
session state management and automatic UI restoration capabilities for industrial manufacturing environments.

KEY FUNCTIONALITY:
The system features intelligent session persistence with automatic restoration of the last used UI interface upon 
application restart, comprehensive system connectivity validation with real-time internet and camera availability 
checking before launching deployment operations, sophisticated configuration management with automatic creation and 
validation of mock configuration files for seamless testing and development workflows, and advanced UI lifecycle 
management with proper session state tracking and enhanced close event handling for reliable state persistence. 
It includes comprehensive system requirements validation through integrated SystemCheckOverlay with visual progress 
indicators and automatic retry mechanisms, flexible UI launching capabilities supporting multiple interface types 
including deployment, image classification, camera feed, demo feed, and manual annotation interfaces, robust error 
handling with comprehensive logging integration and graceful degradation for system failures, and seamless integration 
with the custom logging framework for detailed application monitoring and debugging capabilities.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced session management architecture and comprehensive state persistence mechanisms, the 
module employs sophisticated JSON-based configuration management with automatic file creation and validation for 
organization, location, and defect selection settings, dynamic module loading with runtime class instantiation for 
flexible UI management and extensibility, and comprehensive connectivity validation with multi-protocol testing for 
internet and camera availability. The architecture features intelligent session state management with persistent 
JSON storage for user preferences and UI context restoration, modular UI mapping system enabling easy addition of 
new interface types through the UI_MODULES configuration dictionary, robust overlay dialog implementation 
(SystemCheckOverlay) with real-time progress tracking and automatic retry mechanisms for system validation, and 
comprehensive error handling with detailed logging integration supporting both custom and standard logging frameworks. 
The system includes advanced application lifecycle management with proper resource cleanup and graceful shutdown 
procedures for production deployment scenarios.
"""

import os
import sys
import json
import traceback
import time
import socket
import cv2
import logging
from datetime import datetime
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton

# Import our logging system
try:
    from utils.logging import setup_logging, get_logger, shutdown_logging
    logging_available = True
except ImportError:
    logging_available = False
    print("Warning: Custom logging system not found, using standard logging")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'

# Configuration file paths - these would normally be created by previous screens
CONFIG_PATHS = {
    "organization": os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal_sheet_organizations.json"),
    "location": os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal_sheet_location_config.json"),
    "defects": os.path.join(os.path.dirname(os.path.abspath(__file__)), "defects_config.json"),
    # NEW: Session state file
    "session": os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_state.json"),
}

# Set up logger for this module
logger = None

# UI mapping for session persistence
UI_MODULES = {
    "deploy": {"module": "deploy_ui", "class": "DeploymentUI"},
    "image_classification": {"module": "image_classi_ui", "class": "ImageClassiUI"},
    "camera_feed": {"module": "camera_feed_cloud_ui", "class": "CameraFeedUI"},
    "demo_feed": {"module": "demo_feed_ui", "class": "ModernDemoFeedUI"},
    "manual_annotation": {"module": "manual_annotation_ui", "class": "ImageAnnotationTool"},
    "fabric_deploy": {"module": "fabric_deploy_ui", "class": "FabricDeploymentUI"},
}


def save_session_state(ui_name, user_data=None, additional_data=None):
    """Save the current session state to JSON file"""
    try:
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "user": user_data or create_mock_user(),
            "additional_data": additional_data or {}
        }

        with open(CONFIG_PATHS["session"], 'w') as f:
            json.dump(session_data, f, indent=4)

        if logger:
            logger.info(f"Session state saved: {ui_name}")
        else:
            print(f"Session state saved: {ui_name}")

    except Exception as e:
        error_msg = f"Error saving session state: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)


def load_session_state():
    """Load the last session state from JSON file"""
    try:
        if os.path.exists(CONFIG_PATHS["session"]):
            with open(CONFIG_PATHS["session"], 'r') as f:
                session_data = json.load(f)

            if logger:
                logger.info(f"Session state loaded: {session_data.get('last_ui', 'unknown')}")
            else:
                print(f"Session state loaded: {session_data.get('last_ui', 'unknown')}")

            return session_data
        else:
            if logger:
                logger.info("No previous session found, will start with default UI")
            else:
                print("No previous session found, will start with default UI")
            return None

    except Exception as e:
        error_msg = f"Error loading session state: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None


def create_mock_user():
    """Create a mock user for testing."""
    return {
        "username": "admin",
        "full_name": "Administrator"
    }


def create_mock_configs():
    """Create all necessary configuration files with default values"""
    if logger:
        logger.info("Creating mock configuration files for deployment...")
    else:
        print("Creating mock configuration files for deployment...")

    # Create organization config if needed
    if not os.path.exists(CONFIG_PATHS["organization"]):
        if logger:
            logger.info("Creating mock organization configuration...")
        else:
            print("Creating mock organization configuration...")

        org_config = {
            "Sakarrobotics": {
                "created_by": "admin",
                "creation_date": datetime.now().isoformat(),
                "locations": ["Sakarrobotics_demo"],
                "location_projects": {
                    "Sakarrobotics_demo": ["default", "ai"]
                }
            }
        }
        with open(CONFIG_PATHS["organization"], 'w') as f:
            json.dump(org_config, f, indent=4)

        if logger:
            logger.info("Created organization configuration.")
        else:
            print("Created organization configuration.")

    # Create location config if needed
    if not os.path.exists(CONFIG_PATHS["location"]):
        if logger:
            logger.info("Creating mock location configuration...")
        else:
            print("Creating mock location configuration...")

        location_config = {
            "organization": "Sakarrobotics",
            "location": "Sakarrobotics_demo",
            "project": "ai",
            "user": "admin",
            "timestamp": datetime.now().isoformat(),
            "inspection_type": "metal_sheet"
        }
        with open(CONFIG_PATHS["location"], 'w') as f:
            json.dump(location_config, f, indent=4)

        if logger:
            logger.info("Created location configuration.")
        else:
            print("Created location configuration.")

    # Create defects config if needed
    if not os.path.exists(CONFIG_PATHS["defects"]):
        if logger:
            logger.info("Creating mock defects configuration...")
        else:
            print("Creating mock defects configuration...")

        defects_config = {
            "selected_defects": ["crack", "dent", "scratch", "corrosion", "hole"]
        }
        with open(CONFIG_PATHS["defects"], 'w') as f:
            json.dump(defects_config, f, indent=4)

        if logger:
            logger.info("Created defects configuration.")
        else:
            print("Created defects configuration.")


def simulate_login():
    """Simulate the login process."""
    print("\n======== Simulating Login UI ========")
    print("User: admin")
    print("Authentication successful!")
    print("Login UI completed")
    print("=====================================\n")


def simulate_organization_selection():
    """Simulate the organization selection process."""
    print("\n======== Simulating Organization Selection UI ========")

    # Read organization if it exists, otherwise use default
    org_name = "Sakarrobotics"
    if os.path.exists(CONFIG_PATHS["organization"]):
        try:
            with open(CONFIG_PATHS["organization"], 'r') as f:
                orgs = json.load(f)
                if orgs:
                    org_name = list(orgs.keys())[0]
        except Exception as e:
            print(f"Error reading organization file: {e}")

    print(f"Selected organization: {org_name}")
    print("Organization Selection UI completed")
    print("====================================================\n")
    return org_name


def simulate_location_selection(org_name):
    """Simulate the location selection process."""
    print("\n======== Simulating Location Selection UI ========")

    # Default location and project
    location_name = "Sakarrobotics_demo"
    project_name = "ai"

    # Read from file if exists
    if os.path.exists(CONFIG_PATHS["location"]):
        try:
            with open(CONFIG_PATHS["location"], 'r') as f:
                config = json.load(f)
                location_name = config.get("location", location_name)
                project_name = config.get("project", project_name)
        except Exception as e:
            print(f"Error reading location config: {e}")

    print(f"Organization: {org_name}")
    print(f"Selected location: {location_name}")
    print(f"Selected project: {project_name}")
    print("Location Selection UI completed")
    print("=================================================\n")
    return location_name, project_name


def simulate_defect_selection(org_name, location_name, project_name):
    """Simulate the defect selection process."""
    print("\n======== Simulating Defect Selection UI ========")

    selected_defects = ["crack", "dent", "scratch", "corrosion", "hole"]

    # Read from file if exists
    if os.path.exists(CONFIG_PATHS["defects"]):
        try:
            with open(CONFIG_PATHS["defects"], 'r') as f:
                config = json.load(f)
                selected_defects = config.get("selected_defects", selected_defects)
        except Exception as e:
            print(f"Error reading defects config: {e}")

    print(f"Organization: {org_name}")
    print(f"Location: {location_name}")
    print(f"Project: {project_name}")
    print(f"Selected defects: {', '.join(selected_defects)}")
    print("Defect Selection UI completed")
    print("=================================================\n")
    return selected_defects


def check_internet():
    """Check for internet connectivity."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except Exception as e:
        print(f"Internet check failed: {e}")
        return False


def check_camera(last_used_camera):
    """Check if the last used camera is available."""
    try:
        cap = cv2.VideoCapture(last_used_camera)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception as e:
        print(f"Camera check failed: {e}")
        return False


def continuous_system_check_and_launch():
    """Continuously check system connectivity and launch deployment UI if all checks pass."""
    last_used_camera = 0  # Default to camera index 0; this can be loaded from a config file

    # Load last used camera index from location config if available
    if os.path.exists(CONFIG_PATHS["location"]):
        try:
            with open(CONFIG_PATHS["location"], 'r') as f:
                config = json.load(f)
                last_used_camera = config.get("selected_camera", 0)
        except Exception as e:
            print(f"Error reading location config for camera: {e}")

    while True:
        internet_ok = check_internet()
        camera_ok = check_camera(last_used_camera)

        if internet_ok and camera_ok:
            print("System checks passed. Launching Deployment UI...")
            launch_deployment_ui()
            break
        else:
            if not internet_ok:
                print("Internet connection not available. Retrying...")
            if not camera_ok:
                print(f"Camera {last_used_camera} not available. Retrying...")

            time.sleep(5)  # Retry after 5 seconds


def launch_ui_by_name(ui_name):
    """Launch a specific UI by name"""
    try:
        if ui_name not in UI_MODULES:
            if logger:
                logger.warning(f"Unknown UI name: {ui_name}, falling back to deploy UI")
            else:
                print(f"Unknown UI name: {ui_name}, falling back to deploy UI")
            ui_name = "deploy"

        ui_config = UI_MODULES[ui_name]
        module_name = ui_config["module"]
        class_name = ui_config["class"]

        if logger:
            logger.info(f"Launching {ui_name} UI: {module_name}.{class_name}")
        else:
            print(f"Launching {ui_name} UI: {module_name}.{class_name}")

        # Import the module dynamically
        module = __import__(module_name)
        ui_class = getattr(module, class_name)

        # Create and show the UI
        ui = ui_class()

        # Set up session saving when UI is closed
        original_close_event = ui.closeEvent if hasattr(ui, 'closeEvent') else None

        def enhanced_close_event(event):
            # Save session state before closing
            save_session_state(ui_name, create_mock_user(), {
                "closed_at": datetime.now().isoformat()
            })

            # Call original close event if it exists
            if original_close_event:
                original_close_event(event)
            else:
                event.accept()

        ui.closeEvent = enhanced_close_event

        # Show the UI
        ui.showMaximized()

        # Save session state for this UI launch
        save_session_state(ui_name, create_mock_user(), {
            "launched_at": datetime.now().isoformat()
        })

        if logger:
            logger.info(f"{ui_name} UI launched successfully!")
        else:
            print(f"{ui_name} UI launched successfully!")

        return ui

    except Exception as e:
        error_msg = f"Error launching {ui_name} UI: {e}"
        if logger:
            logger.critical(error_msg, exc_info=True)
        else:
            print(error_msg)
            traceback.print_exc()
        return None


def launch_deployment_ui():
    """Launch the deployment UI directly."""
    return launch_ui_by_name("deploy")


def launch_last_used_ui():
    """Launch the last used UI based on session state"""
    session_data = load_session_state()

    if session_data and 'last_ui' in session_data:
        last_ui = session_data['last_ui']
        if logger:
            logger.info(f"Restoring last UI: {last_ui}")
        else:
            print(f"Restoring last UI: {last_ui}")

        return launch_ui_by_name(last_ui)
    else:
        if logger:
            logger.info("No previous session found, launching default deploy UI")
        else:
            print("No previous session found, launching default deploy UI")

        return launch_deployment_ui()


class SystemCheckOverlay(QDialog):
    """Overlay widget for checking system requirements before launching deployment UI."""

    def __init__(self, last_used_camera, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Connectivity Check")
        self.setFixedSize(500, 450)
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 2px solid #ff914d;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background-color: #f5f5f5;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #ff914d;
                border-radius: 5px;
            }
        """)

        # Set application icon
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'sakar.png')))

        self.last_used_camera = last_used_camera
        self.internet_ok = False
        self.camera_ok = False

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        title = QLabel("System Requirements Check")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #333333;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Internet connectivity section
        self.internet_status = QLabel("Checking internet connectivity...")
        self.internet_status.setStyleSheet("font-size: 14px; color: #555555;")
        layout.addWidget(self.internet_status)

        self.internet_progress = QProgressBar()
        self.internet_progress.setMinimum(0)
        self.internet_progress.setMaximum(100)
        self.internet_progress.setValue(0)
        layout.addWidget(self.internet_progress)

        # Camera availability section
        self.camera_status = QLabel("Checking camera availability...")
        self.camera_status.setStyleSheet("font-size: 14px; color: #555555;")
        layout.addWidget(self.camera_status)

        self.camera_progress = QProgressBar()
        self.camera_progress.setMinimum(0)
        self.camera_progress.setMaximum(100)
        self.camera_progress.setValue(0)
        layout.addWidget(self.camera_progress)

        # Start checks
        self.start_checks()

    def start_checks(self):
        """Start the connectivity checks."""
        self.internet_progress.setValue(0)
        self.camera_progress.setValue(0)
        self.internet_status.setText("Checking internet connectivity...")
        self.camera_status.setText("Checking camera availability...")

        QTimer.singleShot(200, self.check_internet)

    def check_internet(self):
        """Check for internet connectivity."""
        self.internet_progress.setValue(30)
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.internet_ok = True
            self.internet_status.setText("✓ Internet connection available")
            self.internet_status.setStyleSheet("font-size: 14px; color: green;")
            self.internet_progress.setValue(100)
        except Exception as e:
            self.internet_ok = False
            self.internet_status.setText(f"✗ No internet connection: {str(e)}")
            self.internet_status.setStyleSheet("font-size: 14px; color: red;")
            self.internet_progress.setValue(100)

        QTimer.singleShot(500, self.check_camera)

    def check_camera(self):
        """Check for camera availability."""
        self.camera_progress.setValue(30)
        try:
            cap = cv2.VideoCapture(self.last_used_camera)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.camera_ok = True
                    self.camera_status.setText("✓ Camera available")
                    self.camera_status.setStyleSheet("font-size: 14px; color: green;")
                    self.camera_progress.setValue(100)
                    self.check_complete()
                    return
        except Exception as e:
            print(f"Camera check failed: {e}")

        self.camera_ok = False
        self.camera_status.setText("✗ Camera not available")
        self.camera_status.setStyleSheet("font-size: 14px; color: red;")
        self.camera_progress.setValue(100)

        # Automatically retry checks
        QTimer.singleShot(2000, self.start_checks)

    def check_complete(self):
        """Handle completion of all checks."""
        if self.internet_ok and self.camera_ok:
            self.accept()
        else:
            # Automatically retry checks
            QTimer.singleShot(2000, self.start_checks)


# Update the logic to fetch the last used camera from inspection_settings.json

def get_last_used_camera():
    """Fetch the last used camera index from inspection_settings.json."""
    settings_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "inspection_settings.json")
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                return settings.get("selected_camera", 0)  # Default to camera index 0
        except Exception as e:
            print(f"Error reading inspection settings: {e}")
    return 0  # Default to camera index 0


# Replace the main function to include session persistence
def main():
    """Main function with session persistence - launches last used UI"""
    global logger

    # Initialize logging system if available
    if logging_available:
        setup_logging(log_level=logging.INFO)
        logger = get_logger("deploy_launcher")
        logger.info("SAKAR Vision AI Smart Launcher - Starting with session persistence...")
    else:
        print("\n===== SAKAR VISION AI - Smart Launcher =====")
        print("This script will restore your last opened UI after system checks.")
        print("============================================\n")

    try:
        # Create mock configuration files if needed
        create_mock_configs()

        # Initialize QApplication
        app = QApplication(sys.argv)

        # Register logger shutdown with app quit if available
        if logging_available:
            app.aboutToQuit.connect(shutdown_logging)

        # Load last used camera index from location config if available
        last_used_camera = get_last_used_camera()

        # Show system check overlay
        overlay = SystemCheckOverlay(last_used_camera)
        if overlay.exec_() == QDialog.Accepted:
            # Launch last used UI instead of always launching deploy UI
            launch_last_used_ui()

        # Run the application
        return app.exec_()
    except Exception as e:
        error_msg = f"Unhandled exception in launch_deploy: {e}"
        if logger:
            logger.critical(error_msg, exc_info=True)
        else:
            print(error_msg)
            traceback.print_exc()
        return 1
    finally:
        # Ensure logging is shut down properly if available
        if logging_available:
            shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
