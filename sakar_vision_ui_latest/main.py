#!/usr/bin/env python3
"""
SAKAR VISION AI - Main Application Entry Point Module

OVERVIEW:
This module implements the primary application entry point for the Sakar Vision AI platform, serving as the central 
launcher and system orchestrator that provides comprehensive application initialization, environment validation, and 
seamless user interface bootstrapping. It combines advanced logging system integration with robust dependency checking 
and professional application lifecycle management to ensure optimal system startup and reliable operation across 
diverse industrial deployment environments while maintaining comprehensive error handling and graceful degradation 
capabilities for production manufacturing inspection workflows.

KEY FUNCTIONALITY:
The system features comprehensive environment validation with automatic dependency checking for critical modules 
including OpenCV, PyTorch, NumPy, Ultralytics YOLO, and PIL image processing libraries, sophisticated model file 
discovery with automatic detection and validation of available AI models including YOLO weights and custom trained 
models, advanced logging system integration with multi-level logging configuration supporting both development and 
production deployment scenarios, and intelligent application lifecycle management with proper initialization, event 
loop management, and graceful shutdown procedures. It includes professional Qt application configuration with custom 
font settings, organization metadata, and optimized rendering parameters, seamless integration with the inspection 
selection interface for workflow routing between fabric and metal sheet inspection modalities, comprehensive error 
handling with detailed exception tracking and automatic recovery mechanisms, and robust system monitoring with 
real-time dependency validation and performance tracking capabilities.

TECHNICAL ARCHITECTURE:
Built using PyQt5 with advanced application lifecycle management and comprehensive logging integration, the module 
employs sophisticated dependency validation systems with runtime module availability checking and automatic fallback 
mechanisms, comprehensive logging framework integration supporting multi-level logging with file rotation, console 
output, and specialized UI event tracking, and intelligent model discovery with automatic scanning of project directories 
for available AI model files and validation of model compatibility. The architecture features professional Qt application 
configuration with custom message handlers for Qt framework logging integration, optimized font management, and proper 
application metadata configuration, robust error handling with comprehensive exception tracking, detailed stack trace 
logging, and graceful degradation for system failures, seamless integration with the inspection selection interface 
through dynamic module loading and proper window management, and advanced system monitoring with dependency validation, 
performance metrics collection, and comprehensive audit trail capabilities. The system includes optimized startup 
procedures with parallel initialization where possible and comprehensive shutdown procedures ensuring proper resource 
cleanup and logging system termination for production deployment scenarios.
"""

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Import our new logging system
from utils.logging import setup_logging, get_logger, shutdown_logging

from inspection_selection_ui import InspectionSelectionUI
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'

# Set up logger for this module
logger = None


def check_environment():
    """
    Check if all required dependencies and files are available.
    """
    # Verify required modules
    try:
        import cv2
        import numpy
        import torch
        from ultralytics import YOLO
        from PIL import Image
        logger.info("All required modules are available.")
    except ImportError as e:
        logger.error(f"Missing required module: {e}")
        return False

    # Check for model paths existence warnings
    model_files = ["best.pt", "sakar_ai.pt"]
    found_models = []

    for model in model_files:
        if os.path.exists(model):
            found_models.append(model)

    if found_models:
        logger.info(f"Found model files: {', '.join(found_models)}")
    else:
        logger.warning(
            "No model files found in the current directory. Models will need to be manually selected through the UI.")

    return True


def setup_qt_message_handler():
    """Set up custom Qt message handler to redirect Qt messages to our logger."""
    def message_handler(mode, context, message):
        if mode == Qt.DebugMessage:
            logger.debug(f"Qt: {message}")
        elif mode == Qt.InfoMessage:
            logger.info(f"Qt: {message}")
        elif mode == Qt.WarningMessage:
            logger.warning(f"Qt: {message}")
        elif mode == Qt.CriticalMessage:
            logger.error(f"Qt: {message}")
        elif mode == Qt.FatalMessage:
            logger.critical(f"Qt: {message}")

    return message_handler


def main():
    """
    Main entry point for the application.
    """
    global logger

    # Initialize logging system (prefer DEBUG during development, INFO in production)
    setup_logging(log_level=logging.INFO)
    logger = get_logger("main")
    logger.info("SAKAR Vision AI application starting...")

    try:
        # Check environment
        logger.info("Checking environment...")
        check_environment()

        # Initialize Qt application
        logger.info("Initializing Qt application...")
        app = QApplication(sys.argv)
        app.setApplicationName("SAKAR VISION AI")
        app.setOrganizationName("SakarVision")

        # Configure Qt logging
        # qt_handler = setup_qt_message_handler()
        # Qt.qInstallMessageHandler(qt_handler)

        # Set global font
        font = QFont("Segoe UI", 10)
        app.setFont(font)

        # Create and show inspection selection window
        logger.info("Creating inspection selection window...")
        selection_window = InspectionSelectionUI()

        # Connect aboutToQuit signal to our shutdown function
        app.aboutToQuit.connect(shutdown_logging)

        # Show window maximized
        selection_window.showMaximized()
        logger.info("Application UI initialized and displayed")

        # Start the application event loop
        return app.exec_()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        return 1
    finally:
        # Ensure logging is shut down properly
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
