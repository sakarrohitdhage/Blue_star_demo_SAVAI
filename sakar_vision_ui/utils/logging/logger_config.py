#!/usr/bin/env python3
"""
logger_config.py - Central logging configuration for SAKAR Vision AI

This module provides a centralized logging configuration system for the entire SAKAR Vision AI
application. It offers various loggers tailored to different modules and components.

Features:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple log handlers (console, file, rotating file)
- Custom formatting for different logger types
- Component-specific loggers with appropriate settings
"""

import os
import sys
import logging
import logging.handlers
import traceback
from datetime import datetime
from pathlib import Path

# Base directory for logs - create logs folder at application root
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Default formatting settings
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
CONSOLE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Maximum log file size and backup count
MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 10  # Keep 10 backup files


class LoggerFactory:
    """Factory class for creating and configuring loggers."""

    _loggers = {}  # Cache for loggers

    @staticmethod
    def get_logger(name, level=logging.INFO, detailed=False, console=True, file=True):
        """
        Get or create a logger with the specified name and configuration.

        Args:
            name (str): Logger name, typically the module name
            level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            detailed (bool): Whether to use detailed formatting
            console (bool): Whether to add console handler
            file (bool): Whether to add file handler

        Returns:
            logging.Logger: Configured logger
        """
        # If logger already exists, return it
        if name in LoggerFactory._loggers:
            return LoggerFactory._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Determine formatting style
        formatter = logging.Formatter(
            DETAILED_FORMAT if detailed else DEFAULT_FORMAT
        )

        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
            logger.addHandler(console_handler)

        # Add file handler if requested
        if file:
            log_file = os.path.join(LOG_DIR, f"{name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_SIZE_BYTES,
                backupCount=BACKUP_COUNT
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Store logger in cache
        LoggerFactory._loggers[name] = logger
        return logger


# Pre-configured loggers for major modules
def get_ui_logger(component_name):
    """
    Get a logger configured for UI components.

    Args:
        component_name (str): Name of the UI component

    Returns:
        logging.Logger: Configured logger for UI components
    """
    return LoggerFactory.get_logger(
        f"ui.{component_name}",
        level=logging.INFO,
        detailed=False,
        console=True,
        file=True
    )


def get_model_logger(model_name):
    """
    Get a logger configured for model operations.

    Args:
        model_name (str): Name of the model or model operation

    Returns:
        logging.Logger: Configured logger for model operations
    """
    return LoggerFactory.get_logger(
        f"model.{model_name}",
        level=logging.INFO,
        detailed=True,  # Detailed logging for model operations
        console=True,
        file=True
    )


def get_azure_logger(service_name):
    """
    Get a logger configured for Azure service operations.

    Args:
        service_name (str): Name of the Azure service (storage, database, etc.)

    Returns:
        logging.Logger: Configured logger for Azure operations
    """
    return LoggerFactory.get_logger(
        f"azure.{service_name}",
        level=logging.INFO,
        detailed=True,  # Detailed logging for Azure operations
        console=True,
        file=True
    )


def get_data_logger(dataset_name=None):
    """
    Get a logger configured for data operations.

    Args:
        dataset_name (str, optional): Name of the dataset

    Returns:
        logging.Logger: Configured logger for data operations
    """
    name = "data"
    if dataset_name:
        name = f"data.{dataset_name}"

    return LoggerFactory.get_logger(
        name,
        level=logging.INFO,
        detailed=False,
        console=True,
        file=True
    )


# Main application logger
def get_application_logger():
    """
    Get the main application logger.

    Returns:
        logging.Logger: Main application logger
    """
    return LoggerFactory.get_logger(
        "sakar_vision",
        level=logging.INFO,
        detailed=True,
        console=True,
        file=True
    )


# Error logger with exception handling
def get_error_logger():
    """
    Get a logger specifically for error handling.

    Returns:
        logging.Logger: Error logger
    """
    return LoggerFactory.get_logger(
        "error",
        level=logging.ERROR,
        detailed=True,
        console=True,
        file=True
    )


def log_exception(logger, e):
    """
    Helper function to log an exception with traceback.

    Args:
        logger (logging.Logger): Logger to use
        e (Exception): Exception to log
    """
    logger.error(f"Exception: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")


# Create a master log file that aggregates critical logs from all components
def setup_master_logger():
    """
    Configure a master logger that collects all critical logs.

    Returns:
        logging.Logger: Master logger
    """
    logger = logging.getLogger("master")
    logger.setLevel(logging.WARNING)  # Only WARNING and above

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Master log file with date in filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"sakar_vision_master_{date_str}.log")

    # Create a file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE_BYTES * 2,  # Larger size for master log
        backupCount=BACKUP_COUNT * 2       # More backups for master log
    )

    # Create formatter with extra-detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


# Setup master logger at module import time
master_logger = setup_master_logger()


# Add a sample usage function to demonstrate how to use the loggers
def demonstrate_logging():
    """
    Demonstrate how to use the various loggers.
    """
    # Application logger
    app_logger = get_application_logger()
    app_logger.info("This is an application-wide log message")
    app_logger.warning("This is a warning from the main application")

    # UI Component logger
    ui_logger = get_ui_logger("main_window")
    ui_logger.info("Main window initialized")
    ui_logger.debug("Button clicked")

    # Model logger
    model_logger = get_model_logger("YOLO")
    model_logger.info("Model loaded successfully")
    model_logger.debug("Inference took 125ms")

    # Azure logger
    azure_logger = get_azure_logger("storage")
    azure_logger.info("Connected to Azure Storage")
    azure_logger.debug("Uploaded file to container: example.jpg")

    # Data logger
    data_logger = get_data_logger("training")
    data_logger.info("Processed 1000 images for training")

    # Error logging
    error_logger = get_error_logger()
    try:
        # Simulate an error
        1/0
    except Exception as e:
        log_exception(error_logger, e)

    print(f"Log files have been created in {LOG_DIR}")


if __name__ == "__main__":
    demonstrate_logging()
