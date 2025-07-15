#!/usr/bin/env python3
"""
Logging package for SAKAR Vision AI

This package provides comprehensive logging capabilities for the SAKAR Vision AI application.
It includes loggers for general application use, UI events, model operations, and Azure interactions.

Available modules:
- logger_config: Central logging configuration
- performance_logger: Performance monitoring utilities
- ui_event_logger: UI event tracking
- azure_logger: Azure operations logging
- model_logger: AI model operations logging
- console_logger: Console output capture to log.txt

Example usage:
    from utils.logging import setup_logging, get_logger
    
    # Initialize logging for the application
    setup_logging()
    
    # Get a logger for a specific module
    logger = get_logger("my_module")
    logger.info("This is a log message")
"""

import os
import sys
import logging
from pathlib import Path

# Import all logging modules
from .logger_config import (
    LoggerFactory,
    get_ui_logger,
    get_model_logger,
    get_azure_logger,
    get_data_logger,
    get_application_logger,
    get_error_logger,
    log_exception,
    setup_master_logger
)

from .performance_logger import (
    Timer,
    time_function,
    PerformanceMonitor
)

from .ui_event_logger import (
    UIEventTracker,
    UIEventMonitor,
    track_ui_method
)

from .azure_logger import (
    AzureOperationTracker,
    track_azure_operation,
    AzureStorageLogger,
    AzureDatabaseLogger
)

from .model_logger import (
    ModelLogger,
    track_model_inference,
    YOLOModelLogger,
    MobileNetModelLogger
)

# Import the new console logger
from .console_logger import (
    initialize_console_logger,
    shutdown_console_logger,
    get_log_file_path
)

# Define public API
__all__ = [
    # Core logging
    'setup_logging',
    'get_logger',
    'log_exception',

    # Logger factories
    'get_ui_logger',
    'get_model_logger',
    'get_azure_logger',
    'get_data_logger',
    'get_application_logger',
    'get_error_logger',

    # Performance monitoring
    'Timer',
    'time_function',
    'PerformanceMonitor',

    # UI event logging
    'UIEventTracker',
    'UIEventMonitor',
    'track_ui_method',

    # Azure logging
    'AzureOperationTracker',
    'track_azure_operation',
    'AzureStorageLogger',
    'AzureDatabaseLogger',

    # Model logging
    'ModelLogger',
    'track_model_inference',
    'YOLOModelLogger',
    'MobileNetModelLogger',

    # Console logging
    'initialize_console_logger',
    'shutdown_console_logger',
    'get_log_file_path',
]

# Application name for logging
APP_NAME = "SAKAR_Vision_AI"

# Singleton loggers
_app_logger = None
_perf_monitor = None


def setup_logging(log_level=logging.INFO, enable_performance_monitoring=True,
                  monitor_interval=60, capture_console_output=True):
    """
    Set up logging for the entire application.

    Args:
        log_level (int): The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_performance_monitoring (bool): Whether to enable periodic performance monitoring
        monitor_interval (int): Interval between performance measurements in seconds
        capture_console_output (bool): Whether to capture all console output to log.txt

    Returns:
        logging.Logger: The main application logger
    """
    global _app_logger, _perf_monitor

    # Create main application logger
    _app_logger = get_application_logger()
    _app_logger.setLevel(log_level)

    # Initialize console logger if requested to capture all print statements
    if capture_console_output:
        initialize_console_logger(also_to_console=True, daily_logs=True)

    # Log application startup
    _app_logger.info(f"=============== {APP_NAME} STARTUP ===============")
    _app_logger.info(f"Log level: {logging.getLevelName(log_level)}")

    # Set up performance monitoring if enabled
    if enable_performance_monitoring:
        _perf_monitor = PerformanceMonitor(APP_NAME)
        _perf_monitor.log_system_info()

        if monitor_interval > 0:
            _perf_monitor.start_monitoring(interval=monitor_interval)
            _app_logger.info(f"Performance monitoring enabled (interval: {monitor_interval}s)")

    return _app_logger


def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: Configured logger
    """
    return LoggerFactory.get_logger(name)


def shutdown_logging():
    """
    Clean up logging resources and log final messages before application exit.
    """
    global _app_logger, _perf_monitor

    if _perf_monitor:
        _perf_monitor.stop_monitoring()

    if _app_logger:
        _app_logger.info(f"=============== {APP_NAME} SHUTDOWN ===============")

    # Shutdown console logger
    shutdown_console_logger()


# Set up a simple error handler for uncaught exceptions
def _handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by logging them."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception
    logger = get_error_logger()
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Install the global exception handler
sys.excepthook = _handle_exception

# Auto-initialize console logger when the module is imported
# Comment out this line if you want to initialize manually in your main script
initialize_console_logger()
