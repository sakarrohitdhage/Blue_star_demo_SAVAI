#!/usr/bin/env python3
"""
ui_event_logger.py - UI event logging for SAKAR Vision AI

This module provides specialized logging for UI events and user interactions.
It tracks button clicks, form submissions, screen transitions, and other UI activities.
"""

import json
import time
from datetime import datetime
from functools import wraps

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QLineEdit, QWidget

from utils.logging.logger_config import LoggerFactory, log_exception

# Create dedicated UI logger
logger = LoggerFactory.get_logger("ui.events", detailed=True)


class UIEventTracker:
    """Utility class for tracking UI events."""

    def __init__(self, component_name="unnamed"):
        """
        Initialize UI event tracker for a specific component.

        Args:
            component_name (str): Name of the UI component being tracked
        """
        self.component_name = component_name

    def log_button_click(self, button_name):
        """
        Log a button click event.

        Args:
            button_name (str): Name or ID of the button that was clicked
        """
        logger.info(f"Button clicked: {button_name} in {self.component_name}")

    def log_form_submit(self, form_name, data=None):
        """
        Log a form submission event.

        Args:
            form_name (str): Name or ID of the form
            data (dict, optional): Form data (sensitive fields should be redacted)
        """
        if data:
            # Ensure no sensitive data is logged
            safe_data = self._redact_sensitive_data(data)
            logger.info(
                f"Form submitted: {form_name} in {self.component_name} with data: {safe_data}")
        else:
            logger.info(f"Form submitted: {form_name} in {self.component_name}")

    def log_screen_transition(self, from_screen, to_screen):
        """
        Log a screen transition event.

        Args:
            from_screen (str): Name of the previous screen
            to_screen (str): Name of the new screen
        """
        logger.info(f"Screen transition: {from_screen} -> {to_screen}")

    def log_model_change(self, model_name, previous=None, current=None):
        """
        Log a model selection or change event.

        Args:
            model_name (str): Name of the model being changed
            previous (str, optional): Previous value
            current (str, optional): Current value
        """
        if previous and current:
            logger.info(f"Model changed: {model_name} from '{previous}' to '{current}'")
        elif current:
            logger.info(f"Model selected: {model_name} - '{current}'")
        else:
            logger.info(f"Model operation: {model_name}")

    def log_setting_change(self, setting_name, previous=None, current=None):
        """
        Log a setting change event.

        Args:
            setting_name (str): Name of the setting being changed
            previous (str/int/bool, optional): Previous value
            current (str/int/bool, optional): Current value
        """
        if previous is not None and current is not None:
            logger.info(f"Setting changed: {setting_name} from '{previous}' to '{current}'")
        elif current is not None:
            logger.info(f"Setting configured: {setting_name} = '{current}'")
        else:
            logger.info(f"Setting operation: {setting_name}")

    def log_user_action(self, action, details=None):
        """
        Log a general user action.

        Args:
            action (str): Description of the action
            details (dict, optional): Additional details about the action
        """
        if details:
            safe_details = self._redact_sensitive_data(details)
            logger.info(f"User action: {action} - Details: {safe_details}")
        else:
            logger.info(f"User action: {action}")

    def log_error(self, error_type, message, exception=None):
        """
        Log a UI error.

        Args:
            error_type (str): Type of error
            message (str): Error message
            exception (Exception, optional): Exception object
        """
        logger.error(f"UI Error ({error_type}): {message} in {self.component_name}")
        if exception:
            log_exception(logger, exception)

    def _redact_sensitive_data(self, data):
        """
        Redact sensitive fields from data before logging.

        Args:
            data (dict): Data to redact

        Returns:
            dict: Redacted data
        """
        if not isinstance(data, dict):
            return data

        sensitive_fields = [
            'password', 'passwd', 'secret', 'token', 'api_key', 'apikey',
            'access_key', 'auth', 'credentials', 'private', 'key'
        ]

        result = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                result[key] = '******'
            elif isinstance(value, dict):
                result[key] = self._redact_sensitive_data(value)
            else:
                result[key] = value

        return result


class UIEventMonitor(QObject):
    """
    QObject-based monitor that connects to Qt signals to track UI events.
    It can be used to automatically track button clicks and other events.
    """

    event_logged = pyqtSignal(str, str)  # Signal emitted when an event is logged

    def __init__(self, parent=None):
        """
        Initialize UI event monitor.

        Args:
            parent (QObject, optional): Parent QObject
        """
        super().__init__(parent)
        self.tracker = UIEventTracker("event_monitor")
        self.tracked_objects = {}

    def track_button(self, button, name=None):
        """
        Track a button's click events.

        Args:
            button (QPushButton): Button to track
            name (str, optional): Name for the button (defaults to button text)
        """
        if not isinstance(button, QPushButton):
            logger.warning(f"Cannot track non-button object: {button}")
            return

        btn_name = name or button.text() or str(button.objectName())

        # Connect the button's clicked signal
        button.clicked.connect(lambda: self._on_button_clicked(btn_name))
        self.tracked_objects[id(button)] = {"type": "button", "name": btn_name}
        logger.debug(f"Now tracking button: {btn_name}")

    def track_widget(self, widget, name=None):
        """
        Track a widget's shown and hidden events.

        Args:
            widget (QWidget): Widget to track
            name (str, optional): Name for the widget
        """
        if not isinstance(widget, QWidget):
            logger.warning(f"Cannot track non-widget object: {widget}")
            return

        widget_name = name or widget.objectName() or widget.__class__.__name__

        # Connect shown and hidden signals if available
        if hasattr(widget, 'shown'):
            widget.shown.connect(lambda: self._on_widget_shown(widget_name))
        if hasattr(widget, 'hidden'):
            widget.hidden.connect(lambda: self._on_widget_hidden(widget_name))

        # Otherwise track show and hide events
        original_show = widget.show
        original_hide = widget.hide

        @wraps(original_show)
        def tracked_show(*args, **kwargs):
            result = original_show(*args, **kwargs)
            self._on_widget_shown(widget_name)
            return result

        @wraps(original_hide)
        def tracked_hide(*args, **kwargs):
            result = original_hide(*args, **kwargs)
            self._on_widget_hidden(widget_name)
            return result

        widget.show = tracked_show
        widget.hide = tracked_hide

        self.tracked_objects[id(widget)] = {"type": "widget", "name": widget_name}
        logger.debug(f"Now tracking widget: {widget_name}")

    def _on_button_clicked(self, btn_name):
        """Handle button click event."""
        self.tracker.log_button_click(btn_name)
        self.event_logged.emit("button_click", btn_name)

    def _on_widget_shown(self, widget_name):
        """Handle widget shown event."""
        logger.info(f"Widget shown: {widget_name}")
        self.event_logged.emit("widget_shown", widget_name)

    def _on_widget_hidden(self, widget_name):
        """Handle widget hidden event."""
        logger.info(f"Widget hidden: {widget_name}")
        self.event_logged.emit("widget_hidden", widget_name)


# Create a decorator for tracking function calls in UI classes
def track_ui_method(category=None):
    """
    Decorator for tracking UI method calls.

    Args:
        category (str, optional): Category of the UI action

    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get component name from class
            component_name = getattr(self, 'objectName', lambda: self.__class__.__name__)()

            # Create tracker
            tracker = UIEventTracker(component_name)

            # Get method information
            method_name = func.__name__

            # Determine category
            action_category = category or "method"

            # Log the start of the action
            logger.debug(f"UI {action_category} start: {component_name}.{method_name}")

            try:
                # Execute the original method
                result = func(self, *args, **kwargs)

                # Log successful completion
                logger.info(f"UI {action_category} executed: {component_name}.{method_name}")

                return result
            except Exception as e:
                # Log error
                tracker.log_error(f"{action_category}_error",
                                  f"Error in {method_name}", e)
                # Re-raise the exception
                raise

        return wrapper
    return decorator


# Demo function
def demo_ui_logger():
    """Demonstrate UI logging functionality."""
    # Create a tracker
    tracker = UIEventTracker("demo_component")

    # Log various UI events
    tracker.log_button_click("save_button")
    tracker.log_form_submit("login_form", {"username": "demo_user", "password": "secret123"})
    tracker.log_screen_transition("login_screen", "main_dashboard")
    tracker.log_model_change("defect_model", "yolov5s.pt", "yolov8m.pt")
    tracker.log_setting_change("confidence_threshold", 0.5, 0.65)
    tracker.log_user_action(
        "exported_report", {"format": "PDF", "timestamp": datetime.now().isoformat()})

    # Demonstrate error logging
    try:
        1/0
    except Exception as e:
        tracker.log_error("calculation", "Division by zero", e)

    print("UI events logged. Check the logs directory.")


if __name__ == "__main__":
    demo_ui_logger()
