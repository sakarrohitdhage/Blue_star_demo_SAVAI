#!/usr/bin/env python3
"""
console_logger.py - Console output capture for SAKAR Vision AI

This module redirects all stdout and stderr output to a log file,
ensuring that all print statements and terminal output are saved.
It automatically creates a log.txt file in the logs directory.
"""

import os
import sys
import logging
import datetime
import traceback
from io import StringIO

# Get the logs directory path
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, 'log.txt')
DAILY_LOG_FILE_FORMAT = os.path.join(LOG_DIR, 'log_{date}.txt')

# Console logger instance
_console_logger = None


class ConsoleOutputLogger:
    """
    Class that captures all stdout and stderr output and redirects it to a log file.
    """

    def __init__(self, log_file=LOG_FILE, also_to_console=True, daily_logs=True):
        """
        Initialize the console output logger.

        Args:
            log_file (str): Path to the log file
            also_to_console (bool): Whether to also output to the console
            daily_logs (bool): Whether to create daily log files
        """
        self.log_file = log_file
        self.also_to_console = also_to_console
        self.daily_logs = daily_logs
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log_file_handle = None
        self.daily_log_file_handle = None
        self.current_date = None

        # Create initial log file
        self.ensure_log_file()

        # Install the redirectors
        sys.stdout = self
        sys.stderr = self

    def ensure_log_file(self):
        """Ensure the log file is open and ready for writing."""
        # Create or open the main log file
        if self.log_file_handle is None:
            try:
                self.log_file_handle = open(self.log_file, 'a', encoding='utf-8')
            except Exception as e:
                print(f"Error opening log file: {e}", file=self.terminal_stderr)
                self.log_file_handle = None

        # Create or update daily log file if needed
        if self.daily_logs:
            today = datetime.datetime.now().date()
            if self.current_date != today or self.daily_log_file_handle is None:
                self.current_date = today
                date_str = today.strftime("%Y-%m-%d")
                daily_log_path = DAILY_LOG_FILE_FORMAT.format(date=date_str)

                # Close previous daily log if any
                if self.daily_log_file_handle is not None:
                    try:
                        self.daily_log_file_handle.close()
                    except:
                        pass

                # Open new daily log
                try:
                    self.daily_log_file_handle = open(daily_log_path, 'a', encoding='utf-8')
                except Exception as e:
                    print(f"Error opening daily log file: {e}", file=self.terminal_stderr)
                    self.daily_log_file_handle = None

    def write(self, message):
        """
        Write the message to the log file and optionally to the console.

        Args:
            message (str): The message to write
        """
        # Ensure the log files are ready
        self.ensure_log_file()

        # Add timestamp to the message (only for non-empty lines)
        if message and message.strip() and not message.isspace():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} | {message}"
        else:
            log_message = message

        # Write to the main log file
        if self.log_file_handle:
            try:
                self.log_file_handle.write(log_message)
                self.log_file_handle.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}", file=self.terminal_stderr)

        # Write to the daily log file if enabled
        if self.daily_logs and self.daily_log_file_handle:
            try:
                self.daily_log_file_handle.write(log_message)
                self.daily_log_file_handle.flush()
            except Exception as e:
                print(f"Error writing to daily log file: {e}", file=self.terminal_stderr)

        # Also write to the original stdout/stderr if requested
        if self.also_to_console:
            self.terminal_stdout.write(message)

    def flush(self):
        """Flush the log file and the console."""
        if self.log_file_handle:
            try:
                self.log_file_handle.flush()
            except:
                pass

        if self.daily_logs and self.daily_log_file_handle:
            try:
                self.daily_log_file_handle.flush()
            except:
                pass

        if self.also_to_console:
            self.terminal_stdout.flush()

    def close(self):
        """Close the log files and restore stdout and stderr."""
        # Restore stdout and stderr
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr

        # Close log files
        if self.log_file_handle:
            try:
                self.log_file_handle.close()
            except:
                pass
            self.log_file_handle = None

        if self.daily_logs and self.daily_log_file_handle:
            try:
                self.daily_log_file_handle.close()
            except:
                pass
            self.daily_log_file_handle = None


def initialize_console_logger(also_to_console=True, daily_logs=True):
    """
    Initialize the console logger to capture all stdout and stderr output.

    Args:
        also_to_console (bool): Whether to also output to the console
        daily_logs (bool): Whether to create daily log files

    Returns:
        ConsoleOutputLogger: The console logger instance
    """
    global _console_logger

    # Only initialize once
    if _console_logger is None:
        try:
            # Log start of session
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{timestamp} | {'='*50}\n")
                f.write(f"{timestamp} | SAKAR VISION AI - LOG SESSION STARTED\n")
                f.write(f"{timestamp} | {'='*50}\n\n")

            # Create and store the console logger
            _console_logger = ConsoleOutputLogger(
                also_to_console=also_to_console,
                daily_logs=daily_logs
            )

            # Write a test message
            print("Console logger initialized and capturing all terminal output")

        except Exception as e:
            sys.stderr.write(f"Error initializing console logger: {e}\n")
            sys.stderr.write(f"Traceback: {traceback.format_exc()}\n")

    return _console_logger


def shutdown_console_logger():
    """
    Close the console logger and restore stdout and stderr.
    """
    global _console_logger

    if _console_logger:
        try:
            # Log end of session
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*50}")
            print(f"SAKAR VISION AI - LOG SESSION ENDED")
            print(f"{'='*50}\n")

            # Close the logger
            _console_logger.close()
            _console_logger = None

        except Exception as e:
            sys.stderr.write(f"Error shutting down console logger: {e}\n")


# Function to get the current log file path
def get_log_file_path():
    """
    Get the path to the current log file.

    Returns:
        str: Path to the log file
    """
    return LOG_FILE


# Demo function
def test_console_logger():
    """
    Test the console logger by writing some messages.
    """
    initialize_console_logger()

    print("This is a test message from stdout")
    print("This is a test error message", file=sys.stderr)

    try:
        # Generate an exception
        result = 1 / 0
    except Exception as e:
        print(f"Caught exception: {e}")
        traceback.print_exc()

    print(f"Log file path: {get_log_file_path()}")
    shutdown_console_logger()


if __name__ == "__main__":
    test_console_logger()
