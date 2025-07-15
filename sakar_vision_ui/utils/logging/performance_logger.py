#!/usr/bin/env python3
"""
performance_logger.py - Performance monitoring utilities for SAKAR Vision AI

This module provides tools for tracking performance metrics such as execution time,
memory usage, and system resource utilization.
"""

import time
import functools
import os
import psutil
import platform
from functools import wraps
from datetime import datetime

from utils.logging.logger_config import LoggerFactory, log_exception

# Create a dedicated performance logger
logger = LoggerFactory.get_logger("performance", detailed=True)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, operation_name="Operation", log_level="info"):
        """
        Initialize timer with an operation name.

        Args:
            operation_name (str): Name of the operation being timed
            log_level (str): Level to log at (debug, info, warning, error)
        """
        self.operation_name = operation_name
        self.log_level = log_level.lower()

    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}...")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Log elapsed time when exiting the context."""
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        # Choose log level based on parameter
        log_func = getattr(logger, self.log_level, logger.info)

        if exc_type:
            # If there was an exception, log it with the elapsed time
            log_func(f"{self.operation_name} failed after {self.elapsed_time:.4f} seconds")
            log_exception(logger, exc_value)
        else:
            # No exception, log successful completion
            log_func(f"{self.operation_name} completed in {self.elapsed_time:.4f} seconds")


def time_function(level="info"):
    """
    Decorator for timing function execution.

    Args:
        level (str): Log level to use (debug, info, warning, error)

    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            with Timer(operation_name, log_level=level):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceMonitor:
    """Utility for monitoring system and application performance metrics."""

    def __init__(self, app_name="SAKAR_Vision_AI"):
        """
        Initialize performance monitor.

        Args:
            app_name (str): Name of the application being monitored
        """
        self.app_name = app_name
        self.process = psutil.Process(os.getpid())

    def log_system_info(self):
        """Log information about the system."""
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")

        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        logger.info(f"CPU: {cpu_count} physical cores, {cpu_logical} logical cores")

        # Memory info
        mem = psutil.virtual_memory()
        logger.info(f"Memory: Total: {mem.total / (1024**3):.2f} GB, "
                    f"Available: {mem.available / (1024**3):.2f} GB")

        # Disk info
        disk = psutil.disk_usage('/')
        logger.info(f"Disk: Total: {disk.total / (1024**3):.2f} GB, "
                    f"Free: {disk.free / (1024**3):.2f} GB")

        # GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU: {gpu_count} devices available")
                for i in range(gpu_count):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.info("GPU: None available")
        except ImportError:
            logger.info("GPU: PyTorch not installed, cannot detect GPUs")

    def log_process_metrics(self):
        """Log metrics about the current process."""
        # Memory usage
        mem_info = self.process.memory_info()
        logger.info(f"Process memory: RSS: {mem_info.rss / (1024**2):.2f} MB, "
                    f"VMS: {mem_info.vms / (1024**2):.2f} MB")

        # CPU usage
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            logger.info(f"Process CPU: {cpu_percent:.1f}%")
        except Exception as e:
            logger.warning(f"Couldn't get CPU usage: {e}")

        # Thread count
        thread_count = self.process.num_threads()
        logger.info(f"Process threads: {thread_count}")

    def start_monitoring(self, interval=60):
        """
        Start periodic monitoring of system metrics.

        Args:
            interval (int): Time between measurements in seconds

        Returns:
            threading.Timer: Timer object that can be cancelled
        """
        import threading

        def _monitor():
            """Internal monitoring function that reschedules itself."""
            self.log_process_metrics()
            self.timer = threading.Timer(interval, _monitor)
            self.timer.daemon = True
            self.timer.start()

        # Log system info once at the start
        self.log_system_info()

        # Start periodic monitoring
        self.timer = threading.Timer(interval, _monitor)
        self.timer.daemon = True
        self.timer.start()
        logger.info(f"Started performance monitoring with {interval}s interval")
        return self.timer

    def stop_monitoring(self):
        """Stop the periodic monitoring."""
        if hasattr(self, 'timer'):
            self.timer.cancel()
            logger.info("Stopped performance monitoring")


# Simple demo function
def demo_performance_logging():
    """Demonstrate the performance logging utilities."""
    # Log system info
    monitor = PerformanceMonitor()
    monitor.log_system_info()
    monitor.log_process_metrics()

    # Demo the timer context manager
    with Timer("Demo operation", log_level="info"):
        # Simulate some work
        time.sleep(1.5)

    # Demo the function timer decorator
    @time_function(level="info")
    def slow_function():
        time.sleep(0.5)
        return "Done"

    slow_function()

    # Demo periodic monitoring
    timer = monitor.start_monitoring(interval=5)
    time.sleep(6)  # Wait for one monitoring cycle
    monitor.stop_monitoring()


if __name__ == "__main__":
    demo_performance_logging()
