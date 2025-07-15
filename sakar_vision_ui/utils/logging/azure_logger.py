#!/usr/bin/env python3
"""
azure_logger.py - Azure operation logging for SAKAR Vision AI

This module provides specialized logging for Azure operations including storage,
database interactions, and other Azure service integrations.
"""

import json
import time
import uuid
from datetime import datetime
from functools import wraps

from utils.logging.logger_config import LoggerFactory, log_exception
from utils.logging.performance_logger import Timer

# Create dedicated Azure logger
logger = LoggerFactory.get_logger("azure", detailed=True)


class AzureOperationTracker:
    """Utility class for tracking Azure service operations."""

    def __init__(self, service_name):
        """
        Initialize Azure operation tracker for a specific service.

        Args:
            service_name (str): Name of the Azure service (storage, database, etc.)
        """
        self.service_name = service_name

    def log_operation_start(self, operation_name, details=None):
        """
        Log the start of an Azure operation.

        Args:
            operation_name (str): Name of the operation
            details (dict, optional): Additional operation details

        Returns:
            str: Operation ID for correlation
        """
        operation_id = str(uuid.uuid4())

        log_message = f"Azure {self.service_name} operation started: {operation_name} [ID: {operation_id}]"
        if details:
            safe_details = self._sanitize_details(details)
            log_message += f", Details: {safe_details}"

        logger.info(log_message)
        return operation_id

    def log_operation_end(self, operation_id, operation_name, status="success", result=None, error=None):
        """
        Log the completion of an Azure operation.

        Args:
            operation_id (str): Operation ID for correlation
            operation_name (str): Name of the operation
            status (str): Operation status (success, failed)
            result (Any, optional): Operation result
            error (Exception, optional): Error if operation failed
        """
        if status.lower() == "success":
            log_message = f"Azure {self.service_name} operation completed: {operation_name} [ID: {operation_id}]"
            if result:
                # Sanitize and truncate result if needed
                safe_result = self._sanitize_details(result)
                if isinstance(safe_result, str) and len(safe_result) > 200:
                    safe_result = safe_result[:200] + "..."
                log_message += f", Result: {safe_result}"
            logger.info(log_message)
        else:
            log_message = f"Azure {self.service_name} operation failed: {operation_name} [ID: {operation_id}]"
            logger.error(log_message)
            if error:
                log_exception(logger, error)

    def log_connection_attempt(self, connection_string=None, account_name=None):
        """
        Log an Azure connection attempt.

        Args:
            connection_string (str, optional): Connection string (will be redacted)
            account_name (str, optional): Azure account name
        """
        if connection_string:
            # Redact connection string
            redacted = "***REDACTED***"
        else:
            redacted = "None"

        if account_name:
            logger.info(f"Connecting to Azure {self.service_name} account: {account_name}")
        else:
            logger.info(f"Connecting to Azure {self.service_name}")

        logger.debug(f"Connection string: {redacted}")

    def log_connection_result(self, success, error=None):
        """
        Log the result of an Azure connection attempt.

        Args:
            success (bool): Whether the connection was successful
            error (Exception, optional): Error if connection failed
        """
        if success:
            logger.info(f"Successfully connected to Azure {self.service_name}")
        else:
            logger.error(f"Failed to connect to Azure {self.service_name}")
            if error:
                log_exception(logger, error)

    def log_file_operation(self, operation, file_path, container=None, success=True, error=None):
        """
        Log an Azure storage file operation.

        Args:
            operation (str): Operation type (upload, download, delete, etc.)
            file_path (str): Path to the file
            container (str, optional): Container name
            success (bool): Whether the operation was successful
            error (Exception, optional): Error if operation failed
        """
        container_info = f" in container '{container}'" if container else ""

        if success:
            logger.info(
                f"Azure {self.service_name} {operation}: '{file_path}'{container_info} - SUCCESS")
        else:
            logger.error(
                f"Azure {self.service_name} {operation}: '{file_path}'{container_info} - FAILED")
            if error:
                log_exception(logger, error)

    def log_query(self, query_type, query, params=None, results_count=None):
        """
        Log an Azure database query.

        Args:
            query_type (str): Type of query (select, insert, update, delete)
            query (str): The query string
            params (dict, optional): Query parameters
            results_count (int, optional): Number of results returned
        """
        # Truncate long queries for readability
        if len(query) > 500:
            query_log = query[:500] + "..."
        else:
            query_log = query

        log_message = f"Azure {self.service_name} {query_type}: {query_log}"

        if params:
            # Sanitize parameters
            safe_params = self._sanitize_details(params)
            log_message += f", Params: {safe_params}"

        if results_count is not None:
            log_message += f", Results: {results_count}"

        logger.info(log_message)

    def log_batch_operation(self, operation, count, success_count=None, failure_count=None):
        """
        Log an Azure batch operation.

        Args:
            operation (str): Operation type (upload, process, etc.)
            count (int): Total count of items
            success_count (int, optional): Count of successful operations
            failure_count (int, optional): Count of failed operations
        """
        log_message = f"Azure {self.service_name} batch {operation}: {count} items"

        if success_count is not None:
            log_message += f", {success_count} successful"
        if failure_count is not None:
            log_message += f", {failure_count} failed"

        if failure_count and failure_count > 0:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _sanitize_details(self, details):
        """
        Sanitize sensitive details for logging.

        Args:
            details: Details to sanitize

        Returns:
            Any: Sanitized details
        """
        if isinstance(details, dict):
            sanitized = {}
            sensitive_keys = ['key', 'password', 'secret', 'token', 'connection', 'sas', 'auth']

            for k, v in details.items():
                if any(sensitive in k.lower() for sensitive in sensitive_keys):
                    sanitized[k] = "***REDACTED***"
                elif isinstance(v, dict):
                    sanitized[k] = self._sanitize_details(v)
                else:
                    sanitized[k] = v
            return sanitized
        return details


def track_azure_operation(service, operation=None):
    """
    Decorator for tracking Azure operations.

    Args:
        service (str): Azure service name
        operation (str, optional): Operation name (defaults to function name)

    Returns:
        function: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a tracker
            tracker = AzureOperationTracker(service)

            # Determine operation name
            op_name = operation or func.__name__

            # Generate operation ID and log start
            op_id = tracker.log_operation_start(op_name)

            try:
                # Time the operation
                with Timer(f"Azure {service} {op_name}", log_level="debug"):
                    # Execute the operation
                    result = func(*args, **kwargs)

                # Log operation completion
                tracker.log_operation_end(op_id, op_name, "success", result)
                return result
            except Exception as e:
                # Log operation failure
                tracker.log_operation_end(op_id, op_name, "failed", error=e)
                # Re-raise the exception
                raise

        return wrapper
    return decorator


class AzureStorageLogger(AzureOperationTracker):
    """Specialized logger for Azure Storage operations."""

    def __init__(self):
        """Initialize Azure Storage logger."""
        super().__init__("storage")

    def log_upload(self, file_path, container, success=True, error=None):
        """
        Log a file upload operation.

        Args:
            file_path (str): Path to the file
            container (str): Container name
            success (bool): Whether the upload was successful
            error (Exception, optional): Error if upload failed
        """
        self.log_file_operation("upload", file_path, container, success, error)

    def log_download(self, file_path, container, success=True, error=None):
        """
        Log a file download operation.

        Args:
            file_path (str): Path to the file
            container (str): Container name
            success (bool): Whether the download was successful
            error (Exception, optional): Error if download failed
        """
        self.log_file_operation("download", file_path, container, success, error)

    def log_delete(self, file_path, container, success=True, error=None):
        """
        Log a file delete operation.

        Args:
            file_path (str): Path to the file
            container (str): Container name
            success (bool): Whether the delete was successful
            error (Exception, optional): Error if delete failed
        """
        self.log_file_operation("delete", file_path, container, success, error)

    def log_list(self, container, prefix=None, results_count=None):
        """
        Log a file listing operation.

        Args:
            container (str): Container name
            prefix (str, optional): File prefix
            results_count (int, optional): Number of files listed
        """
        message = f"Listed files in container '{container}'"
        if prefix:
            message += f" with prefix '{prefix}'"
        if results_count is not None:
            message += f" - {results_count} files found"

        logger.info(message)


class AzureDatabaseLogger(AzureOperationTracker):
    """Specialized logger for Azure Database operations."""

    def __init__(self):
        """Initialize Azure Database logger."""
        super().__init__("database")

    def log_select(self, query, params=None, results_count=None):
        """
        Log a SELECT query.

        Args:
            query (str): Query string
            params (dict, optional): Query parameters
            results_count (int, optional): Number of results returned
        """
        self.log_query("SELECT", query, params, results_count)

    def log_insert(self, table, data=None, success=True, error=None):
        """
        Log an INSERT operation.

        Args:
            table (str): Table name
            data (dict, optional): Data being inserted
            success (bool): Whether the insert was successful
            error (Exception, optional): Error if insert failed
        """
        if success:
            message = f"Inserted data into table '{table}'"
            if data:
                safe_data = self._sanitize_details(data)
                message += f" - Data: {safe_data}"
            logger.info(message)
        else:
            logger.error(f"Failed to insert data into table '{table}'")
            if error:
                log_exception(logger, error)

    def log_update(self, table, condition=None, data=None, rows_affected=None, success=True, error=None):
        """
        Log an UPDATE operation.

        Args:
            table (str): Table name
            condition (str, optional): Update condition
            data (dict, optional): Data being updated
            rows_affected (int, optional): Number of rows affected
            success (bool): Whether the update was successful
            error (Exception, optional): Error if update failed
        """
        if success:
            message = f"Updated table '{table}'"
            if condition:
                message += f" with condition '{condition}'"
            if rows_affected is not None:
                message += f" - {rows_affected} rows affected"
            logger.info(message)
        else:
            logger.error(f"Failed to update table '{table}'")
            if error:
                log_exception(logger, error)

    def log_delete(self, table, condition=None, rows_affected=None, success=True, error=None):
        """
        Log a DELETE operation.

        Args:
            table (str): Table name
            condition (str, optional): Delete condition
            rows_affected (int, optional): Number of rows affected
            success (bool): Whether the delete was successful
            error (Exception, optional): Error if delete failed
        """
        if success:
            message = f"Deleted from table '{table}'"
            if condition:
                message += f" with condition '{condition}'"
            if rows_affected is not None:
                message += f" - {rows_affected} rows affected"
            logger.info(message)
        else:
            logger.error(f"Failed to delete from table '{table}'")
            if error:
                log_exception(logger, error)


# Demo function
def demo_azure_logger():
    """Demonstrate Azure logging functionality."""
    # Storage logger demo
    storage_logger = AzureStorageLogger()
    storage_logger.log_connection_attempt(account_name="sakarvisiondatabase")
    storage_logger.log_connection_result(True)
    storage_logger.log_upload("local/path/image.jpg", "images", True)
    storage_logger.log_download("remote/path/config.json", "configs", True)
    storage_logger.log_list("models", prefix="yolo", results_count=3)
    storage_logger.log_batch_operation("upload", 10, 8, 2)

    # Database logger demo
    db_logger = AzureDatabaseLogger()
    db_logger.log_connection_attempt(account_name="sakar-sql-db")
    db_logger.log_connection_result(True)
    db_logger.log_select("SELECT * FROM defects WHERE severity > ?", {"severity": 3}, 15)
    db_logger.log_insert("inspection_results", {"defect_id": 123, "confidence": 0.95})
    db_logger.log_update("defects", "id = 42", {"status": "reviewed"}, 1)
    db_logger.log_delete("temp_results", "created_at < '2023-01-01'", 56)

    # Decorator demo
    @track_azure_operation("storage", "file_upload")
    def upload_file(file_path, container):
        # Simulate file upload
        time.sleep(0.5)
        return {"status": "success", "path": file_path, "container": container}

    try:
        result = upload_file("test.jpg", "images")
        print(f"Upload result: {result}")
    except Exception as e:
        print(f"Upload failed: {e}")

    print("Azure operations logged. Check the logs directory.")


if __name__ == "__main__":
    demo_azure_logger()
