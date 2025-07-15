"""
Azure Storage Module for SAKAR Vision AI Application

This module provides functions to interact with Azure Blob Storage and Data Lake Storage Gen2.
It follows Azure best practices for error handling, logging, and security.
"""

import os
import socket
from datetime import datetime
from typing import List, Tuple, Optional

from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, ServiceRequestError, ClientAuthenticationError
from azure.identity import DefaultAzureCredential

# Import our custom logging
from utils.logging import get_azure_logger, track_azure_operation, log_exception
from utils.logging.azure_logger import AzureStorageLogger

# Set up logger for this module using our new logging system
logger = get_azure_logger("storage")
storage_logger = AzureStorageLogger()

# Azure connection string - in production, this should be retrieved from a key vault or environment variable
# TODO: Move this to environment variable or Azure Key Vault in production
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=sakarvisiondatabase;AccountKey=zbhMWPTVvClF1rJG/s2BDGK6ayk5366EmqZko8jLgaJstMkpbaOrucCytDls+AgMqJ/DYdaJ/KMq+AStchZNbA==;EndpointSuffix=core.windows.net"

# Azure Data Lake storage file system name
DEFAULT_FILE_SYSTEM_NAME = "sakarvisiondatabase"

# Azure API version - using a supported version from the error message
API_VERSION = "2023-01-03"

# Flag to track network connectivity - initialized at module level
has_network_connectivity = True

# Common Azure hosts to check connectivity
AZURE_HOSTS = ["sakarvisiondatabase.blob.core.windows.net", "microsoft.com"]

# Initialize Azure storage clients
blob_service_client = None
datalake_service_client = None


def check_internet_connection(timeout=2):
    """
    Check if there's an active internet connection by trying to connect to Azure endpoints.

    Args:
        timeout: Connection timeout in seconds

    Returns:
        bool: True if internet connection is available, False otherwise
    """
    global has_network_connectivity

    for host in AZURE_HOSTS:
        try:
            # Try to establish a connection to the host
            socket.create_connection((host, 80), timeout=timeout)
            has_network_connectivity = True
            logger.debug(f"Successfully connected to {host}")
            return True
        except (socket.timeout, socket.gaierror, OSError):
            continue

    # If we've tried all hosts and none worked
    has_network_connectivity = False
    logger.warning("Network connectivity test failed for all Azure endpoints")
    return False


@track_azure_operation("storage", "initialize_clients")
def initialize_clients():
    """Initialize Azure clients if not already initialized and network is available"""
    global blob_service_client, datalake_service_client, has_network_connectivity

    # Only initialize if we have network connectivity
    if not check_internet_connection():
        logger.warning(
            "No internet connection available. Azure storage operations will be unavailable.")
        return False

    # If clients are already initialized, return
    if blob_service_client and datalake_service_client:
        logger.debug("Azure storage clients already initialized")
        return True

    try:
        # Log connection attempt with redacted credentials
        storage_logger.log_connection_attempt(account_name="sakarvisiondatabase")

        # Create clients with the supported API version
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_CONNECTION_STRING,
            api_version=API_VERSION
        )
        datalake_service_client = DataLakeServiceClient.from_connection_string(
            AZURE_CONNECTION_STRING,
            api_version=API_VERSION
        )

        # Log successful connection
        storage_logger.log_connection_result(True)
        logger.info(
            f"Azure storage clients initialized successfully with API version {API_VERSION}")
        return True
    except Exception as e:
        # Log failed connection with error details
        storage_logger.log_connection_result(False, error=e)
        logger.error(f"Failed to initialize Azure storage clients: {str(e)}")
        has_network_connectivity = False
        return False


# Try to initialize clients at module load time, but don't block if it fails
try:
    logger.info("Initializing Azure storage clients at module load time")
    initialize_clients()
except Exception as e:
    logger.warning(f"Initial client initialization skipped: {str(e)}")


@track_azure_operation("storage", "create_directory")
def create_directory_in_azure(directory_path: str, file_system_name: str = DEFAULT_FILE_SYSTEM_NAME) -> bool:
    """
    Creates a directory in Azure Data Lake Storage Gen2 if it doesn't exist.

    Args:
        directory_path: The path of the directory to create in Azure Data Lake
        file_system_name: The name of the file system (container) in Azure Data Lake

    Returns:
        bool: True if successful, False otherwise
    """
    global has_network_connectivity

    if not has_network_connectivity or not initialize_clients():
        logger.warning("Network unavailable. Skipping Azure directory creation.")
        return False

    try:
        # Get the file system client
        file_system_client = datalake_service_client.get_file_system_client(
            file_system=file_system_name)

        # Check if the file system exists, create it if not
        if not file_system_client.exists():
            file_system_client.create_file_system()
            logger.info(f"Created file system in Azure Data Lake: {file_system_name}")

        # Create the directory if it doesn't exist
        directory_client = file_system_client.get_directory_client(directory_path)
        if not directory_client.exists():
            directory_client.create_directory()
            logger.info(f"Created directory in Azure Data Lake: {directory_path}")
        else:
            logger.debug(f"Directory already exists in Azure Data Lake: {directory_path}")

        return True
    except (ServiceRequestError, ClientAuthenticationError) as e:
        # These are network-related errors
        has_network_connectivity = False
        logger.error(f"Network error creating directory in Azure Data Lake Storage: {str(e)}")
        log_exception(logger, e)
        return False
    except Exception as e:
        logger.error(f"Error creating directory in Azure Data Lake Storage: {str(e)}")
        log_exception(logger, e)
        return False


@track_azure_operation("storage", "upload_file")
def upload_to_azure_datalake(
    local_file_path: str,
    azure_directory: str,
    file_name: str,
    file_system_name: str = DEFAULT_FILE_SYSTEM_NAME
) -> bool:
    """
    Uploads a file to Azure Data Lake Storage Gen2.

    Args:
        local_file_path: Path to the local file to upload
        azure_directory: Directory in Azure Data Lake to upload to
        file_name: Name to give the file in Azure Data Lake
        file_system_name: The name of the file system (container) in Azure Data Lake

    Returns:
        bool: True if successful, False otherwise
    """
    global has_network_connectivity

    if not has_network_connectivity or not initialize_clients():
        logger.warning("Network unavailable. Skipping Azure upload.")
        return False

    op_id = storage_logger.log_operation_start("file_upload", {
        "file": file_name,
        "directory": azure_directory,
        "container": file_system_name
    })

    try:
        # Get the file system client
        file_system_client = datalake_service_client.get_file_system_client(
            file_system=file_system_name)

        # Ensure the directory exists
        directory_client = file_system_client.get_directory_client(azure_directory)
        if not directory_client.exists():
            directory_client.create_directory()
            logger.info(f"Created directory in Azure Data Lake: {azure_directory}")

        # Get a file client for the upload
        file_client = directory_client.get_file_client(file_name)

        # Read the file and upload
        with open(local_file_path, "rb") as data:
            file_client.upload_data(data, overwrite=True)

        # Log the successful upload
        storage_logger.log_upload(file_name, azure_directory, True)
        storage_logger.log_operation_end(op_id, "file_upload", "success")

        logger.info(
            f"Successfully uploaded {file_name} to Azure Data Lake Storage in {azure_directory}")
        return True
    except (ServiceRequestError, ClientAuthenticationError) as e:
        # These are network-related errors
        has_network_connectivity = False
        storage_logger.log_upload(file_name, azure_directory, False, error=e)
        storage_logger.log_operation_end(op_id, "file_upload", "failed", error=e)
        logger.error(f"Network error uploading to Azure Data Lake Storage: {str(e)}")
        return False
    except Exception as e:
        storage_logger.log_upload(file_name, azure_directory, False, error=e)
        storage_logger.log_operation_end(op_id, "file_upload", "failed", error=e)
        logger.error(f"Error uploading to Azure Data Lake Storage: {str(e)}")
        return False


@track_azure_operation("storage", "batch_upload")
def batch_upload_to_azure(
    pending_uploads: List[Tuple[str, str, str]],
    file_system_name: str = DEFAULT_FILE_SYSTEM_NAME,
    progress_callback: callable = None
) -> Tuple[int, int]:
    """
    Performs batch upload of files to Azure Data Lake Storage.

    Args:
        pending_uploads: List of tuples (local_path, azure_folder, filename)
        file_system_name: The name of the file system (container) in Azure Data Lake
        progress_callback: Optional callback function to report upload progress (current, total)

    Returns:
        Tuple[int, int]: Count of successful and failed uploads
    """
    global has_network_connectivity

    successful = 0
    failed = 0
    total_files = len(pending_uploads)

    if not has_network_connectivity or not initialize_clients():
        logger.warning("Network unavailable. Skipping Azure batch upload.")
        return 0, len(pending_uploads)

    # Log batch operation start
    storage_logger.log_batch_operation("upload", total_files)
    batch_op_id = storage_logger.log_operation_start("batch_upload", {
        "total_files": total_files,
        "container": file_system_name
    })

    # Create a dictionary to group uploads by directory
    uploads_by_directory = {}
    for local_path, azure_dir, filename in pending_uploads:
        if azure_dir not in uploads_by_directory:
            uploads_by_directory[azure_dir] = []
        uploads_by_directory[azure_dir].append((local_path, filename))

    # Process uploads directory by directory to minimize directory existence checks
    for azure_dir, files in uploads_by_directory.items():
        # Ensure the directory exists
        if not create_directory_in_azure(azure_dir, file_system_name):
            # If directory creation failed, mark all files in this directory as failed
            failed += len(files)
            if progress_callback:
                progress_callback(successful, total_files)
            continue

        # Get file system and directory clients
        file_system_client = datalake_service_client.get_file_system_client(file_system_name)
        directory_client = file_system_client.get_directory_client(azure_dir)

        # Upload each file in the directory
        for local_path, filename in files:
            try:
                if os.path.exists(local_path):
                    file_client = directory_client.get_file_client(filename)
                    with open(local_path, "rb") as data:
                        file_client.upload_data(data, overwrite=True)
                    successful += 1
                    storage_logger.log_upload(filename, azure_dir, True)
                else:
                    logger.warning(f"Local file not found: {local_path}")
                    failed += 1

                # Update progress after each file
                if progress_callback:
                    progress_callback(successful, total_files)
            except (ServiceRequestError, ClientAuthenticationError) as e:
                # Network-related errors
                has_network_connectivity = False
                storage_logger.log_upload(filename, azure_dir, False, error=e)
                logger.error(f"Network error uploading {filename}: {str(e)}")
                failed += 1
                # Stop trying more files if we've lost connectivity
                if progress_callback:
                    progress_callback(successful, total_files)
                break
            except Exception as e:
                storage_logger.log_upload(filename, azure_dir, False, error=e)
                logger.error(f"Error uploading {filename}: {str(e)}")
                failed += 1
                if progress_callback:
                    progress_callback(successful, total_files)

    # Log batch operation end with results
    storage_logger.log_batch_operation("upload", total_files, successful, failed)
    storage_logger.log_operation_end(batch_op_id, "batch_upload", "completed", {
        "successful": successful,
        "failed": failed
    })

    # Only log a single summary message instead of individual file uploads
    if successful > 0:
        logger.info(f"Successfully uploaded {successful} files to Azure storage")

    if failed > 0:
        logger.warning(f"Failed to upload {failed} files to Azure storage")

    return successful, failed


@track_azure_operation("storage", "download_file")
def download_from_azure_datalake(
    azure_directory: str,
    file_name: str,
    local_file_path: str,
    file_system_name: str = DEFAULT_FILE_SYSTEM_NAME
) -> bool:
    """
    Downloads a file from Azure Data Lake Storage Gen2.

    Args:
        azure_directory: Directory in Azure Data Lake where the file is stored
        file_name: Name of the file in Azure Data Lake
        local_file_path: Path where the downloaded file should be saved
        file_system_name: The name of the file system (container) in Azure Data Lake

    Returns:
        bool: True if successful, False otherwise
    """
    global has_network_connectivity

    if not has_network_connectivity or not initialize_clients():
        logger.warning("Network unavailable. Skipping Azure download.")
        return False

    op_id = storage_logger.log_operation_start("file_download", {
        "file": file_name,
        "directory": azure_directory,
        "container": file_system_name,
        "local_path": local_file_path
    })

    try:
        # Get the file system client
        file_system_client = datalake_service_client.get_file_system_client(
            file_system_name)

        # Get the directory client
        directory_client = file_system_client.get_directory_client(azure_directory)
        if not directory_client.exists():
            logger.error(f"Directory does not exist in Azure Data Lake: {azure_directory}")
            storage_logger.log_operation_end(op_id, "file_download", "failed", {
                "reason": "Directory not found"
            })
            return False

        # Get a file client for the download
        file_client = directory_client.get_file_client(file_name)

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        with open(local_file_path, "wb") as local_file:
            download = file_client.download_file()
            local_file.write(download.readall())

        # Log the successful download
        storage_logger.log_download(file_name, azure_directory, True)
        storage_logger.log_operation_end(op_id, "file_download", "success")

        logger.info(f"Successfully downloaded {file_name} from Azure Data Lake Storage")
        return True
    except (ServiceRequestError, ClientAuthenticationError) as e:
        # Network-related errors
        has_network_connectivity = False
        storage_logger.log_download(file_name, azure_directory, False, error=e)
        storage_logger.log_operation_end(op_id, "file_download", "failed", error=e)
        logger.error(f"Network error downloading from Azure Data Lake Storage: {str(e)}")
        return False
    except Exception as e:
        storage_logger.log_download(file_name, azure_directory, False, error=e)
        storage_logger.log_operation_end(op_id, "file_download", "failed", error=e)
        logger.error(f"Error downloading from Azure Data Lake Storage: {str(e)}")
        return False


@track_azure_operation("storage", "list_files")
def list_files_in_azure_directory(
    azure_directory: str,
    file_system_name: str = DEFAULT_FILE_SYSTEM_NAME
) -> List[str]:
    """
    Lists all files in a specified Azure Data Lake directory.

    Args:
        azure_directory: Directory in Azure Data Lake to list files from
        file_system_name: The name of the file system (container) in Azure Data Lake

    Returns:
        List[str]: List of filenames in the directory, empty list if error
    """
    global has_network_connectivity

    if not has_network_connectivity or not initialize_clients():
        logger.warning("Network unavailable. Skipping Azure directory listing.")
        return []

    op_id = storage_logger.log_operation_start("list_files", {
        "directory": azure_directory,
        "container": file_system_name
    })

    try:
        # Get the file system client
        file_system_client = datalake_service_client.get_file_system_client(
            file_system=file_system_name)

        # Get the directory client
        directory_client = file_system_client.get_directory_client(azure_directory)
        if not directory_client.exists():
            logger.error(f"Directory does not exist in Azure Data Lake: {azure_directory}")
            storage_logger.log_operation_end(op_id, "list_files", "failed", {
                "reason": "Directory not found"
            })
            return []

        # List all files in the directory
        files = []
        paths = directory_client.get_paths(recursive=False)
        for path in paths:
            if not path.is_directory:
                # Extract filename from the path
                filename = path.name.split('/')[-1]
                files.append(filename)

        # Log successful listing with file count
        storage_logger.log_list(azure_directory, None, len(files))
        storage_logger.log_operation_end(op_id, "list_files", "success", {
            "files_count": len(files)
        })

        logger.info(f"Listed {len(files)} files in Azure Data Lake directory: {azure_directory}")
        return files
    except (ServiceRequestError, ClientAuthenticationError) as e:
        # Network-related errors
        has_network_connectivity = False
        storage_logger.log_operation_end(op_id, "list_files", "failed", error=e)
        logger.error(f"Network error listing files in Azure Data Lake Storage: {str(e)}")
        return []
    except Exception as e:
        storage_logger.log_operation_end(op_id, "list_files", "failed", error=e)
        logger.error(f"Error listing files in Azure Data Lake Storage: {str(e)}")
        return []


@track_azure_operation("storage", "delete_file")
def delete_file_from_azure(
    azure_directory: str,
    file_name: str,
    file_system_name: str = DEFAULT_FILE_SYSTEM_NAME
) -> bool:
    """
    Deletes a file from Azure Data Lake Storage Gen2.

    Args:
        azure_directory: Directory in Azure Data Lake where the file is stored
        file_name: Name of the file in Azure Data Lake to delete
        file_system_name: The name of the file system (container) in Azure Data Lake

    Returns:
        bool: True if successful, False otherwise
    """
    global has_network_connectivity

    if not has_network_connectivity or not initialize_clients():
        logger.warning("Network unavailable. Skipping Azure file deletion.")
        return False

    op_id = storage_logger.log_operation_start("delete_file", {
        "file": file_name,
        "directory": azure_directory,
        "container": file_system_name
    })

    try:
        # Get the file system client
        file_system_client = datalake_service_client.get_file_system_client(
            file_system=file_system_name)

        # Get the directory client
        directory_client = file_system_client.get_directory_client(azure_directory)
        if not directory_client.exists():
            logger.error(f"Directory does not exist in Azure Data Lake: {azure_directory}")
            storage_logger.log_operation_end(op_id, "delete_file", "failed", {
                "reason": "Directory not found"
            })
            return False

        # Get a file client for the delete operation
        file_client = directory_client.get_file_client(file_name)

        # Delete the file
        file_client.delete_file()

        # Log successful deletion
        storage_logger.log_delete(file_name, azure_directory, True)
        storage_logger.log_operation_end(op_id, "delete_file", "success")

        logger.info(f"Successfully deleted {file_name} from Azure Data Lake Storage")
        return True
    except ResourceNotFoundError:
        storage_logger.log_operation_end(op_id, "delete_file", "failed", {
            "reason": "File not found"
        })
        logger.warning(f"File not found in Azure Data Lake: {file_name}")
        return False
    except (ServiceRequestError, ClientAuthenticationError) as e:
        # Network-related errors
        has_network_connectivity = False
        storage_logger.log_delete(file_name, azure_directory, False, error=e)
        storage_logger.log_operation_end(op_id, "delete_file", "failed", error=e)
        logger.error(f"Network error deleting file from Azure Data Lake Storage: {str(e)}")
        return False
    except Exception as e:
        storage_logger.log_delete(file_name, azure_directory, False, error=e)
        storage_logger.log_operation_end(op_id, "delete_file", "failed", error=e)
        logger.error(f"Error deleting file from Azure Data Lake Storage: {str(e)}")
        return False


def reset_connectivity_status():
    """Reset connectivity status and attempt to reconnect to Azure services"""
    global has_network_connectivity
    has_network_connectivity = True
    logger.info("Resetting Azure connectivity status and attempting to reconnect")
    return initialize_clients()
