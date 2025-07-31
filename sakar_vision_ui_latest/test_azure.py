# azure_image_upload.py

import os

import time

from datetime import datetime

from azure.storage.blob import BlobServiceClient

from azure.storage.filedatalake import DataLakeServiceClient

# Azure Storage connection string and configuration

connect_str = "DefaultEndpointsProtocol=https;AccountName=sakarvisiondatabase;AccountKey=zbhMWPTVvClF1rJG/s2BDGK6ayk5366EmqZko8jLgaJstMkpbaOrucCytDls+AgMqJ/DYdaJ/KMq+AStchZNbA==;EndpointSuffix=core.windows.net"

# Set API version explicitly to a supported one
AZURE_API_VERSION = "2023-11-03"

blob_service_client = BlobServiceClient.from_connection_string(connect_str, api_version=AZURE_API_VERSION)

datalake_service_client = DataLakeServiceClient.from_connection_string(connect_str, api_version=AZURE_API_VERSION)

# Azure Data Lake storage configuration

AZURE_FILE_SYSTEM_NAME = "railway"  # must be lowercase, no special chars
DIRECTORY_NAME = "pune"             # recommended lowercase, no special chars

# Temporary directory to store images before upload

TEMP_DIR = "temp_images"

os.makedirs(TEMP_DIR, exist_ok=True)

# Clean up any existing files in the temporary directory


def cleanup_temp_dir():

    for file in os.listdir(TEMP_DIR):

        file_path = os.path.join(TEMP_DIR, file)

        try:

            if os.path.isfile(file_path):

                os.remove(file_path)

                print(f"Cleaned up old temporary file: {file_path}")

        except Exception as e:

            print(f"Error cleaning up {file_path}: {str(e)}")

# Create the directory in Data Lake if it doesn't exist


def ensure_directory_exists():

    try:

        file_system_client = datalake_service_client.get_file_system_client(file_system=AZURE_FILE_SYSTEM_NAME)

        # Check if directory exists, create if it doesn't

        try:

            directory_client = file_system_client.get_directory_client(DIRECTORY_NAME)

            directory_client.get_properties()

            print(f"Directory {DIRECTORY_NAME} already exists in {AZURE_FILE_SYSTEM_NAME}")

        except Exception:

            directory_client = file_system_client.create_directory(DIRECTORY_NAME)

            print(f"Created directory {DIRECTORY_NAME} in {AZURE_FILE_SYSTEM_NAME}")

    except Exception as e:

        print(f"Error ensuring directory exists: {str(e)}")

        # Try to create the file system if it doesn't exist

        try:

            datalake_service_client.create_file_system(file_system=AZURE_FILE_SYSTEM_NAME)

            print(f"Created file system {AZURE_FILE_SYSTEM_NAME}")

            # Now create the directory

            file_system_client = datalake_service_client.get_file_system_client(file_system=AZURE_FILE_SYSTEM_NAME)

            directory_client = file_system_client.create_directory(DIRECTORY_NAME)

            print(f"Created directory {DIRECTORY_NAME} in {AZURE_FILE_SYSTEM_NAME}")

        except Exception as e2:

            print(f"Failed to create file system and directory: {str(e2)}")

# Upload image to Azure Data Lake and delete local copy after successful upload


def upload_to_azure(image_path, label="image"):

    try:

        # Generate unique filename with timestamp and label

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_name = f"{label}_{timestamp}.jpg"

        # Get file system client

        file_system_client = datalake_service_client.get_file_system_client(file_system=AZURE_FILE_SYSTEM_NAME)

        # Get directory client

        directory_client = file_system_client.get_directory_client(DIRECTORY_NAME)

        # Get file client

        file_client = directory_client.get_file_client(file_name)

        # Upload the file

        with open(image_path, "rb") as data:

            file_client.upload_data(data, overwrite=True)

        print(f"Uploaded {file_name} to Azure Data Lake: {AZURE_FILE_SYSTEM_NAME}/{DIRECTORY_NAME}")

        # Delete the local file after successful upload

        os.remove(image_path)

        print(f"Deleted local temporary file: {image_path}")

        return True

    except Exception as e:

        print(f"Error uploading to Azure: {str(e)}")

        return False

# Example usage function - upload a specific image


def upload_image(image_path, label="image"):

    # Make sure the Azure directory exists

    ensure_directory_exists()

    # Create a copy in temp directory (optional)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    temp_path = os.path.join(TEMP_DIR, f"{label}_{timestamp}.jpg")

    try:

        # If you want to copy the image to temp dir first (optional)

        # import shutil

        # shutil.copy2(image_path, temp_path)

        # Then upload the copy: upload_to_azure(temp_path, label)

        # Or upload directly

        return upload_to_azure(image_path, label)

    except Exception as e:

        print(f"Error processing image: {str(e)}")

        return False

# Example usage


if __name__ == "__main__":

    # Clean up temporary directory

    cleanup_temp_dir()

    # Make sure the Azure directory exists

    ensure_directory_exists()

    # Example: Upload a specific image (replace with your image path)

    # image_path = "path/to/your/image.jpg"

    # upload_image(image_path, "example_label")

    print("Azure upload module ready!")

    print("Use upload_image(image_path, label) to upload images")
