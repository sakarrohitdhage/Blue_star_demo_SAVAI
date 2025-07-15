"""
SAKAR VISION AI - Data Integration Module

OVERVIEW:
This module serves as the central data orchestration system for the Sakar Vision AI platform, providing comprehensive 
functionality for integrating, combining, and synchronizing training datasets across local storage and Azure cloud 
infrastructure. It handles the complex workflow of merging manually annotated data, automatically generated annotations, 
and cloud-stored datasets into unified training datasets suitable for machine learning model development and deployment.

KEY FUNCTIONALITY:
The system manages bidirectional data flow between local annotation workflows and Azure Data Lake Storage, featuring 
intelligent dataset combination with automatic class mapping and label conversion, batch upload/download operations 
with progress tracking and cancellation support, and dynamic YAML configuration file generation for YOLO-format datasets. 
It includes robust file categorization and statistics tracking, duplicate filename prevention through intelligent prefixing, 
and seamless integration with the ClassManager system for consistent class definitions across the platform. The module 
also provides comprehensive error handling and logging for data pipeline operations, ensuring data integrity throughout 
the annotation-to-training workflow.

TECHNICAL ARCHITECTURE:
Built with modular functions supporting both synchronous and asynchronous operations, the system employs configurable 
local path management with automatic directory structure creation, Azure Data Lake Storage integration using the 
azure_storage module, and sophisticated progress callback mechanisms for UI integration. It features thread-safe 
operations with cancellation support for long-running data transfers, automatic dataset structure validation and 
conversion between different annotation formats, and intelligent file conflict resolution through source-based prefixing 
(manual_, auto_, azure_). The architecture supports scalable dataset combination operations with memory-efficient 
processing and comprehensive statistics tracking for dataset composition analysis.
"""

import os
import shutil
import yaml
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import Azure storage module
from azure_storage import (
    download_from_azure_datalake,
    list_files_in_azure_directory,
    create_directory_in_azure,
    upload_to_azure_datalake,
    batch_upload_to_azure,
)

# Import ClassManager for handling class definitions
from utils import ClassManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_integration')

# Azure storage configuration
AZURE_SAVAI_FOLDER = "SAVAI_METAL_INSPECTION_GEN"
DEFAULT_FILE_SYSTEM = "sakarvisiondatabase"

# Local storage paths - will be configured dynamically
LOCAL_MANUAL_ANNOTATED = None
LOCAL_AUTO_ANNOTATED = None
LOCAL_COMBINED_DATA = None

# Flag to control automatic initialization - disabled by default
AUTO_INIT = False


def configure_local_paths(root_folder: str, auto_initialize: bool = False):
    """
    Configure local paths based on the project root folder.

    Args:
        root_folder: Root folder of the project
        auto_initialize: Whether to automatically initialize and combine data
    """
    global LOCAL_MANUAL_ANNOTATED, LOCAL_AUTO_ANNOTATED, LOCAL_COMBINED_DATA

    LOCAL_MANUAL_ANNOTATED = os.path.join(root_folder, "manually_annotated")
    LOCAL_AUTO_ANNOTATED = os.path.join(root_folder, "auto_annotated")
    LOCAL_COMBINED_DATA = os.path.join(root_folder, "combined_data")

    # Create directories if they don't exist
    for folder in [LOCAL_MANUAL_ANNOTATED, LOCAL_AUTO_ANNOTATED, LOCAL_COMBINED_DATA]:
        os.makedirs(folder, exist_ok=True)

    logger.info(
        f"Local paths configured: manual={LOCAL_MANUAL_ANNOTATED}, auto={LOCAL_AUTO_ANNOTATED}, combined={LOCAL_COMBINED_DATA}")

    # Only perform auto-initialization if explicitly requested
    if auto_initialize:
        try:
            # Get class names from ClassManager
            class_manager = ClassManager()
            if class_manager.initialized or class_manager.load_from_file():
                selected_classes = class_manager.get_classes()

                # Initialize the combined dataset
                initialize_combined_data(selected_classes)
        except Exception as e:
            logger.error(f"Error during auto initialization: {e}")


def initialize_combined_data(selected_classes: List[str]):
    """
    Initialize the combined data folder with a dataset.yaml file.
    This avoids automatic combining of data which can cause thread safety issues.

    Args:
        selected_classes: List of class names to include
    """
    global LOCAL_COMBINED_DATA

    if not LOCAL_COMBINED_DATA:
        logger.warning("Cannot initialize combined data: LOCAL_COMBINED_DATA not set")
        return

    # Create needed subdirectories but don't copy any data
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(LOCAL_COMBINED_DATA, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(LOCAL_COMBINED_DATA, split, 'labels'), exist_ok=True)

    # Create a dataset.yaml file
    yaml_path = os.path.join(LOCAL_COMBINED_DATA, 'dataset.yaml')

    update_dataset_yaml(yaml_path, selected_classes)
    logger.info(f"Created dataset.yaml with {len(selected_classes)} classes")

    # Update stats in log but don't actually combine data which might invoke UI
    stats = {
        'manual': {'images': 0, 'labels': 0},
        'auto': {'images': 0, 'labels': 0},
        'azure': {'images': 0, 'labels': 0}
    }

    # Count files in azure_data if it exists
    azure_data_folder = os.path.join(os.path.dirname(LOCAL_COMBINED_DATA), "azure_data")
    if os.path.exists(azure_data_folder):
        for split in ['train', 'valid', 'test']:
            images_dir = os.path.join(azure_data_folder, split, 'images')
            if os.path.exists(images_dir):
                stats['azure']['images'] += len([f for f in os.listdir(images_dir)
                                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            labels_dir = os.path.join(azure_data_folder, split, 'labels')
            if os.path.exists(labels_dir):
                stats['azure']['labels'] += len([f for f in os.listdir(labels_dir)
                                                if f.endswith('.txt')])

    total_images = sum(data['images'] for data in stats.values())
    total_labels = sum(data['labels'] for data in stats.values())

    logger.info(
        f"Combined {total_images} images and {total_labels} labels in {LOCAL_COMBINED_DATA}")
    for source, data in stats.items():
        logger.info(f"  {source}: {data['images']} images, {data['labels']} labels")


def download_azure_training_data(
    local_output_folder: str,
    azure_folder: str = AZURE_SAVAI_FOLDER,
    cancel_check_func: callable = None
) -> bool:
    """
    Download training data from Azure Data Lake Storage with frequent cancellation checks.

    Args:
        local_output_folder: Local folder to save downloaded data
        azure_folder: Azure folder path to download from
        cancel_check_func: Optional function that returns True if download should be cancelled

    Returns:
        bool or dict: False if failed/cancelled, dict with stats if successful
    """
    import time

    logger.info(f"Starting Azure download to {local_output_folder}")

    # Create the local output folder if it doesn't exist
    os.makedirs(local_output_folder, exist_ok=True)

    # Structure to track what we've downloaded
    downloaded_files = {
        'train': {'images': 0, 'labels': 0},
        'valid': {'images': 0, 'labels': 0},
        'test': {'images': 0, 'labels': 0}
    }

    # Initial cancellation check
    if cancel_check_func and cancel_check_func():
        logger.info("Download cancelled before starting")
        return False

    # Download data for each split (train, valid, test)
    for split_idx, split in enumerate(['train', 'valid', 'test']):
        logger.info(f"Processing {split} split ({split_idx + 1}/3)")

        # Frequent cancellation check
        if cancel_check_func and cancel_check_func():
            logger.info(f"Download cancelled before processing {split} split")
            return False

        # Create local directories for images and labels
        split_images_dir = os.path.join(local_output_folder, split, 'images')
        split_labels_dir = os.path.join(local_output_folder, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

        # Download images
        azure_images_path = f"{azure_folder}/{split}/images"

        # Another cancellation check before listing files
        if cancel_check_func and cancel_check_func():
            logger.info(f"Download cancelled before listing images for {split}")
            return False

        try:
            logger.info(f"Listing image files for {split}")
            image_files = list_files_in_azure_directory(azure_images_path)
            logger.info(f"Found {len(image_files)} image files for {split}")

            # Check cancellation after listing
            if cancel_check_func and cancel_check_func():
                logger.info(f"Download cancelled after listing images for {split}")
                return False

        except Exception as e:
            logger.warning(f"Could not list images for {split}: {e}")
            image_files = []

        # Process images with frequent cancellation checks
        for i, image_file in enumerate(image_files):
            # Check cancellation before each file AND every 10 files
            if cancel_check_func and cancel_check_func():
                logger.info(
                    f"Download cancelled while downloading images for {split} (file {i+1}/{len(image_files)})")
                return False

            local_image_path = os.path.join(split_images_dir, image_file)

            # Skip if file already exists
            if os.path.exists(local_image_path):
                downloaded_files[split]['images'] += 1
                # Still check for cancellation even on skipped files
                if i % 10 == 0 and cancel_check_func and cancel_check_func():
                    logger.info(f"Download cancelled during skip check for {split}")
                    return False
                continue

            try:
                # Download with timeout
                logger.debug(f"Downloading image {i+1}/{len(image_files)}: {image_file}")
                success = download_from_azure_datalake(
                    azure_directory=azure_images_path,
                    file_name=image_file,
                    local_file_path=local_image_path
                )

                if success:
                    downloaded_files[split]['images'] += 1
                else:
                    logger.warning(f"Failed to download {image_file}")

                # Check cancellation after each download
                if cancel_check_func and cancel_check_func():
                    logger.info(f"Download cancelled after downloading {image_file}")
                    return False

                # Small delay to allow cancellation check to be more responsive
                if i % 5 == 0:  # Every 5 files, add a small delay
                    time.sleep(0.01)  # 10ms delay

            except Exception as e:
                logger.warning(f"Exception downloading image {image_file}: {e}")
                # Check cancellation even after exceptions
                if cancel_check_func and cancel_check_func():
                    logger.info(f"Download cancelled after exception with {image_file}")
                    return False

        # Check cancellation before processing labels
        if cancel_check_func and cancel_check_func():
            logger.info(f"Download cancelled before processing labels for {split}")
            return False

        # Download labels with similar logic
        azure_labels_path = f"{azure_folder}/{split}/labels"
        try:
            logger.info(f"Listing label files for {split}")
            label_files = list_files_in_azure_directory(azure_labels_path)
            logger.info(f"Found {len(label_files)} label files for {split}")

            # Check cancellation after listing labels
            if cancel_check_func and cancel_check_func():
                logger.info(f"Download cancelled after listing labels for {split}")
                return False

        except Exception as e:
            logger.warning(f"Could not list labels for {split}: {e}")
            label_files = []

        for i, label_file in enumerate(label_files):
            # Frequent cancellation checks
            if cancel_check_func and cancel_check_func():
                logger.info(
                    f"Download cancelled while downloading labels for {split} (file {i+1}/{len(label_files)})")
                return False

            local_label_path = os.path.join(split_labels_dir, label_file)

            # Skip if file already exists
            if os.path.exists(local_label_path):
                downloaded_files[split]['labels'] += 1
                if i % 10 == 0 and cancel_check_func and cancel_check_func():
                    logger.info(f"Download cancelled during label skip check for {split}")
                    return False
                continue

            try:
                logger.debug(f"Downloading label {i+1}/{len(label_files)}: {label_file}")
                success = download_from_azure_datalake(
                    azure_directory=azure_labels_path,
                    file_name=label_file,
                    local_file_path=local_label_path
                )

                if success:
                    downloaded_files[split]['labels'] += 1
                else:
                    logger.warning(f"Failed to download {label_file}")

                # Check cancellation after each download
                if cancel_check_func and cancel_check_func():
                    logger.info(f"Download cancelled after downloading {label_file}")
                    return False

                # Small delay for responsiveness
                if i % 5 == 0:
                    time.sleep(0.01)

            except Exception as e:
                logger.warning(f"Exception downloading label {label_file}: {e}")
                if cancel_check_func and cancel_check_func():
                    logger.info(f"Download cancelled after exception with {label_file}")
                    return False

    # Final cancellation check before YAML download
    if cancel_check_func and cancel_check_func():
        logger.info("Download cancelled before downloading YAML file")
        return False

    # Download the dataset.yaml file if it exists
    logger.info("Downloading dataset.yaml file")
    azure_yaml_path = f"{azure_folder}/dataset.yaml"
    local_yaml_path = os.path.join(local_output_folder, "dataset.yaml")
    try:
        success = download_from_azure_datalake(
            azure_directory=os.path.dirname(azure_yaml_path),
            file_name=os.path.basename(azure_yaml_path),
            local_file_path=local_yaml_path
        )
        if success:
            logger.info(f"Downloaded dataset.yaml from Azure")

        # Final cancellation check
        if cancel_check_func and cancel_check_func():
            logger.info("Download cancelled after YAML download")
            return False

    except Exception as e:
        logger.warning(f"Could not download dataset.yaml: {e}")

    # Log summary of downloaded files
    total_images = sum(data['images'] for data in downloaded_files.values())
    total_labels = sum(data['labels'] for data in downloaded_files.values())

    logger.info(f"Download completed: {total_images} images and {total_labels} labels")
    for split, data in downloaded_files.items():
        logger.info(f"  {split}: {data['images']} images, {data['labels']} labels")

    return downloaded_files if total_images > 0 or total_labels > 0 else False


def combine_local_and_azure_data(
    local_manual_folder: str,
    local_auto_folder: str,
    azure_download_folder: str,
    output_folder: str,
    selected_classes: Optional[List[str]] = None,
    update_config: bool = True,
    prevent_overwrite: bool = True  # New parameter to prevent overwriting
) -> Tuple[bool, Dict[str, int]]:
    """
    Combine local manually annotated data, auto-annotated data, and Azure downloaded data.

    Args:
        local_manual_folder: Path to local manually annotated data
        local_auto_folder: Path to local auto-annotated data
        azure_download_folder: Path to downloaded Azure data
        output_folder: Path to output combined data
        selected_classes: List of class names to include (if None, include all)
        update_config: Whether to update defects_config.json (should be False in worker threads)
        prevent_overwrite: Whether to add source prefixes to prevent filename collisions

    Returns:
        Tuple[bool, Dict]: Success flag and stats of copied files
    """
    # Get class manager to handle class mapping
    class_manager = ClassManager()
    if not class_manager.initialized:
        class_manager.load_from_file()

    # If selected_classes is None, use all classes
    if selected_classes is None:
        selected_classes = class_manager.get_classes()

    # Create mapping of class names to indices (new combined order)
    new_class_mapping = {name: idx for idx, name in enumerate(selected_classes)}

    # Ensure the output directories exist
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'labels'), exist_ok=True)

    # Stats for tracking copies
    stats = {
        'manual': {'images': 0, 'labels': 0},
        'auto': {'images': 0, 'labels': 0},
        'azure': {'images': 0, 'labels': 0}
    }

    # Keep track of used filenames to avoid duplicates
    used_filenames = {
        'train': set(),
        'valid': set(),
        'test': set()
    }

    # Process manually annotated data (assuming it's already split or we'll put it in train)
    if os.path.exists(local_manual_folder):
        manual_split_dirs = [d for d in os.listdir(local_manual_folder)
                             if os.path.isdir(os.path.join(local_manual_folder, d))
                             and d in ['train', 'valid', 'test']]

        if manual_split_dirs:  # If already split
            for split in manual_split_dirs:
                images_dir = os.path.join(local_manual_folder, split, 'images')
                labels_dir = os.path.join(local_manual_folder, split, 'labels')

                if os.path.exists(images_dir) and os.path.exists(labels_dir):
                    for img_file in os.listdir(images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            # Add prefix to prevent overwriting if enabled
                            dst_img_file = f"manual_{img_file}" if prevent_overwrite else img_file

                            # If filename already used, add unique identifier
                            if prevent_overwrite and dst_img_file in used_filenames[split]:
                                base, ext = os.path.splitext(dst_img_file)
                                dst_img_file = f"{base}_{stats['manual']['images']}{ext}"

                            # Mark filename as used
                            if prevent_overwrite:
                                used_filenames[split].add(dst_img_file)

                            # Copy image
                            src_img = os.path.join(images_dir, img_file)
                            dst_img = os.path.join(output_folder, split, 'images', dst_img_file)
                            shutil.copy2(src_img, dst_img)
                            stats['manual']['images'] += 1

                            # Look for corresponding label file
                            label_file = os.path.splitext(img_file)[0] + '.txt'
                            dst_label_file = os.path.splitext(dst_img_file)[0] + '.txt'
                            src_label = os.path.join(labels_dir, label_file)
                            if os.path.exists(src_label):
                                # Map class indices if needed
                                dst_label = os.path.join(
                                    output_folder, split, 'labels', dst_label_file)
                                _convert_and_copy_label(
                                    src_label, dst_label, new_class_mapping, selected_classes)
                                stats['manual']['labels'] += 1
        else:  # If not split, assume all in root
            # Find label files
            label_files = [f for f in os.listdir(local_manual_folder)
                           if f.endswith('.txt') and os.path.isfile(os.path.join(local_manual_folder, f))]

            for label_file in label_files:
                img_base = os.path.splitext(label_file)[0]
                found_img = False

                # Look for matching image with various extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_file = img_base + ext
                    img_path = os.path.join(local_manual_folder, img_file)

                    if os.path.exists(img_path):
                        # Add prefix to prevent overwriting if enabled
                        dst_img_file = f"manual_{img_file}" if prevent_overwrite else img_file
                        dst_label_file = f"manual_{label_file}" if prevent_overwrite else label_file

                        # If filename already used, add unique identifier
                        if prevent_overwrite and dst_img_file in used_filenames['train']:
                            base, ext = os.path.splitext(dst_img_file)
                            dst_img_file = f"{base}_{stats['manual']['images']}{ext}"
                            base, ext = os.path.splitext(dst_label_file)
                            dst_label_file = f"{base}_{stats['manual']['labels']}{ext}"

                        # Mark filename as used
                        if prevent_overwrite:
                            used_filenames['train'].add(dst_img_file)

                        # Copy image to train split
                        dst_img = os.path.join(output_folder, 'train', 'images', dst_img_file)
                        shutil.copy2(img_path, dst_img)
                        stats['manual']['images'] += 1

                        # Copy and convert label
                        src_label = os.path.join(local_manual_folder, label_file)
                        dst_label = os.path.join(output_folder, 'train', 'labels', dst_label_file)
                        _convert_and_copy_label(src_label, dst_label,
                                                new_class_mapping, selected_classes)
                        stats['manual']['labels'] += 1

                        found_img = True
                        break

                if not found_img:
                    logger.warning(f"No matching image found for label: {label_file}")

    # Process auto-annotated data (similar logic as manual data)
    if os.path.exists(local_auto_folder):
        auto_split_dirs = [d for d in os.listdir(local_auto_folder)
                           if os.path.isdir(os.path.join(local_auto_folder, d))
                           and d in ['train', 'valid', 'test']]

        if auto_split_dirs:  # If already split
            for split in auto_split_dirs:
                images_dir = os.path.join(local_auto_folder, split, 'images')
                labels_dir = os.path.join(local_auto_folder, split, 'labels')

                if os.path.exists(images_dir) and os.path.exists(labels_dir):
                    for img_file in os.listdir(images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            # Add prefix to prevent overwriting if enabled
                            dst_img_file = f"auto_{img_file}" if prevent_overwrite else img_file

                            # If filename already used, add unique identifier
                            if prevent_overwrite and dst_img_file in used_filenames[split]:
                                base, ext = os.path.splitext(dst_img_file)
                                dst_img_file = f"{base}_{stats['auto']['images']}{ext}"

                            # Mark filename as used
                            if prevent_overwrite:
                                used_filenames[split].add(dst_img_file)

                            # Copy image
                            src_img = os.path.join(images_dir, img_file)
                            dst_img = os.path.join(output_folder, split, 'images', dst_img_file)
                            shutil.copy2(src_img, dst_img)
                            stats['auto']['images'] += 1

                            # Look for corresponding label file
                            label_file = os.path.splitext(img_file)[0] + '.txt'
                            dst_label_file = os.path.splitext(dst_img_file)[0] + '.txt'
                            src_label = os.path.join(labels_dir, label_file)
                            if os.path.exists(src_label):
                                # Map class indices if needed
                                dst_label = os.path.join(
                                    output_folder, split, 'labels', dst_label_file)
                                _convert_and_copy_label(
                                    src_label, dst_label, new_class_mapping, selected_classes)
                                stats['auto']['labels'] += 1
        else:  # If not split, assume all in root
            # Find label files
            label_files = [f for f in os.listdir(local_auto_folder)
                           if f.endswith('.txt') and os.path.isfile(os.path.join(local_auto_folder, f))]

            for label_file in label_files:
                img_base = os.path.splitext(label_file)[0]
                found_img = False

                # Look for matching image with various extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_file = img_base + ext
                    img_path = os.path.join(local_auto_folder, img_file)

                    if os.path.exists(img_path):
                        # Add prefix to prevent overwriting if enabled
                        dst_img_file = f"auto_{img_file}" if prevent_overwrite else img_file
                        dst_label_file = f"auto_{label_file}" if prevent_overwrite else label_file

                        # If filename already used, add unique identifier
                        if prevent_overwrite and dst_img_file in used_filenames['train']:
                            base, ext = os.path.splitext(dst_img_file)
                            dst_img_file = f"{base}_{stats['auto']['images']}{ext}"
                            base, ext = os.path.splitext(dst_label_file)
                            dst_label_file = f"{base}_{stats['auto']['labels']}{ext}"

                        # Mark filename as used
                        if prevent_overwrite:
                            used_filenames['train'].add(dst_img_file)

                        # Copy image to train split
                        dst_img = os.path.join(output_folder, 'train', 'images', dst_img_file)
                        shutil.copy2(img_path, dst_img)
                        stats['auto']['images'] += 1

                        # Copy and convert label
                        src_label = os.path.join(local_auto_folder, label_file)
                        dst_label = os.path.join(output_folder, 'train', 'labels', dst_label_file)
                        _convert_and_copy_label(src_label, dst_label,
                                                new_class_mapping, selected_classes)
                        stats['auto']['labels'] += 1

                        found_img = True
                        break

                if not found_img:
                    logger.warning(f"No matching image found for label: {label_file}")

    # Process Azure downloaded data
    if os.path.exists(azure_download_folder):
        for split in ['train', 'valid', 'test']:
            images_dir = os.path.join(azure_download_folder, split, 'images')
            labels_dir = os.path.join(azure_download_folder, split, 'labels')

            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                for img_file in os.listdir(images_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # Add prefix to prevent overwriting if enabled
                        dst_img_file = f"azure_{img_file}" if prevent_overwrite else img_file

                        # If filename already used, add unique identifier
                        if prevent_overwrite and dst_img_file in used_filenames[split]:
                            base, ext = os.path.splitext(dst_img_file)
                            dst_img_file = f"{base}_{stats['azure']['images']}{ext}"

                        # Mark filename as used
                        if prevent_overwrite:
                            used_filenames[split].add(dst_img_file)

                        # Copy image
                        src_img = os.path.join(images_dir, img_file)
                        dst_img = os.path.join(output_folder, split, 'images', dst_img_file)
                        shutil.copy2(src_img, dst_img)
                        stats['azure']['images'] += 1

                        # Look for corresponding label file
                        label_file = os.path.splitext(img_file)[0] + '.txt'
                        dst_label_file = os.path.splitext(dst_img_file)[0] + '.txt'
                        src_label = os.path.join(labels_dir, label_file)
                        if os.path.exists(src_label):
                            # Map class indices if needed
                            dst_label = os.path.join(output_folder, split,
                                                     'labels', dst_label_file)
                            _convert_and_copy_label(src_label, dst_label,
                                                    new_class_mapping, selected_classes)
                            stats['azure']['labels'] += 1

    # Create a dataset.yaml file in the output folder
    yaml_path = os.path.join(output_folder, 'dataset.yaml')

    with open(yaml_path, 'w') as f:
        yaml.dump({
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': len(selected_classes),
            'names': selected_classes
        }, f)

    logger.info(f"Created dataset.yaml with {len(selected_classes)} classes")

    # Log summary of combined data
    total_images = sum(data['images'] for data in stats.values())
    total_labels = sum(data['labels'] for data in stats.values())

    logger.info(f"Combined {total_images} images and {total_labels} labels in {output_folder}")
    for source, data in stats.items():
        logger.info(f"  {source}: {data['images']} images, {data['labels']} labels")

    # Update defects_config.json with these classes
    if update_config:
        update_defects_config(selected_classes, selected_classes)

    return True, stats


def combine_three_datasets_with_yaml(dataset1_path, dataset2_path, dataset3_path, output_path):
    """
    Combines three YOLO-format datasets into a single output directory,
    and merges their data.yaml files (union of all class names).
    """
    import os
    import shutil
    import yaml

    splits = ['train', 'valid', 'test']
    stats = {
        'dataset1': {'images': 0, 'labels': 0},
        'dataset2': {'images': 0, 'labels': 0},
        'dataset3': {'images': 0, 'labels': 0},
        'total': {'images': 0, 'labels': 0}
    }

    def copy_files(src_dir, dst_dir, file_exts):
        if not os.path.exists(src_dir):
            return 0
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        for file in os.listdir(src_dir):
            if file.lower().endswith(file_exts):
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
                count += 1
        return count

    # Copy images and labels for each split and dataset
    for split in splits:
        for idx, dataset_path in enumerate([dataset1_path, dataset2_path, dataset3_path], 1):
            images_src = os.path.join(dataset_path, split, 'images')
            labels_src = os.path.join(dataset_path, split, 'labels')
            images_dst = os.path.join(output_path, split, 'images')
            labels_dst = os.path.join(output_path, split, 'labels')

            img_count = copy_files(images_src, images_dst, ('.jpg', '.jpeg', '.png', '.bmp'))
            lbl_count = copy_files(labels_src, labels_dst, ('.txt',))

            stats[f'dataset{idx}']['images'] += img_count
            stats[f'dataset{idx}']['labels'] += lbl_count
            stats['total']['images'] += img_count
            stats['total']['labels'] += lbl_count

    # Merge data.yaml files
    def get_class_names(yaml_path):
        if not os.path.exists(yaml_path):
            return []
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])

    yaml_paths = [
        os.path.join(dataset1_path, 'data.yaml'),
        os.path.join(dataset2_path, 'data.yaml'),
        os.path.join(dataset3_path, 'data.yaml')
    ]
    all_classes = set()
    for ypath in yaml_paths:
        all_classes.update(get_class_names(ypath))
    all_classes = sorted(list(all_classes))

    # Write merged data.yaml
    merged_yaml = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(all_classes),
        'names': all_classes
    }
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        yaml.dump(merged_yaml, f)

    return True, stats


def combine_two_datasets_with_yaml(dataset1_path, dataset2_path, output_path):
    """
    Combines three YOLO-format datasets into a single output directory,
    and merges their data.yaml files (union of all class names).
    """
    import os
    import shutil
    import yaml

    splits = ['train', 'valid', 'test']
    stats = {
        'dataset1': {'images': 0, 'labels': 0},
        'dataset2': {'images': 0, 'labels': 0},
        'total': {'images': 0, 'labels': 0}
    }

    def copy_files(src_dir, dst_dir, file_exts):
        if not os.path.exists(src_dir):
            return 0
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        for file in os.listdir(src_dir):
            if file.lower().endswith(file_exts):
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
                count += 1
        return count

    # Copy images and labels for each split and dataset
    for split in splits:
        for idx, dataset_path in enumerate([dataset1_path, dataset2_path], 1):
            images_src = os.path.join(dataset_path, split, 'images')
            labels_src = os.path.join(dataset_path, split, 'labels')
            images_dst = os.path.join(output_path, split, 'images')
            labels_dst = os.path.join(output_path, split, 'labels')

            img_count = copy_files(images_src, images_dst, ('.jpg', '.jpeg', '.png', '.bmp'))
            lbl_count = copy_files(labels_src, labels_dst, ('.txt',))

            stats[f'dataset{idx}']['images'] += img_count
            stats[f'dataset{idx}']['labels'] += lbl_count
            stats['total']['images'] += img_count
            stats['total']['labels'] += lbl_count

    # Merge data.yaml files
    def get_class_names(yaml_path):
        if not os.path.exists(yaml_path):
            return []
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])

    yaml_paths = [
        os.path.join(dataset1_path, 'data.yaml'),
        os.path.join(dataset2_path, 'data.yaml')
    ]
    all_classes = set()
    for ypath in yaml_paths:
        all_classes.update(get_class_names(ypath))
    all_classes = sorted(list(all_classes))

    # Write merged data.yaml
    merged_yaml = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(all_classes),
        'names': all_classes
    }
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        yaml.dump(merged_yaml, f)

    return True, stats


def combine_two_datasets(dataset1_path: str, dataset2_path: str, output_path: str) -> Tuple[bool, Dict[str, int]]:
    """
    Combine two datasets by merging their images and labels into a single output directory.

    Args:
        dataset1_path (str): Path to the first dataset.
        dataset2_path (str): Path to the second dataset.
        output_path (str): Path to the output directory where the combined dataset will be stored.

    Returns:
        Tuple[bool, Dict[str, int]]: A tuple containing a success flag and a dictionary with stats of combined files.
    """
    try:
        # Ensure output directories exist
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

        # Stats for tracking combined files
        stats = {
            'images': 0,
            'labels': 0
        }

        # Helper function to copy files
        def copy_files(src_dir, dst_dir, file_ext):
            nonlocal stats
            if os.path.exists(src_dir):
                for file_name in os.listdir(src_dir):
                    if file_name.lower().endswith(file_ext):
                        src_file = os.path.join(src_dir, file_name)
                        dst_file = os.path.join(dst_dir, file_name)
                        shutil.copy2(src_file, dst_file)
                        stats['images' if file_ext in [
                            '.jpg', '.jpeg', '.png', '.bmp'] else 'labels'] += 1

        # Combine images and labels for each split
        for split in ['train', 'valid', 'test']:
            # Combine images
            copy_files(os.path.join(dataset1_path, split, 'images'), os.path.join(
                output_path, split, 'images'), ('.jpg', '.jpeg', '.png', '.bmp'))
            copy_files(os.path.join(dataset2_path, split, 'images'), os.path.join(
                output_path, split, 'images'), ('.jpg', '.jpeg', '.png', '.bmp'))

            # Combine labels
            copy_files(os.path.join(dataset1_path, split, 'labels'),
                       os.path.join(output_path, split, 'labels'), ('.txt',))
            copy_files(os.path.join(dataset2_path, split, 'labels'),
                       os.path.join(output_path, split, 'labels'), ('.txt',))

        return True, stats

    except Exception as e:
        logging.error(f"Error combining datasets: {e}")
        return False, {}


def upload_combined_dataset_to_azure(
    local_folder: str,
    azure_folder: str = AZURE_SAVAI_FOLDER,
    file_system_name: str = DEFAULT_FILE_SYSTEM,
    progress_callback: callable = None
) -> Tuple[bool, Dict[str, int]]:
    """
    Upload any folder structure to Azure Data Lake Storage with real progress and cancellation.

    Args:
        local_folder: Local folder to upload (any structure)
        azure_folder: Azure folder path to upload to
        file_system_name: Azure file system name
        progress_callback: Optional callback for progress updates (message, current_file, total_files)
                         Should return True to continue, False to cancel

    Returns:
        Tuple[bool, Dict]: Success flag and stats of uploaded files
    """
    import time
    import os

    logger.info(f"Starting upload from {local_folder} to Azure folder: {azure_folder}")

    # Stats for tracking uploads - more generic categories
    stats = {
        'images': 0,      # jpg, jpeg, png, bmp, gif, tiff, webp
        'text_files': 0,  # txt, yaml, yml, json, csv, md
        'documents': 0,   # pdf, doc, docx, xls, xlsx, ppt, pptx
        'code_files': 0,  # py, js, html, css, cpp, java, etc.
        'other_files': 0  # everything else
    }

    # File extension mappings for categorization
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.svg'}
    text_extensions = {'.txt', '.yaml', '.yml', '.json', '.csv', '.md', '.xml', '.log'}
    document_extensions = {'.pdf', '.doc', '.docx', '.xls',
                           '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp'}
    code_extensions = {'.py', '.js', '.html', '.css', '.cpp', '.c', '.java', '.php',
                       '.rb', '.go', '.rs', '.ts', '.jsx', '.tsx', '.vue', '.sql', '.sh', '.bat', '.ps1'}

    def categorize_file(filename: str) -> str:
        """Categorize file based on extension"""
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            return 'images'
        elif ext in text_extensions:
            return 'text_files'
        elif ext in document_extensions:
            return 'documents'
        elif ext in code_extensions:
            return 'code_files'
        else:
            return 'other_files'

    # First pass: count all files to upload for accurate progress
    total_files = 0
    file_queue = []  # List of (local_path, azure_path, filename, file_category)

    def check_cancellation():
        """Check if upload should be cancelled"""
        if progress_callback:
            try:
                return not progress_callback("Checking for cancellation...", None, None)
            except:
                return True  # If callback fails, assume cancellation
        return False

    # Initial cancellation check
    if check_cancellation():
        logger.info("Upload cancelled before starting")
        return False, stats

    # Validate local folder exists
    if not os.path.exists(local_folder):
        logger.error(f"Local folder does not exist: {local_folder}")
        if progress_callback:
            progress_callback(f"Error: Local folder not found: {local_folder}", 0, 0)
        return False, stats

    if not os.path.isdir(local_folder):
        logger.error(f"Path is not a directory: {local_folder}")
        if progress_callback:
            progress_callback(f"Error: Path is not a directory: {local_folder}", 0, 0)
        return False, stats

    # Count and queue files for upload
    if progress_callback:
        if not progress_callback("Analyzing files to upload...", None, None):
            logger.info("Upload cancelled during file analysis")
            return False, stats

    def scan_directory(current_local_path: str, current_azure_path: str):
        """Recursively scan directory and queue files for upload"""
        nonlocal total_files

        try:
            items = os.listdir(current_local_path)
        except PermissionError:
            logger.warning(f"Permission denied accessing: {current_local_path}")
            return
        except Exception as e:
            logger.warning(f"Error accessing directory {current_local_path}: {e}")
            return

        for item in items:
            # Check cancellation periodically during scanning
            if total_files % 50 == 0 and check_cancellation():
                logger.info(f"Upload cancelled while scanning directory: {current_local_path}")
                return

            item_local_path = os.path.join(current_local_path, item)

            # Skip hidden files and directories (starting with .)
            if item.startswith('.'):
                continue

            if os.path.isfile(item_local_path):
                # It's a file - add to queue
                try:
                    # Get file size to skip empty files if needed
                    file_size = os.path.getsize(item_local_path)
                    if file_size == 0:
                        logger.debug(f"Skipping empty file: {item_local_path}")
                        continue

                    file_category = categorize_file(item)
                    file_queue.append((item_local_path, current_azure_path, item, file_category))
                    total_files += 1

                    # Progress update during scanning for large directories
                    if total_files % 100 == 0 and progress_callback:
                        if not progress_callback(f"Found {total_files} files to upload...", None, None):
                            logger.info("Upload cancelled during file counting")
                            return

                except Exception as e:
                    logger.warning(f"Error processing file {item_local_path}: {e}")

            elif os.path.isdir(item_local_path):
                # It's a directory - recurse into it
                # Maintain relative path structure in Azure
                relative_path = os.path.relpath(item_local_path, local_folder)
                # Normalize path separators for Azure (use forward slashes)
                relative_azure_path = relative_path.replace(os.sep, '/')
                new_azure_path = f"{current_azure_path}/{relative_azure_path}"

                scan_directory(item_local_path, azure_folder)

    # Start scanning from the root folder
    scan_directory(local_folder, azure_folder)

    # Final check before starting upload
    if check_cancellation():
        logger.info("Upload cancelled before starting file upload")
        return False, stats

    if not file_queue:
        logger.warning("No files to upload")
        if progress_callback:
            progress_callback("No files found to upload", 0, 0)
        return True, stats  # Return True since no error occurred, just no files

    logger.info(f"Starting upload of {total_files} files to Azure folder: {azure_folder}")

    # Upload files one by one with progress tracking
    uploaded_files = 0
    failed_files = 0

    for file_idx, (local_path, azure_dir, filename, file_category) in enumerate(file_queue):
        # Check cancellation before each file
        if check_cancellation():
            logger.info(
                f"Upload cancelled during file upload ({uploaded_files}/{total_files} completed)")
            return False, stats

        try:
            # Calculate relative path for Azure
            relative_local_path = os.path.relpath(local_path, local_folder)
            relative_dir = os.path.dirname(relative_local_path)

            # Construct Azure directory path
            if relative_dir and relative_dir != '.':
                # Normalize path separators for Azure
                relative_dir = relative_dir.replace(os.sep, '/')
                final_azure_dir = f"{azure_folder}/{relative_dir}"
            else:
                final_azure_dir = azure_folder

            # Update progress with current file
            if progress_callback:
                if not progress_callback(
                    f"Uploading {file_category}: {relative_local_path}",
                    file_idx,
                    total_files
                ):
                    logger.info(f"Upload cancelled while uploading {filename}")
                    return False, stats

            # Perform the actual upload
            logger.debug(
                f"Uploading {file_category} {file_idx + 1}/{total_files}: {relative_local_path}")

            success = upload_to_azure_datalake(
                azure_directory=final_azure_dir,
                file_name=filename,
                local_file_path=local_path,
                file_system_name=file_system_name
            )

            if success:
                uploaded_files += 1
                # Update stats based on file category
                stats[file_category] += 1
                logger.debug(f"Successfully uploaded {relative_local_path}")
            else:
                failed_files += 1
                logger.warning(f"Failed to upload {relative_local_path}")

            # Check cancellation after each upload
            if check_cancellation():
                logger.info(f"Upload cancelled after uploading {filename}")
                return False, stats

            # Small delay to make cancellation more responsive
            if file_idx % 10 == 0:  # Every 10 files
                time.sleep(0.01)  # 10ms delay

        except Exception as e:
            failed_files += 1
            logger.error(f"Exception uploading {relative_local_path}: {e}")

            # Check cancellation even after exceptions
            if check_cancellation():
                logger.info(f"Upload cancelled after exception with {filename}")
                return False, stats

    # Final progress update
    if progress_callback:
        if not progress_callback(
            f"Upload complete: {uploaded_files} successful, {failed_files} failed",
            total_files,
            total_files
        ):
            logger.info("Cancellation detected at completion")

    # Final cancellation check
    if check_cancellation():
        logger.info("Upload cancelled at completion")
        return False, stats

    # Log detailed statistics
    logger.info(f"Upload statistics: {stats}")

    # Determine success
    if failed_files > 0:
        logger.warning(f"Upload completed with {failed_files} failures out of {total_files} files")
        return False, stats
    else:
        logger.info(
            f"Successfully uploaded all {uploaded_files} files to Azure folder: {azure_folder}")
        return True, stats


def update_defects_config(selected_classes: List[str], new_classes: List[str]) -> bool:
    """
    Update the defects_config.json file with new classes.

    Args:
        selected_classes: List of selected class names
        new_classes: List of new class names to add

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from datetime import datetime
        config_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "defects_config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get the current selected defects list or create it if it doesn't exist
            existing_defects = config.get('selected_defects', [])

            # Add new classes that don't exist yet
            updated_defects = list(existing_defects)  # Create a copy
            classes_added = False

            for class_name in new_classes:
                if class_name not in updated_defects:
                    updated_defects.append(class_name)
                    classes_added = True
                    logger.info(f"Added new class to defects_config.json: {class_name}")

            # Update the config with all defects
            config['selected_defects'] = updated_defects

            # Update timestamp
            config['timestamp'] = datetime.now().isoformat()

            # Write updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Updated defects_config.json with {len(new_classes)} new classes")

            # If we added classes, also update the ClassManager
            if classes_added:
                from utils import ClassManager
                class_manager = ClassManager()
                class_manager.update_classes(updated_defects, "data_integration_update")

            return True
        else:
            # If the file doesn't exist, create it with the new classes
            config = {
                "selected_defects": new_classes,
                "user": "system",
                "timestamp": datetime.now().isoformat()
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Created new defects_config.json with {len(new_classes)} classes")

            # Update ClassManager
            from utils import ClassManager
            class_manager = ClassManager()
            class_manager.update_classes(new_classes, "data_integration_create")

            return True
    except Exception as e:
        logger.error(f"Error updating defects_config.json: {e}")
        return False


def update_dataset_yaml(yaml_path: str, class_names: List[str]) -> bool:
    """
    Updates or creates a dataset.yaml file with the proper class names.

    Args:
        yaml_path: Path to the dataset.yaml file
        class_names: List of class names to include

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if the yaml file exists
        yaml_exists = os.path.exists(yaml_path)

        if yaml_exists:
            # Load existing yaml data
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Update class names and count
            yaml_data['names'] = class_names
            yaml_data['nc'] = len(class_names)
        else:
            # Create default yaml structure
            yaml_data = {
                'train': '../train/images',
                'val': '../valid/images',
                'test': '../test/images',
                'nc': len(class_names),
                'names': class_names
            }

        # Write yaml file
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)

        logger.info(f"Updated dataset.yaml with {len(class_names)} classes at {yaml_path}")
        return True

    except Exception as e:
        logger.error(f"Error updating dataset.yaml: {e}")
        return False


def find_dataset_yaml_files() -> List[str]:
    """
    Finds all dataset.yaml files in the standard locations.

    Returns:
        List[str]: List of paths to dataset.yaml files
    """
    yaml_files = []

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Check for dataset.yaml in standard locations
    standard_locations = [
        current_dir,  # Current directory
        os.path.join(current_dir, 'manually_annotated'),  # Manually annotated folder
        os.path.join(current_dir, 'auto_annotated'),  # Auto annotated folder
        os.path.join(current_dir, 'combined_data')  # Combined data folder
    ]

    # Check if LOCAL paths are configured
    if LOCAL_MANUAL_ANNOTATED:
        standard_locations.append(LOCAL_MANUAL_ANNOTATED)
    if LOCAL_AUTO_ANNOTATED:
        standard_locations.append(LOCAL_AUTO_ANNOTATED)
    if LOCAL_COMBINED_DATA:
        standard_locations.append(LOCAL_COMBINED_DATA)

    # Look for dataset.yaml in standard locations
    for location in standard_locations:
        yaml_path = os.path.join(location, 'dataset.yaml')
        if os.path.exists(yaml_path):
            yaml_files.append(yaml_path)
            logger.info(f"Found dataset.yaml at {yaml_path}")

    # Look in subdirectories under standard locations
    for location in standard_locations:
        if os.path.exists(location):
            for subdir in os.listdir(location):
                subdir_path = os.path.join(location, subdir)
                if os.path.isdir(subdir_path):
                    yaml_path = os.path.join(subdir_path, 'dataset.yaml')
                    if os.path.exists(yaml_path) and yaml_path not in yaml_files:
                        yaml_files.append(yaml_path)
                        logger.info(f"Found dataset.yaml at {yaml_path}")

    return yaml_files


def update_all_yaml_files(class_names: List[str]) -> Tuple[bool, int]:
    """
    Updates all found dataset.yaml files with the given class names.

    Args:
        class_names: List of class names to include

    Returns:
        Tuple[bool, int]: Success flag and count of updated files
    """
    yaml_files = find_dataset_yaml_files()

    if not yaml_files:
        logger.warning("No dataset.yaml files found to update")
        return False, 0

    success_count = 0
    for yaml_path in yaml_files:
        if update_dataset_yaml(yaml_path, class_names):
            success_count += 1

    logger.info(f"Updated {success_count} of {len(yaml_files)} dataset.yaml files")
    return success_count > 0, success_count


def sync_annotations_to_azure(
    local_folder: str,
    azure_folder: str = AZURE_SAVAI_FOLDER,
    file_system_name: str = DEFAULT_FILE_SYSTEM
) -> Tuple[bool, Dict[str, int]]:
    """
    Synchronizes local annotations with Azure storage.

    This function scans the local folder for image and annotation pairs,
    then uploads them to the corresponding Azure folder.

    Args:
        local_folder: Path to the local folder containing annotations
        azure_folder: Path to the Azure folder to sync with
        file_system_name: Azure file system name

    Returns:
        Tuple[bool, Dict]: Success flag and stats of uploads
    """
    import os

    # Stats for tracking uploads
    stats = {
        'images': 0,
        'labels': 0
    }

    # Check if the local folder exists
    if not os.path.exists(local_folder):
        logger.error(f"Local folder does not exist: {local_folder}")
        return False, stats

    # List all files in the local folder
    local_files = os.listdir(local_folder)

    # Find all annotation files (.txt)
    annotation_files = [f for f in local_files if f.endswith('.txt')]

    # Image extensions to check
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Prepare batch upload queue
    pending_uploads = []

    for annotation_file in annotation_files:
        # Get the base name (without extension)
        base_name = os.path.splitext(annotation_file)[0]

        # Look for a matching image file
        image_file = None
        for ext in img_extensions:
            img_name = base_name + ext
            if img_name in local_files:
                image_file = img_name
                break

        if image_file:
            # Found a matching image-annotation pair
            image_path = os.path.join(local_folder, image_file)
            annotation_path = os.path.join(local_folder, annotation_file)

            # Queue them for upload
            pending_uploads.append((image_path, azure_folder, image_file))
            pending_uploads.append((annotation_path, azure_folder, annotation_file))

            stats['images'] += 1
            stats['labels'] += 1

    # Upload the files in batch
    if pending_uploads:
        logger.info(
            f"Starting Azure sync with {len(pending_uploads)} files ({stats['images']} image-label pairs)")
        success_count, fail_count = batch_upload_to_azure(pending_uploads, file_system_name)

        if fail_count > 0:
            logger.warning(
                f"Failed to upload {fail_count} of {len(pending_uploads)} files to Azure")
            return False, stats

        logger.info(f"Successfully uploaded {success_count} files to Azure")
        return True, stats
    else:
        logger.info(f"No annotation pairs found to sync in {local_folder}")
        return True, stats
