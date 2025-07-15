# Sakar Vision UI

Sakar Vision UI is a comprehensive computer vision toolkit featuring camera feed handling, manual/automatic annotation, and image classification capabilities.

## Features

### 1. Camera Feed Interface
- Live camera feed display
- Image capture functionality
- Progress tracking for image capture
- Integration with annotation tools

### 2. Manual Annotation Tool
- Custom bounding box creation
- Box editing capabilities (move, resize, delete)
- Class label management
- YOLO format annotation export
- Custom RangeSlider for dataset splitting

### 3. Auto Annotation Tool
- YOLO model integration for automatic detection
- Manual verification/editing of auto-generated annotations 
- Confidence threshold adjustment
- Batch processing capabilities
- Dataset splitting functionality
kiok
### 4. Image Classification UI
- TensorFlow/Keras based classification
- Model training interface
- Real-time training progress display
- Test image viewer with predictions
- Session management for multiple training runs

## Entry Point

The entry point of the application is `inspection_selection_ui.py`. This script allows users to select the type of inspection they want to perform (Fabric or Metal Sheet) and proceed through the necessary steps.
# System Requirements
Python 3.8+
Qt6 (for demo feed) or Qt5 (for other components)

# Required Python Packages
pip install -r requirements.txt
```

Contents of requirements.txt:
```txt
PyQt5==5.15.6
opencv-python==4.11.0.86
numpy==1.26.4
tensorflow>=2.0.0
ultralytics
```

## Environment Setup

### Linux Setup
Set the Qt plugin path:
```bash
sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0

# For Qt5 components
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins

# For Qt6 components (demo feed)
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt6/plugins
```

## Usage

### 1. Camera Feed Interface
```bash
python3 camera_feed_ui.py
```
- Click "Capture Images Setup" to start/stop image capture
- Adjust capture settings as needed
- Monitor progress through the progress bar

### 2. Manual Annotation Tool
```bash
python3 manual_annotation_ui.py
```
- Load images from directory
- Create/edit bounding boxes
- Set class labels
- Export annotations in YOLO format

### 3. Auto Annotation Tool
```bash
python3 auto_annotation.py
```
- Select YOLO model (.pt file)
- Choose input/output directories
- Set confidence threshold
- Review and edit auto-generated annotations

### 4. Image Classification
```bash
python3 image_classi_ui.py
```
- Select training data directory
- Configure and train model
- View training progress
- Test on new images

## Project Structure

```
sakar_vision_ui/
├── camera_feed_ui.py      # Camera interface
├── manual_annotation_ui.py # Manual annotation tool
├── auto_annotation.py      # Auto annotation using YOLO
├── image_classi_ui.py     # Classification interface
├── demo_feed_ui.py        # Demo feed implementation
└── requirements.txt       # Package dependencies
```

## Key Components

### RangeSlider
- Custom widget for dataset splitting
- Visual adjustment of train/validation/test ratios
- Implemented in multiple UIs

### Annotation Tools
- Support for YOLO format
- Bounding box manipulation
- Class management
- Dataset organization

### Image Classification
- CNN-based architecture
- Real-time training monitoring
- Model evaluation tools
- Test image navigation

## Troubleshooting

1. Qt Plugin Issues:
   - Verify correct Qt version installation
   - Check environment variables
   - Ensure matching PyQt version

2. Camera Access:
   - Check camera permissions
   - Verify device ID
   - Test camera with other applications

3. YOLO Model:
   - Confirm model file (.pt) exists
   - Check class names match dataset
   - Verify ultralytics installation

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License

This project is provided as-is under the MIT License.

## Contact

For issues and suggestions, please use the GitHub issue tracker.

