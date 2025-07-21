# SAKAR Vision AI - Machine Vision Platform

A comprehensive industrial machine vision platform for real-time defect detection and quality control in manufacturing environments. This system combines advanced AI/ML capabilities with user-friendly interfaces for fabric and metal sheet inspection workflows.

## 🚀 Features

- **Real-time Defect Detection**: YOLO-based AI models for instant defect identification
- **Dual Inspection Modes**: Support for both fabric and metal sheet inspection workflows
- **Advanced Object Tracking**: DeepSORT integration to prevent duplicate counting
- **Azure Cloud Integration**: Database synchronization and model storage capabilities
- **Session Persistence**: Automatic crash recovery and state restoration
- **Professional UI**: Modern PyQt5-based interface with responsive design
- **Comprehensive Logging**: Multi-level logging system for debugging and monitoring

## 📋 Table of Contents

- [Installation & Setup](#installation--setup)
- [Project Flow & Architecture](#project-flow--architecture)
- [Crash Recovery Feature](#crash-recovery-feature)
- [File Structure](#file-structure)
- [Usage Instructions](#usage-instructions)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Webcam or USB camera for live inspection
- Internet connection for Azure services

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd machine-vision-platform/sakar_vision_ui
```

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv sakar_vision_env

# Activate virtual environment
# On Linux/Mac:
source sakar_vision_env/bin/activate
# On Windows:
sakar_vision_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify critical dependencies
python -c "import cv2, torch, ultralytics; print('All dependencies installed successfully')"
```

### Step 4: Environment Configuration

#### Azure Configuration (Optional)
If using Azure services, create a `.env` file in the project root:

```bash
AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
AZURE_DATABASE_HOST="your_database_host"
AZURE_DATABASE_USER="your_username"
AZURE_DATABASE_PASSWORD="your_password"
AZURE_DATABASE_NAME="your_database_name"
```

#### Model Files
Ensure you have the required AI model files:
- `best.pt` - Primary YOLO detection model
- `sakar_ai.pt` - Secondary detection model
- `sakar_cls.pth` - Classification model

### Step 5: Initial Setup

```bash
# Run the application for the first time
python main.py
```

The system will automatically create necessary configuration files during first run.

## 🏗️ Project Flow & Architecture

### Complete Application Workflow

The software follows a comprehensive workflow with multiple paths depending on user needs:

#### **Main Workflow Path:**
```
main.py → inspection_selection_ui.py → login_ui.py → 
organization_creation_ui.py → location_selection_ui.py → 
defect_selection_ui.py → deploy_ui.py
```

#### **Alternative Workflow Paths:**

**1. Image Classification Path:**
```
defect_selection_ui.py → [Click "Defective"] → image_classi_ui.py → 
[Images Split Train] → demo_feed_ui.py
```

**2. New Defect Training Path:**
```
defect_selection_ui.py → [Click "New Defects"] → 
[Capture Images] → manual_annotation_ui.py → auto_annotation.py → 
[Accurate Training & Fast Training] → deploy_ui.py
```

**3. Direct Deployment Path:**
```
defect_selection_ui.py → [Select Existing Defects] → deploy_ui.py
```

### Core Modules Explained

#### 1. **main.py**
- **Purpose**: Primary application entry point
- **Function**: Environment validation, dependency checking, logging setup
- **Connections**: Launches `inspection_selection_ui.py`

#### 2. **inspection_selection_ui.py**
- **Purpose**: Initial interface for choosing inspection type
- **Function**: Provides fabric vs. metal sheet inspection selection
- **Features**: System connectivity validation, camera detection
- **Connections**: Routes to `login_ui.py`

#### 3. **login_ui.py**
- **Purpose**: User authentication system
- **Function**: Secure login with SHA-256 encryption, session management
- **Features**: Login attempt tracking, account lockout protection
- **Connections**: Forwards to organization setup

#### 4. **organization_creation_ui.py**
- **Purpose**: Organization management interface
- **Function**: Create or select organization for the inspection workflow
- **Features**: Organization data validation, database integration
- **Connections**: Routes to `location_selection_ui.py`

#### 5. **location_selection_ui.py** / **metal_sheet_location_ui.py**
- **Purpose**: Location and project configuration
- **Function**: Define inspection location and project settings
- **Features**: Location-specific configurations, project assignment
- **Connections**: Forwards to `defect_selection_ui.py`

#### 6. **defect_selection_ui.py**
- **Purpose**: Central hub for defect management and workflow routing
- **Function**: Select defects, manage training data, route to different modules
- **Features**: 
  - Defect type selection for existing defects
  - **"Defective" button**: Routes to image classification workflow
  - **"New Defects" button**: Initiates new defect training workflow
  - Dynamic defect grid, Azure database integration
- **Connections**: Multiple paths based on user selection

#### 7. **Image Classification & Demo Modules**
- **image_classi_ui.py**: Image classification interface with train/test splitting
- **demo_feed_ui.py**: Demo and testing interface for model validation
- **Function**: Model testing, data validation, performance assessment
- **Features**: Real-time classification, accuracy metrics, data export

#### 8. **Training & Annotation Modules**
- **manual_annotation_ui.py**: Manual image annotation tools
- **auto_annotation.py**: Automated annotation and model training
- **Function**: 
  - Manual annotation for new defect types
  - Automated training pipeline with accurate and fast training options
  - Model optimization and validation
- **Features**: Annotation tools, training progress tracking, model export

#### 9. **deploy_ui.py** / **fabric_deploy_ui.py**
- **Purpose**: Main deployment interfaces for real-time detection
- **Function**: Live camera feed analysis, defect counting, reporting
- **Features**: Dual-dashboard design, object tracking, database synchronization
- **Connections**: Integrates with Azure services and logging systems

#### 10. **azure_database.py**
- **Purpose**: Database operations and cloud synchronization
- **Function**: 
  - Defect count storage and retrieval
  - User management and authentication data
  - Training data and model metadata storage
  - Organization and project data management
- **Features**: Batch updates, transaction management, error handling

#### 11. **launch_deploy.py**
- **Purpose**: Smart launcher with session persistence
- **Function**: System validation, automatic UI restoration
- **Features**: Crash recovery, connectivity checking, mock data generation

### Supporting Modules

- **utils/logging/**: Comprehensive logging framework
- **azure_storage.py**: Cloud storage operations for models and datasets
- **data_integration.py**: Data processing and integration utilities
- **metal_sheet_organization_ui.py**: Metal sheet specific organization management

## 🔄 Crash Recovery & Restore Feature

### How It Works

The system implements intelligent session persistence to handle crashes gracefully:

#### Session State Management
- **Location**: `session_state.json`
- **Function**: Stores last active UI, user information, timestamps
- **Update Frequency**: Real-time on UI transitions

#### Crash Recovery Process

1. **Detection**: `launch_deploy.py` checks for existing session data
2. **Validation**: System verifies connectivity and camera availability
3. **Restoration**: Automatically launches the last active UI interface
4. **Data Integrity**: Preserves user sessions, defect counts, and configurations

#### Code Implementation

```python
# In launch_deploy.py
def launch_last_used_ui():
    """Launch the last used UI based on session state"""
    session_data = load_session_state()
    
    if session_data and 'last_ui' in session_data:
        last_ui = session_data['last_ui']
        return launch_ui_by_name(last_ui)
    else:
        return launch_deployment_ui()  # Default fallback

# Session saving on UI close
def enhanced_close_event(event):
    save_session_state(ui_name, create_mock_user(), {
        "closed_at": datetime.now().isoformat()
    })
```

#### What Users Can Expect

- **Automatic Restoration**: Software reopens at the exact point of interruption
- **Data Preservation**: Defect counts, user sessions, and settings remain intact
- **Seamless Experience**: No manual intervention required for recovery
- **Fallback Safety**: Defaults to Deploy UI if session data is corrupted

## 📁 File Structure

```
sakar_vision_ui/
├── 📄 main.py                          # Application entry point
├── 📄 inspection_selection_ui.py       # Initial inspection type selection
├── 📄 login_ui.py                      # User authentication interface
├── 📄 defect_selection_ui.py           # Defect type selection UI
├── 📄 deploy_ui.py                     # Main deployment interface (metal sheet)
├── 📄 fabric_deploy_ui.py              # Fabric inspection interface
├── 📄 launch_deploy.py                 # Smart launcher with session persistence
├── 📄 azure_database.py                # Database operations and cloud sync
├── 📄 azure_storage.py                 # Cloud storage management
├── 📄 data_integration.py              # Data processing utilities
├── 📄 auto_annotation.py               # Automated annotation and training
├── 📄 manual_annotation_ui.py          # Manual annotation tools
├── 📄 image_classi_ui.py               # Image classification interface
├── 📄 camera_feed_cloud_ui.py          # Camera feed with cloud integration
├── 📄 demo_feed_ui.py                  # Demo/testing interface
├── 📄 utils.py                         # Utility functions and helpers
├── 📄 requirements.txt                 # Python dependencies
├── 📄 requirements_current.txt         # Current environment dependencies
├── 📄 LICENSE                          # License information
│
├── 📂 utils/                           # Utility modules
│   ├── 📂 logging/                     # Comprehensive logging system
│   │   ├── 📄 logger_config.py         # Logging configuration
│   │   ├── 📄 ui_event_logger.py       # UI event tracking
│   │   ├── 📄 azure_logger.py          # Azure operations logging
│   │   ├── 📄 performance_logger.py    # Performance monitoring
│   │   └── 📄 console_logger.py        # Console output management
│   └── 📄 __init__.py
│
├── 📂 logs/                            # Application logs
│   ├── 📄 sakar_vision_master_*.log    # Master log files
│   ├── 📄 azure.log                    # Azure operations log
│   ├── 📄 ui.events.log               # UI interaction logs
│   └── 📄 performance.log             # Performance metrics
│
├── 📂 Configuration Files              # Auto-generated configuration
│   ├── 📄 session_state.json          # Session persistence data
│   ├── 📄 defects_config.json         # Selected defect types
│   ├── 📄 organizations.json          # Organization data
│   ├── 📄 location_config.json        # Location settings
│   ├── 📄 inspection_settings.json    # Inspection parameters
│   └── 📄 users.json                  # User account data
│
├── 📂 AI Models                        # Machine learning models
│   ├── 📄 best.pt                      # Primary YOLO detection model
│   ├── 📄 sakar_ai.pt                 # Secondary detection model
│   ├── 📄 sakar_cls.pth               # Classification model
│   └── 📄 yolo11n.pt                  # YOLO11 nano model
│
└── 📂 Assets                           # UI assets and resources
    ├── 📄 sakar_logo.png              # Application logo
    ├── 📄 bg.png                       # Background images
    ├── 📄 fabric.jpeg                  # Material reference images
    └── 📄 metal.jpeg                   # Metal sheet reference
```

## 📖 Usage Instructions

### Step 1: Launch the Application

```bash
# Standard launch
python main.py

# Or use the smart launcher (recommended for crash recovery)
python launch_deploy.py
```

### Step 2: Inspection Type Selection

1. **Choose Inspection Mode**:
   - **Fabric Inspection**: For textile defect detection
   - **Metal Sheet Inspection**: For metal surface analysis

2. **System Validation**:
   - Internet connectivity check
   - Camera availability verification
   - Hardware compatibility validation

### Step 3: User Authentication

1. **Login Credentials**:
   - Enter username and password
   - System validates against user database
   - Session tracking begins

2. **Organization Selection**:
   - Choose your organization from the dropdown
   - Create new organization if needed
   - Configure organization settings

3. **Location & Project Setup**:
   - Select or create inspection location
   - Define project parameters
   - Configure location-specific settings

### Step 4: Defect Configuration & Workflow Selection

The **defect_selection_ui.py** is the central hub that offers multiple workflow paths:

#### **Option 1: Use Existing Defects (Direct Deployment)**
1. **Select Defect Types**:
   - Choose from existing defects in the grid
   - Configure detection thresholds
   - Set counting parameters

2. **Proceed to Deployment**:
   - Click "Proceed" to go directly to Deploy UI
   - Start live detection immediately

#### **Option 2: Image Classification Workflow**
1. **Click "Defective" Button**:
   - Routes to image classification interface
   - Access `image_classi_ui.py`

2. **Image Processing**:
   - Click "Images Split Train" to process datasets
   - Train/test data splitting
   - Model validation

3. **Demo & Testing**:
   - Access `demo_feed_ui.py` for model testing
   - Performance validation
   - Accuracy assessment

#### **Option 3: New Defect Training Workflow**
1. **Click "New Defects" Button**:
   - Initiates new defect training pipeline
   - Capture new defect images

2. **Manual Annotation**:
   - Use `manual_annotation_ui.py` for image labeling
   - Create bounding boxes and classifications
   - Quality control for annotations

3. **Automated Training**:
   - Access `auto_annotation.py` for model training
   - **Accurate Training**: High-precision, longer training time
   - **Fast Training**: Quick training for rapid prototyping
   - Progress tracking and model optimization

4. **Deploy Trained Model**:
   - Proceed to Deploy UI with newly trained model
   - Integration with existing workflow

### Step 5: Live Deployment

1. **Camera Setup**:
   - Select camera source (USB/integrated)
   - Adjust camera settings (resolution, FPS)
   - Preview camera feed

2. **Model Configuration**:
   - Choose AI model (automatic detection of latest trained models)
   - Set confidence thresholds
   - Configure tracking parameters

3. **Real-time Monitoring**:
   - Start live detection
   - Monitor defect counts in dual-dashboard
   - View real-time statistics and reports

### Step 6: Results and Reporting

1. **Live Statistics**:
   - Real-time defect counting
   - Performance metrics (FPS, inference time)
   - Classification accuracy

2. **Data Export**:
   - Azure database synchronization
   - Local report generation
   - Batch data processing

## ⚙️ Configuration

### Camera Settings

Edit `inspection_settings.json`:

```json
{
  "camera_index": 0,
  "resolution": [1280, 720],
  "fps": 30,
  "auto_exposure": true
}
```

### Detection Parameters

Edit `defects_config.json`:

```json
{
  "selected_defects": ["hole", "stain", "tear"],
  "confidence_threshold": 0.5,
  "tracking_enabled": true,
  "user": "admin"
}
```

### Azure Integration

Configure Azure services in environment variables or directly in configuration files.

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Camera Not Detected**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Solution: Update camera_index in inspection_settings.json
```

#### 2. **Model Loading Errors**
```bash
# Verify model files exist
ls -la *.pt *.pth

# Solution: Download required models or update model paths
```

#### 3. **Dependency Issues**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check specific packages
pip show opencv-python torch ultralytics
```

#### 4. **Azure Connection Problems**
- Verify internet connectivity
- Check Azure credentials in environment variables
- Review firewall settings

#### 5. **Performance Issues**
- Enable GPU acceleration (CUDA)
- Reduce camera resolution
- Lower detection confidence threshold
- Close unnecessary applications

### Log Analysis

Check log files for detailed error information:

```bash
# View latest log
tail -f logs/sakar_vision_master_$(date +%Y-%m-%d).log

# Check specific components
tail -f logs/azure.log        # Azure operations
tail -f logs/ui.events.log    # UI interactions
tail -f logs/performance.log  # Performance metrics
```

### System Requirements Check

```bash
# Run system validation
python -c "
import cv2, torch, ultralytics
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
"
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

## 🔧 Advanced Features

### Session Persistence Architecture

The crash recovery system uses a sophisticated state management approach:

- **Automatic State Saving**: Every UI transition updates `session_state.json`
- **Smart Recovery**: `launch_deploy.py` intelligently restores the last active interface
- **Data Integrity**: User sessions, defect counts, and configurations remain intact
- **Fallback Mechanisms**: Graceful degradation when session data is corrupted

### Real-time Performance Monitoring

- **FPS Tracking**: Monitor real-time processing speed
- **Inference Time**: Track AI model performance
- **Memory Usage**: Monitor system resource consumption
- **Database Latency**: Track Azure synchronization performance

### Azure Cloud Integration

- **Database Synchronization**: Real-time defect count storage
- **Model Management**: Cloud-based AI model storage and retrieval
- **Report Generation**: Automated report creation and distribution
- **User Management**: Centralized authentication and authorization

---

**For additional support or questions, please refer to the application logs or contact the development team.**
