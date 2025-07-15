from PyQt5.QtGui import QIcon
import os
from ultralytics import YOLO  # Added import for YOLO class


def set_window_icon(widget, icon_path="sakar.png"):
    if not os.path.exists(icon_path):
        print(f"Icon file not found: {icon_path}")
    else:
        widget.setWindowIcon(QIcon(icon_path))


class ClassManager:
    """
    Singleton class manager to ensure consistent class definitions across all UI components.
    This provides a central, always-up-to-date class list that all components reference.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassManager, cls).__new__(cls)
            cls._instance.classes = []
            cls._instance.initialized = False
            cls._instance.source = ""
            cls._instance.last_updated = 0
        return cls._instance

    def initialize(self):
        """Initialize with default classes if not already initialized."""
        if not self.initialized:
            default_classes = ['corrosion', 'hole', 'pitting', 'inclusion', 'crack', 'patch',
                               'rolling', 'spot', 'burr', 'crazing', 'dent', 'punch', 'casting', 'scratch', 'wrinkle']
            self.classes = self.normalize_class_names(default_classes)
            self.initialized = True
            self.source = "default"
            import time
            self.last_updated = time.time()
            self._save_to_file()

    def normalize_class_names(self, classes):
        """
        Normalize class names to ensure consistent capitalization.
        We'll standardize on first letter capitalized format.
        """
        if not classes:
            return []

        # Convert all class names to have first letter capitalized
        normalized = [name[0].upper() + name[1:].lower() if name else "" for name in classes]
        return normalized

    def update_classes(self, classes, source="unknown"):
        """Update the class list with new classes."""
        if classes and isinstance(classes, list):
            # Normalize class names for consistency
            normalized_classes = self.normalize_class_names(classes)

            # Only update if the class list actually changed (compare normalized)
            if set(normalized_classes) != set(self.classes):
                # Simply replace the classes without sorting
                self.classes = normalized_classes
                self.source = source
                import time
                self.last_updated = time.time()
                self._save_to_file()
                print(f"ClassManager: Updated classes from {source}: {self.classes}")

                # Update defects_config.json and YAML files
                try:
                    # Update defects_config.json
                    from data_integration import update_defects_config
                    update_defects_config(self.classes, self.classes)

                    # Update all dataset.yaml files
                    from data_integration import update_all_yaml_files
                    success, count = update_all_yaml_files(self.classes)
                    if success:
                        print(f"ClassManager: Updated {count} YAML file(s) with new classes")
                except ImportError:
                    print("ClassManager: data_integration module not available, skipping config updates")
                except Exception as e:
                    print(f"ClassManager: Error updating configurations: {e}")

                return True
        return False

    def get_classes(self):
        """Get the current class list."""
        if not self.initialized:
            self.initialize()
        return self.classes.copy()

    def _save_to_file(self):
        """Save classes to a persistent file."""
        try:
            import json
            import os

            data = {
                'classes': self.classes,
                'source': self.source,
                'timestamp': self.last_updated
            }

            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "shared_classes.json")

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"ClassManager: Saved classes to {file_path}")
        except Exception as e:
            print(f"ClassManager: Error saving classes to file: {e}")

    def load_from_file(self):
        """Load classes from persistent file if it exists."""
        try:
            import json
            import os
            import time

            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "shared_classes.json")

            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if 'classes' in data and data['classes']:
                    self.classes = data['classes']
                    self.source = data.get('source', "file")
                    self.last_updated = data.get('timestamp', time.time())
                    self.initialized = True
                    print(f"ClassManager: Loaded classes from file: {self.classes}")
                    return True
        except Exception as e:
            print(f"ClassManager: Error loading classes from file: {e}")

        return False

    def extract_from_model(self, model_path):
        """Extract classes from a model file."""
        try:
            # Load model temporarily
            model = YOLO(model_path)

            # Try to get class names
            if hasattr(model, 'names'):
                # Handle both dictionary and list formats
                if isinstance(model.names, dict):
                    # For dictionary format, preserve order by index
                    extracted_classes = []
                    class_dict = model.names
                    # Get the maximum index to make sure we don't miss any classes
                    max_index = max(int(key) for key in class_dict.keys()
                                    if isinstance(key, (int, str)) and str(key).isdigit())

                    # Create a list with the correct order based on indices
                    for i in range(max_index + 1):
                        if i in class_dict:
                            extracted_classes.append(class_dict[i])
                        elif str(i) in class_dict:
                            extracted_classes.append(class_dict[str(i)])
                else:
                    # For list format, just convert to list
                    extracted_classes = list(model.names)

                if extracted_classes:
                    # Normalize but don't sort
                    normalized_classes = self.normalize_class_names(extracted_classes)

                    # Only update if we actually got classes and they're different
                    if set(normalized_classes) != set(self.classes):
                        self.classes = normalized_classes
                        self.source = f"model:{model_path}"
                        import time
                        self.last_updated = time.time()
                        self._save_to_file()
                        print(
                            f"ClassManager: Extracted classes from model in original order: {self.classes}")
                        return True

                print(f"ClassManager: Could not extract classes from model {model_path}")
            else:
                print(f"ClassManager: Model has no 'names' attribute")

        except Exception as e:
            print(f"ClassManager: Error extracting classes from model: {e}")
            import traceback
            traceback.print_exc()

        return False

    def add_class(self, class_name, source="manual"):
        """Add a single new class if it doesn't exist."""
        if class_name:
            # Normalize the class name
            normalized_name = class_name[0].upper() + class_name[1:].lower() if class_name else ""

            if normalized_name not in self.classes:
                self.classes.append(normalized_name)
                # Remove this line: self.classes.sort()  # Keep alphabetically sorted
                self.source = source
                import time
                self.last_updated = time.time()
                self._save_to_file()
                print(
                    f"ClassManager: Added new class {normalized_name}, classes now: {self.classes}")
                return True
        return False


def extract_classes_from_model(model_path):
    """
    Extract and display class names with their IDs from a YOLO model.

    Args:
        model_path (str): Path to the YOLO model file (.pt)

    Returns:
        tuple: (success, class_dict) where:
            - success is a boolean indicating if extraction was successful
            - class_dict is a dictionary mapping class IDs to class names
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False, {}

    try:
        # Load the YOLO model
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)

        # Extract class names
        if hasattr(model, 'names'):
            names = model.names

            # Print header
            print("\n" + "=" * 50)
            print(f"CLASSES IN MODEL: {model_path}")
            print("=" * 50)

            class_dict = {}
            # Handle different formats (dict or list)
            if isinstance(names, dict):
                # Sort by class ID
                sorted_items = sorted(names.items(), key=lambda x: int(
                    x[0]) if str(x[0]).isdigit() else x[0])
                for class_id, class_name in sorted_items:
                    print(f"Class ID: {class_id:<5} | Class Name: {class_name}")
                    class_dict[class_id] = class_name
            else:
                # For list format, the index is the class ID
                for class_id, class_name in enumerate(names):
                    print(f"Class ID: {class_id:<5} | Class Name: {class_name}")
                    class_dict[class_id] = class_name

            print("=" * 50)
            print(f"Total Classes: {len(names)}")
            return True, class_dict
        else:
            print("Error: This model doesn't have class names attribute.")
            return False, {}

    except Exception as e:
        print(f"Error loading model: {e}")
        return False, {}
