#!/usr/bin/env python3
"""
Visual Data Export Module for SAKAR Vision AI
Creates visual representations of defective/non-defective classification data
"""

import json
import os
import sys
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

class VisualDataExporter:
    """Create visual representations of classification data"""
    
    def __init__(self, predictions_file="predictions_log.json"):
        self.predictions_file = predictions_file
        self.output_dir = "visual_exports"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_session_data(self, session_id=None):
        """Load data from predictions file"""
        try:
            if not os.path.exists(self.predictions_file):
                print(f"No predictions file found: {self.predictions_file}")
                return None
                
            with open(self.predictions_file, 'r') as f:
                data = json.load(f)
            
            if session_id:
                if session_id in data.get("sessions", {}):
                    return data["sessions"][session_id]["predictions"]
                else:
                    print(f"Session {session_id} not found")
                    return None
            else:
                # Get latest session
                sessions = data.get("sessions", {})
                if not sessions:
                    print("No sessions found")
                    return None
                    
                latest_session = max(sessions.items(), 
                                   key=lambda x: x[1]["session_info"]["start_time"])
                return latest_session[1]["predictions"]
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_visual_summary(self, predictions, output_filename=None):
        """Create a comprehensive visual summary"""
        if not predictions:
            print("No predictions to visualize")
            return None
            
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"visual_summary_{timestamp}.pdf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Extract data for visualization
        classes = [p["predicted_class"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        timestamps = [p["timestamp"] for p in predictions]
        
        # Create PDF with multiple pages
        with PdfPages(output_path) as pdf:
            # Page 1: Overall Summary
            self._create_summary_page(pdf, classes, confidences, timestamps)
            
            # Page 2: Timeline Analysis
            self._create_timeline_page(pdf, classes, timestamps)
            
            # Page 3: Confidence Analysis
            self._create_confidence_page(pdf, classes, confidences)
            
            # Page 4: Simple Visual Status
            self._create_simple_status_page(pdf, classes)
        
        print(f"Visual summary created: {output_path}")
        return output_path
    
    def _create_summary_page(self, pdf, classes, confidences, timestamps):
        """Create overall summary page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('SAKAR Vision AI - Classification Summary', fontsize=16, fontweight='bold')
        
        # 1. Pie chart of classifications
        class_counts = Counter(classes)
        colors = ['#28a745' if 'non_defective' in cls.lower() else '#dc3545' 
                 for cls in class_counts.keys()]
        
        ax1.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Classification Distribution')
        
        # 2. Bar chart with counts
        bars = ax2.bar(class_counts.keys(), class_counts.values(), color=colors)
        ax2.set_title('Classification Counts')
        ax2.set_ylabel('Count')
        for bar, count in zip(bars, class_counts.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 3. Confidence distribution
        ax3.hist(confidences, bins=20, alpha=0.7, color='#17a2b8', edgecolor='black')
        ax3.set_title('Confidence Score Distribution')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.2%}')
        ax3.legend()
        
        # 4. Summary statistics table
        ax4.axis('off')
        stats_data = [
            ['Total Predictions', str(len(classes))],
            ['Defective', str(sum(1 for c in classes if 'defective' in c.lower() and 'non' not in c.lower()))],
            ['Non-Defective', str(sum(1 for c in classes if 'non_defective' in c.lower()))],
            ['Avg Confidence', f"{np.mean(confidences):.2%}"],
            ['Min Confidence', f"{min(confidences):.2%}"],
            ['Max Confidence', f"{max(confidences):.2%}"]
        ]
        
        table = ax4.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_timeline_page(self, pdf, classes, timestamps):
        """Create timeline analysis page"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Timeline Analysis', fontsize=16, fontweight='bold')
        
        # Convert timestamps to datetime objects
        dt_timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        
        # 1. Timeline scatter plot
        y_values = [1 if 'defective' in cls.lower() and 'non' not in cls.lower() else 0 
                   for cls in classes]
        colors = ['#dc3545' if y == 1 else '#28a745' for y in y_values]
        
        ax1.scatter(dt_timestamps, y_values, c=colors, alpha=0.7, s=50)
        ax1.set_ylabel('Classification\n(0=Non-Defective, 1=Defective)')
        ax1.set_title('Classification Timeline')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # 2. Cumulative counts over time
        defective_count = 0
        non_defective_count = 0
        defective_cumulative = []
        non_defective_cumulative = []
        
        for cls in classes:
            if 'defective' in cls.lower() and 'non' not in cls.lower():
                defective_count += 1
            else:
                non_defective_count += 1
            defective_cumulative.append(defective_count)
            non_defective_cumulative.append(non_defective_count)
        
        ax2.plot(dt_timestamps, defective_cumulative, 'r-', label='Defective', linewidth=2)
        ax2.plot(dt_timestamps, non_defective_cumulative, 'g-', label='Non-Defective', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Count')
        ax2.set_title('Cumulative Classifications Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_page(self, pdf, classes, confidences):
        """Create confidence analysis page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Confidence Analysis', fontsize=16, fontweight='bold')
        
        # Separate by class
        defective_conf = [conf for cls, conf in zip(classes, confidences) 
                         if 'defective' in cls.lower() and 'non' not in cls.lower()]
        non_defective_conf = [conf for cls, conf in zip(classes, confidences) 
                             if 'non_defective' in cls.lower()]
        
        # 1. Box plot comparison
        data_to_plot = []
        labels = []
        if defective_conf:
            data_to_plot.append(defective_conf)
            labels.append('Defective')
        if non_defective_conf:
            data_to_plot.append(non_defective_conf)
            labels.append('Non-Defective')
        
        if data_to_plot:
            ax1.boxplot(data_to_plot, labels=labels)
            ax1.set_title('Confidence by Classification')
            ax1.set_ylabel('Confidence')
        
        # 2. Confidence vs sequence
        sequence = list(range(1, len(confidences) + 1))
        colors = ['#dc3545' if 'defective' in cls.lower() and 'non' not in cls.lower() 
                 else '#28a745' for cls in classes]
        ax2.scatter(sequence, confidences, c=colors, alpha=0.7)
        ax2.set_title('Confidence Over Sequence')
        ax2.set_xlabel('Prediction Sequence')
        ax2.set_ylabel('Confidence')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence ranges
        high_conf = sum(1 for c in confidences if c >= 0.8)
        med_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        conf_ranges = ['High (≥80%)', 'Medium (60-80%)', 'Low (<60%)']
        conf_counts = [high_conf, med_conf, low_conf]
        conf_colors = ['#28a745', '#ffc107', '#dc3545']
        
        bars = ax3.bar(conf_ranges, conf_counts, color=conf_colors)
        ax3.set_title('Confidence Range Distribution')
        ax3.set_ylabel('Count')
        for bar, count in zip(bars, conf_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 4. Statistics table
        ax4.axis('off')
        if defective_conf and non_defective_conf:
            stats_data = [
                ['Defective Avg Conf', f"{np.mean(defective_conf):.2%}"],
                ['Non-Def Avg Conf', f"{np.mean(non_defective_conf):.2%}"],
                ['Overall Avg Conf', f"{np.mean(confidences):.2%}"],
                ['Std Deviation', f"{np.std(confidences):.2%}"],
                ['High Confidence', f"{high_conf}/{len(confidences)}"],
                ['Low Confidence', f"{low_conf}/{len(confidences)}"]
            ]
        else:
            stats_data = [
                ['Overall Avg Conf', f"{np.mean(confidences):.2%}"],
                ['Std Deviation', f"{np.std(confidences):.2%}"],
                ['Min Confidence', f"{min(confidences):.2%}"],
                ['Max Confidence', f"{max(confidences):.2%}"],
                ['High Confidence', f"{high_conf}/{len(confidences)}"],
                ['Low Confidence', f"{low_conf}/{len(confidences)}"]
            ]
        
        table = ax4.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Confidence Statistics')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_simple_status_page(self, pdf, classes):
        """Create a simple visual status representation"""
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        fig.suptitle('Simple Visual Status', fontsize=16, fontweight='bold')
        
        # Create a simple visual representation like: 1=D, 2=ND, 3=ND, 4=D, 5=D
        # where D=Defective, ND=Non-Defective
        
        # Create visual blocks
        n_items = len(classes)
        cols = min(10, n_items)  # Max 10 columns
        rows = (n_items + cols - 1) // cols  # Calculate needed rows
        
        # Create grid
        for i, cls in enumerate(classes):
            row = i // cols
            col = i % cols
            
            # Determine color
            if 'defective' in cls.lower() and 'non' not in cls.lower():
                color = '#dc3545'  # Red for defective
                label = 'D'
            else:
                color = '#28a745'  # Green for non-defective
                label = 'ND'
            
            # Create rectangle
            rect = patches.Rectangle((col, rows - row - 1), 0.8, 0.8, 
                                   facecolor=color, alpha=0.8, edgecolor='black')
            ax.add_patch(rect)
            
            # Add text
            ax.text(col + 0.4, rows - row - 0.6, f"{i+1}\n{label}", 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Set up the plot
        ax.set_xlim(-0.5, cols)
        ax.set_ylim(-0.5, rows)
        ax.set_aspect('equal')
        ax.set_title(f'Visual Sequence: {n_items} Items\n(Numbers show sequence, D=Defective, ND=Non-Defective)', 
                     fontsize=14, pad=20)
        
        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        defective_patch = patches.Patch(color='#dc3545', label='Defective')
        non_defective_patch = patches.Patch(color='#28a745', label='Non-Defective')
        ax.legend(handles=[defective_patch, non_defective_patch], 
                 loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def create_simple_text_export(self, predictions, output_filename=None):
        """Create a simple text-based export for systems without matplotlib"""
        if not predictions:
            print("No predictions to export")
            return None
            
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"simple_export_{timestamp}.txt"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Extract data
        classes = [p["predicted_class"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        
        with open(output_path, 'w') as f:
            f.write("SAKAR VISION AI - CLASSIFICATION EXPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            defective_count = sum(1 for c in classes if 'defective' in c.lower() and 'non' not in c.lower())
            non_defective_count = len(classes) - defective_count
            
            f.write("SUMMARY:\n")
            f.write(f"Total Predictions: {len(classes)}\n")
            f.write(f"Defective: {defective_count}\n")
            f.write(f"Non-Defective: {non_defective_count}\n")
            f.write(f"Average Confidence: {np.mean(confidences):.2%}\n\n")
            
            # Visual sequence
            f.write("VISUAL SEQUENCE:\n")
            f.write("(D=Defective, N=Non-Defective)\n\n")
            
            for i, cls in enumerate(classes, 1):
                if 'defective' in cls.lower() and 'non' not in cls.lower():
                    symbol = 'D'
                else:
                    symbol = 'N'
                f.write(f"{i}={symbol}, ")
                
                if i % 10 == 0:  # New line every 10 items
                    f.write("\n")
            
            f.write("\n\n")
            
            # Detailed list
            f.write("DETAILED LIST:\n")
            f.write("-" * 40 + "\n")
            for i, (cls, conf) in enumerate(zip(classes, confidences), 1):
                status = "DEFECTIVE" if 'defective' in cls.lower() and 'non' not in cls.lower() else "NON-DEFECTIVE"
                f.write(f"{i:3d}. {status:15s} ({conf:.1%})\n")
        
        print(f"Simple text export created: {output_path}")
        return output_path


def main():
    """Example usage"""
    exporter = VisualDataExporter()
    
    # Example data - replace with your actual data
    example_predictions = [
        {"predicted_class": "defective", "confidence": 0.95, "timestamp": "2025-07-10T10:00:00"},
        {"predicted_class": "non_defective", "confidence": 0.88, "timestamp": "2025-07-10T10:01:00"},
        {"predicted_class": "non_defective", "confidence": 0.92, "timestamp": "2025-07-10T10:02:00"},
        {"predicted_class": "defective", "confidence": 0.85, "timestamp": "2025-07-10T10:03:00"},
        {"predicted_class": "defective", "confidence": 0.90, "timestamp": "2025-07-10T10:04:00"},
    ]
    
    # Create visual summary
    pdf_path = exporter.create_visual_summary(example_predictions)
    
    # Create simple text export
    text_path = exporter.create_simple_text_export(example_predictions)
    
    print(f"Exports created in: {exporter.output_dir}")


if __name__ == "__main__":
    main()