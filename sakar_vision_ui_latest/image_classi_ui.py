#!/usr/bin/env python3
"""
SAKAR VISION AI - Complete Image Classification UI with Visual Reporting

OVERVIEW:
This module implements a sophisticated real-time image classification interface for the Sakar Vision AI platform, 
serving as an advanced quality control and validation system that combines live camera feeds with deep learning-based 
classification capabilities. It provides a professional dual-dashboard architecture for monitoring and analyzing 
classification results in real-time, with comprehensive visual reporting including charts, graphs, and analytics.

ENHANCED FEATURES:
- Automatic loading of best_model.pth from current directory
- Comprehensive prediction storage system with JSON export
- Intelligent class name detection from model checkpoints
- Motion detection for stable frame prediction
- Dual dashboard interface for defective/non-defective monitoring
- Professional visual reporting with charts and graphs
- Quality metrics dashboards and analytics
- Timeline analysis and statistical insights
- Performance monitoring and control charts
- Interactive HTML reports
- Session-based prediction tracking
"""

import io
import os
import sys
import time
import json
import glob
import uuid
import base64
import statistics
import subprocess
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog,
                             QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QProgressDialog, QPushButton, QStackedWidget,
                             QTextEdit, QVBoxLayout, QWidget, QFrame, QGroupBox,
                             QSizePolicy, QScrollArea, QSlider, QSpinBox, QLineEdit,
                             QInputDialog)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms

# Visual reporting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    VISUAL_REPORTING_AVAILABLE = True
    
    # Set matplotlib and seaborn styling
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    print("✅ Visual reporting with charts and graphs available")
except ImportError as e:
    VISUAL_REPORTING_AVAILABLE = False
    print(f"⚠️ Visual reporting not available: {e}")
    print("📦 Install with: pip install matplotlib seaborn")

# Environment setup
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt6/plugins'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration paths
REPORTS_STORAGE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "classification_reports.json")
SESSION_STATE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "session_state.json")
PREDICTIONS_STORAGE_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "predictions_log.json")


class DefectClassifier(nn.Module):
    """
    Enhanced DefectClassifier class using MobileNetV2 architecture
    """
    def __init__(self, num_classes=2):
        super(DefectClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Use MobileNetV2 architecture
        self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class VisualReportGenerator:
    """Professional visual report generator with charts and analytics"""
    
    def __init__(self, prediction_storage):
        self.prediction_storage = prediction_storage
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'normal': '#28a745',
            'defective': '#dc3545',
            'neutral': '#6c757d'
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 5,    # <5% defect rate
            'good': 15,        # 5-15% defect rate
            'acceptable': 30,  # 15-30% defect rate
            'poor': 100        # >30% defect rate
        }

    def generate_visual_report(self, export_path=None, report_format='png'):
        """Generate comprehensive visual report with multiple charts"""
        try:
            # Flush any cached predictions first
            self.prediction_storage.flush_cache()
            
            # Load session data
            with open(self.prediction_storage.storage_path, 'r') as f:
                data = json.load(f)
            
            session_id = self.prediction_storage.session_id
            if session_id not in data["sessions"]:
                return None, "Session data not found"
            
            session_data = data["sessions"][session_id]
            predictions = session_data["predictions"]
            
            if not predictions:
                return None, "No predictions available for visual report generation"
            
            # Determine export path
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"SAKAR_AI_Visual_Report_{session_id}_{timestamp}"
            
            # Create comprehensive visual report
            generated_files = []
            
            # 1. Executive Dashboard (single comprehensive view)
            dashboard_file = self.create_executive_dashboard(predictions, f"{export_path}_dashboard.{report_format}")
            if dashboard_file:
                generated_files.append(dashboard_file)
            
            # 2. Quality Metrics Dashboard
            quality_file = self.create_quality_metrics_dashboard(predictions, f"{export_path}_quality.{report_format}")
            if quality_file:
                generated_files.append(quality_file)
            
            # 3. Timeline Analysis Dashboard
            timeline_file = self.create_timeline_dashboard(predictions, f"{export_path}_timeline.{report_format}")
            if timeline_file:
                generated_files.append(timeline_file)
            
            # 4. Statistical Analysis Dashboard
            stats_file = self.create_statistical_dashboard(predictions, f"{export_path}_statistics.{report_format}")
            if stats_file:
                generated_files.append(stats_file)
            
            # 5. Performance Trends Dashboard
            performance_file = self.create_performance_dashboard(predictions, f"{export_path}_performance.{report_format}")
            if performance_file:
                generated_files.append(performance_file)
            
            # 6. Generate summary HTML report
            html_file = self.create_html_summary(predictions, session_data, generated_files, f"{export_path}_summary.html")
            if html_file:
                generated_files.append(html_file)
            
            return generated_files, None
            
        except Exception as e:
            return None, f"Error generating visual report: {e}"

    def create_executive_dashboard(self, predictions, filepath):
        """Create comprehensive executive dashboard with key metrics"""
        try:
            # Set up the figure with professional layout
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('SAKAR VISION AI - EXECUTIVE DASHBOARD', fontsize=20, fontweight='bold', y=0.98)
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3, top=0.92, bottom=0.08, left=0.06, right=0.94)
            
            # Prepare data
            classes = [p["predicted_class"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            timestamps = [datetime.fromisoformat(p["timestamp"]) for p in predictions]
            
            normal_keywords = ['normal', 'good', 'ok', 'non_defective']
            normal_count = sum(1 for c in classes if c.lower() in normal_keywords)
            defective_count = len(predictions) - normal_count
            defect_rate = (defective_count / len(predictions)) * 100
            avg_confidence = statistics.mean(confidences)
            
            # 1. Key Metrics Cards (Top Row)
            metrics = [
                ("Total Predictions", len(predictions), self.colors['primary']),
                ("Normal Items", normal_count, self.colors['normal']),
                ("Defective Items", defective_count, self.colors['defective']),
                ("Avg Confidence", f"{avg_confidence:.1%}", self.colors['secondary'])
            ]
            
            for i, (title, value, color) in enumerate(metrics):
                ax = fig.add_subplot(gs[0, i])
                ax.text(0.5, 0.7, str(value), ha='center', va='center', fontsize=24, fontweight='bold', color=color)
                ax.text(0.5, 0.3, title, ha='center', va='center', fontsize=12, color='#333333')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Add border
                rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            
            # 2. Classification Distribution (Pie Chart)
            ax_pie = fig.add_subplot(gs[1, :2])
            class_counts = Counter(classes)
            colors_pie = [self.colors['normal'] if cls.lower() in normal_keywords else self.colors['defective'] 
                         for cls in class_counts.keys()]
            
            wedges, texts, autotexts = ax_pie.pie(class_counts.values(), labels=class_counts.keys(), 
                                                 autopct='%1.1f%%', colors=colors_pie, startangle=90)
            ax_pie.set_title('Classification Distribution', fontsize=14, fontweight='bold', pad=20)
            
            # 3. Confidence Distribution (Histogram)
            ax_hist = fig.add_subplot(gs[1, 2:])
            ax_hist.hist(confidences, bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='white')
            ax_hist.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
            ax_hist.set_xlabel('Confidence Level')
            ax_hist.set_ylabel('Number of Predictions')
            ax_hist.grid(True, alpha=0.3)
            
            # Add confidence statistics
            ax_hist.axvline(avg_confidence, color=self.colors['warning'], linestyle='--', linewidth=2, label=f'Mean: {avg_confidence:.2f}')
            ax_hist.legend()
            
            # 4. Quality Status Gauge
            ax_gauge = fig.add_subplot(gs[2, 0])
            self.create_quality_gauge(ax_gauge, defect_rate)
            
            # 5. Timeline Trend
            ax_timeline = fig.add_subplot(gs[2, 1:])
            
            # Group predictions by hour for trend analysis
            hourly_data = defaultdict(list)
            for pred in predictions:
                hour = datetime.fromisoformat(pred["timestamp"]).hour
                hourly_data[hour].append(pred["confidence"])
            
            hours = sorted(hourly_data.keys())
            avg_confidences = [statistics.mean(hourly_data[hour]) for hour in hours]
            prediction_counts = [len(hourly_data[hour]) for hour in hours]
            
            # Plot confidence trend
            ax_timeline.plot(hours, avg_confidences, marker='o', linewidth=2, color=self.colors['primary'], label='Avg Confidence')
            ax_timeline.set_title('Hourly Performance Trend', fontsize=14, fontweight='bold')
            ax_timeline.set_xlabel('Hour of Day')
            ax_timeline.set_ylabel('Average Confidence', color=self.colors['primary'])
            ax_timeline.tick_params(axis='y', labelcolor=self.colors['primary'])
            ax_timeline.grid(True, alpha=0.3)
            
            # Add prediction count on secondary y-axis
            ax2 = ax_timeline.twinx()
            ax2.bar(hours, prediction_counts, alpha=0.3, color=self.colors['secondary'], label='Prediction Count')
            ax2.set_ylabel('Prediction Count', color=self.colors['secondary'])
            ax2.tick_params(axis='y', labelcolor=self.colors['secondary'])
            
            # Combined legend
            lines1, labels1 = ax_timeline.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_timeline.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 6. Defect Analysis (Bottom Left)
            ax_defects = fig.add_subplot(gs[3, :2])
            defect_classes = [cls for cls in class_counts.keys() if cls.lower() not in normal_keywords]
            if defect_classes:
                defect_counts = [class_counts[cls] for cls in defect_classes]
                bars = ax_defects.bar(defect_classes, defect_counts, color=self.colors['defective'], alpha=0.7)
                ax_defects.set_title('Defect Types Analysis', fontsize=14, fontweight='bold')
                ax_defects.set_xlabel('Defect Type')
                ax_defects.set_ylabel('Count')
                ax_defects.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax_defects.annotate(f'{int(height)}',
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3),  # 3 points vertical offset
                                      textcoords="offset points",
                                      ha='center', va='bottom')
            else:
                ax_defects.text(0.5, 0.5, 'No Defects Detected\n✓ Excellent Quality!', 
                              ha='center', va='center', fontsize=16, color=self.colors['normal'],
                              transform=ax_defects.transAxes)
                ax_defects.set_title('Defect Types Analysis', fontsize=14, fontweight='bold')
            ax_defects.grid(True, alpha=0.3)
            
            # 7. Quality Score Card (Bottom Right)
            ax_score = fig.add_subplot(gs[3, 2:])
            quality_score = self.calculate_quality_score(defect_rate, avg_confidence)
            self.create_quality_scorecard(ax_score, quality_score, defect_rate, avg_confidence)
            
            # Add footer with session info
            session_start = datetime.fromisoformat(predictions[0]["timestamp"])
            session_end = datetime.fromisoformat(predictions[-1]["timestamp"])
            duration = session_end - session_start
            
            footer_text = (f"Session Duration: {duration} | "
                          f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                          f"Session ID: {self.prediction_storage.session_id}")
            fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', fontsize=8, color='#666666')
            
            # Save the figure
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating executive dashboard: {e}")
            return None

    def create_quality_gauge(self, ax, defect_rate):
        """Create a quality gauge visualization"""
        # Define quality levels and colors
        levels = [
            (0, 5, self.colors['normal'], 'Excellent'),
            (5, 15, '#FFD700', 'Good'),  # Gold
            (15, 30, '#FFA500', 'Acceptable'),  # Orange
            (30, 100, self.colors['defective'], 'Poor')
        ]
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        for i, (start, end, color, label) in enumerate(levels):
            start_angle = np.pi * (1 - start/100)
            end_angle = np.pi * (1 - end/100)
            theta_range = np.linspace(end_angle, start_angle, 50)
            
            x = np.cos(theta_range)
            y = np.sin(theta_range)
            ax.fill_between(x, y, y*0.7, color=color, alpha=0.7, label=label)
        
        # Add needle for current defect rate
        needle_angle = np.pi * (1 - min(defect_rate, 100)/100)
        needle_x = [0, 0.8 * np.cos(needle_angle)]
        needle_y = [0, 0.8 * np.sin(needle_angle)]
        ax.plot(needle_x, needle_y, 'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=8)
        
        # Styling
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Quality Status\n{defect_rate:.1f}% Defect Rate', fontsize=12, fontweight='bold')
        
        # Add legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=8)

    def create_quality_scorecard(self, ax, quality_score, defect_rate, avg_confidence):
        """Create quality scorecard with key metrics"""
        ax.axis('off')
        
        # Main quality score
        score_color = self.colors['normal'] if quality_score > 80 else self.colors['warning'] if quality_score > 60 else self.colors['defective']
        ax.text(0.5, 0.8, f"{quality_score:.0f}", ha='center', va='center', fontsize=36, 
                fontweight='bold', color=score_color, transform=ax.transAxes)
        ax.text(0.5, 0.65, "QUALITY SCORE", ha='center', va='center', fontsize=12, 
                fontweight='bold', transform=ax.transAxes)
        
        # Performance indicators
        indicators = [
            ("Defect Rate", f"{defect_rate:.1f}%", defect_rate < 5),
            ("Avg Confidence", f"{avg_confidence:.1%}", avg_confidence > 0.8),
            ("Performance", "Excellent" if quality_score > 80 else "Good" if quality_score > 60 else "Needs Attention", quality_score > 60)
        ]
        
        y_positions = [0.45, 0.35, 0.25]
        for i, (label, value, is_good) in enumerate(indicators):
            color = self.colors['normal'] if is_good else self.colors['defective']
            ax.text(0.1, y_positions[i], label + ":", ha='left', va='center', fontsize=10, 
                   transform=ax.transAxes)
            ax.text(0.9, y_positions[i], value, ha='right', va='center', fontsize=10, 
                   fontweight='bold', color=color, transform=ax.transAxes)

    def calculate_quality_score(self, defect_rate, avg_confidence):
        """Calculate overall quality score"""
        # Defect rate component (0-50 points, lower is better)
        defect_score = max(0, 50 - defect_rate * 1.5)
        
        # Confidence component (0-50 points, higher is better)
        confidence_score = avg_confidence * 50
        
        return min(100, defect_score + confidence_score)

    def create_quality_metrics_dashboard(self, predictions, filepath):
        """Create detailed quality metrics dashboard"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Quality Metrics Dashboard', fontsize=16, fontweight='bold')
            
            # Prepare data
            confidences = [p["confidence"] for p in predictions]
            classes = [p["predicted_class"] for p in predictions]
            
            # 1. Confidence Distribution with Statistics
            ax1.hist(confidences, bins=30, color=self.colors['primary'], alpha=0.7, edgecolor='white')
            ax1.axvline(statistics.mean(confidences), color=self.colors['warning'], linestyle='--', 
                       linewidth=2, label=f'Mean: {statistics.mean(confidences):.3f}')
            ax1.axvline(statistics.median(confidences), color=self.colors['success'], linestyle='--', 
                       linewidth=2, label=f'Median: {statistics.median(confidences):.3f}')
            ax1.set_title('Confidence Distribution Analysis')
            ax1.set_xlabel('Confidence Level')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Quality Tiers
            high_conf = sum(1 for c in confidences if c >= 0.9)
            medium_conf = sum(1 for c in confidences if 0.7 <= c < 0.9)
            low_conf = sum(1 for c in confidences if c < 0.7)
            
            tiers = ['High (≥90%)', 'Medium (70-89%)', 'Low (<70%)']
            counts = [high_conf, medium_conf, low_conf]
            colors = [self.colors['normal'], self.colors['warning'], self.colors['defective']]
            
            bars = ax2.bar(tiers, counts, color=colors, alpha=0.7)
            ax2.set_title('Confidence Quality Tiers')
            ax2.set_ylabel('Number of Predictions')
            
            # Add percentage labels
            total = len(predictions)
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.annotate(f'{count}\n({count/total*100:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            ax2.grid(True, alpha=0.3)
            
            # 3. Class-wise Confidence Analysis
            class_confidence = defaultdict(list)
            for pred in predictions:
                class_confidence[pred["predicted_class"]].append(pred["confidence"])
            
            class_names = list(class_confidence.keys())
            class_means = [statistics.mean(class_confidence[cls]) for cls in class_names]
            class_stds = [statistics.stdev(class_confidence[cls]) if len(class_confidence[cls]) > 1 else 0 
                         for cls in class_names]
            
            x_pos = np.arange(len(class_names))
            bars = ax3.bar(x_pos, class_means, yerr=class_stds, capsize=5, 
                          color=self.colors['secondary'], alpha=0.7)
            ax3.set_title('Class-wise Confidence Analysis')
            ax3.set_xlabel('Classes')
            ax3.set_ylabel('Average Confidence')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # 4. Performance Over Time
            if len(predictions) > 10:
                timestamps = [datetime.fromisoformat(p["timestamp"]) for p in predictions]
                
                # Group by time intervals
                time_intervals = []
                confidence_intervals = []
                interval_size = max(1, len(predictions) // 20)
                
                for i in range(0, len(predictions), interval_size):
                    interval_predictions = predictions[i:i+interval_size]
                    if interval_predictions:
                        avg_time = timestamps[i + len(interval_predictions)//2]
                        avg_conf = statistics.mean([p["confidence"] for p in interval_predictions])
                        time_intervals.append(avg_time)
                        confidence_intervals.append(avg_conf)
                
                ax4.plot(time_intervals, confidence_intervals, marker='o', linewidth=2, 
                        color=self.colors['primary'])
                ax4.set_title('Confidence Trend Over Time')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Average Confidence')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient Data\nfor Trend Analysis\n(Need >10 predictions)', 
                        ha='center', va='center', fontsize=12, transform=ax4.transAxes)
                ax4.set_title('Confidence Trend Over Time')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating quality metrics dashboard: {e}")
            return None

    def create_timeline_dashboard(self, predictions, filepath):
        """Create timeline analysis dashboard"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Timeline Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # Prepare time data
            timestamps = [datetime.fromisoformat(p["timestamp"]) for p in predictions]
            classes = [p["predicted_class"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            
            # 1. Hourly Distribution
            hours = [t.hour for t in timestamps]
            hourly_counts = Counter(hours)
            
            all_hours = list(range(24))
            counts = [hourly_counts.get(h, 0) for h in all_hours]
            
            ax1.bar(all_hours, counts, color=self.colors['primary'], alpha=0.7)
            ax1.set_title('Hourly Prediction Distribution')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Number of Predictions')
            ax1.set_xticks(range(0, 24, 2))
            ax1.grid(True, alpha=0.3)
            
            # 2. Prediction Timeline
            normal_keywords = ['normal', 'good', 'ok', 'non_defective']
            colors_timeline = [self.colors['normal'] if cls.lower() in normal_keywords 
                             else self.colors['defective'] for cls in classes]
            
            ax2.scatter(timestamps, confidences, c=colors_timeline, alpha=0.6, s=30)
            ax2.set_title('Prediction Timeline')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Confidence Level')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add legend
            normal_patch = plt.scatter([], [], c=self.colors['normal'], label='Normal')
            defect_patch = plt.scatter([], [], c=self.colors['defective'], label='Defective')
            ax2.legend(handles=[normal_patch, defect_patch])
            
            # 3. Prediction Frequency Analysis
            if len(timestamps) > 1:
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                            for i in range(len(timestamps)-1)]
                
                ax3.hist(intervals, bins=20, color=self.colors['secondary'], alpha=0.7)
                ax3.set_title('Prediction Interval Distribution')
                ax3.set_xlabel('Interval (seconds)')
                ax3.set_ylabel('Frequency')
                ax3.axvline(statistics.mean(intervals), color=self.colors['warning'], 
                           linestyle='--', linewidth=2, label=f'Mean: {statistics.mean(intervals):.1f}s')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Single Prediction\nNo Interval Data', ha='center', va='center', 
                        fontsize=12, transform=ax3.transAxes)
                ax3.set_title('Prediction Interval Distribution')
            
            # 4. Session Summary
            ax4.axis('off')
            
            session_start = min(timestamps)
            session_end = max(timestamps)
            duration = session_end - session_start
            total_predictions = len(predictions)
            avg_confidence = statistics.mean(confidences)
            
            normal_count = sum(1 for c in classes if c.lower() in normal_keywords)
            defect_count = total_predictions - normal_count
            defect_rate = (defect_count / total_predictions) * 100
            
            session_info = f"""
SESSION SUMMARY

Duration: {duration}
Total Predictions: {total_predictions:,}
Average Confidence: {avg_confidence:.1%}

Quality Metrics:
• Normal Items: {normal_count:,} ({(normal_count/total_predictions)*100:.1f}%)
• Defective Items: {defect_count:,} ({defect_rate:.1f}%)

Performance:
• Prediction Rate: {total_predictions / max(duration.total_seconds() / 60, 1):.1f}/min
• Quality Status: {"Excellent" if defect_rate < 5 else "Good" if defect_rate < 15 else "Needs Attention"}
            """
            
            ax4.text(0.1, 0.9, session_info, ha='left', va='top', fontsize=11, 
                    fontfamily='monospace', transform=ax4.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['primary'], alpha=0.1))
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating timeline dashboard: {e}")
            return None

    def create_statistical_dashboard(self, predictions, filepath):
        """Create statistical analysis dashboard"""
        try:
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle('Statistical Analysis Dashboard', fontsize=16, fontweight='bold')
            
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            confidences = [p["confidence"] for p in predictions]
            classes = [p["predicted_class"] for p in predictions]
            
            # 1. Box Plot of Confidence by Class
            ax1 = fig.add_subplot(gs[0, :2])
            class_confidence = defaultdict(list)
            for pred in predictions:
                class_confidence[pred["predicted_class"]].append(pred["confidence"])
            
            box_data = [class_confidence[cls] for cls in class_confidence.keys()]
            box_labels = list(class_confidence.keys())
            
            bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(self.colors['primary'])
                patch.set_alpha(0.7)
            
            ax1.set_title('Confidence Distribution by Class')
            ax1.set_ylabel('Confidence Level')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 2. Statistical Summary
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis('off')
            
            p25 = np.percentile(confidences, 25)
            p50 = np.percentile(confidences, 50)
            p75 = np.percentile(confidences, 75)
            p95 = np.percentile(confidences, 95)
            
            stats_text = f"""
STATISTICAL SUMMARY

Confidence Statistics:
• Mean: {statistics.mean(confidences):.3f}
• Median: {statistics.median(confidences):.3f}
• Std Dev: {statistics.stdev(confidences) if len(confidences) > 1 else 0:.3f}

Percentiles:
• 25th: {p25:.3f}
• 50th: {p50:.3f}
• 75th: {p75:.3f}
• 95th: {p95:.3f}

Range:
• Min: {min(confidences):.3f}
• Max: {max(confidences):.3f}
            """
            
            ax2.text(0.1, 0.9, stats_text, ha='left', va='top', fontsize=10, 
                    fontfamily='monospace', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['neutral'], alpha=0.1))
            
            # 3. Correlation Analysis
            ax3 = fig.add_subplot(gs[1, :])
            
            timestamps = [datetime.fromisoformat(p["timestamp"]) for p in predictions]
            time_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]
            
            ax3.scatter(time_numeric, confidences, alpha=0.6, color=self.colors['primary'])
            
            if len(time_numeric) > 2:
                z = np.polyfit(time_numeric, confidences, 1)
                p = np.poly1d(z)
                ax3.plot(time_numeric, p(time_numeric), "r--", alpha=0.8, 
                        label=f'Trend (slope: {z[0]:.2e})')
                ax3.legend()
            
            ax3.set_title('Confidence vs Time Correlation Analysis')
            ax3.set_xlabel('Time from Start (seconds)')
            ax3.set_ylabel('Confidence Level')
            ax3.grid(True, alpha=0.3)
            
            # 4. Distribution Comparison
            ax4 = fig.add_subplot(gs[2, 0])
            
            normal_keywords = ['normal', 'good', 'ok', 'non_defective']
            normal_confidences = [p["confidence"] for p in predictions 
                                if p["predicted_class"].lower() in normal_keywords]
            defect_confidences = [p["confidence"] for p in predictions 
                                if p["predicted_class"].lower() not in normal_keywords]
            
            if normal_confidences and defect_confidences:
                ax4.hist(normal_confidences, bins=15, alpha=0.7, color=self.colors['normal'], 
                        label=f'Normal (n={len(normal_confidences)})')
                ax4.hist(defect_confidences, bins=15, alpha=0.7, color=self.colors['defective'], 
                        label=f'Defective (n={len(defect_confidences)})')
                ax4.set_title('Normal vs Defective\nConfidence Comparison')
                ax4.set_xlabel('Confidence Level')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient Data\nfor Comparison', ha='center', va='center', 
                        fontsize=12, transform=ax4.transAxes)
                ax4.set_title('Normal vs Defective\nConfidence Comparison')
            
            # 5. Quality Score Evolution
            ax5 = fig.add_subplot(gs[2, 1:])
            
            window_size = max(5, len(predictions) // 10)
            rolling_scores = []
            rolling_times = []
            
            for i in range(window_size, len(predictions) + 1):
                window_preds = predictions[i-window_size:i]
                window_classes = [p["predicted_class"] for p in window_preds]
                window_confidences = [p["confidence"] for p in window_preds]
                
                normal_count = sum(1 for c in window_classes if c.lower() in normal_keywords)
                defect_rate = ((len(window_preds) - normal_count) / len(window_preds)) * 100
                avg_conf = statistics.mean(window_confidences)
                
                quality_score = self.calculate_quality_score(defect_rate, avg_conf)
                rolling_scores.append(quality_score)
                rolling_times.append(timestamps[i-1])
            
            if rolling_scores:
                ax5.plot(rolling_times, rolling_scores, marker='o', linewidth=2, 
                        color=self.colors['success'])
                ax5.axhline(y=80, color=self.colors['normal'], linestyle='--', alpha=0.7, label='Excellent (80+)')
                ax5.axhline(y=60, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Good (60+)')
                ax5.set_title(f'Quality Score Evolution (Window: {window_size})')
                ax5.set_xlabel('Time')
                ax5.set_ylabel('Quality Score')
                ax5.tick_params(axis='x', rotation=45)
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Insufficient Data\nfor Evolution Analysis', ha='center', va='center', 
                        fontsize=12, transform=ax5.transAxes)
                ax5.set_title('Quality Score Evolution')
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating statistical dashboard: {e}")
            return None

    def create_performance_dashboard(self, predictions, filepath):
        """Create performance analysis dashboard"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='bold')
            
            confidences = [p["confidence"] for p in predictions]
            classes = [p["predicted_class"] for p in predictions]
            timestamps = [datetime.fromisoformat(p["timestamp"]) for p in predictions]
            
            # 1. Performance Heatmap
            hourly_performance = defaultdict(lambda: defaultdict(list))
            for pred in predictions:
                hour = datetime.fromisoformat(pred["timestamp"]).hour
                day = datetime.fromisoformat(pred["timestamp"]).strftime('%m-%d')
                hourly_performance[day][hour].append(pred["confidence"])
            
            if len(hourly_performance) > 1:
                days = sorted(hourly_performance.keys())
                hours = range(24)
                heatmap_data = []
                
                for day in days:
                    day_data = []
                    for hour in hours:
                        if hourly_performance[day][hour]:
                            day_data.append(statistics.mean(hourly_performance[day][hour]))
                        else:
                            day_data.append(np.nan)
                    heatmap_data.append(day_data)
                
                im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                ax1.set_title('Performance Heatmap (Confidence by Hour)')
                ax1.set_xlabel('Hour of Day')
                ax1.set_ylabel('Date')
                ax1.set_xticks(range(0, 24, 2))
                ax1.set_xticklabels(range(0, 24, 2))
                ax1.set_yticks(range(len(days)))
                ax1.set_yticklabels(days)
                
                cbar = plt.colorbar(im, ax=ax1)
                cbar.set_label('Average Confidence')
            else:
                ax1.text(0.5, 0.5, 'Insufficient Data\nfor Heatmap\n(Need multiple days)', 
                        ha='center', va='center', fontsize=12, transform=ax1.transAxes)
                ax1.set_title('Performance Heatmap (Confidence by Hour)')
            
            # 2. Efficiency Metrics
            ax2.axis('off')
            
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0
            predictions_per_hour = len(predictions) / max(total_duration / 3600, 1/3600)
            
            normal_keywords = ['normal', 'good', 'ok', 'non_defective']
            accuracy_proxy = statistics.mean(confidences)
            
            class_efficiency = len(set(classes)) / len(predictions) * 100
            
            efficiency_text = f"""
EFFICIENCY METRICS

Throughput:
• Predictions/Hour: {predictions_per_hour:.1f}
• Total Duration: {timedelta(seconds=int(total_duration))}
• Average Interval: {total_duration/max(len(predictions)-1, 1):.1f}s

Performance:
• Average Confidence: {accuracy_proxy:.1%}
• Classification Diversity: {class_efficiency:.1f}%
• High Confidence Rate: {sum(1 for c in confidences if c >= 0.9)/len(confidences)*100:.1f}%

Quality Indicators:
• Normal Rate: {sum(1 for c in classes if c.lower() in normal_keywords)/len(classes)*100:.1f}%
• Defect Rate: {sum(1 for c in classes if c.lower() not in normal_keywords)/len(classes)*100:.1f}%
• Consistency: {1-statistics.stdev(confidences) if len(confidences) > 1 else 1:.1%}
            """
            
            ax2.text(0.1, 0.9, efficiency_text, ha='left', va='top', fontsize=11, 
                    fontfamily='monospace', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['primary'], alpha=0.1))
            
            # 3. Prediction Confidence Trends
            if len(predictions) > 10:
                window = max(5, len(predictions) // 20)
                moving_avg = []
                moving_times = []
                
                for i in range(window, len(predictions) + 1):
                    avg_conf = statistics.mean(confidences[i-window:i])
                    moving_avg.append(avg_conf)
                    moving_times.append(timestamps[i-1])
                
                ax3.plot(moving_times, moving_avg, linewidth=2, color=self.colors['primary'], 
                        label=f'Moving Average (n={window})')
                ax3.scatter(timestamps, confidences, alpha=0.3, s=20, color=self.colors['secondary'])
                ax3.set_title('Confidence Trend Analysis')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Confidence Level')
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient Data\nfor Trend Analysis', ha='center', va='center', 
                        fontsize=12, transform=ax3.transAxes)
                ax3.set_title('Confidence Trend Analysis')
            
            # 4. Quality Control Chart
            ax4.plot(range(len(confidences)), confidences, 'o-', linewidth=1, markersize=3, 
                    color=self.colors['primary'], alpha=0.7)
            
            mean_conf = statistics.mean(confidences)
            std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0
            
            ax4.axhline(y=mean_conf, color=self.colors['success'], linestyle='-', linewidth=2, label='Mean')
            ax4.axhline(y=mean_conf + 2*std_conf, color=self.colors['warning'], linestyle='--', 
                       alpha=0.7, label='UCL (+2σ)')
            ax4.axhline(y=mean_conf - 2*std_conf, color=self.colors['warning'], linestyle='--', 
                       alpha=0.7, label='LCL (-2σ)')
            
            for i, conf in enumerate(confidences):
                if abs(conf - mean_conf) > 2 * std_conf:
                    ax4.plot(i, conf, 'ro', markersize=6, alpha=0.8)
            
            ax4.set_title('Quality Control Chart')
            ax4.set_xlabel('Prediction Number')
            ax4.set_ylabel('Confidence Level')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error creating performance dashboard: {e}")
            return None

    def create_html_summary(self, predictions, session_data, generated_files, filepath):
        """Create HTML summary report with embedded images"""
        try:
            classes = [p["predicted_class"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            
            normal_keywords = ['normal', 'good', 'ok', 'non_defective']
            normal_count = sum(1 for c in classes if c.lower() in normal_keywords)
            defective_count = len(predictions) - normal_count
            avg_confidence = statistics.mean(confidences)
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAKAR Vision AI - Visual Analytics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid {self.colors['primary']};
        }}
        .header h1 {{
            color: {self.colors['primary']};
            margin: 0;
            font-size: 2.5em;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, {self.colors['primary']}, {self.colors['secondary']});
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
        }}
        .dashboard-item {{
            text-align: center;
        }}
        .dashboard-item h2 {{
            color: {self.colors['primary']};
            margin-bottom: 15px;
        }}
        .dashboard-item img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SAKAR VISION AI</h1>
            <p>Visual Analytics Report - Session {session_data['session_info']['session_id']}</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{len(predictions)}</h3>
                <p>Total Predictions</p>
            </div>
            <div class="summary-card">
                <h3>{avg_confidence:.1%}</h3>
                <p>Average Confidence</p>
            </div>
            <div class="summary-card">
                <h3>{normal_count}</h3>
                <p>Normal Items</p>
            </div>
            <div class="summary-card">
                <h3>{defective_count}</h3>
                <p>Defective Items</p>
            </div>
        </div>
        
        <div class="dashboard-grid">
            {self._generate_dashboard_sections(generated_files)}
        </div>
        
        <div class="footer">
            <p><strong>🎨 Professional Visual Analytics Features:</strong></p>
            <p>• High-resolution charts (300 DPI) • Quality control gauges • Statistical analysis</p>
            <p>• Performance trends • Interactive dashboards • Export capabilities</p>
            <br>
            <p>SAKAR Vision AI - Advanced Quality Control System</p>
        </div>
    </div>
</body>
</html>
            """
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return filepath
            
        except Exception as e:
            print(f"Error creating HTML summary: {e}")
            return None

    def _generate_dashboard_sections(self, generated_files):
        """Generate dashboard sections HTML"""
        dashboard_names = {
            'dashboard': ('📊 Executive Dashboard', 'Key metrics, quality gauges, and performance overview'),
            'quality': ('📈 Quality Metrics Analysis', 'Confidence distributions, quality tiers, and class comparisons'),
            'timeline': ('⏱️ Timeline & Patterns', 'Hourly patterns, prediction timeline, and session analysis'),
            'statistics': ('📊 Statistical Analysis', 'Box plots, correlations, and advanced statistics'),
            'performance': ('🎯 Performance Metrics', 'Efficiency metrics, trends, and quality control charts')
        }
        
        html = ""
        for filepath in generated_files:
            if filepath.endswith(('.png', '.jpg', '.jpeg')):
                filename = os.path.basename(filepath)
                dashboard_info = None
                
                for key in dashboard_names:
                    if key in filename:
                        dashboard_info = dashboard_names[key]
                        break
                
                if dashboard_info:
                    title, description = dashboard_info
                    html += f"""
                    <div class="dashboard-item">
                        <h2>{title}</h2>
                        <p>{description}</p>
                        <img src="{filename}" alt="{title}">
                    </div>
                    """
        
        return html


class PredictionStorage:
    """Enhanced prediction storage system with comprehensive logging and management"""
    
    def __init__(self, storage_path=PREDICTIONS_STORAGE_PATH, max_predictions=10000):
        self.storage_path = storage_path
        self.max_predictions = max_predictions
        self.session_id = str(uuid.uuid4())[:8]  # Unique session identifier
        self.session_start_time = datetime.now().isoformat()
        self.predictions_cache = []
        
        # Initialize storage file if it doesn't exist
        self.initialize_storage()
        
    def initialize_storage(self):
        """Initialize the predictions storage file"""
        try:
            if not os.path.exists(self.storage_path):
                initial_data = {
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "version": "1.0",
                        "description": "SAKAR Vision AI - Prediction Storage System"
                    },
                    "sessions": {}
                }
                with open(self.storage_path, 'w') as f:
                    json.dump(initial_data, f, indent=4)
                print(f"✓ Initialized prediction storage: {self.storage_path}")
            else:
                print(f"✓ Using existing prediction storage: {self.storage_path}")
                
        except Exception as e:
            print(f"✗ Error initializing prediction storage: {e}")
    
    def save_prediction(self, predicted_class, confidence, additional_data=None):
        """
        Save a single prediction with comprehensive metadata
        
        Args:
            predicted_class (str): The predicted class name
            confidence (float): Confidence score (0.0 to 1.0)
            additional_data (dict): Optional additional metadata
        """
        try:
            # Create prediction record
            prediction_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_percentage": f"{confidence:.2%}",
                "metadata": {
                    "device": str(device),
                    "session_start": self.session_start_time
                }
            }
            
            # Add any additional data
            if additional_data:
                prediction_record["additional_data"] = additional_data
            
            # Add to cache for batch saving
            self.predictions_cache.append(prediction_record)
            
            # Save immediately for critical predictions or when cache is full
            if len(self.predictions_cache) >= 50:  # Batch save every 50 predictions
                self.flush_cache()
                
            return prediction_record["id"]
            
        except Exception as e:
            print(f"✗ Error saving prediction: {e}")
            return None
    
    def flush_cache(self):
        """Flush cached predictions to storage file"""
        if not self.predictions_cache:
            return
            
        try:
            # Load existing data
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Initialize session data if it doesn't exist
            if self.session_id not in data["sessions"]:
                data["sessions"][self.session_id] = {
                    "session_info": {
                        "session_id": self.session_id,
                        "start_time": self.session_start_time,
                        "last_update": datetime.now().isoformat()
                    },
                    "predictions": []
                }
            
            # Add cached predictions to session
            data["sessions"][self.session_id]["predictions"].extend(self.predictions_cache)
            data["sessions"][self.session_id]["session_info"]["last_update"] = datetime.now().isoformat()
            data["sessions"][self.session_id]["session_info"]["total_predictions"] = len(data["sessions"][self.session_id]["predictions"])
            
            # Manage storage size - keep only recent sessions if needed
            self.manage_storage_size(data)
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"✓ Saved {len(self.predictions_cache)} predictions to storage")
            self.predictions_cache = []  # Clear cache
            
        except Exception as e:
            print(f"✗ Error flushing predictions cache: {e}")
    
    def manage_storage_size(self, data):
        """Manage storage size by removing old sessions if necessary"""
        try:
            total_predictions = sum(len(session["predictions"]) for session in data["sessions"].values())
            
            if total_predictions > self.max_predictions:
                # Sort sessions by start time and remove oldest
                sessions_by_time = sorted(
                    data["sessions"].items(),
                    key=lambda x: x[1]["session_info"]["start_time"]
                )
                
                while total_predictions > self.max_predictions and len(sessions_by_time) > 1:
                    # Remove oldest session (but keep current session)
                    oldest_session_id, oldest_session = sessions_by_time.pop(0)
                    if oldest_session_id != self.session_id:
                        predictions_removed = len(oldest_session["predictions"])
                        del data["sessions"][oldest_session_id]
                        total_predictions -= predictions_removed
                        print(f"✓ Removed old session {oldest_session_id} with {predictions_removed} predictions")
                        
        except Exception as e:
            print(f"✗ Error managing storage size: {e}")
    
    def get_session_statistics(self):
        """Get statistics for current session"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.session_id in data["sessions"]:
                session_data = data["sessions"][self.session_id]
                predictions = session_data["predictions"]
                
                if not predictions:
                    return {"total": 0, "classes": {}, "avg_confidence": 0}
                
                # Calculate statistics
                total_predictions = len(predictions)
                class_counts = {}
                total_confidence = 0
                
                for pred in predictions:
                    class_name = pred["predicted_class"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_confidence += pred["confidence"]
                
                avg_confidence = total_confidence / total_predictions
                
                return {
                    "session_id": self.session_id,
                    "total": total_predictions,
                    "classes": class_counts,
                    "avg_confidence": avg_confidence,
                    "start_time": session_data["session_info"]["start_time"]
                }
            else:
                return {"total": 0, "classes": {}, "avg_confidence": 0}
                
        except Exception as e:
            print(f"✗ Error getting session statistics: {e}")
            return {"total": 0, "classes": {}, "avg_confidence": 0}
    
    def export_session_data(self, export_path=None):
        """Export current session data to a separate file"""
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"session_{self.session_id}_{timestamp}.json"
            
            # Flush any cached predictions first
            self.flush_cache()
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.session_id in data["sessions"]:
                session_export = {
                    "export_info": {
                        "exported_at": datetime.now().isoformat(),
                        "session_id": self.session_id,
                        "source_file": self.storage_path
                    },
                    "session_data": data["sessions"][self.session_id]
                }
                
                with open(export_path, 'w') as f:
                    json.dump(session_export, f, indent=4)
                
                print(f"✓ Session data exported to: {export_path}")
                return export_path
            else:
                print(f"✗ Session {self.session_id} not found")
                return None
                
        except Exception as e:
            print(f"✗ Error exporting session data: {e}")
            return None
    
    def close_session(self):
        """Close current session and flush all cached data"""
        try:
            # Flush any remaining cached predictions
            self.flush_cache()
            
            # Update session end time
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.session_id in data["sessions"]:
                data["sessions"][self.session_id]["session_info"]["end_time"] = datetime.now().isoformat()
                
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f"✓ Session {self.session_id} closed successfully")
            
        except Exception as e:
            print(f"✗ Error closing session: {e}")


def save_session_state(ui_name="image_classification"):
    """Save session state when this UI is opened"""
    try:
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "user": {
                "username": "admin",
                "full_name": "Administrator"
            },
            "additional_data": {
                "opened_at": datetime.now().isoformat()
            }
        }

        with open(SESSION_STATE_PATH, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session state saved: {ui_name}")

    except Exception as e:
        print(f"Error saving session state: {e}")


def save_session_on_close(ui_name="image_classification"):
    """Save session state when this UI is closed"""
    try:
        session_data = {
            "last_ui": ui_name,
            "timestamp": datetime.now().isoformat(),
            "user": {
                "username": "admin",
                "full_name": "Administrator"
            },
            "additional_data": {
                "closed_at": datetime.now().isoformat()
            }
        }

        with open(SESSION_STATE_PATH, 'w') as f:
            json.dump(session_data, f, indent=4)

        print(f"Session state saved on close: {ui_name}")

    except Exception as e:
        print(f"Error saving session state on close: {e}")


class ClassificationDashboard(QWidget):
    """Dashboard for displaying classification results and statistics"""

    def __init__(self, parent=None, dashboard_type="non_defective"):
        super().__init__(parent)
        self.dashboard_type = dashboard_type  # "non_defective" or "defective"
        self.setMinimumWidth(300)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            QLabel {
                padding: 6px;
                color: #495057;
                font-size: 11pt;
            }
            QLabel#heading {
                background-color: #ff914d;
                color: white;
                font-weight: bold;
                font-size: 14pt;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 10px;
                border-bottom: 1px solid #ff7730;
            }
            QLabel#subheading {
                background-color: #e9ecef;
                font-weight: bold;
                color: #343a40;
                padding: 8px;
                border-top: 1px solid #dee2e6;
                border-bottom: 1px solid #dee2e6;
                margin-top: 5px;
            }
            QScrollArea {
                border: none;
            }
            QWidget#classificationListWidget {
                background-color: #ffffff;
                border: none;
            }
        """)

        # Initialize recent classifications
        self.recent_classifications = []  # Store recent classifications for display

        self.init_ui()

    def init_ui(self):
        """Initialize the dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Dashboard title based on type
        if self.dashboard_type == "non_defective":
            title_text = "Non-Defective"
            self.title_color = "#28a745"  # Green
        else:
            title_text = "Defective"
            self.title_color = "#dc3545"  # Red

        self.title_label = QLabel(title_text)
        self.title_label.setObjectName("heading")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"""
            background-color: {self.title_color};
            color: white;
            font-weight: bold;
            font-size: 14pt;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            padding: 10px;
        """)
        layout.addWidget(self.title_label)

        # Main content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        # Recent classifications section
        recent_label = QLabel("Recent Classifications")
        recent_label.setObjectName("subheading")
        content_layout.addWidget(recent_label)

        # Scroll area for recent classifications
        self.recent_scroll_area = QScrollArea()
        self.recent_scroll_area.setWidgetResizable(True)
        self.recent_scroll_area.setMinimumHeight(150)
        self.recent_scroll_area.setMaximumHeight(200)
        self.recent_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.recent_list_widget = QWidget()
        self.recent_list_widget.setObjectName("classificationListWidget")
        self.recent_layout = QVBoxLayout(self.recent_list_widget)
        self.recent_layout.setContentsMargins(5, 5, 5, 5)
        self.recent_layout.setSpacing(5)
        self.recent_layout.addStretch(1)

        self.recent_scroll_area.setWidget(self.recent_list_widget)
        content_layout.addWidget(self.recent_scroll_area)

        content_layout.addStretch()
        layout.addWidget(content_widget)

    def update_classification(self, class_name, confidence):
        """Update classification based on class name"""
        self.add_recent_classification(class_name, confidence)
        self.update_ui()

    def add_recent_classification(self, class_name, confidence):
        """Add a recent classification to the list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.recent_classifications.append({
            "class": class_name,
            "confidence": confidence,
            "time": timestamp
        })

        # Keep only last 10 classifications
        if len(self.recent_classifications) > 10:
            self.recent_classifications = self.recent_classifications[-10:]

    def update_ui(self):
        """Update the UI"""
        # Clear recent classifications display
        while self.recent_layout.count() > 1:  # Keep the stretch
            child = self.recent_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add recent classifications
        if not self.recent_classifications:
            placeholder_label = QLabel("No classifications yet")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("color: #6c757d; font-style: italic;")
            self.recent_layout.insertWidget(0, placeholder_label)
        else:
            for i, item in enumerate(reversed(self.recent_classifications)):
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 2, 5, 2)

                time_label = QLabel(item["time"])
                time_label.setStyleSheet("color: #6c757d; font-size: 9pt;")
                time_label.setMinimumWidth(60)

                class_label = QLabel(item["class"])
                class_label.setStyleSheet("color: #495057; font-size: 10pt;")

                conf_label = QLabel(f"{item['confidence']:.1%}")
                conf_label.setStyleSheet(
                    f"color: {self.title_color}; font-weight: bold; font-size: 9pt;")
                conf_label.setMinimumWidth(50)

                item_layout.addWidget(time_label)
                item_layout.addWidget(class_label, 1)
                item_layout.addWidget(conf_label)

                self.recent_layout.insertWidget(0, item_widget)

        self.recent_list_widget.updateGeometry()


class CameraWorker(QObject):
    """Enhanced camera worker with motion detection and stationary frame prediction"""
    finished = pyqtSignal()
    image_captured = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float)
    motion_status_changed = pyqtSignal(bool)  # New signal for motion status

    def __init__(self, model=None, transform=None, class_names=None, idx_to_class=None):
        super().__init__()
        self.running = True
        self.capture = None
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.idx_to_class = idx_to_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Motion detection parameters
        self.motion_threshold = 2000  # Threshold for motion detection
        self.min_stationary_frames = 10  # Minimum frames to be stationary before prediction
        self.motion_sensitivity = 0.02  # Sensitivity for motion detection (0.01-0.1)
        
        # Motion detection state
        self.previous_frame = None
        self.stationary_frame_count = 0
        self.is_motion_detected = False
        self.frame_count = 0
        self.last_prediction_frame = None

    def detect_motion(self, current_frame):
        """
        Detect motion between consecutive frames using frame difference
        
        Args:
            current_frame: Current OpenCV frame
            
        Returns:
            bool: True if motion detected, False if stationary
        """
        try:
            # Convert to grayscale for motion detection
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
            
            # If this is the first frame, store it and return no motion
            if self.previous_frame is None:
                self.previous_frame = gray_current
                return False
            
            # Compute the absolute difference between frames
            frame_diff = cv2.absdiff(self.previous_frame, gray_current)
            
            # Apply threshold to get binary image
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate the percentage of changed pixels
            total_pixels = thresh.shape[0] * thresh.shape[1]
            changed_pixels = cv2.countNonZero(thresh)
            motion_percentage = changed_pixels / total_pixels
            
            # Update previous frame
            self.previous_frame = gray_current
            
            # Determine if motion is detected based on threshold
            motion_detected = motion_percentage > self.motion_sensitivity
            
            return motion_detected
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return True  # Assume motion on error to be safe

    def is_frame_suitable_for_prediction(self):
        """
        Check if current frame is suitable for prediction
        
        Returns:
            bool: True if frame is suitable (camera is stationary)
        """
        return (not self.is_motion_detected and 
                self.stationary_frame_count >= self.min_stationary_frames)

    def run(self):
        """Main camera loop with motion detection"""
        try:
            print("📹 Starting camera feed...")

            self.capture = cv2.VideoCapture(6)
            if not self.capture.isOpened():
                print("Failed to open camera 2, trying camera 1")
                self.capture = cv2.VideoCapture(6)

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            print("📹 Camera started with motion detection enabled")
            print(f"🎯 Predictions will only occur when camera is stationary for {self.min_stationary_frames} frames")

            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    continue

                self.frame_count += 1

                # Detect motion in current frame
                motion_detected = self.detect_motion(frame)
                
                # Update motion state
                if motion_detected != self.is_motion_detected:
                    self.is_motion_detected = motion_detected
                    self.motion_status_changed.emit(motion_detected)
                    
                    if motion_detected:
                        self.stationary_frame_count = 0
                    
                # Update stationary frame count
                if not motion_detected:
                    self.stationary_frame_count += 1
                else:
                    self.stationary_frame_count = 0

                # Convert to RGB for display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add motion status overlay to the image
                rgb_image_with_overlay = self.add_motion_overlay(rgb_image, motion_detected)
                
                # Convert to Qt image and emit
                h, w, ch = rgb_image_with_overlay.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image_with_overlay.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_captured.emit(qt_image)

                # Perform prediction only when camera is stationary and model is available
                if (self.model is not None and self.transform is not None and
                        self.is_frame_suitable_for_prediction()):
                    
                    # Only predict every 30 frames when stationary to avoid too frequent predictions
                    if self.stationary_frame_count % 30 == 0:
                        self.predict_frame(frame)

                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            print(f"Error in camera feed: {e}")
        finally:
            if self.capture is not None:
                self.capture.release()
            self.finished.emit()

    def add_motion_overlay(self, image, motion_detected):
        """
        Add visual overlay to indicate motion status
        
        Args:
            image: RGB image array
            motion_detected: Boolean indicating if motion is detected
            
        Returns:
            Modified image with overlay
        """
        try:
            # Create a copy to avoid modifying original
            overlay_image = image.copy()
            
            # Define overlay parameters
            overlay_height = 30
            overlay_width = 200
            
            # Position overlay at top-right corner
            y_start = 10
            x_start = image.shape[1] - overlay_width - 10
            y_end = y_start + overlay_height
            x_end = x_start + overlay_width
            
            # Choose color and text based on motion status
            if motion_detected:
                color = (255, 100, 100)  # Red for motion
                text = "MOTION DETECTED"
            elif self.stationary_frame_count < self.min_stationary_frames:
                color = (255, 255, 100)  # Yellow for stabilizing
                text = f"STABILIZING ({self.stationary_frame_count}/{self.min_stationary_frames})"
            else:
                color = (100, 255, 100)  # Green for ready
                text = "READY FOR PREDICTION"
            
            # Draw semi-transparent overlay rectangle
            overlay = overlay_image.copy()
            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
            overlay_image = cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0)
            
            # Add text
            cv2.putText(overlay_image, text, (x_start + 5, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
            return overlay_image
            
        except Exception as e:
            print(f"Error adding motion overlay: {e}")
            return image

    def predict_frame(self, frame):
        """Make prediction on current frame (only called when stationary)"""
        try:
            if self.model is None or self.transform is None:
                return

            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                probability = probabilities[predicted_idx].item()

                # Use idx_to_class mapping if available
                if self.idx_to_class and predicted_idx in self.idx_to_class:
                    predicted_class = self.idx_to_class[predicted_idx]
                elif predicted_idx < len(self.class_names):
                    predicted_class = self.class_names[predicted_idx]
                else:
                    predicted_class = f"Class {predicted_idx}"

                print(f"🎯 Prediction: {predicted_class} ({probability:.2%}) - Frame: {self.frame_count}")
                self.prediction_ready.emit(predicted_class, probability)

        except Exception as e:
            print(f"Error in prediction: {e}")

    def stop(self):
        """Stop camera worker"""
        self.running = False


class ClassNamesDialog(QDialog):
    """Custom dialog for entering class names"""
    
    def __init__(self, num_classes, current_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Class Names")
        self.setModal(True)
        self.resize(400, min(600, 100 + num_classes * 35))
        
        layout = QVBoxLayout()
        
        # Instructions
        instruction = QLabel(f"Enter names for {num_classes} classes:")
        instruction.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instruction)
        
        # Scroll area for many classes
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Input fields
        self.line_edits = []
        for i in range(num_classes):
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Class {i} name...")
            line_edit.setText(current_names[i] if i < len(current_names) else f"class_{i}")
            self.line_edits.append(line_edit)
            scroll_layout.addWidget(line_edit)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        preset_button = QPushButton("Use Defect Presets")
        preset_button.clicked.connect(self.use_defect_presets)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        
        button_layout.addWidget(preset_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def use_defect_presets(self):
        """Fill with common defect detection class names"""
        if len(self.line_edits) == 10:
            presets = [
                'normal', 'clamp_miss', 'screw_miss', 'scratch', 'dent', 
                'crack', 'spot', 'hole', 'incomplete', 'damaged'
            ]
        elif len(self.line_edits) == 2:
            presets = ['normal', 'defective']
        elif len(self.line_edits) == 6:
            presets = ['normal', 'clamp_miss', 'screw_miss', 'scratch', 'dent', 'crack']
        elif len(self.line_edits) == 8:
            presets = ['normal', 'clamp_miss', 'screw_miss', 'scratch', 'dent', 'crack', 'spot', 'hole']
        else:
            presets = [f"defect_type_{i}" for i in range(len(self.line_edits))]
            presets[0] = 'normal'  # First class is usually normal
        
        for i, preset in enumerate(presets):
            if i < len(self.line_edits):
                self.line_edits[i].setText(preset)
    
    def get_class_names(self):
        return [line_edit.text().strip() or f"class_{i}" for i, line_edit in enumerate(self.line_edits)]


class ImageClassiUI(QWidget):
    """Enhanced Image Classification UI with sophisticated interface and visual reporting"""

    def __init__(self, parent=None, good_dir=None, bad_dir=None):
        super().__init__(parent)
        self.demo_feed_ui = parent
        self.good_dir = good_dir
        self.bad_dir = bad_dir
        self.model = None
        self.class_names = []
        self.class_mapping = {}
        self.idx_to_class = {}
        self.img_height, self.img_width = 224, 224
        self.camera_feed_running = False

        # Initialize prediction storage system
        self.prediction_storage = PredictionStorage()
        
        # Initialize visual report generator if available
        if VISUAL_REPORTING_AVAILABLE:
            self.visual_report_generator = VisualReportGenerator(self.prediction_storage)
        else:
            self.visual_report_generator = None

        # Save session state when this UI is opened
        save_session_state("image_classification")

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.init_ui()
        self.auto_load_latest_model()

    def init_ui(self):
        """Initialize the enhanced UI"""
        self.setWindowTitle('SAKAR VISION AI - Image Classification with Visual Analytics')
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a80d2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Main content area with dashboards and camera
        content_main_layout = QHBoxLayout()
        content_main_layout.setSpacing(10)

        # Left dashboard (Non-defective)
        self.left_dashboard = ClassificationDashboard(dashboard_type="non_defective")
        content_main_layout.addWidget(self.left_dashboard, 1)

        # Camera feed group
        camera_group = QGroupBox("Live Camera Classification")
        camera_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #FFA500;
                border-radius: 10px;
                padding: 10px;
                background-color: #FFFFFF;
            }
        """)
        camera_layout = QVBoxLayout()

        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("""
            border: 3px solid #FFA500;
            border-radius: 10px;
            background-color: #222222;
            color: white;
            font-size: 16px;
        """)
        self.camera_label.setText("Camera feed will appear here when started")
        camera_layout.addWidget(self.camera_label, 0, Qt.AlignCenter)

        # Control panel
        controls_container = QWidget()
        controls_container.setStyleSheet("""
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 15px;
        """)
        controls_layout = QVBoxLayout(controls_container)

        # Prediction display
        self.camera_prediction_label = QLabel("Prediction: Camera not started")
        self.camera_prediction_label.setAlignment(Qt.AlignCenter)
        self.camera_prediction_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #333333;
            padding: 15px;
            margin: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 2px solid #e9ecef;
        """)
        controls_layout.addWidget(self.camera_prediction_label)

        # Session statistics display
        self.session_stats_label = QLabel("Session: 0 predictions")
        self.session_stats_label.setAlignment(Qt.AlignCenter)
        self.session_stats_label.setStyleSheet("""
            font-size: 14px;
            color: #6c757d;
            padding: 8px;
            margin: 5px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        """)
        controls_layout.addWidget(self.session_stats_label)

        # Camera control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.start_camera_button = QPushButton("Start Camera")
        self.start_camera_button.clicked.connect(self.start_camera_feed)
        self.start_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 130px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.stop_camera_button = QPushButton("Stop Camera")
        self.stop_camera_button.clicked.connect(self.stop_camera_feed)
        self.stop_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 130px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        # Visual Report Button (main feature)
        self.visual_report_button = QPushButton("📊 Visual Analytics")
        self.visual_report_button.clicked.connect(self.generate_visual_report)
        self.visual_report_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)

        # Export buttons
        self.export_button = QPushButton("📁 Save Data")
        self.export_button.clicked.connect(self.export_session_data)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 130px;
            }
            QPushButton:hover {
                background-color: #5a35a3;
            }
        """)

        self.quick_export_button = QPushButton("📄 Text Summary")
        self.quick_export_button.clicked.connect(self.export_visual_data)
        self.quick_export_button.setStyleSheet("""
            QPushButton {
                background-color: #fd7e14;
                color: white;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 130px;
            }
            QPushButton:hover {
                background-color: #e8690b;
            }
        """)

        # Arrange buttons in the layout
        button_layout.addStretch(1)
        button_layout.addWidget(self.start_camera_button)
        button_layout.addWidget(self.stop_camera_button)
        button_layout.addWidget(self.visual_report_button)  # MAIN VISUAL REPORT BUTTON
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.quick_export_button)
        button_layout.addStretch(1)

        controls_layout.addLayout(button_layout)
        camera_layout.addWidget(controls_container)
        camera_group.setLayout(camera_layout)

        content_main_layout.addWidget(camera_group, 4)

        # Right dashboard (Defective)
        self.right_dashboard = ClassificationDashboard(dashboard_type="defective")
        content_main_layout.addWidget(self.right_dashboard, 1)

        content_layout.addLayout(content_main_layout)

        # Status bar
        status_bar = QWidget()
        status_bar.setStyleSheet("""
            background-color: #f8f9fa;
            border-top: 1px solid #e9ecef;
            padding: 5px;
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 5, 10, 5)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #6c757d; font-size: 12px;")
        status_layout.addWidget(self.status_label)

        content_layout.addWidget(status_bar)
        main_layout.addWidget(content_widget)

        # Timer for updating session statistics
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_session_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds

    def update_button_states(self):
        """Update button states based on available data"""
        stats = self.prediction_storage.get_session_statistics()
        total = stats.get("total", 0)
        
        # Enable visual report button only if we have predictions and visual reporting is available
        if hasattr(self, 'visual_report_button'):
            self.visual_report_button.setEnabled(total > 0 and VISUAL_REPORTING_AVAILABLE)
            
            # Update button text to show prediction count
            if total > 0:
                self.visual_report_button.setText(f"📊 Visual Analytics ({total})")
            else:
                self.visual_report_button.setText("📊 Visual Analytics")
            
            # Show warning if visual reporting not available
            if not VISUAL_REPORTING_AVAILABLE:
                self.visual_report_button.setToolTip("Install matplotlib and seaborn for visual reporting")

    def update_session_stats(self):
        """Update session statistics display and button states"""
        try:
            stats = self.prediction_storage.get_session_statistics()
            total = stats.get("total", 0)
            avg_conf = stats.get("avg_confidence", 0)
            
            if total > 0:
                self.session_stats_label.setText(
                    f"Session: {total} predictions | Avg Confidence: {avg_conf:.1%}"
                )
            else:
                self.session_stats_label.setText("Session: 0 predictions")
            
            # Update button states
            self.update_button_states()
                
        except Exception as e:
            print(f"Error updating session stats: {e}")

    def generate_visual_report(self):
        """Generate comprehensive visual analytics report with charts and graphs"""
        try:
            # Check if visual reporting is available
            if not VISUAL_REPORTING_AVAILABLE:
                QMessageBox.critical(
                    self,
                    "Visual Reporting Not Available",
                    "📊 Visual reporting with charts and graphs is not available.\n\n"
                    "📦 Please install required packages:\n"
                    "pip install matplotlib seaborn\n\n"
                    "🔄 Restart the application after installation."
                )
                return
            
            stats = self.prediction_storage.get_session_statistics()
            total = stats.get("total", 0)
            
            if total == 0:
                QMessageBox.information(
                    self,
                    "No Data Available",
                    "📊 No predictions available for visual report generation.\n\n"
                    "💡 Please start the camera feed and make some predictions first.\n\n"
                    "🎨 The visual report includes:\n"
                    "• Executive dashboard with key metrics\n"
                    "• Quality analysis charts and graphs\n"
                    "• Timeline patterns and trends\n"
                    "• Statistical analysis visualizations\n"
                    "• Performance metrics and control charts\n"
                    "• Interactive HTML summary report"
                )
                return
            
            # Show progress dialog
            progress = QProgressDialog("🎨 Generating visual charts and graphs...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(10)
            progress.setStyleSheet("""
                QProgressDialog {
                    background-color: #f8f9fa;
                }
                QProgressBar {
                    border: 2px solid #17a2b8;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #17a2b8;
                    border-radius: 3px;
                }
            """)
            QApplication.processEvents()
            
            # Get export directory from user
            file_dialog = QFileDialog()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = self.prediction_storage.session_id
            
            progress.setValue(20)
            progress.setLabelText("📁 Selecting save location...")
            QApplication.processEvents()
            
            # Ask for directory instead of specific file
            export_dir = file_dialog.getExistingDirectory(
                self,
                "Select Directory for Visual Analytics Report",
                "",
                QFileDialog.ShowDirsOnly
            )
            
            if not export_dir:
                progress.close()
                return
            
            export_path = os.path.join(export_dir, f"SAKAR_AI_Visual_Report_{session_id}_{timestamp}")
            
            progress.setValue(40)
            progress.setLabelText("📈 Creating executive dashboard...")
            QApplication.processEvents()
            
            # Generate visual report
            generated_files, error = self.visual_report_generator.generate_visual_report(export_path, 'png')
            
            progress.setValue(80)
            progress.setLabelText("✨ Finalizing visual report...")
            QApplication.processEvents()
            
            if error:
                progress.close()
                QMessageBox.critical(
                    self,
                    "Visual Report Generation Failed",
                    f"❌ Failed to generate visual report:\n\n{error}\n\n"
                    f"Please check the logs for more information."
                )
                return
            
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()
            
            # Calculate report statistics for success message
            avg_conf = stats.get("avg_confidence", 0)
            classes = stats.get("classes", {})
            defect_types = len([k for k in classes.keys() if k.lower() not in ['normal', 'good', 'ok', 'non_defective']])
            
            # Show success message with detailed information
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("✅ Visual Analytics Report Generated")
            msg_box.setText("🎉 Professional visual analytics report with charts and graphs has been generated!")
            
            # Count different types of files generated
            png_files = [f for f in generated_files if f.endswith('.png')]
            html_files = [f for f in generated_files if f.endswith('.html')]
            
            msg_box.setInformativeText(
                f"📊 <b>Report Summary:</b><br>"
                f"• Total predictions analyzed: <b>{total:,}</b><br>"
                f"• Average confidence: <b>{avg_conf:.1%}</b><br>"
                f"• Unique classes detected: <b>{len(classes)}</b><br>"
                f"• Defect types identified: <b>{defect_types}</b><br><br>"
                
                f"🎨 <b>Visual Components Generated:</b><br>"
                f"• Executive Dashboard (Key Metrics & Gauges)<br>"
                f"• Quality Metrics (Charts & Distributions)<br>"
                f"• Timeline Analysis (Trends & Patterns)<br>"
                f"• Statistical Analysis (Box Plots & Correlations)<br>"
                f"• Performance Metrics (Control Charts & Heatmaps)<br>"
                f"• Interactive HTML Summary Report<br><br>"
                
                f"📁 <b>Files Created:</b><br>"
                f"• {len(png_files)} High-Resolution Chart Images<br>"
                f"• {len(html_files)} Interactive HTML Report<br>"
                f"• Location: <code>{os.path.basename(export_dir)}</code>"
            )
            
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setIcon(QMessageBox.Information)
            
            # Add custom action buttons
            open_folder_button = msg_box.addButton("📂 Open Folder", QMessageBox.ActionRole)
            view_dashboard_button = msg_box.addButton("📊 View Dashboard", QMessageBox.ActionRole)
            view_summary_button = msg_box.addButton("🌐 Open HTML Report", QMessageBox.ActionRole)
            
            # Style the message box
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #f8f9fa;
                }
                QMessageBox QPushButton {
                    background-color: #17a2b8;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 120px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #138496;
                }
            """)
            
            msg_box.exec_()
            
            # Handle custom button clicks
            clicked_button = msg_box.clickedButton()
            
            if clicked_button == open_folder_button:
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(export_dir)
                    elif os.name == 'posix':  # macOS and Linux
                        if sys.platform == 'darwin':  # macOS
                            subprocess.call(['open', export_dir])
                        else:  # Linux
                            subprocess.call(['xdg-open', export_dir])
                except Exception as e:
                    print(f"Unable to open folder: {e}")
                    QMessageBox.information(self, "Folder Location", f"Report saved at:\n{export_dir}")
            
            elif clicked_button == view_dashboard_button:
                # Open the executive dashboard image
                dashboard_files = [f for f in png_files if 'dashboard' in f]
                if dashboard_files:
                    try:
                        if os.name == 'nt':  # Windows
                            os.startfile(dashboard_files[0])
                        elif os.name == 'posix':  # macOS and Linux
                            if sys.platform == 'darwin':  # macOS
                                subprocess.call(['open', dashboard_files[0]])
                            else:  # Linux
                                subprocess.call(['xdg-open', dashboard_files[0]])
                    except Exception as e:
                        print(f"Unable to open dashboard: {e}")
            
            elif clicked_button == view_summary_button:
                # Find and open HTML summary
                if html_files:
                    try:
                        webbrowser.open(f'file://{html_files[0]}')
                    except Exception as e:
                        print(f"Unable to open HTML summary: {e}")
                        QMessageBox.information(self, "HTML Report", f"HTML report located at:\n{html_files[0]}")
            
            # Update status display
            self.display_message(f"✅ Visual analytics report generated: {len(generated_files)} files created")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Visual Report Generation Error",
                f"❌ An unexpected error occurred while generating the visual report:\n\n{e}\n\n"
                f"Please ensure matplotlib and seaborn are installed:\n"
                f"pip install matplotlib seaborn"
            )
            print(f"Visual report generation error: {e}")
            import traceback
            traceback.print_exc()

    def export_session_data(self):
        """Export current session data"""
        try:
            file_dialog = QFileDialog()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"session_export_{timestamp}.json"
            
            filepath, _ = file_dialog.getSaveFileName(
                self,
                "Export Session Data",
                default_filename,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if filepath:
                exported_file = self.prediction_storage.export_session_data(filepath)
                if exported_file:
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Session data exported successfully to:\n{exported_file}"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Failed to export session data. Please check the logs."
                    )
                    
        except Exception as e:
            print(f"Error in export dialog: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during export:\n{e}"
            )

    def export_visual_data(self):
        """Export visual data - simple text-based export"""
        try:
            stats = self.prediction_storage.get_session_statistics()
            total = stats.get("total", 0)
            
            if total == 0:
                QMessageBox.information(
                    self,
                    "No Data",
                    "No predictions available to export. Start camera feed and make some predictions first."
                )
                return
            
            # Get file path from user
            file_dialog = QFileDialog()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"text_summary_{timestamp}.txt"
            
            filepath, _ = file_dialog.getSaveFileName(
                self,
                "Export Text Summary",
                default_filename,
                "Text Files (*.txt);;All Files (*)"
            )
            
            if not filepath:
                return
            
            # Load session data
            with open(self.prediction_storage.storage_path, 'r') as f:
                data = json.load(f)
            
            if self.prediction_storage.session_id not in data["sessions"]:
                QMessageBox.warning(self, "Export Failed", "Session data not found.")
                return
            
            predictions = data["sessions"][self.prediction_storage.session_id]["predictions"]
            classes = [p["predicted_class"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            
            # Create simple text export
            with open(filepath, 'w') as f:
                f.write("SAKAR VISION AI - TEXT SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Summary
                normal_keywords = ['normal', 'good', 'ok', 'non_defective']
                normal_count = sum(1 for c in classes if c.lower() in normal_keywords)
                defective_count = len(classes) - normal_count
                
                f.write("SUMMARY:\n")
                f.write(f"Total Predictions: {len(classes)}\n")
                f.write(f"Normal/Good: {normal_count}\n")
                f.write(f"Defective: {defective_count}\n")
                f.write(f"Average Confidence: {np.mean(confidences):.2%}\n\n")
                
                # Class breakdown
                class_counts = {}
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                f.write("CLASS BREAKDOWN:\n")
                for cls, count in sorted(class_counts.items()):
                    percentage = (count / len(classes)) * 100
                    f.write(f"{cls}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Visual representation
                f.write("VISUAL REPRESENTATION:\n")
                f.write("(N=Normal/Good, D=Defective)\n\n")
                
                for i, cls in enumerate(classes):
                    if i % 10 == 0 and i > 0:
                        f.write("\n")
                    
                    if cls.lower() in normal_keywords:
                        f.write("[N] ")
                    else:
                        f.write("[D] ")
                
                f.write("\n\n")
                
                # Detailed list
                f.write("DETAILED PREDICTIONS:\n")
                f.write("-" * 40 + "\n")
                for i, (cls, conf) in enumerate(zip(classes, confidences), 1):
                    f.write(f"{i:3d}. {cls:15s} ({conf:.1%})\n")
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Text summary created:\n{filepath}"
            )
            
        except Exception as e:
            print(f"Error in text export: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during export:\n{e}"
            )

    def display_message(self, message):
        """Display status message"""
        self.status_label.setText(message)
        print(message)  # Also print to console
        QApplication.processEvents()

    def auto_load_latest_model(self):
        """Automatically load the best_model.pth or latest trained model"""
        # First try to load best_model.pth
        if os.path.exists("best_model.pth"):
            self.display_message("Loading best_model.pth...")
            return self.load_model("best_model.pth")
        
        # If best_model.pth not found, look for other .pth files
        model_files = glob.glob("*.pth")
        if not model_files:
            self.display_message("No trained model (.pth files) found in current directory.")
            return False

        latest_model = max(model_files, key=os.path.getmtime)
        self.display_message(f"Loading latest model: {latest_model}")
        return self.load_model(latest_model)

    def load_model(self, model_path):
        """Enhanced model loading with best_model.pth support"""
        if not os.path.exists(model_path):
            self.display_message(f"Model file not found: {model_path}")
            return False

        try:
            self.display_message(f"Loading model from: {model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                
                # Check for class mapping (best_model.pth format)
                if 'class_mapping' in checkpoint:
                    self.display_message("✓ Found class_mapping in checkpoint (best_model.pth format)")
                    self.class_mapping = checkpoint['class_mapping']
                    self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
                    num_classes = len(self.class_mapping)
                    self.class_names = list(self.class_mapping.keys())
                    
                    # Load model state
                    if 'model_state_dict' in checkpoint:
                        model_state_dict = checkpoint['model_state_dict']
                    else:
                        model_state_dict = checkpoint
                        
                    # Display additional info
                    if 'epoch' in checkpoint:
                        self.display_message(f"Model from epoch: {checkpoint['epoch']}")
                    if 'val_accuracy' in checkpoint:
                        self.display_message(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
                
                # Handle other checkpoint formats
                elif 'model_state_dict' in checkpoint:
                    self.display_message("Loading from standard checkpoint format")
                    model_state_dict = checkpoint['model_state_dict']
                    
                    # Try to get number of classes from the model structure
                    num_classes = self.detect_num_classes_from_state_dict(model_state_dict)
                    
                    # Check for other class name keys
                    class_keys = ['class_names', 'classes', 'idx_to_class']
                    for key in class_keys:
                        if key in checkpoint:
                            if isinstance(checkpoint[key], list):
                                self.class_names = checkpoint[key]
                                self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                                self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                            elif isinstance(checkpoint[key], dict):
                                if key == 'idx_to_class':
                                    self.idx_to_class = checkpoint[key]
                                    self.class_mapping = {v: k for k, v in self.idx_to_class.items()}
                                    self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
                            break
                    
                    # If no class names found, create generic ones
                    if not self.class_names:
                        self.class_names = [f"class_{i}" for i in range(num_classes)]
                        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                        
                else:
                    # Assume entire checkpoint is state dict
                    model_state_dict = checkpoint
                    num_classes = self.detect_num_classes_from_state_dict(model_state_dict)
                    self.class_names = [f"class_{i}" for i in range(num_classes)]
                    self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                    self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                    
            else:
                # Entire model object
                self.model = checkpoint
                self.model.to(device)
                self.model.eval()
                
                # Try to get number of classes
                if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'classifier'):
                    if hasattr(self.model.backbone.classifier, '1'):
                        num_classes = self.model.backbone.classifier[1].out_features
                    else:
                        num_classes = self.model.backbone.classifier[-1].out_features
                else:
                    num_classes = 2
                
                self.class_names = [f"class_{i}" for i in range(num_classes)]
                self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
                self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
                
                self.display_message(f"✓ Model loaded: {os.path.basename(model_path)} | Classes: {num_classes}")
                self.display_message(f"✓ Classes: {', '.join(self.class_names)}")
                self.start_camera_button.setEnabled(True)
                return True

            # Create model architecture using MobileNetV2
            self.model = DefectClassifier(num_classes)
            
            # Load state dict
            try:
                # Remove any 'backbone.' prefix if present
                cleaned_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('backbone.'):
                        cleaned_state_dict[key] = value
                    else:
                        cleaned_state_dict[f'backbone.{key}'] = value
                
                self.model.load_state_dict(cleaned_state_dict, strict=False)
            except:
                # Try direct loading
                self.model.load_state_dict(model_state_dict, strict=False)
                
            self.model.to(device)
            self.model.eval()

            # Display success message
            self.display_message(f"✓ Model loaded: {os.path.basename(model_path)}")
            self.display_message(f"✓ Classes ({num_classes}): {', '.join(self.class_names)}")
            
            self.start_camera_button.setEnabled(True)
            return True

        except Exception as e:
            self.model = None
            error_msg = f"✗ Error loading model: {e}"
            self.display_message(error_msg)
            print(f"Full error details: {e}")
            
            QMessageBox.critical(self, "Model Load Error", f"Failed to load model.\nError: {e}")
            return False

    def detect_num_classes_from_state_dict(self, state_dict):
        """Detect number of classes from model state dict"""
        try:
            # Look for final layer weights in MobileNetV2 structure
            for key, tensor in state_dict.items():
                if any(layer_name in key for layer_name in ['classifier.1.weight', 'classifier.weight']):
                    return tensor.shape[0]
            return 2  # Default fallback
        except:
            return 2

    def start_camera_feed(self):
        """Start camera feed for real-time classification"""
        if self.camera_feed_running or self.model is None:
            if self.model is None:
                QMessageBox.warning(self, "No Model", "Please load a trained model first.")
            return

        # Disable the Start Camera button immediately
        self.start_camera_button.setEnabled(False)
        self.camera_feed_running = True
        self.display_message("Starting camera feed...")

        # Enable the Stop Camera button
        self.stop_camera_button.setEnabled(True)

        # Start camera worker thread
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(
            model=self.model,
            transform=self.transform,
            class_names=self.class_names,
            idx_to_class=self.idx_to_class
        )
        self.camera_worker.moveToThread(self.camera_thread)

        # Connect signals
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.image_captured.connect(self.update_camera_view)
        self.camera_worker.prediction_ready.connect(self.update_prediction)
        self.camera_worker.finished.connect(self.stop_camera_feed)
        self.camera_worker.motion_status_changed.connect(self.update_motion_status)

        self.camera_thread.start()

    def stop_camera_feed(self):
        """Stop camera feed"""
        if not self.camera_feed_running:
            return

        # Disable the Stop Camera button immediately
        self.stop_camera_button.setEnabled(False)
        self.display_message("Stopping camera feed...")
        self.camera_feed_running = False

        if hasattr(self, 'camera_worker'):
            self.camera_worker.stop()

        if hasattr(self, 'camera_thread'):
            self.camera_thread.quit()
            self.camera_thread.wait()

        # Enable the Start Camera button
        self.start_camera_button.setEnabled(True)
        self.display_message("Camera feed stopped")
        self.camera_prediction_label.setText("Prediction: Camera stopped")

    def update_camera_view(self, image):
        """Update camera view with captured image - maintain aspect ratio"""
        if self.camera_label:
            # Get the QImage dimensions
            img_width = image.width()
            img_height = image.height()

            # Get the label dimensions
            label_width = self.camera_label.width()
            label_height = self.camera_label.height()

            # Calculate scaling factor while maintaining aspect ratio
            if label_width > 0 and label_height > 0 and img_width > 0 and img_height > 0:
                aspect_ratio = img_width / img_height
                label_ratio = label_width / label_height

                if aspect_ratio > label_ratio:
                    # Image is wider than label
                    new_width = label_width
                    new_height = int(label_width / aspect_ratio)
                else:
                    # Image is taller than label
                    new_height = label_height
                    new_width = int(label_height * aspect_ratio)

                # Scale the image
                scaled_image = image.scaled(new_width, new_height,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pixmap = QPixmap.fromImage(scaled_image)
            else:
                pixmap = QPixmap.fromImage(image)

            self.camera_label.setPixmap(pixmap)

    def update_prediction(self, predicted_class, probability):
        """Update prediction display and store in JSON"""
        confidence_text = f"{probability:.2%}"
        self.camera_prediction_label.setText(f"Prediction: {predicted_class} ({confidence_text})")

        # Store prediction in JSON file
        additional_data = {
            "model_classes": self.class_names,
            "camera_feed": True,
            "ui_component": "real_time_classification"
        }
        
        prediction_id = self.prediction_storage.save_prediction(
            predicted_class, 
            probability, 
            additional_data
        )
        
        if prediction_id:
            print(f"✓ Prediction stored with ID: {prediction_id}")

        # Update only the relevant dashboard based on actual class names
        if predicted_class.lower() in ['normal', 'good', 'ok', 'non_defective']:
            self.left_dashboard.update_classification(predicted_class, probability)
        else:
            # All other classes (defects) go to right dashboard
            self.right_dashboard.update_classification(predicted_class, probability)

    def update_motion_status(self, motion_detected):
        """Update UI or status based on motion detection"""
        if motion_detected:
            self.display_message("Motion detected - predictions paused")
        else:
            self.display_message("Camera stationary - ready for prediction")

    def closeEvent(self, event):
        """Handle application close event with enhanced cleanup"""
        try:
            # Stop camera feed if running
            if hasattr(self, 'camera_worker'):
                self.camera_worker.stop()
            if hasattr(self, 'camera_thread'):
                self.camera_thread.quit()
                self.camera_thread.wait(3000)  # Wait up to 3 seconds
            
            # Close prediction storage session and flush any pending data
            if hasattr(self, 'prediction_storage'):
                self.prediction_storage.close_session()
                
            # Save session state on close
            save_session_on_close("image_classification")
            
            print("✅ Application closed gracefully with all data saved")
            
        except Exception as e:
            print(f"⚠️  Warning during close: {e}")
        finally:
            event.accept()


def main():
    """Main function to run the SAKAR Vision AI Image Classification UI"""
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("SAKAR Vision AI")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("SAKAR Technologies")
        
        # Create and show the main window
        window = ImageClassiUI()
        window.show()
        
        print("🚀 SAKAR Vision AI - Image Classification Interface Started")
        print("=" * 70)
        print("📊 Features Available:")
        print("  • Real-time camera feed classification")
        print("  • Motion detection for stable predictions")
        print("  • Dual dashboard monitoring")
        if VISUAL_REPORTING_AVAILABLE:
            print("  • Professional visual reporting with charts and graphs")
        else:
            print("  • Visual reporting (install matplotlib and seaborn)")
        print("  • Session-based prediction tracking")
        print("  • JSON data export capabilities")
        print("  • Quality analytics and insights")
        print("=" * 70)
        
        if not VISUAL_REPORTING_AVAILABLE:
            print("📦 To enable visual reporting with charts:")
            print("   pip install matplotlib seaborn")
            print("   Then restart the application")
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
