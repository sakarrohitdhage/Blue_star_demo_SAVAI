#!/usr/bin/env python3
"""
Test Visual Export Functionality
Demonstrates the visual export system with sample defective/non-defective data
"""

import sys
import os
from datetime import datetime

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from visual_data_export import VisualDataExporter
    print("✓ Visual data export module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import visual data export: {e}")
    sys.exit(1)

def test_visual_export():
    """Test the visual export functionality with sample data"""
    print("\n🧪 Testing Visual Export Functionality")
    print("=" * 50)
    
    # Create sample data exactly like your example: 1, 2, 3, 4, 5 as defective, non-defective, non-defective, defective, defective
    sample_predictions = [
        {
            "predicted_class": "defective",
            "confidence": 0.95,
            "timestamp": "2025-07-10T10:00:00",
            "id": "test_1"
        },
        {
            "predicted_class": "non_defective", 
            "confidence": 0.88,
            "timestamp": "2025-07-10T10:01:00",
            "id": "test_2"
        },
        {
            "predicted_class": "non_defective",
            "confidence": 0.92,
            "timestamp": "2025-07-10T10:02:00", 
            "id": "test_3"
        },
        {
            "predicted_class": "defective",
            "confidence": 0.85,
            "timestamp": "2025-07-10T10:03:00",
            "id": "test_4"
        },
        {
            "predicted_class": "defective",
            "confidence": 0.90,
            "timestamp": "2025-07-10T10:04:00",
            "id": "test_5"
        }
    ]
    
    print(f"📊 Sample data: {len(sample_predictions)} predictions")
    print("   Sequence: 1=D, 2=N, 3=N, 4=D, 5=D")
    
    # Initialize exporter
    exporter = VisualDataExporter()
    
    # Test 1: Create visual summary (PDF with charts)
    print("\n📈 Test 1: Creating visual summary (PDF)...")
    try:
        pdf_path = exporter.create_visual_summary(sample_predictions, "test_visual_summary.pdf")
        if pdf_path:
            print(f"✓ PDF visual summary created: {pdf_path}")
        else:
            print("✗ Failed to create PDF visual summary")
    except Exception as e:
        print(f"✗ Error creating PDF: {e}")
        print("   Falling back to text export...")
    
    # Test 2: Create simple text export
    print("\n📝 Test 2: Creating simple text export...")
    try:
        text_path = exporter.create_simple_text_export(sample_predictions, "test_simple_export.txt")
        if text_path:
            print(f"✓ Simple text export created: {text_path}")
            
            # Show the content
            with open(text_path, 'r') as f:
                content = f.read()
                print("\n📄 Preview of exported content:")
                print("-" * 40)
                # Show first few lines
                lines = content.split('\n')
                for i, line in enumerate(lines[:20]):  # Show first 20 lines
                    print(line)
                if len(lines) > 20:
                    print("... (truncated)")
        else:
            print("✗ Failed to create simple text export")
    except Exception as e:
        print(f"✗ Error creating text export: {e}")
    
    # Test 3: Show what the visual sequence looks like
    print("\n🎯 Test 3: Visual sequence demonstration")
    classes = [p["predicted_class"] for p in sample_predictions]
    
    # Create visual sequence string
    visual_sequence = []
    ascii_visual = []
    
    for i, cls in enumerate(classes, 1):
        if 'defective' in cls.lower() and 'non' not in cls.lower():
            visual_sequence.append(f"{i}=D")
            ascii_visual.append("[X]")
        else:
            visual_sequence.append(f"{i}=N")
            ascii_visual.append("[O]")
    
    print(f"   Visual Sequence: {', '.join(visual_sequence)}")
    print(f"   ASCII Visual:    {' '.join(ascii_visual)}")
    print("   Legend: D=Defective, N=Non-Defective, [X]=Defective, [O]=Non-Defective")
    
    # Statistics
    defective_count = sum(1 for c in classes if 'defective' in c.lower() and 'non' not in c.lower())
    non_defective_count = len(classes) - defective_count
    
    print(f"\n📊 Statistics:")
    print(f"   Total predictions: {len(classes)}")
    print(f"   Defective: {defective_count} ({defective_count/len(classes)*100:.1f}%)")
    print(f"   Non-Defective: {non_defective_count} ({non_defective_count/len(classes)*100:.1f}%)")
    
    print(f"\n✅ Test completed! Check the '{exporter.output_dir}' folder for exported files.")

if __name__ == "__main__":
    test_visual_export()