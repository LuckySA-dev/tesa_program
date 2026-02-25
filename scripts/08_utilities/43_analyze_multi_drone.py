"""
Analyze multi-drone detections and handle cases with multiple drones
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

print("="*80)
print("🔍 Analyzing Multi-Drone Detection")
print("="*80)

# Load YOLO model
print("\n📂 Loading YOLO model...")
yolo_model = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

# Find test images
test_dir = Path('datasets/DATA_TEST')
test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))
print(f"✅ Found {len(test_images)} test images")

# Analyze all images
print("\n🔍 Analyzing detections...")
detection_stats = {
    '0_drones': [],
    '1_drone': [],
    '2_drones': [],
    '3+_drones': []
}

for img_path in test_images:
    results = yolo_model(img_path, verbose=False)
    num_detections = len(results[0].boxes)
    
    if num_detections == 0:
        detection_stats['0_drones'].append(img_path.name)
    elif num_detections == 1:
        detection_stats['1_drone'].append(img_path.name)
    elif num_detections == 2:
        detection_stats['2_drones'].append(img_path.name)
    else:
        detection_stats['3+_drones'].append(img_path.name)

print("\n📊 Detection Statistics:")
print(f"  0 drones detected: {len(detection_stats['0_drones'])} images")
print(f"  1 drone detected:  {len(detection_stats['1_drone'])} images")
print(f"  2 drones detected: {len(detection_stats['2_drones'])} images")
print(f"  3+ drones detected: {len(detection_stats['3+_drones'])} images")

# Visualize images with 2+ drones
if len(detection_stats['2_drones']) > 0 or len(detection_stats['3+_drones']) > 0:
    print("\n" + "="*80)
    print("🎨 Visualizing Multi-Drone Detections")
    print("="*80)
    
    multi_drone_images = detection_stats['2_drones'] + detection_stats['3+_drones']
    print(f"\nFound {len(multi_drone_images)} images with multiple drones")
    
    # Show first 12 multi-drone images
    sample_size = min(12, len(multi_drone_images))
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx in range(sample_size):
        img_name = multi_drone_images[idx]
        img_path = test_dir / img_name
        
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = yolo_model(img_path, verbose=False)
        boxes = results[0].boxes
        
        # Draw all detections
        img_annotated = img_rgb.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            # Different color for each detection
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            color = colors[i % len(colors)]
            
            # Draw bbox
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f'Drone {i+1}: {conf:.2f}'
            cv2.putText(img_annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img_annotated, (cx, cy), 5, color, -1)
        
        # Plot
        ax = axes[idx]
        ax.imshow(img_annotated)
        ax.axis('off')
        ax.set_title(f"{img_name}\n{len(boxes)} drones detected", 
                    fontsize=10, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(sample_size, 12):
        axes[idx].axis('off')
    
    plt.suptitle('Multi-Drone Detections', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_results/multi_drone_detections.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: visualization_results/multi_drone_detections.png")
    plt.close()

print("\n" + "="*80)
print("💡 Recommendation for Multi-Drone Cases")
print("="*80)

print("""
สำหรับภาพที่มีหลายโดรน มีตัวเลือก:

1. ใช้โดรนที่มี confidence สูงสุด (วิธีปัจจุบัน)
   ✅ ง่าย, ทำอยู่แล้ว
   ❌ อาจพลาดโดรนที่เป็นเป้าหมายจริง

2. ใช้โดรนที่ใกล้กล้องที่สุด (bbox ใหญ่สุด)
   ✅ สมเหตุสมผล (ใกล้ = เห็นชัด = เป้าหมาย?)
   ⚠️  ต้องปรับ code

3. ใช้โดรนที่อยู่กึ่งกลางภาพ
   ✅ photographer มักเล็งกล้องไปที่เป้าหมาย
   ⚠️  ต้องปรับ code

4. Predict ทุกโดรนแล้วเฉลี่ย
   ⚠️  อาจทำให้ผิดพลาดมากขึ้น

5. Train โมเดลแยก main/secondary drone
   ❌ ต้อง re-label data

แนะนำ: ลองวิธีที่ 2 (bbox ใหญ่สุด) หรือ 3 (ใกล้กึ่งกลางสุด)
""")

print("\n" + "="*80)
