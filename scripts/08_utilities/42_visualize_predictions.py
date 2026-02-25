"""
Visualize YOLO detection and predictions on test set
Show bbox, confidence, and predicted GPS coordinates
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Constants
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees"""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad)*np.sin(lat2_rad) - np.sin(lat1_rad)*np.cos(lat2_rad)*np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

print("="*80)
print("🔍 Visualizing YOLO Detections and Predictions")
print("="*80)

# Load predictions
print("\n📂 Loading predictions...")
df_pred = pd.read_csv('test_predictions_residual.csv')
print(f"✅ Loaded {len(df_pred)} predictions")

# Load YOLO model
print("\n📂 Loading YOLO model...")
yolo_model = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')
print("✅ YOLO model loaded")

# Find test images
test_dir = Path('datasets/DATA_TEST')
if not test_dir.exists():
    print("❌ Test directory not found!")
    exit(1)

test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))
print(f"✅ Found {len(test_images)} test images")

# Create output directory
output_dir = Path('visualization_results')
output_dir.mkdir(exist_ok=True)

# Visualize random samples
print("\n" + "="*80)
print("🎨 Creating Visualizations")
print("="*80)

# Select 12 random images
np.random.seed(42)
sample_indices = np.random.choice(len(test_images), size=min(12, len(test_images)), replace=False)
sample_images = [test_images[i] for i in sorted(sample_indices)]

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

for idx, img_path in enumerate(sample_images):
    img_name = img_path.name
    
    # Read image
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Run YOLO detection
    results = yolo_model(img_path, verbose=False)
    
    # Get prediction
    pred_row = df_pred[df_pred['ImageName'] == img_name].iloc[0]
    pred_lat = pred_row['Latitude']
    pred_lon = pred_row['Longitude']
    pred_alt = pred_row['Altitude']
    
    # Calculate distance and bearing from camera
    distance = haversine_distance(CAMERA_LAT, CAMERA_LON, pred_lat, pred_lon)
    bearing = calculate_bearing(CAMERA_LAT, CAMERA_LON, pred_lat, pred_lon)
    
    # Draw detection on image
    img_annotated = img_rgb.copy()
    
    if len(results[0].boxes) > 0:
        # Get first detection
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        
        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw confidence
        label = f'Drone {conf:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_annotated, (x1, y1-label_h-10), (x1+label_w+10, y1), (0, 255, 0), -1)
        cv2.putText(img_annotated, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 2)
        
        # Draw center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(img_annotated, (cx, cy), 5, (255, 0, 0), -1)
        
        detection_status = "✓ Detected"
        status_color = 'green'
    else:
        detection_status = "✗ Not Detected"
        status_color = 'red'
    
    # Plot in subplot
    ax = fig.add_subplot(gs[idx // 4, idx % 4])
    ax.imshow(img_annotated)
    ax.axis('off')
    
    # Title with detection info
    title = f"{img_name}\n{detection_status}"
    ax.set_title(title, fontsize=10, fontweight='bold', color=status_color)
    
    # Add text annotations
    info_text = (
        f"Lat: {pred_lat:.6f}°\n"
        f"Lon: {pred_lon:.6f}°\n"
        f"Alt: {pred_alt:.1f} m\n"
        f"Dist: {distance:.1f} m\n"
        f"Bear: {bearing:.1f}°"
    )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('YOLO Drone Detection + GPS Predictions (Residual Learning)', 
            fontsize=16, fontweight='bold')
plt.savefig(output_dir / 'detection_visualization_grid.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'detection_visualization_grid.png'}")

# Create detailed visualization for top 6 images
print("\n📊 Creating detailed visualizations...")

fig2, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, img_path in enumerate(sample_images[:6]):
    img_name = img_path.name
    
    # Read image
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Run YOLO detection
    results = yolo_model(img_path, verbose=False)
    
    # Get prediction
    pred_row = df_pred[df_pred['ImageName'] == img_name].iloc[0]
    pred_lat = pred_row['Latitude']
    pred_lon = pred_row['Longitude']
    pred_alt = pred_row['Altitude']
    
    distance = haversine_distance(CAMERA_LAT, CAMERA_LON, pred_lat, pred_lon)
    bearing = calculate_bearing(CAMERA_LAT, CAMERA_LON, pred_lat, pred_lon)
    
    # Draw on image
    img_annotated = img_rgb.copy()
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        
        # Get bbox features
        cx_norm = ((x1 + x2) / 2) / w
        cy_norm = ((y1 + y2) / 2) / h
        w_norm = (x2 - x1) / w
        h_norm = (y2 - y1) / h
        area = w_norm * h_norm
        
        # Draw bbox
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # Draw center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(img_annotated, (cx, cy), 8, (255, 0, 0), -1)
        
        # Draw crosshair
        cv2.line(img_annotated, (cx-20, cy), (cx+20, cy), (255, 0, 0), 2)
        cv2.line(img_annotated, (cx, cy-20), (cx, cy+20), (255, 0, 0), 2)
    
    # Plot
    ax = axes[idx]
    ax.imshow(img_annotated)
    ax.axis('off')
    
    # Detailed title
    if len(results[0].boxes) > 0:
        title = (f"{img_name}\n"
                f"✓ Detected (conf={conf:.3f}) | "
                f"bbox area={area:.4f}")
    else:
        title = f"{img_name}\n✗ Not Detected"
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Detailed info box
    if len(results[0].boxes) > 0:
        info_text = (
            f"📍 GPS Prediction:\n"
            f"  Lat: {pred_lat:.6f}°\n"
            f"  Lon: {pred_lon:.6f}°\n"
            f"  Alt: {pred_alt:.2f} m\n\n"
            f"📏 From Camera:\n"
            f"  Distance: {distance:.2f} m\n"
            f"  Bearing: {bearing:.2f}°\n\n"
            f"📦 Bbox Features:\n"
            f"  Center: ({cx_norm:.3f}, {cy_norm:.3f})\n"
            f"  Size: {w_norm:.3f} × {h_norm:.3f}\n"
            f"  Area: {area:.4f}"
        )
    else:
        info_text = (
            f"📍 GPS Prediction:\n"
            f"  Lat: {pred_lat:.6f}°\n"
            f"  Lon: {pred_lon:.6f}°\n"
            f"  Alt: {pred_alt:.2f} m\n\n"
            f"⚠️ No detection\n"
            f"Using default bbox"
        )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

plt.suptitle('Detailed Detection Analysis - Residual Learning Predictions', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'detection_detailed.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'detection_detailed.png'}")

# Statistics
print("\n" + "="*80)
print("📊 Detection Statistics")
print("="*80)

detection_count = 0
total_confidence = 0

for img_path in test_images:
    results = yolo_model(img_path, verbose=False)
    if len(results[0].boxes) > 0:
        detection_count += 1
        total_confidence += float(results[0].boxes[0].conf[0])

detection_rate = detection_count / len(test_images) * 100
avg_confidence = total_confidence / detection_count if detection_count > 0 else 0

print(f"\nTotal images: {len(test_images)}")
print(f"Detected: {detection_count} ({detection_rate:.1f}%)")
print(f"Not detected: {len(test_images) - detection_count}")
print(f"Average confidence: {avg_confidence:.3f}")

# Prediction statistics
print(f"\n📍 Prediction Statistics:")
print(f"   Latitude range: {df_pred['Latitude'].min():.6f}° - {df_pred['Latitude'].max():.6f}°")
print(f"   Longitude range: {df_pred['Longitude'].min():.6f}° - {df_pred['Longitude'].max():.6f}°")
print(f"   Altitude range: {df_pred['Altitude'].min():.2f} - {df_pred['Altitude'].max():.2f} m")

# Calculate distances and bearings for all
distances = []
bearings = []

for _, row in df_pred.iterrows():
    dist = haversine_distance(CAMERA_LAT, CAMERA_LON, row['Latitude'], row['Longitude'])
    bear = calculate_bearing(CAMERA_LAT, CAMERA_LON, row['Latitude'], row['Longitude'])
    distances.append(dist)
    bearings.append(bear)

print(f"\n📏 From Camera:")
print(f"   Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f} m")
print(f"            (range: {np.min(distances):.2f} - {np.max(distances):.2f} m)")
print(f"   Bearing: {np.mean(bearings):.2f} ± {np.std(bearings):.2f}°")
print(f"            (range: {np.min(bearings):.2f} - {np.max(bearings):.2f}°)")

# Create distribution plots
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))

# Distance distribution
axes3[0, 0].hist(distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes3[0, 0].axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.1f}m')
axes3[0, 0].set_xlabel('Distance from Camera (m)', fontsize=12)
axes3[0, 0].set_ylabel('Frequency', fontsize=12)
axes3[0, 0].set_title('Distance Distribution', fontsize=14, fontweight='bold')
axes3[0, 0].legend()
axes3[0, 0].grid(True, alpha=0.3)

# Bearing distribution
axes3[0, 1].hist(bearings, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
axes3[0, 1].axvline(np.mean(bearings), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(bearings):.1f}°')
axes3[0, 1].set_xlabel('Bearing from Camera (°)', fontsize=12)
axes3[0, 1].set_ylabel('Frequency', fontsize=12)
axes3[0, 1].set_title('Bearing Distribution', fontsize=14, fontweight='bold')
axes3[0, 1].legend()
axes3[0, 1].grid(True, alpha=0.3)

# Altitude distribution
axes3[1, 0].hist(df_pred['Altitude'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes3[1, 0].axvline(df_pred['Altitude'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {df_pred["Altitude"].mean():.1f}m')
axes3[1, 0].set_xlabel('Altitude (m)', fontsize=12)
axes3[1, 0].set_ylabel('Frequency', fontsize=12)
axes3[1, 0].set_title('Altitude Distribution', fontsize=14, fontweight='bold')
axes3[1, 0].legend()
axes3[1, 0].grid(True, alpha=0.3)

# 2D scatter: Distance vs Bearing
scatter = axes3[1, 1].scatter(bearings, distances, c=df_pred['Altitude'], 
                             cmap='viridis', s=50, alpha=0.6, edgecolor='black')
axes3[1, 1].set_xlabel('Bearing from Camera (°)', fontsize=12)
axes3[1, 1].set_ylabel('Distance from Camera (m)', fontsize=12)
axes3[1, 1].set_title('Distance vs Bearing (colored by Altitude)', fontsize=14, fontweight='bold')
axes3[1, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes3[1, 1])
cbar.set_label('Altitude (m)', fontsize=10)

plt.suptitle('Prediction Distributions - Test Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'prediction_distributions.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'prediction_distributions.png'}")

plt.close('all')

print("\n" + "="*80)
print("✅ Visualization Complete!")
print("="*80)
print(f"\nGenerated files in '{output_dir}/':")
print("  1. detection_visualization_grid.png  - 12 sample detections")
print("  2. detection_detailed.png            - 6 detailed analyses")
print("  3. prediction_distributions.png      - Statistical distributions")
print("\n🎉 All visualizations saved successfully!")
print("="*80)
