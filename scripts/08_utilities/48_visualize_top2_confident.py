"""
Visualize TOP-2 Confident Predictions
Show distribution, multi-drone handling, and sample detections
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import cv2

print("="*80)
print("🎨 Visualizing TOP-2 Confident Predictions")
print("="*80)

# Load predictions
df = pd.read_csv('test_predictions_top2_confident_detailed.csv')
print(f"\n📂 Loaded {len(df)} predictions")

# Statistics
print("\n📊 Basic Statistics:")
print(f"  Total images: {df['ImageName'].nunique()}")
print(f"  Total predictions: {len(df)}")
print(f"  Avg drones/image: {len(df)/df['ImageName'].nunique():.2f}")

# Count drones per image
drones_per_image = df.groupby('ImageName').size()
print(f"\n🔢 Drones per image:")
print(f"  1 drone:  {(drones_per_image == 1).sum()} images ({(drones_per_image == 1).sum()/len(drones_per_image)*100:.1f}%)")
print(f"  2 drones: {(drones_per_image == 2).sum()} images ({(drones_per_image == 2).sum()/len(drones_per_image)*100:.1f}%)")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Confidence Distribution
ax1 = plt.subplot(2, 4, 1)
sns.histplot(df['Confidence'], bins=30, kde=True, ax=ax1, color='skyblue', edgecolor='black')
ax1.axvline(df['Confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Confidence"].mean():.3f}')
ax1.axvline(df['Confidence'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Confidence"].median():.3f}')
ax1.set_title('Confidence Distribution (TOP-2)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Confidence', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Latitude Distribution
ax2 = plt.subplot(2, 4, 2)
sns.histplot(df['Latitude'], bins=30, kde=True, ax=ax2, color='lightcoral', edgecolor='black')
ax2.axvline(df['Latitude'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Latitude"].mean():.6f}')
ax2.set_title('Latitude Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Latitude', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Longitude Distribution
ax3 = plt.subplot(2, 4, 3)
sns.histplot(df['Longitude'], bins=30, kde=True, ax=ax3, color='lightgreen', edgecolor='black')
ax3.axvline(df['Longitude'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Longitude"].mean():.6f}')
ax3.set_title('Longitude Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Longitude', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Altitude Distribution
ax4 = plt.subplot(2, 4, 4)
sns.histplot(df['Altitude'], bins=30, kde=True, ax=ax4, color='gold', edgecolor='black')
ax4.axvline(df['Altitude'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Altitude"].mean():.2f}m')
ax4.set_title('Altitude Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Altitude (m)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Drones per Image
ax5 = plt.subplot(2, 4, 5)
drones_count = drones_per_image.value_counts().sort_index()
colors_bar = ['#FF6B6B', '#4ECDC4']
bars = ax5.bar(drones_count.index, drones_count.values, color=colors_bar, edgecolor='black', linewidth=2)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(drones_per_image)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax5.set_title('Drones per Image Distribution', fontsize=14, fontweight='bold')
ax5.set_xlabel('Number of Drones', fontsize=12)
ax5.set_ylabel('Number of Images', fontsize=12)
ax5.set_xticks([1, 2])
ax5.grid(True, alpha=0.3, axis='y')

# 6. Confidence by DroneID
ax6 = plt.subplot(2, 4, 6)
drone1_conf = df[df['DroneID'] == 1]['Confidence']
drone2_conf = df[df['DroneID'] == 2]['Confidence']
bp = ax6.boxplot([drone1_conf, drone2_conf], labels=['Drone 1\n(Highest)', 'Drone 2\n(2nd Highest)'],
                  patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax6.set_title('Confidence by Drone Rank', fontsize=14, fontweight='bold')
ax6.set_ylabel('Confidence', fontsize=12)
ax6.grid(True, alpha=0.3, axis='y')
ax6.text(1, drone1_conf.mean(), f'{drone1_conf.mean():.3f}', ha='center', va='bottom', fontweight='bold')
ax6.text(2, drone2_conf.mean(), f'{drone2_conf.mean():.3f}', ha='center', va='bottom', fontweight='bold')

# 7. BboxArea Distribution
ax7 = plt.subplot(2, 4, 7)
sns.histplot(df['BboxArea'], bins=30, kde=True, ax=ax7, color='plum', edgecolor='black')
ax7.axvline(df['BboxArea'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["BboxArea"].mean():.0f}px²')
ax7.set_title('Bounding Box Area Distribution', fontsize=14, fontweight='bold')
ax7.set_xlabel('Bbox Area (pixels²)', fontsize=12)
ax7.set_ylabel('Count', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Confidence vs BboxArea
ax8 = plt.subplot(2, 4, 8)
scatter = ax8.scatter(df['BboxArea'], df['Confidence'], c=df['DroneID'], 
                     cmap='coolwarm', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax8.set_title('Confidence vs Bbox Area', fontsize=14, fontweight='bold')
ax8.set_xlabel('Bbox Area (pixels²)', fontsize=12)
ax8.set_ylabel('Confidence', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('DroneID', fontsize=11)
ax8.grid(True, alpha=0.3)

# Add correlation text
corr = df['BboxArea'].corr(df['Confidence'])
ax8.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax8.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('visualization_results/top2_confident_statistics.png', dpi=300, bbox_inches='tight')
print(f"\n💾 Saved: visualization_results/top2_confident_statistics.png")

# ========== PART 2: Sample Detections Visualization ==========
print("\n🖼️  Creating sample detections visualization...")

# Load YOLO
yolo = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

# Select diverse samples
test_dir = Path('datasets/DATA_TEST')
sample_images = [
    'img_0001.jpg',  # Multiple drones
    'img_0006.jpg',  # 2 drones
    'img_0009.jpg',  # 2 drones
    'img_0015.jpg',  # Check detection
    'img_0050.jpg',  # Middle range
    'img_0100.jpg',  # Different position
    'img_0150.jpg',  # Different position
    'img_0200.jpg',  # Different position
    'img_0250.jpg',  # Different position
]

fig2, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for idx, img_name in enumerate(sample_images):
    if idx >= len(axes):
        break
    
    img_path = test_dir / img_name
    if not img_path.exists():
        continue
    
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get YOLO detections
    results = yolo(img_path, verbose=False, conf=0.25)
    boxes = results[0].boxes
    
    # Get predictions for this image
    img_preds = df[df['ImageName'] == img_name].sort_values('DroneID')
    
    # Draw detections
    for box_idx, box in enumerate(boxes[:2], start=1):  # Only top-2
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Color by rank
        color = (0, 255, 0) if box_idx == 1 else (255, 165, 0)  # Green for #1, Orange for #2
        
        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Get prediction for this drone
        if box_idx <= len(img_preds):
            pred = img_preds.iloc[box_idx-1]
            label = f"#{box_idx} | Conf:{conf:.3f}\nLat:{pred['Latitude']:.6f}\nAlt:{pred['Altitude']:.1f}m"
        else:
            label = f"#{box_idx} | Conf:{conf:.3f}"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label.split('\n')[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-60), (x1+150, y1), color, -1)
        
        # Draw text (multi-line)
        y_offset = y1 - 10
        for line in label.split('\n'):
            cv2.putText(img, line, (x1+5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 15
    
    # Show image
    axes[idx].imshow(img)
    axes[idx].set_title(f"{img_name}\n{len(img_preds)} drones detected (TOP-2)", 
                       fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('visualization_results/top2_confident_detections.png', dpi=200, bbox_inches='tight')
print(f"💾 Saved: visualization_results/top2_confident_detections.png")

# ========== PART 3: GPS Heatmap ==========
print("\n🗺️  Creating GPS heatmap...")

fig3, ax = plt.subplots(1, 1, figsize=(12, 10))

# Create 2D histogram (heatmap)
h = ax.hist2d(df['Longitude'], df['Latitude'], bins=50, cmap='YlOrRd', cmin=1)
plt.colorbar(h[3], ax=ax, label='Number of Predictions')

# Add camera position
camera_lon = 101.172718
camera_lat = 14.304750
ax.scatter(camera_lon, camera_lat, c='blue', s=300, marker='^', 
          edgecolors='white', linewidth=2, label='Camera', zorder=5)

# Add mean position
mean_lon = df['Longitude'].mean()
mean_lat = df['Latitude'].mean()
ax.scatter(mean_lon, mean_lat, c='green', s=200, marker='*',
          edgecolors='white', linewidth=2, label='Mean Prediction', zorder=5)

ax.set_title('GPS Predictions Heatmap (TOP-2 Confident)', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_results/top2_confident_gps_heatmap.png', dpi=300, bbox_inches='tight')
print(f"💾 Saved: visualization_results/top2_confident_gps_heatmap.png")

# ========== Summary Statistics ==========
print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

print(f"\n🎯 Overall:")
print(f"  Total images: {df['ImageName'].nunique()}")
print(f"  Total predictions: {len(df)}")
print(f"  Avg drones/image: {len(df)/df['ImageName'].nunique():.2f}")

print(f"\n🔢 Drones per image:")
for n_drones, count in drones_count.items():
    print(f"  {n_drones} drone(s): {count} images ({count/len(drones_per_image)*100:.1f}%)")

print(f"\n📈 Confidence:")
print(f"  Mean: {df['Confidence'].mean():.3f}")
print(f"  Median: {df['Confidence'].median():.3f}")
print(f"  Std: {df['Confidence'].std():.3f}")
print(f"  Min: {df['Confidence'].min():.3f}")
print(f"  Max: {df['Confidence'].max():.3f}")

print(f"\n📊 By Drone Rank:")
print(f"  Drone #1 (Highest conf): {drone1_conf.mean():.3f} ± {drone1_conf.std():.3f}")
print(f"  Drone #2 (2nd highest):  {drone2_conf.mean():.3f} ± {drone2_conf.std():.3f}")
print(f"  Difference: {drone1_conf.mean() - drone2_conf.mean():.3f}")

print(f"\n🌍 GPS Coordinates:")
print(f"  Latitude:  {df['Latitude'].mean():.6f} ± {df['Latitude'].std():.6f}")
print(f"  Longitude: {df['Longitude'].mean():.6f} ± {df['Longitude'].std():.6f}")
print(f"  Altitude:  {df['Altitude'].mean():.2f} ± {df['Altitude'].std():.2f} m")

print(f"\n📦 Bbox Area:")
print(f"  Mean: {df['BboxArea'].mean():.0f} px²")
print(f"  Min: {df['BboxArea'].min():.0f} px²")
print(f"  Max: {df['BboxArea'].max():.0f} px²")

print(f"\n🔗 Correlation (Confidence vs BboxArea): {corr:.3f}")

print("\n" + "="*80)
print("✅ Visualization Complete!")
print("="*80)
print("\n📁 Files created:")
print("  1. visualization_results/top2_confident_statistics.png")
print("  2. visualization_results/top2_confident_detections.png")
print("  3. visualization_results/top2_confident_gps_heatmap.png")
