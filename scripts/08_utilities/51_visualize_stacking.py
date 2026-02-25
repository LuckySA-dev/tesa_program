"""
Visualize Stacking Ensemble Predictions
Show sample detections with bounding boxes and predicted GPS coordinates
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import cv2

print("="*80)
print("🎨 Visualizing Stacking Ensemble Predictions")
print("="*80)

# Load predictions
df = pd.read_csv('test_predictions_stacking_detailed.csv')
print(f"\n📂 Loaded {len(df)} predictions from {df['ImageName'].nunique()} images")

# Load YOLO
yolo = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

# Statistics
print("\n📊 Statistics:")
drones_per_image = df.groupby('ImageName').size()
print(f"  1 drone:  {(drones_per_image == 1).sum()} images")
print(f"  2 drones: {(drones_per_image == 2).sum()} images")
print(f"  Avg drones/image: {len(df)/df['ImageName'].nunique():.2f}")

# ============================================================================
# PART 1: Statistical Visualizations
# ============================================================================
fig1 = plt.figure(figsize=(20, 12))

# 1. Confidence Distribution
ax1 = plt.subplot(2, 4, 1)
sns.histplot(df['Confidence'], bins=30, kde=True, ax=ax1, color='skyblue', edgecolor='black')
ax1.axvline(df['Confidence'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["Confidence"].mean():.3f}')
ax1.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Confidence', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Latitude Distribution
ax2 = plt.subplot(2, 4, 2)
sns.histplot(df['Latitude'], bins=30, kde=True, ax=ax2, color='lightcoral', edgecolor='black')
ax2.axvline(df['Latitude'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["Latitude"].mean():.6f}')
ax2.set_title('Latitude Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Latitude', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Longitude Distribution
ax3 = plt.subplot(2, 4, 3)
sns.histplot(df['Longitude'], bins=30, kde=True, ax=ax3, color='lightgreen', edgecolor='black')
ax3.axvline(df['Longitude'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["Longitude"].mean():.6f}')
ax3.set_title('Longitude Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Longitude', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Altitude Distribution
ax4 = plt.subplot(2, 4, 4)
sns.histplot(df['Altitude'], bins=30, kde=True, ax=ax4, color='gold', edgecolor='black')
ax4.axvline(df['Altitude'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["Altitude"].mean():.2f}m')
ax4.set_title('Altitude Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Altitude (m)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Drones per Image
ax5 = plt.subplot(2, 4, 5)
counts = drones_per_image.value_counts().sort_index()
bars = ax5.bar(counts.index, counts.values, color=['#FF6B6B', '#4ECDC4'], 
              edgecolor='black', linewidth=2)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(drones_per_image)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax5.set_title('Drones per Image', fontsize=14, fontweight='bold')
ax5.set_xlabel('Number of Drones', fontsize=12)
ax5.set_ylabel('Number of Images', fontsize=12)
ax5.set_xticks([1, 2])
ax5.grid(True, alpha=0.3, axis='y')

# 6. Confidence by Drone Rank
ax6 = plt.subplot(2, 4, 6)
drone1 = df[df['DroneID'] == 1]['Confidence']
drone2 = df[df['DroneID'] == 2]['Confidence']
bp = ax6.boxplot([drone1, drone2], labels=['Drone #1', 'Drone #2'],
                 patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax6.set_title('Confidence by Drone Rank', fontsize=14, fontweight='bold')
ax6.set_ylabel('Confidence', fontsize=12)
ax6.grid(True, alpha=0.3, axis='y')
ax6.text(1, drone1.mean(), f'{drone1.mean():.3f}', ha='center', va='bottom', fontweight='bold')
ax6.text(2, drone2.mean(), f'{drone2.mean():.3f}', ha='center', va='bottom', fontweight='bold')

# 7. Altitude by Drone Rank
ax7 = plt.subplot(2, 4, 7)
alt1 = df[df['DroneID'] == 1]['Altitude']
alt2 = df[df['DroneID'] == 2]['Altitude']
bp2 = ax7.boxplot([alt1, alt2], labels=['Drone #1', 'Drone #2'],
                  patch_artist=True, showmeans=True)
bp2['boxes'][0].set_facecolor('lightgreen')
bp2['boxes'][1].set_facecolor('plum')
ax7.set_title('Altitude by Drone Rank', fontsize=14, fontweight='bold')
ax7.set_ylabel('Altitude (m)', fontsize=12)
ax7.grid(True, alpha=0.3, axis='y')

# 8. GPS Heatmap
ax8 = plt.subplot(2, 4, 8)
h = ax8.hist2d(df['Longitude'], df['Latitude'], bins=40, cmap='YlOrRd', cmin=1)
plt.colorbar(h[3], ax=ax8, label='Count')
ax8.scatter(101.172718, 14.304750, c='blue', s=300, marker='^', 
           edgecolors='white', linewidth=2, label='Camera', zorder=5)
ax8.set_title('GPS Predictions Heatmap', fontsize=14, fontweight='bold')
ax8.set_xlabel('Longitude', fontsize=12)
ax8.set_ylabel('Latitude', fontsize=12)
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)

plt.tight_layout()
Path('visualization_results').mkdir(exist_ok=True)
plt.savefig('visualization_results/stacking_statistics.png', dpi=300, bbox_inches='tight')
print(f"\n💾 Saved: visualization_results/stacking_statistics.png")
plt.close()

# ============================================================================
# PART 2: Sample Detection Visualizations
# ============================================================================
print("\n🖼️  Creating detection visualizations...")

test_dir = Path('datasets/DATA_TEST')

# Select diverse samples (mix of 1-drone and 2-drone images)
sample_images = [
    'img_0001.jpg',  # 2 drones
    'img_0006.jpg',  # 2 drones
    'img_0015.jpg',  # 1 or 2
    'img_0050.jpg',  # Check
    'img_0100.jpg',  # Check
    'img_0150.jpg',  # Check
    'img_0200.jpg',  # Check
    'img_0250.jpg',  # Check
    'img_0264.jpg',  # Last image
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
    
    # Draw TOP-2 detections only
    for box_idx, box in enumerate(boxes[:2], start=1):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Color by rank
        color = (0, 255, 0) if box_idx == 1 else (255, 165, 0)  # Green/#1, Orange/#2
        
        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Get prediction
        if box_idx <= len(img_preds):
            pred = img_preds.iloc[box_idx-1]
            label_lines = [
                f"Drone #{box_idx} | Conf:{conf:.3f}",
                f"Lat: {pred['Latitude']:.6f}",
                f"Lon: {pred['Longitude']:.6f}",
                f"Alt: {pred['Altitude']:.1f}m"
            ]
        else:
            label_lines = [f"Drone #{box_idx} | Conf:{conf:.3f}"]
        
        # Draw label background
        label_height = 15 * len(label_lines) + 5
        cv2.rectangle(img, (x1, y1-label_height-5), (x1+200, y1), color, -1)
        
        # Draw text (multi-line)
        y_offset = y1 - 5
        for line in label_lines:
            cv2.putText(img, line, (x1+5, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 15
    
    # Show image
    axes[idx].imshow(img)
    axes[idx].set_title(f"{img_name}\n{len(img_preds)} drones (Stacking Ensemble)",
                       fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('visualization_results/stacking_detections.png', dpi=200, bbox_inches='tight')
print(f"💾 Saved: visualization_results/stacking_detections.png")
plt.close()

# ============================================================================
# PART 3: Detailed Grid (More Samples)
# ============================================================================
print("\n🖼️  Creating detailed detection grid (20 samples)...")

# Get 20 random samples
all_images = df['ImageName'].unique()
np.random.seed(42)
random_samples = np.random.choice(all_images, min(20, len(all_images)), replace=False)

fig3, axes = plt.subplots(4, 5, figsize=(25, 20))
axes = axes.flatten()

for idx, img_name in enumerate(random_samples):
    img_path = test_dir / img_name
    if not img_path.exists():
        continue
    
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get YOLO detections
    results = yolo(img_path, verbose=False, conf=0.25)
    boxes = results[0].boxes
    
    # Get predictions
    img_preds = df[df['ImageName'] == img_name].sort_values('DroneID')
    
    # Draw detections
    for box_idx, box in enumerate(boxes[:2], start=1):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, xyxy)
        
        color = (0, 255, 0) if box_idx == 1 else (255, 140, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Simple label
        if box_idx <= len(img_preds):
            pred = img_preds.iloc[box_idx-1]
            label = f"#{box_idx}:{conf:.2f}"
            cv2.rectangle(img, (x1, y1-25), (x1+80, y1), color, -1)
            cv2.putText(img, label, (x1+3, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    axes[idx].imshow(img)
    axes[idx].set_title(f"{img_name} ({len(img_preds)}D)", fontsize=9, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('visualization_results/stacking_grid_20samples.png', dpi=150, bbox_inches='tight')
print(f"💾 Saved: visualization_results/stacking_grid_20samples.png")
plt.close()

# ============================================================================
# PART 4: Compare Top-2 Predictions
# ============================================================================
print("\n📊 Analyzing TOP-2 predictions...")

# Images with 2 drones
df_2drones = df[df.groupby('ImageName')['ImageName'].transform('count') == 2]
img_2drones = df_2drones['ImageName'].unique()

print(f"\nImages with 2 drones: {len(img_2drones)}")

# Analyze differences between Drone #1 and #2
drone1_data = df[df['DroneID'] == 1]
drone2_data = df[df['DroneID'] == 2]

print(f"\n🔍 Drone #1 vs Drone #2 Analysis:")
print(f"  Drone #1:")
print(f"    Confidence: {drone1_data['Confidence'].mean():.3f} ± {drone1_data['Confidence'].std():.3f}")
print(f"    Latitude:   {drone1_data['Latitude'].mean():.6f} ± {drone1_data['Latitude'].std():.6f}")
print(f"    Longitude:  {drone1_data['Longitude'].mean():.6f} ± {drone1_data['Longitude'].std():.6f}")
print(f"    Altitude:   {drone1_data['Altitude'].mean():.2f} ± {drone1_data['Altitude'].std():.2f}")

print(f"  Drone #2:")
print(f"    Confidence: {drone2_data['Confidence'].mean():.3f} ± {drone2_data['Confidence'].std():.3f}")
print(f"    Latitude:   {drone2_data['Latitude'].mean():.6f} ± {drone2_data['Latitude'].std():.6f}")
print(f"    Longitude:  {drone2_data['Longitude'].mean():.6f} ± {drone2_data['Longitude'].std():.6f}")
print(f"    Altitude:   {drone2_data['Altitude'].mean():.2f} ± {drone2_data['Altitude'].std():.2f}")

print(f"\n  Differences:")
print(f"    Confidence: {drone1_data['Confidence'].mean() - drone2_data['Confidence'].mean():.3f}")
print(f"    Altitude:   {abs(drone1_data['Altitude'].mean() - drone2_data['Altitude'].mean()):.2f}m")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

print(f"\n🎯 Prediction Statistics:")
print(f"  Total predictions: {len(df)}")
print(f"  Total images: {df['ImageName'].nunique()}")
print(f"  Avg drones/image: {len(df)/df['ImageName'].nunique():.2f}")
print(f"  1-drone images: {(drones_per_image == 1).sum()}")
print(f"  2-drone images: {(drones_per_image == 2).sum()}")

print(f"\n📈 Overall Statistics:")
print(f"  Confidence: {df['Confidence'].mean():.3f} ± {df['Confidence'].std():.3f}")
print(f"  Latitude:   {df['Latitude'].mean():.6f} ± {df['Latitude'].std():.6f}")
print(f"  Longitude:  {df['Longitude'].mean():.6f} ± {df['Longitude'].std():.6f}")
print(f"  Altitude:   {df['Altitude'].mean():.2f} ± {df['Altitude'].std():.2f}m")

print(f"\n📁 Files Created:")
print(f"  1. visualization_results/stacking_statistics.png (8 statistical plots)")
print(f"  2. visualization_results/stacking_detections.png (9 sample detections)")
print(f"  3. visualization_results/stacking_grid_20samples.png (20 random samples)")

print("\n" + "="*80)
print("✅ Visualization Complete!")
print("="*80)
