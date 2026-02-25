"""
Compare YOLO Predictions vs Ground Truth Labels
Visualize OBB predictions on validation set
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

if __name__ == '__main__':
    print("=" * 70)
    print("YOLO Predictions vs Ground Truth Comparison")
    print("=" * 70)

    # Load best model
    MODEL_PATH = "runs/detect/drone_detect_v15/weights/best.pt"
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")

    # Get validation images
    VAL_IMAGE_DIR = Path("yolo_dataset/valid/images")
    VAL_LABEL_DIR = Path("yolo_dataset/valid/labels")
    
    val_images = sorted(list(VAL_IMAGE_DIR.glob("*.jpg")))[:6]  # First 6 images
    print(f"✅ Found {len(val_images)} validation images")

    # Create comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle("YOLO Predictions (Red) vs Ground Truth Labels (Green)", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, img_path in enumerate(val_images):
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Get label file
        label_name = img_path.stem.replace('_jpg.rf', '_jpg_rf') + '.txt'
        # Try different label name patterns
        possible_labels = [
            VAL_LABEL_DIR / f"{img_path.stem}.txt",
            VAL_LABEL_DIR / label_name,
            VAL_LABEL_DIR / f"{img_path.name.replace('.jpg', '.txt')}"
        ]
        
        label_file = None
        for lf in possible_labels:
            if lf.exists():
                label_file = lf
                break
        
        # Plot image
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{img_path.name}\n(Red=Prediction, Green=Ground Truth)", fontsize=10)
        axes[idx].axis('off')
        
        # Draw ground truth labels (GREEN)
        if label_file and label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, bw, bh = map(float, parts[:5])
                        
                        # Convert normalized to pixel coordinates
                        cx_px = cx * w
                        cy_px = cy * h
                        bw_px = bw * w
                        bh_px = bh * h
                        
                        # Draw bbox
                        x1 = cx_px - bw_px / 2
                        y1 = cy_px - bh_px / 2
                        
                        rect = patches.Rectangle(
                            (x1, y1), bw_px, bh_px,
                            linewidth=2, edgecolor='lime', facecolor='none',
                            label='Ground Truth'
                        )
                        axes[idx].add_patch(rect)
                        
                        # Draw center point
                        axes[idx].plot(cx_px, cy_px, 'go', markersize=8, label='GT Center')
        
        # Run YOLO prediction (RED)
        results = model.predict(img_path, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            for box in boxes:
                # Get normalized bbox
                xywhn = box.xywhn[0].cpu().numpy()  # [cx, cy, w, h] normalized
                conf = box.conf[0].item()
                
                # Convert to pixel coordinates
                cx_px = xywhn[0] * w
                cy_px = xywhn[1] * h
                bw_px = xywhn[2] * w
                bh_px = xywhn[3] * h
                
                # Draw bbox
                x1 = cx_px - bw_px / 2
                y1 = cy_px - bh_px / 2
                
                rect = patches.Rectangle(
                    (x1, y1), bw_px, bh_px,
                    linewidth=2, edgecolor='red', facecolor='none',
                    linestyle='--', label='Prediction'
                )
                axes[idx].add_patch(rect)
                
                # Draw center point
                axes[idx].plot(cx_px, cy_px, 'r*', markersize=10, label='Pred Center')
                
                # Add confidence text
                axes[idx].text(
                    x1, y1 - 10, f'conf: {conf:.2f}',
                    color='red', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        # Add info text
        num_gt = sum(1 for line in open(label_file, 'r')) if label_file and label_file.exists() else 0
        num_pred = len(boxes)
        
        info_text = f"GT: {num_gt} | Pred: {num_pred}"
        axes[idx].text(
            10, 30, info_text,
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lime', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Prediction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=8, label='GT Center'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Pred Center')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("yolo_vs_groundtruth_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: yolo_vs_groundtruth_comparison.png")

    # Calculate accuracy metrics
    print("\n" + "=" * 70)
    print("📊 Detailed Comparison Statistics")
    print("=" * 70)
    
    total_gt = 0
    total_pred = 0
    total_images = 0
    
    for img_path in val_images:
        label_name = img_path.stem.replace('_jpg.rf', '_jpg_rf') + '.txt'
        possible_labels = [
            VAL_LABEL_DIR / f"{img_path.stem}.txt",
            VAL_LABEL_DIR / label_name,
            VAL_LABEL_DIR / f"{img_path.name.replace('.jpg', '.txt')}"
        ]
        
        label_file = None
        for lf in possible_labels:
            if lf.exists():
                label_file = lf
                break
        
        results = model.predict(img_path, conf=0.25, verbose=False)
        num_pred = len(results[0].boxes)
        num_gt = sum(1 for line in open(label_file, 'r')) if label_file and label_file.exists() else 0
        
        total_gt += num_gt
        total_pred += num_pred
        total_images += 1
        
        print(f"\n{img_path.name}:")
        print(f"   Ground Truth: {num_gt} drones")
        print(f"   Predicted: {num_pred} drones")
        print(f"   Status: {'✅ Match' if num_gt == num_pred else '⚠️ Mismatch'}")

    print(f"\n{'='*70}")
    print(f"Total Ground Truth: {total_gt}")
    print(f"Total Predictions: {total_pred}")
    print(f"Average GT per image: {total_gt/total_images:.2f}")
    print(f"Average Pred per image: {total_pred/total_images:.2f}")
    
    print("\n✅ Comparison complete!")
