"""
Phase 2 - Task 2.3: Evaluate YOLO Model
Validate trained model and analyze performance
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 2 - Task 2.3: Evaluate YOLO Model")
    print("=" * 70)

    # Load best model
    MODEL_PATH = "runs/detect/drone_detect_v15/weights/best.pt"

    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Model not found: {MODEL_PATH}")
        print("Please train the model first (04_train_yolo_obb.py)")
        exit(1)

    print(f"\n📦 Loading trained model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")

    # Validate on validation set
    print("\n" + "=" * 70)
    print("📊 Running Validation")
    print("=" * 70)

    metrics = model.val(data='data.yaml', split='val', workers=0)

    print(f"\n🎯 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

    # Run predictions on sample validation images
    print("\n" + "=" * 70)
    print("🔍 Sample Predictions")
    print("=" * 70)

    VAL_DIR = Path("yolo_dataset/valid/images")
    sample_images = list(VAL_DIR.glob("*.jpg"))[:5]  # First 5 validation images

    print(f"\nProcessing {len(sample_images)} sample images...")

    for img_path in sample_images:
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        # Get detections
        boxes = results[0].boxes
        
        print(f"\n{img_path.name}:")
        print(f"   Detections: {len(boxes)}")
        
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                conf = box.conf[0].item()
                x, y, w, h = box.xywhn[0].tolist()  # Normalized coordinates
                print(f"   [{i+1}] Confidence: {conf:.3f}, Center: ({x:.3f}, {y:.3f}), Size: ({w:.3f}, {h:.3f})")

    # Visualize predictions
    print("\n" + "=" * 70)
    print("🎨 Creating Visualizations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("YOLO Drone Detection - Sample Predictions", fontsize=16)

    for idx, img_path in enumerate(sample_images):
        if idx >= 6:
            break
        
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        # Plot result
        plotted = results[0].plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(plotted_rgb)
        axes[row, col].set_title(f"{img_path.name}\nDetections: {len(results[0].boxes)}")
        axes[row, col].axis('off')

    # Hide empty subplots
    if len(sample_images) < 6:
        for idx in range(len(sample_images), 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("05_yolo_predictions.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: 05_yolo_predictions.png")

    print("\n" + "=" * 70)
    print("✅ Task 2.3 Complete!")
    print("\n📊 Key Results:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    print("\n🚀 Next: Task 2.4 - Extract bbox features from all training images")
