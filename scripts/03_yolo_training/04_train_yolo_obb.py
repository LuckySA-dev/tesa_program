"""
Phase 2 - Task 2.2: Train YOLOv8-OBB Model
Train Oriented Bounding Box model for drone detection
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 2 - Task 2.2: Train YOLOv8-OBB Model")
    print("=" * 70)

    # Auto-detect device and force GPU usage
    device = 0 if torch.cuda.is_available() else 'cpu'  # Use GPU 0 if available
    print(f"\n🖥️  Device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"   ⚠️ Running on CPU (GPU not available)")

    # Verify data.yaml
    print("\n📄 Checking data.yaml configuration...")
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
        print(f"   Dataset path: {config['path']}")
        print(f"   Train: {config['train']}")
        print(f"   Valid: {config['val']}")
        print(f"   Classes: {config['nc']} ({config['names']})")

    # Verify dataset exists
    train_images = Path(config['path']) / config['train']
    val_images = Path(config['path']) / config['val']
    print(f"\n✅ Train images: {len(list(train_images.glob('*.jpg')))} files")
    print(f"✅ Valid images: {len(list(val_images.glob('*.jpg')))} files")

    # Load YOLOv8 model (standard detection, not OBB)
    print("\n🚀 Loading YOLOv8 model (yolov8n.pt)...")
    model = YOLO('yolov8n.pt')  # Nano model for faster training
    print("✅ Model loaded successfully")

    # Training configuration
    print("\n⚙️  Training Configuration:")
    print("   Model: yolov8n (Standard Detection)")
    print("   Epochs: 100")
    print("   Image size: 1280 (High Resolution)")
    print("   Batch size: 8 (reduced for higher resolution)")
    print("   Patience: 20 (early stopping)")
    print("   Optimizer: AdamW")
    print(f"   Device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    print("\n" + "=" * 70)
    print("🏋️  Starting Training...")
    print("=" * 70)

    # Train model
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=1280,  # High resolution for better small object detection
        batch=8,  # Reduced batch size to fit GPU memory
        name='drone_detect_v1',
        patience=20,
        save=True,
        plots=True,
        device=device,
        verbose=True,
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic augmentation in last 10 epochs
        amp=True,  # Automatic Mixed Precision for faster training
        workers=0  # Fix Windows multiprocessing issue
    )

    print("\n" + "=" * 70)
    print("📊 Training Results Summary")
    print("=" * 70)

    # Get final metrics
    best_metrics = results.results_dict
    print(f"\n🎯 Best Metrics:")
    print(f"   mAP50: {best_metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95: {best_metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision: {best_metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall: {best_metrics.get('metrics/recall(B)', 0):.4f}")

    print(f"\n📁 Saved Models:")
    print(f"   Best: runs/detect/drone_detect_v1/weights/best.pt")
    print(f"   Last: runs/detect/drone_detect_v1/weights/last.pt")

    print(f"\n📈 Training Plots:")
    print(f"   Results: runs/detect/drone_detect_v1/results.png")
    print(f"   Confusion Matrix: runs/detect/drone_detect_v1/confusion_matrix.png")

    print("\n✅ Task 2.2 Complete!")
    print("\n🚀 Next: Task 2.3 - Evaluate model on validation set")
