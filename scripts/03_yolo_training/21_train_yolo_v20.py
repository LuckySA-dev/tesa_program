"""
Phase 8.0: Train YOLO with Real Labels (150 train + 42 valid = 192 total)
Major improvement from 50 → 150 training labels (+200%)
"""

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 8.0: Train YOLO with Expanded Real Labels (150)")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️ Using CPU")

    # Load YOLOv8n
    print("\n📦 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("✅ Model loaded")

    # Training configuration
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    
    config = {
        'data': 'data.yaml',
        'epochs': 150,
        'imgsz': 1280,
        'batch': 8,
        'patience': 30,
        'device': device,
        'workers': 0,
        'project': 'runs/detect',
        'name': 'drone_detect_v20_real_labels',
        'exist_ok': True,
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        'verbose': True,
        'save': True,
        'save_period': -1,
        'cache': False,
        'plots': True,
    }
    
    print("\n📊 Dataset:")
    print("   Train: 150 labels (100 from Roboflow + 50 original)")
    print("   Valid: 42 labels")
    print("   Total: 192 labels (+109% from v15 baseline)")
    print("\n   ✅ All REAL labels (no pseudo-labels!)")
    
    print("\n⚙️  Key settings:")
    print("   Epochs: 150")
    print("   Batch: 8")
    print("   Patience: 30")
    print("   Image size: 1280")

    # Train
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print("⏱️  Estimated: 12-15 minutes on GPU")
    print()

    results = model.train(**config)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Validate
    print("\n📊 Running validation...")
    metrics = model.val(data='data.yaml', workers=0)
    
    print("\n" + "=" * 70)
    print("Validation Metrics (Real Labels)")
    print("=" * 70)
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    # Compare
    print("\n" + "=" * 70)
    print("📈 Comparison")
    print("=" * 70)
    print("\n   v15 (50 labels):      mAP50=0.475, Recall=0.598")
    print(f"   v20 (150 labels):     mAP50={metrics.box.map50:.4f}, Recall={metrics.box.mr:.4f}")
    
    improvement = (metrics.box.map50 - 0.475) / 0.475 * 100
    print(f"\n   mAP50 Improvement:    {improvement:+.1f}%")
    
    if metrics.box.map50 > 0.475:
        print("\n   ✅ Real labels IMPROVED performance!")
        print("   🎯 Next: Extract features & retrain XGBoost")
    else:
        print("\n   ⚠️ Need to check data quality")
    
    print(f"\n✅ Best model: runs/detect/drone_detect_v20_real_labels/weights/best.pt")
