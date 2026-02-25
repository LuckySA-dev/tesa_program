"""
Phase 7.1: Tune YOLO Model - Version 2
Improvements:
- Increase epochs (100 -> 200)
- Add more augmentation
- Adjust confidence threshold
- Use larger model (YOLOv8s instead of YOLOv8n)
"""

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.1: YOLO Model Tuning - Version 2")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️ GPU not available, using CPU")

    # Use YOLOv8s (small) - better than nano but still fast
    print("\n📦 Loading YOLOv8s model...")
    model = YOLO("yolov8s.pt")
    print("✅ Model loaded: YOLOv8s")

    # Enhanced training configuration
    print("\n" + "=" * 70)
    print("Training Configuration (Enhanced)")
    print("=" * 70)
    
    config = {
        'data': 'data.yaml',
        'epochs': 200,              # Increased from 100
        'imgsz': 1280,              # Keep high resolution
        'batch': 4,                 # Reduced for larger model
        'patience': 30,             # More patience for convergence
        'device': device,
        'workers': 0,
        'project': 'runs/detect',
        'name': 'drone_detect_v2_tuned',
        'exist_ok': True,
        
        # Optimizer settings
        'optimizer': 'AdamW',
        'lr0': 0.001,              # Lower initial learning rate
        'lrf': 0.01,               # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        
        # Augmentation (enhanced)
        'hsv_h': 0.015,            # Hue augmentation
        'hsv_s': 0.7,              # Saturation augmentation
        'hsv_v': 0.4,              # Value augmentation
        'degrees': 5.0,            # Rotation augmentation
        'translate': 0.1,          # Translation augmentation
        'scale': 0.2,              # Scale augmentation
        'shear': 0.0,              # Shear augmentation
        'perspective': 0.0001,     # Perspective augmentation
        'flipud': 0.0,             # No vertical flip (drones don't flip)
        'fliplr': 0.5,             # Horizontal flip
        'mosaic': 1.0,             # Mosaic augmentation
        'mixup': 0.1,              # Mixup augmentation
        
        # Loss weights
        'box': 7.5,                # Box loss weight
        'cls': 0.5,                # Class loss weight
        'dfl': 1.5,                # Distribution focal loss weight
        
        # Other settings
        'verbose': True,
        'save': True,
        'save_period': -1,
        'cache': False,
        'plots': True,
    }
    
    print("\nKey improvements:")
    print("   1. Model: YOLOv8s (more parameters than YOLOv8n)")
    print("   2. Epochs: 200 (vs 100 before)")
    print("   3. Patience: 30 (vs 20 before)")
    print("   4. Enhanced augmentation: hsv, rotation, scale, mixup")
    print("   5. Lower learning rate: 0.001 (vs 0.01 default)")
    print("   6. Adjusted loss weights for better bbox regression")

    # Train model
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print("⏱️  Estimated time: 15-20 minutes on GPU")
    print()

    results = model.train(**config)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Validate
    print("\n📊 Running validation...")
    metrics = model.val(data='data.yaml', workers=0)
    
    print("\n" + "=" * 70)
    print("Validation Metrics (YOLOv8s Tuned)")
    print("=" * 70)
    print(f"   mAP50:     {metrics.box.map50:.4f} (target: > 0.60)")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    # Compare with previous version
    print("\n📈 Comparison with v15:")
    print("   Previous (YOLOv8n): mAP50 = 0.475, Recall = 0.598")
    print(f"   Current (YOLOv8s):  mAP50 = {metrics.box.map50:.4f}, Recall = {metrics.box.mr:.4f}")
    improvement = (metrics.box.map50 - 0.475) / 0.475 * 100
    print(f"   Improvement: {improvement:+.1f}%")
    
    print("\n✅ Model saved: runs/detect/drone_detect_v2_tuned/weights/best.pt")
    print("\n🎯 Next: Re-extract features and retrain XGBoost with improved YOLO")
