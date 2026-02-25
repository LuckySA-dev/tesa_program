"""
Phase 7.7: Train YOLO with Augmented Dataset (368 images)
Expect: Better generalization, less overfitting, higher mAP50
"""

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.7: Train YOLO with Augmented Dataset")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️ GPU not available, using CPU")

    # Load YOLOv8n
    print("\n📦 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("✅ Model loaded: YOLOv8n")

    # Training configuration
    print("\n" + "=" * 70)
    print("Training Configuration (Augmented Dataset)")
    print("=" * 70)
    
    config = {
        'data': 'data_augmented.yaml',  # Use augmented dataset
        'epochs': 150,                   # More epochs for larger dataset
        'imgsz': 1280,
        'batch': 8,
        'patience': 30,
        'device': device,
        'workers': 0,
        'project': 'runs/detect',
        'name': 'drone_detect_v16_augmented',
        'exist_ok': True,
        
        # Optimizer settings
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Augmentation (moderate - data already includes pseudo-labels)
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
        'mixup': 0.0,          # Reduce mixup for pseudo-labels
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Other settings
        'verbose': True,
        'save': True,
        'save_period': -1,
        'cache': False,
        'plots': True,
    }
    
    print("\nDataset:")
    print("   Train: 368 images (50 original + 318 pseudo-labeled)")
    print("   Valid: 42 images")
    print("   Total: 410 images (+446% from baseline)")
    
    print("\nKey settings:")
    print("   Epochs: 150")
    print("   Batch: 8")
    print("   Patience: 30")
    print("   Augmentation: Moderate (mosaic, hsv, flip)")

    # Train model
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print("⏱️  Estimated time: 10-15 minutes on GPU")
    print()

    results = model.train(**config)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Validate
    print("\n📊 Running validation...")
    metrics = model.val(data='data_augmented.yaml', workers=0)
    
    print("\n" + "=" * 70)
    print("Validation Metrics (Augmented Dataset)")
    print("=" * 70)
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    # Compare with v15 baseline
    print("\n📈 Comparison with v15 (50 images):")
    print("   v15 Baseline:      mAP50 = 0.475, Recall = 0.598")
    print(f"   v16 Augmented:     mAP50 = {metrics.box.map50:.4f}, Recall = {metrics.box.mr:.4f}")
    improvement = (metrics.box.map50 - 0.475) / 0.475 * 100
    print(f"   Improvement:       {improvement:+.1f}%")
    
    if metrics.box.map50 > 0.475:
        print("\n✅ Augmented dataset IMPROVED performance!")
        print("   Next: Re-extract features and retrain XGBoost")
    else:
        print("\n⚠️ No improvement from pseudo-labels")
        print("   Possible reasons:")
        print("   - Pseudo-labels quality too low")
        print("   - Need higher confidence threshold")
        print("   - Original labels already sufficient")
    
    print(f"\n✅ Model saved: runs/detect/drone_detect_v16_augmented/weights/best.pt")
