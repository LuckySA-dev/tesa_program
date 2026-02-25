"""
Phase 8.1: Train YOLO v21 with Maximum Data (192 train + 10 valid)
Use all available real labels for training
"""

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 8.1: Train YOLO v21 - Maximum Data Usage")
    print("=" * 70)

    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ Using CPU")

    print("\n📦 Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    config = {
        'data': 'data.yaml',
        'epochs': 150,
        'imgsz': 1280,
        'batch': 8,
        'patience': 30,
        'device': device,
        'workers': 0,
        'project': 'runs/detect',
        'name': 'drone_detect_v21_max_data',
        'exist_ok': True,
        
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        'verbose': True,
        'save': True,
        'cache': False,
        'plots': True,
    }
    
    print("\n📊 Dataset:")
    print("   Train: 192 images (150 original + 42 from validation)")
    print("   Valid: 10 images (minimal)")
    print("   Total: 202 images")
    print("   ✅ Using ALL available real labels!")
    
    print("\n⚙️  Configuration:")
    print("   Epochs: 150, Batch: 8, Image size: 1280")
    print("   Early stopping patience: 30")

    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print("⏱️  Estimated: 12-15 minutes\n")

    results = model.train(**config)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    print("\n📊 Validating...")
    metrics = model.val(data='data.yaml', workers=0)
    
    print("\n" + "=" * 70)
    print("📊 YOLO v21 Results (192 training images)")
    print("=" * 70)
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    print("\n📈 Comparison:")
    print("   v15 (50 labels):   mAP50=0.475, Recall=0.598")
    print("   v20 (150 labels):  mAP50=0.475, Recall=0.622")
    print(f"   v21 (192 labels):  mAP50={metrics.box.map50:.4f}, Recall={metrics.box.mr:.4f}")
    
    improvement_v15 = (metrics.box.map50 - 0.475) / 0.475 * 100
    improvement_v20 = (metrics.box.map50 - 0.475) / 0.475 * 100
    
    print(f"\n   vs v15: {improvement_v15:+.1f}%")
    print(f"   vs v20: {improvement_v20:+.1f}%")
    
    if metrics.box.map50 > 0.475:
        print("\n   ✅ Best model yet!")
    
    print(f"\n💾 Saved: runs/detect/drone_detect_v21_max_data/weights/best.pt")
