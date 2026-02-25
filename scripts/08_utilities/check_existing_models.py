"""
Quick Training Script - Use existing trained model or train minimal epochs
"""

from ultralytics import YOLO
from pathlib import Path

print("=" * 70)
print("Quick YOLO Training Check")
print("=" * 70)

# Check for existing trained models
existing_models = [
    "runs/obb/drone_obb_v12/weights/best.pt",
    "runs/obb/drone_obb_v1/weights/best.pt",
    "runs/detect/drone_detect_v1/weights/best.pt"
]

print("\n🔍 Checking for existing trained models...")
for model_path in existing_models:
    if Path(model_path).exists():
        print(f"✅ Found: {model_path}")
        
        # Test the model
        print(f"\n📊 Testing model: {model_path}")
        model = YOLO(model_path)
        
        # Quick validation
        try:
            metrics = model.val(data='data.yaml', split='val', verbose=False)
            print(f"   mAP50: {metrics.box.map50:.4f}")
            print(f"   mAP50-95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.mp:.4f}")
            print(f"   Recall: {metrics.box.mr:.4f}")
            
            print(f"\n✅ This model can be used for Phase 2!")
            print(f"\nTo use this model, update scripts to use:")
            print(f"   MODEL_PATH = '{model_path}'")
            break
        except Exception as e:
            print(f"   ⚠️ Validation failed: {e}")
    else:
        print(f"❌ Not found: {model_path}")

print("\n" + "=" * 70)
print("Alternative: Train quick model with fewer epochs")
print("=" * 70)
print("\nOption 1: Use existing model if validation passed")
print("Option 2: Train new model with epochs=20 (faster)")
print("Option 3: Wait for GPU PyTorch installation to complete")
