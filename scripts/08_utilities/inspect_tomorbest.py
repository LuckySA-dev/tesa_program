"""
Check tomorbest.pt model type and configuration
"""

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("=" * 70)
    print("Inspecting tomorbest.pt Model")
    print("=" * 70)

    model = YOLO('tomorbest.pt')
    
    # Check model info
    print("\n📊 Model Information:")
    print(f"   Task:       {model.task}")
    print(f"   Model type: {type(model.model).__name__}")
    
    # Check class names
    if hasattr(model.model, 'names'):
        print(f"\n📋 Class Names ({len(model.model.names)} classes):")
        for idx, name in model.model.names.items():
            print(f"      {idx}: {name}")
    
    # Check model layers
    print(f"\n🔧 Model Architecture:")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"   Layers:     {len(list(model.model.modules()))}")
    
    # Try prediction on one image to see output format
    print("\n🧪 Testing prediction...")
    import os
    test_img = 'datasets/DATA_TRAIN/image/img_0001.png'
    if os.path.exists(test_img):
        results = model.predict(test_img, verbose=False)
        if len(results) > 0:
            result = results[0]
            print(f"   Image shape: {result.orig_shape}")
            print(f"   Detections:  {len(result.boxes) if hasattr(result, 'boxes') else 'N/A'}")
            if hasattr(result, 'obb'):
                print(f"   OBB boxes:   {len(result.obb) if result.obb is not None else 0}")
                print("   ✅ This is an OBB model!")
            else:
                print("   ⚠️ This is a regular detection model")
            
            # Check box format
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                box = result.boxes[0]
                print(f"\n   First detection:")
                print(f"      Class: {int(box.cls)}")
                print(f"      Conf:  {float(box.conf):.4f}")
                print(f"      Box shape: {box.xyxy.shape if hasattr(box, 'xyxy') else 'N/A'}")
    else:
        print(f"   ⚠️ Test image not found: {test_img}")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
