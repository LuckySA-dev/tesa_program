"""
Test tomorbest.pt model on validation set
Compare with current best (YOLOv8n v15)
"""

from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    print("=" * 70)
    print("Testing tomorbest.pt Model")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU")

    # Check if model exists
    if not os.path.exists('tomorbest.pt'):
        print("❌ tomorbest.pt not found!")
        exit(1)

    print("\n" + "=" * 70)
    print("1. Testing tomorbest.pt")
    print("=" * 70)

    model_tomor = YOLO('tomorbest.pt')
    
    # Validate on our validation set
    print("\n📊 Running validation...")
    metrics_tomor = model_tomor.val(data='data.yaml', workers=0, verbose=False)
    
    print("\n📊 tomorbest.pt Results:")
    print(f"   mAP50:     {metrics_tomor.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics_tomor.box.map:.4f}")
    print(f"   Precision: {metrics_tomor.box.mp:.4f}")
    print(f"   Recall:    {metrics_tomor.box.mr:.4f}")

    # Load current best for comparison
    print("\n" + "=" * 70)
    print("2. Testing Current Best (YOLOv8n v15)")
    print("=" * 70)

    best_v15_path = 'runs/detect/drone_detect_v15/weights/best.pt'
    if os.path.exists(best_v15_path):
        model_v15 = YOLO(best_v15_path)
        metrics_v15 = model_v15.val(data='data.yaml', workers=0, verbose=False)
        
        print("\n📊 YOLOv8n v15 Results:")
        print(f"   mAP50:     {metrics_v15.box.map50:.4f}")
        print(f"   mAP50-95:  {metrics_v15.box.map:.4f}")
        print(f"   Precision: {metrics_v15.box.mp:.4f}")
        print(f"   Recall:    {metrics_v15.box.mr:.4f}")

        # Comparison
        print("\n" + "=" * 70)
        print("📊 Comparison")
        print("=" * 70)
        
        print(f"\n   mAP50:")
        print(f"      tomorbest.pt:  {metrics_tomor.box.map50:.4f}")
        print(f"      v15:           {metrics_v15.box.map50:.4f}")
        diff_map50 = metrics_tomor.box.map50 - metrics_v15.box.map50
        print(f"      Difference:    {diff_map50:+.4f} ({diff_map50/metrics_v15.box.map50*100:+.1f}%)")
        
        print(f"\n   Recall:")
        print(f"      tomorbest.pt:  {metrics_tomor.box.mr:.4f}")
        print(f"      v15:           {metrics_v15.box.mr:.4f}")
        diff_recall = metrics_tomor.box.mr - metrics_v15.box.mr
        print(f"      Difference:    {diff_recall:+.4f} ({diff_recall/metrics_v15.box.mr*100:+.1f}%)")
        
        print(f"\n   Precision:")
        print(f"      tomorbest.pt:  {metrics_tomor.box.mp:.4f}")
        print(f"      v15:           {metrics_v15.box.mp:.4f}")
        diff_prec = metrics_tomor.box.mp - metrics_v15.box.mp
        print(f"      Difference:    {diff_prec:+.4f} ({diff_prec/metrics_v15.box.mp*100:+.1f}%)")

        print("\n" + "=" * 70)
        if metrics_tomor.box.map50 > metrics_v15.box.map50:
            print("✅ tomorbest.pt is BETTER! (+{:.1f}%)".format(diff_map50/metrics_v15.box.map50*100))
            print("   Recommendation: Use tomorbest.pt for feature extraction")
        elif metrics_tomor.box.map50 < metrics_v15.box.map50:
            print("⚠️ v15 is still BETTER (+{:.1f}%)".format(-diff_map50/metrics_v15.box.map50*100))
            print("   Recommendation: Keep using v15")
        else:
            print("🤝 Both models are similar")
        print("=" * 70)
    else:
        print(f"\n⚠️ v15 model not found at: {best_v15_path}")
        print("   Only tomorbest.pt results available")

    print("\n✅ Testing Complete")
