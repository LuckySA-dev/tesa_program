"""
Evaluate YOLO v21 Performance
"""

from ultralytics import YOLO

if __name__ == '__main__':
    print("=" * 70)
    print("📊 YOLO v21 Performance Evaluation")
    print("=" * 70)

    # Load model
    model = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')
    
    # Validate
    print("\n⏳ Running validation...")
    metrics = model.val(data='data.yaml', workers=0, verbose=False)
    
    # Results
    print("\n" + "=" * 70)
    print("📊 YOLO v21 Results (192 Training Images)")
    print("=" * 70)
    print(f"\n   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("📈 Comparison with Previous Versions")
    print("=" * 70)
    
    v15_map = 0.4750
    v15_recall = 0.5980
    v20_map = 0.4745
    v20_recall = 0.6220
    
    print(f"\n   v15 (50 labels):   mAP50={v15_map:.4f}, Recall={v15_recall:.4f}")
    print(f"   v20 (150 labels):  mAP50={v20_map:.4f}, Recall={v20_recall:.4f}")
    print(f"   v21 (192 labels):  mAP50={metrics.box.map50:.4f}, Recall={metrics.box.mr:.4f}")
    
    # Calculate improvements
    map_improve_v15 = (metrics.box.map50 - v15_map) / v15_map * 100
    recall_improve_v15 = (metrics.box.mr - v15_recall) / v15_recall * 100
    map_improve_v20 = (metrics.box.map50 - v20_map) / v20_map * 100
    recall_improve_v20 = (metrics.box.mr - v20_recall) / v20_recall * 100
    
    print(f"\n   📊 vs v15 (baseline):")
    print(f"      mAP50:  {map_improve_v15:+.1f}%")
    print(f"      Recall: {recall_improve_v15:+.1f}%")
    
    print(f"\n   📊 vs v20 (150 labels):")
    print(f"      mAP50:  {map_improve_v20:+.1f}%")
    print(f"      Recall: {recall_improve_v20:+.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 Summary")
    print("=" * 70)
    
    if metrics.box.map50 > v15_map:
        improvement = (metrics.box.map50 - v15_map) / v15_map * 100
        print(f"\n   ✅ v21 is BETTER than baseline!")
        print(f"   ✨ mAP50 improved by {improvement:.1f}%")
        print(f"   ✨ Detection improved from {v15_recall:.1%} to {metrics.box.mr:.1%}")
    else:
        print(f"\n   ⚠️ v21 performance similar to baseline")
        print(f"   💡 May need different approach (hyperparameter tuning, data augmentation)")
    
    print("\n" + "=" * 70)
