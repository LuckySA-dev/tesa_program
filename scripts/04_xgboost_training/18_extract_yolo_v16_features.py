"""
Phase 7.8: Extract Features from YOLO v16 (Augmented)
Use improved YOLO model to re-extract all features
"""

import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.8: Extract Features from YOLO v16")
    print("=" * 70)

    # Load YOLO v16 model
    model_path = "runs/detect/drone_detect_v16_augmented/weights/best.pt"
    print(f"\n📦 Loading YOLO v16: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded")

    # Load existing metadata
    df = pd.read_csv('train_metadata.csv')
    print(f"\n✅ Loaded metadata: {df.shape}")

    # Run predictions on all training images
    print("\n" + "=" * 70)
    print("Extracting YOLO Features")
    print("=" * 70)
    
    image_dir = Path('datasets/DATA_TRAIN/image')
    
    yolo_features = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"   Processing: {idx+1}/{len(df)} images...")
        
        img_name = row['image_name']
        img_path = image_dir / img_name
        
        if not img_path.exists():
            print(f"   ⚠️ Image not found: {img_name}")
            yolo_features.append({
                'has_detection': 0,
                'confidence': 0.0,
                'bbox_x': 0.5,
                'bbox_y': 0.5,
                'bbox_w': 0.0,
                'bbox_h': 0.0,
                'bbox_area': 0.0
            })
            continue
        
        # Run prediction
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            # No detection
            yolo_features.append({
                'has_detection': 0,
                'confidence': 0.0,
                'bbox_x': 0.5,
                'bbox_y': 0.5,
                'bbox_w': 0.0,
                'bbox_h': 0.0,
                'bbox_area': 0.0
            })
            continue
        
        # Get best detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        best_box = boxes[best_idx]
        
        # Extract features
        x_center, y_center, width, height = best_box.xywhn[0].cpu().numpy()
        conf = float(best_box.conf)
        
        yolo_features.append({
            'has_detection': 1,
            'confidence': conf,
            'bbox_x': float(x_center),
            'bbox_y': float(y_center),
            'bbox_w': float(width),
            'bbox_h': float(height),
            'bbox_area': float(width * height)
        })
    
    print(f"\n✅ Extracted features from {len(yolo_features)} images")
    
    # Add to dataframe
    yolo_df = pd.DataFrame(yolo_features)
    
    # Combine with original metadata
    result_df = pd.concat([df, yolo_df], axis=1)
    
    # Save
    output_path = 'train_metadata_with_yolo_v16.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   Shape: {result_df.shape}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 YOLO v16 Feature Statistics")
    print("=" * 70)
    
    detection_rate = yolo_df['has_detection'].mean()
    mean_conf = yolo_df[yolo_df['has_detection'] == 1]['confidence'].mean()
    
    print(f"\n   Detection rate:    {detection_rate:.1%} ({yolo_df['has_detection'].sum()}/{len(yolo_df)})")
    print(f"   Mean confidence:   {mean_conf:.3f}")
    print(f"   Bbox area (mean):  {yolo_df['bbox_area'].mean():.4f}")
    print(f"   Bbox area (std):   {yolo_df['bbox_area'].std():.4f}")
    
    # Compare with v15
    if os.path.exists('train_metadata_with_yolo.csv'):
        try:
            df_v15 = pd.read_csv('train_metadata_with_yolo.csv')
            if 'has_detection' in df_v15.columns:
                detection_rate_v15 = df_v15['has_detection'].mean()
                mean_conf_v15 = df_v15[df_v15['has_detection'] == 1]['confidence'].mean()
                
                print(f"\n📈 Comparison with v15:")
                print(f"   Detection rate: {detection_rate_v15:.1%} → {detection_rate:.1%} ({(detection_rate-detection_rate_v15)/detection_rate_v15*100:+.1f}%)")
                print(f"   Mean confidence: {mean_conf_v15:.3f} → {mean_conf:.3f} ({(mean_conf-mean_conf_v15)/mean_conf_v15*100:+.1f}%)")
        except Exception as e:
            print(f"\n⚠️ Could not compare with v15: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Phase 7.8 Complete!")
    print("=" * 70)
    print("\n🎯 Next: Feature engineering with v16 features")
    print("   Run: python 07_feature_engineering.py (will use new YOLO features)")
