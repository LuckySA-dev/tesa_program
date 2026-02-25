"""
Phase 2 - Task 2.4: Extract YOLO Features from All Training Images
Run inference on all 438 training images to extract bbox features for XGBoost
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 2 - Task 2.4: Extract YOLO Features")
    print("=" * 70)

    # Load best model
    MODEL_PATH = "runs/detect/drone_detect_v15/weights/best.pt"

    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Model not found: {MODEL_PATH}")
        print("Please train the model first (04_train_yolo_obb.py)")
        exit(1)

    print(f"\n📦 Loading trained model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")

    # Load metadata
    print(f"\n📄 Loading train metadata...")
    metadata_df = pd.read_csv('train_metadata_enhanced.csv')
    print(f"✅ Loaded {len(metadata_df)} training records")

    # Process all images
    print("\n" + "=" * 70)
    print("🔍 Running Inference on All Training Images")
    print("=" * 70)

    IMAGE_DIR = Path("datasets/DATA_TRAIN/image")

    results_list = []

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing images"):
        img_name = row['image_name']
        img_path = IMAGE_DIR / img_name
        
        if not img_path.exists():
            print(f"\n⚠️ Image not found: {img_path}")
            # Use default values
            results_list.append({
                'image_num': row['image_num'],
                'image_name': img_name,
                'yolo_detected': False,
                'yolo_conf': 0.0,
                'yolo_cx': 0.5,
                'yolo_cy': 0.5,
                'yolo_w': 0.0,
                'yolo_h': 0.0,
                'num_detections': 0
            })
            continue
        
        # Run inference
        results = model.predict(img_path, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            # No detection - use image center as fallback
            results_list.append({
                'image_num': row['image_num'],
                'image_name': img_name,
                'yolo_detected': False,
                'yolo_conf': 0.0,
                'yolo_cx': 0.5,
                'yolo_cy': 0.5,
                'yolo_w': 0.0,
                'yolo_h': 0.0,
                'num_detections': 0
            })
        else:
            # Use highest confidence detection
            best_box = boxes[0]
            conf = best_box.conf[0].item()
            xywhn = best_box.xywhn[0].tolist()  # Normalized [cx, cy, w, h]
            
            results_list.append({
                'image_num': row['image_num'],
                'image_name': img_name,
                'yolo_detected': True,
                'yolo_conf': conf,
                'yolo_cx': xywhn[0],
                'yolo_cy': xywhn[1],
                'yolo_w': xywhn[2],
                'yolo_h': xywhn[3],
                'num_detections': len(boxes)
            })

    # Create DataFrame
    yolo_features_df = pd.DataFrame(results_list)

    # Merge with original metadata
    full_df = metadata_df.merge(yolo_features_df, on=['image_num', 'image_name'], how='left')

    # Save results
    OUTPUT_FILE = 'train_metadata_with_yolo.csv'
    full_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("📊 Extraction Summary")
    print("=" * 70)

    total_images = len(full_df)
    detected = full_df['yolo_detected'].sum()
    not_detected = total_images - detected

    print(f"\nTotal images processed: {total_images}")
    print(f"Successfully detected: {detected} ({detected/total_images*100:.1f}%)")
    print(f"Not detected (fallback): {not_detected} ({not_detected/total_images*100:.1f}%)")

    print(f"\n📈 Detection Statistics:")
    print(f"   Mean confidence: {full_df[full_df['yolo_detected']]['yolo_conf'].mean():.3f}")
    print(f"   Mean bbox width: {full_df[full_df['yolo_detected']]['yolo_w'].mean():.3f}")
    print(f"   Mean bbox height: {full_df[full_df['yolo_detected']]['yolo_h'].mean():.3f}")

    print(f"\n📁 Output saved: {OUTPUT_FILE}")
    print(f"   Columns: {list(full_df.columns)}")
    print(f"   Shape: {full_df.shape}")

    print("\n" + "=" * 70)
    print("✅ Task 2.4 Complete!")
    print("\n🎯 Phase 2 Summary:")
    print("   ✅ Task 2.1: YOLO dataset prepared (92 images)")
    print("   ✅ Task 2.2: Model trained (YOLOv8n)")
    print("   ✅ Task 2.3: Model evaluated")
    print("   ✅ Task 2.4: Features extracted (438 images)")
    print("\n🚀 Next: Phase 3 - Feature Engineering for XGBoost")
