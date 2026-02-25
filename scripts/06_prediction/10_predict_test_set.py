"""
Phase 6: Predict Test Set
Run YOLO + XGBoost pipeline on test images to generate submission file
"""

import pandas as pd
import numpy as np
import json
import pickle
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 6: Test Set Prediction")
    print("=" * 70)

    # Camera position
    CAMERA_LAT = 14.305029
    CAMERA_LON = 101.173010
    CAMERA_ALT = 0

    # Load YOLO model
    print("\n📦 Loading YOLO model...")
    yolo_model = YOLO("runs/detect/drone_detect_v15/weights/best.pt")
    print("✅ Loaded: runs/detect/drone_detect_v15/weights/best.pt")

    # Load XGBoost models
    print("\n📦 Loading XGBoost models...")
    with open('xgb_model_latitude.pkl', 'rb') as f:
        model_lat = pickle.load(f)
    with open('xgb_model_longitude.pkl', 'rb') as f:
        model_lon = pickle.load(f)
    with open('xgb_model_altitude.pkl', 'rb') as f:
        model_alt = pickle.load(f)
    print("✅ Loaded all XGBoost models")

    # Load feature columns
    with open('feature_columns.json', 'r') as f:
        feature_info = json.load(f)
    feature_columns = feature_info['feature_columns']
    print(f"✅ Feature columns: {len(feature_columns)}")

    # Get test images
    TEST_IMAGE_DIR = Path("datasets/DATA_TEST")
    test_images = sorted(list(TEST_IMAGE_DIR.glob("*.jpg")))
    print(f"\n✅ Found {len(test_images)} test images")

    if len(test_images) == 0:
        print("⚠️ No test images found! Checking directory structure...")
        print(f"   Looking in: {TEST_IMAGE_DIR}")
        # Try alternative locations
        alt_dirs = [
            Path("datasets/DATA_TEST/image"),
            Path("datasets/DATA_TEST/images"),
            Path("datasets/test"),
        ]
        for alt_dir in alt_dirs:
            if alt_dir.exists():
                test_images = sorted(list(alt_dir.glob("*.jpg")))
                if len(test_images) > 0:
                    print(f"✅ Found {len(test_images)} images in: {alt_dir}")
                    TEST_IMAGE_DIR = alt_dir
                    break

    # Step 1: Run YOLO on test images
    print("\n" + "=" * 70)
    print("Step 1: Running YOLO Detection on Test Images")
    print("=" * 70)

    test_data = []
    for img_path in tqdm(test_images, desc="Processing test images"):
        # Run YOLO
        results = yolo_model.predict(img_path, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        # Extract bbox features
        if len(boxes) > 0:
            # Take the highest confidence detection
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            box = boxes[best_idx]
            
            xywhn = box.xywhn[0].cpu().numpy()
            conf = box.conf[0].item()
            
            cx, cy, w, h = xywhn
            detected = True
        else:
            # Fallback: assume center of image
            cx, cy, w, h = 0.5, 0.5, 0.02, 0.02
            conf = 0.0
            detected = False
        
        test_data.append({
            'image_name': img_path.name,
            'yolo_detected': detected,
            'yolo_conf': conf,
            'yolo_cx': cx,
            'yolo_cy': cy,
            'yolo_w': w,
            'yolo_h': h,
            'num_detections': len(boxes)
        })

    df_test = pd.DataFrame(test_data)
    print(f"\n✅ YOLO Detection Complete:")
    print(f"   Total images: {len(df_test)}")
    print(f"   Detected: {df_test['yolo_detected'].sum()} ({df_test['yolo_detected'].mean():.1%})")
    print(f"   Mean confidence: {df_test['yolo_conf'].mean():.3f}")

    # Step 2: Engineer features for test set
    print("\n" + "=" * 70)
    print("Step 2: Engineering Features for Test Set")
    print("=" * 70)

    # Since we don't have ground truth for test set, we need to estimate some features
    # For features that depend on ground truth (distance, bearing, altitude), 
    # we'll use median values from training set or make reasonable estimates
    
    # Load training metadata for reference statistics
    df_train = pd.read_csv('train_metadata_engineered.csv')
    
    # Use median values from training set for missing features
    median_distance = df_train['distance_m'].median()
    median_bearing = df_train['bearing_deg'].median()
    median_altitude = df_train['altitude_m'].median()
    
    print(f"   Using training set medians:")
    print(f"      Distance: {median_distance:.2f} m")
    print(f"      Bearing: {median_bearing:.2f} deg")
    print(f"      Altitude: {median_altitude:.2f} m")

    # Add placeholder values for geospatial features
    df_test['distance_m'] = median_distance
    df_test['bearing_deg'] = median_bearing
    df_test['altitude_m'] = median_altitude

    # Extract image number
    df_test['image_num'] = df_test['image_name'].str.extract(r'(\d+)').astype(int)
    df_test['image_num_normalized'] = df_test['image_num'] / df_test['image_num'].max()

    # YOLO bbox features
    df_test['yolo_area'] = df_test['yolo_w'] * df_test['yolo_h']
    df_test['yolo_aspect_ratio'] = df_test['yolo_w'] / (df_test['yolo_h'] + 1e-8)
    df_test['yolo_dist_from_center'] = np.sqrt((df_test['yolo_cx'] - 0.5)**2 + (df_test['yolo_cy'] - 0.5)**2)
    df_test['yolo_angle_from_center'] = np.arctan2(df_test['yolo_cy'] - 0.5, df_test['yolo_cx'] - 0.5)
    df_test['yolo_in_center'] = ((df_test['yolo_cx'] > 0.3) & (df_test['yolo_cx'] < 0.7) & 
                                  (df_test['yolo_cy'] > 0.3) & (df_test['yolo_cy'] < 0.7)).astype(int)

    # Geospatial features
    df_test['bearing_rad'] = np.deg2rad(df_test['bearing_deg'])
    df_test['bearing_sin'] = np.sin(df_test['bearing_rad'])
    df_test['bearing_cos'] = np.cos(df_test['bearing_rad'])

    # Interaction features
    df_test['distance_x_conf'] = df_test['distance_m'] * df_test['yolo_conf']
    df_test['distance_x_area'] = df_test['distance_m'] * df_test['yolo_area']
    df_test['bearing_x_cx'] = df_test['bearing_deg'] * df_test['yolo_cx']
    df_test['bearing_x_cy'] = df_test['bearing_deg'] * df_test['yolo_cy']
    df_test['altitude_diff_from_camera'] = df_test['altitude_m'] - CAMERA_ALT
    df_test['altitude_x_distance'] = df_test['altitude_m'] * df_test['distance_m']

    # Confidence features
    df_test['conf_squared'] = df_test['yolo_conf'] ** 2
    df_test['conf_sqrt'] = np.sqrt(df_test['yolo_conf'])
    df_test['is_high_conf'] = (df_test['yolo_conf'] > 0.5).astype(int)
    df_test['is_detected'] = df_test['yolo_detected'].astype(int)

    # Position estimation
    df_test['estimated_lat_offset'] = (df_test['yolo_cy'] - 0.5) * df_test['distance_m'] / 111000
    df_test['estimated_lon_offset'] = (df_test['yolo_cx'] - 0.5) * df_test['distance_m'] / (111000 * np.cos(np.deg2rad(CAMERA_LAT)))
    df_test['estimated_lat'] = CAMERA_LAT + df_test['estimated_lat_offset']
    df_test['estimated_lon'] = CAMERA_LON + df_test['estimated_lon_offset']

    # Bins (use training set bins)
    df_test['distance_bin'] = pd.cut(df_test['distance_m'], 
                                       bins=pd.qcut(df_train['distance_m'], q=5, retbins=True)[1],
                                       labels=False)
    df_test['altitude_bin'] = pd.cut(df_test['altitude_m'],
                                       bins=pd.qcut(df_train['altitude_m'], q=5, retbins=True)[1],
                                       labels=False)
    
    # Fill NaN in bins (values outside training range)
    df_test['distance_bin'] = df_test['distance_bin'].fillna(2)  # Use middle bin
    df_test['altitude_bin'] = df_test['altitude_bin'].fillna(2)

    print(f"✅ Features engineered: {df_test.shape[1]} columns")

    # Step 3: Predict coordinates
    print("\n" + "=" * 70)
    print("Step 3: Predicting Coordinates with XGBoost")
    print("=" * 70)

    X_test = df_test[feature_columns].values
    
    pred_lat = model_lat.predict(X_test)
    pred_lon = model_lon.predict(X_test)
    pred_alt = model_alt.predict(X_test)
    
    print(f"✅ Predictions complete!")
    print(f"   Latitude range: {pred_lat.min():.6f} - {pred_lat.max():.6f}")
    print(f"   Longitude range: {pred_lon.min():.6f} - {pred_lon.max():.6f}")
    print(f"   Altitude range: {pred_alt.min():.1f} - {pred_alt.max():.1f} m")

    # Step 4: Create submission file
    print("\n" + "=" * 70)
    print("Step 4: Creating Submission File")
    print("=" * 70)

    submission = pd.DataFrame({
        'image_name': df_test['image_name'],
        'latitude': pred_lat,
        'longitude': pred_lon,
        'altitude': pred_alt
    })

    submission_file = 'test_predictions.csv'
    submission.to_csv(submission_file, index=False)
    print(f"✅ Saved: {submission_file}")
    print(f"   Shape: {submission.shape}")
    print(f"\n   First 5 predictions:")
    print(submission.head())

    # Also save detailed test results with YOLO features
    df_test['pred_latitude'] = pred_lat
    df_test['pred_longitude'] = pred_lon
    df_test['pred_altitude'] = pred_alt
    
    detailed_file = 'test_predictions_detailed.csv'
    df_test.to_csv(detailed_file, index=False)
    print(f"\n✅ Saved detailed results: {detailed_file}")

    print("\n" + "=" * 70)
    print("✅ Phase 6 Complete: Test Set Prediction")
    print("=" * 70)
    print(f"\n📋 Submission file ready: {submission_file}")
    print(f"   Total predictions: {len(submission)}")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review predictions: {submission_file}")
    print(f"   2. Submit to competition platform")
    print(f"   3. Proceed to Phase 7: Model Tuning (if needed)")
