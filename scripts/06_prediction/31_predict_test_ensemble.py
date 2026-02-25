"""
Generate Test Set Predictions using Ensemble (v1=0.1, v21=0.9)
Best score: 5.2838 (11% better than baseline v1)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from ultralytics import YOLO
from math import radians, cos, sin, asin, sqrt, atan2, degrees

# Camera position
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360

print("="*70)
print("🎯 Test Set Prediction - Ensemble (v1=0.1, v21=0.9)")
print("="*70)

# Load ensemble config
with open('ensemble_config.json', 'r') as f:
    ensemble_config = json.load(f)

w1 = ensemble_config['v1_weight']
w2 = ensemble_config['v21_weight']

print(f"\n✅ Ensemble weights: v1={w1:.1f}, v21={w2:.1f}")
print(f"   Validation score: {ensemble_config['score']:.4f}")

# ============================================================================
# Step 1: Load v1 Models
# ============================================================================
print("\n" + "="*70)
print("Step 1: Load Baseline v1 Models")
print("="*70)

with open('xgb_model_latitude.pkl', 'rb') as f:
    model_lat_v1 = pickle.load(f)
with open('xgb_model_longitude.pkl', 'rb') as f:
    model_lon_v1 = pickle.load(f)
with open('xgb_model_altitude.pkl', 'rb') as f:
    model_alt_v1 = pickle.load(f)

with open('feature_columns.json', 'r') as f:
    feature_data_v1 = json.load(f)
    feature_cols_v1 = feature_data_v1['feature_columns']

print(f"✅ Loaded v1 models ({len(feature_cols_v1)} features)")

# ============================================================================
# Step 2: Load v21 Models
# ============================================================================
print("\n" + "="*70)
print("Step 2: Load YOLO v21 Models")
print("="*70)

with open('xgb_model_latitude_v21.pkl', 'rb') as f:
    model_lat_v21 = pickle.load(f)
with open('xgb_model_longitude_v21.pkl', 'rb') as f:
    model_lon_v21 = pickle.load(f)
with open('xgb_model_altitude_v21.pkl', 'rb') as f:
    model_alt_v21 = pickle.load(f)

with open('feature_columns_v21.json', 'r') as f:
    feature_data_v21 = json.load(f)
    feature_cols_v21 = feature_data_v21['feature_columns']

print(f"✅ Loaded v21 models ({len(feature_cols_v21)} features)")

# ============================================================================
# Step 3: Load YOLO Models
# ============================================================================
print("\n" + "="*70)
print("Step 3: Load YOLO Models")
print("="*70)

# YOLO v15 (for v1)
yolo_v15 = YOLO('runs/detect/drone_detect_v15/weights/best.pt')
print("✅ Loaded YOLO v15")

# YOLO v21
yolo_v21 = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')
print("✅ Loaded YOLO v21")

# ============================================================================
# Step 4: Load Test Images
# ============================================================================
print("\n" + "="*70)
print("Step 4: Load Test Images")
print("="*70)

test_dir = 'datasets/DATA_TEST'
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

print(f"✅ Found {len(test_images)} test images")

# ============================================================================
# Step 5: Extract Features and Predict
# ============================================================================
print("\n" + "="*70)
print("Step 5: Extract Features and Generate Predictions")
print("="*70)

predictions = []

print("\n   Processing test images...")
for idx, img_name in enumerate(test_images):
    if (idx + 1) % 50 == 0:
        print(f"   {idx + 1}/{len(test_images)}...")
    
    img_path = os.path.join(test_dir, img_name)
    
    # ========================================================================
    # Extract v1 features (YOLO v15)
    # ========================================================================
    results_v15 = yolo_v15(img_path, verbose=False)
    
    if len(results_v15[0].boxes) > 0:
        box = results_v15[0].boxes[0]
        bbox = box.xywhn[0].cpu().numpy()
        conf_v15 = float(box.conf[0])
        
        yolo_data_v15 = {
            'yolo_detected': 1,
            'yolo_conf': conf_v15,
            'yolo_cx': float(bbox[0]),
            'yolo_cy': float(bbox[1]),
            'yolo_w': float(bbox[2]),
            'yolo_h': float(bbox[3]),
        }
    else:
        yolo_data_v15 = {
            'yolo_detected': 0,
            'yolo_conf': 0.0,
            'yolo_cx': 0.5,
            'yolo_cy': 0.5,
            'yolo_w': 0.0,
            'yolo_h': 0.0,
        }
    
    # Feature engineering for v1
    # Note: For test set, we don't have ground truth distance/bearing
    # So we'll use estimated values from YOLO bbox position
    
    # Estimate distance based on bbox size (rough approximation)
    bbox_area = yolo_data_v15['yolo_w'] * yolo_data_v15['yolo_h']
    estimated_distance = 100.0 if bbox_area == 0 else max(10, min(200, 50 / (bbox_area + 0.001)))
    
    # Estimate bearing from bbox position (camera FOV ~60 degrees)
    # Center of image = bearing 220 degrees (approximate from training data)
    horizontal_offset = (yolo_data_v15['yolo_cx'] - 0.5) * 60  # FOV angle
    estimated_bearing = (220 + horizontal_offset) % 360
    
    features_v1 = {
        'yolo_cx': yolo_data_v15['yolo_cx'],
        'yolo_cy': yolo_data_v15['yolo_cy'],
        'yolo_w': yolo_data_v15['yolo_w'],
        'yolo_h': yolo_data_v15['yolo_h'],
        'yolo_conf': yolo_data_v15['yolo_conf'],
        'yolo_area': bbox_area,
        'yolo_aspect_ratio': yolo_data_v15['yolo_w'] / (yolo_data_v15['yolo_h'] + 1e-6),
        'yolo_dist_from_center': np.sqrt((yolo_data_v15['yolo_cx'] - 0.5)**2 + (yolo_data_v15['yolo_cy'] - 0.5)**2),
        'yolo_angle_from_center': np.degrees(np.arctan2(yolo_data_v15['yolo_cy'] - 0.5, yolo_data_v15['yolo_cx'] - 0.5)),
        'yolo_in_center': 1 if np.sqrt((yolo_data_v15['yolo_cx'] - 0.5)**2 + (yolo_data_v15['yolo_cy'] - 0.5)**2) < 0.2 else 0,
        'distance_m': estimated_distance,
        'bearing_deg': estimated_bearing,
        'bearing_sin': np.sin(np.radians(estimated_bearing)),
        'bearing_cos': np.cos(np.radians(estimated_bearing)),
        'distance_x_conf': estimated_distance * yolo_data_v15['yolo_conf'],
        'distance_x_area': estimated_distance * bbox_area,
        'bearing_x_cx': estimated_bearing * yolo_data_v15['yolo_cx'],
        'bearing_x_cy': estimated_bearing * yolo_data_v15['yolo_cy'],
        'altitude_x_distance': 60.0 * estimated_distance,  # Assume avg altitude 60m
        'conf_squared': yolo_data_v15['yolo_conf'] ** 2,
        'conf_sqrt': np.sqrt(yolo_data_v15['yolo_conf']),
        'is_high_conf': 1 if yolo_data_v15['yolo_conf'] > 0.5 else 0,
        'is_detected': yolo_data_v15['yolo_detected'],
        'estimated_lat_offset': (yolo_data_v15['yolo_cy'] - 0.5) * estimated_distance / 111000,
        'estimated_lon_offset': (yolo_data_v15['yolo_cx'] - 0.5) * estimated_distance / (111000 * np.cos(np.radians(CAMERA_LAT))),
        'distance_bin': 2,  # Middle bin
        'altitude_bin': 2,  # Middle bin
        'image_num_normalized': idx / len(test_images)
    }
    
    # ========================================================================
    # Extract v21 features (YOLO v21)
    # ========================================================================
    results_v21 = yolo_v21(img_path, verbose=False)
    
    if len(results_v21[0].boxes) > 0:
        box = results_v21[0].boxes[0]
        bbox = box.xywhn[0].cpu().numpy()
        conf_v21 = float(box.conf[0])
        
        yolo_data_v21 = {
            'yolo_detected': 1,
            'yolo_conf': conf_v21,
            'yolo_cx': float(bbox[0]),
            'yolo_cy': float(bbox[1]),
            'yolo_w': float(bbox[2]),
            'yolo_h': float(bbox[3]),
        }
    else:
        yolo_data_v21 = {
            'yolo_detected': 0,
            'yolo_conf': 0.0,
            'yolo_cx': 0.5,
            'yolo_cy': 0.5,
            'yolo_w': 0.0,
            'yolo_h': 0.0,
        }
    
    # Feature engineering for v21 (same logic)
    bbox_area_v21 = yolo_data_v21['yolo_w'] * yolo_data_v21['yolo_h']
    estimated_distance_v21 = 100.0 if bbox_area_v21 == 0 else max(10, min(200, 50 / (bbox_area_v21 + 0.001)))
    horizontal_offset_v21 = (yolo_data_v21['yolo_cx'] - 0.5) * 60
    estimated_bearing_v21 = (220 + horizontal_offset_v21) % 360
    
    features_v21 = {
        'yolo_cx': yolo_data_v21['yolo_cx'],
        'yolo_cy': yolo_data_v21['yolo_cy'],
        'yolo_w': yolo_data_v21['yolo_w'],
        'yolo_h': yolo_data_v21['yolo_h'],
        'yolo_conf': yolo_data_v21['yolo_conf'],
        'yolo_area': bbox_area_v21,
        'yolo_aspect_ratio': yolo_data_v21['yolo_w'] / (yolo_data_v21['yolo_h'] + 1e-6),
        'yolo_dist_from_center': np.sqrt((yolo_data_v21['yolo_cx'] - 0.5)**2 + (yolo_data_v21['yolo_cy'] - 0.5)**2),
        'yolo_angle_from_center': np.degrees(np.arctan2(yolo_data_v21['yolo_cy'] - 0.5, yolo_data_v21['yolo_cx'] - 0.5)),
        'yolo_in_center': 1 if np.sqrt((yolo_data_v21['yolo_cx'] - 0.5)**2 + (yolo_data_v21['yolo_cy'] - 0.5)**2) < 0.2 else 0,
        'distance_m': estimated_distance_v21,
        'bearing_deg': estimated_bearing_v21,
        'bearing_sin': np.sin(np.radians(estimated_bearing_v21)),
        'bearing_cos': np.cos(np.radians(estimated_bearing_v21)),
        'distance_x_conf': estimated_distance_v21 * yolo_data_v21['yolo_conf'],
        'distance_x_area': estimated_distance_v21 * bbox_area_v21,
        'bearing_x_cx': estimated_bearing_v21 * yolo_data_v21['yolo_cx'],
        'bearing_x_cy': estimated_bearing_v21 * yolo_data_v21['yolo_cy'],
        'altitude_x_distance': 60.0 * estimated_distance_v21,
        'conf_squared': yolo_data_v21['yolo_conf'] ** 2,
        'conf_sqrt': np.sqrt(yolo_data_v21['yolo_conf']),
        'is_high_conf': 1 if yolo_data_v21['yolo_conf'] > 0.5 else 0,
        'is_detected': yolo_data_v21['yolo_detected'],
        'estimated_lat_offset': (yolo_data_v21['yolo_cy'] - 0.5) * estimated_distance_v21 / 111000,
        'estimated_lon_offset': (yolo_data_v21['yolo_cx'] - 0.5) * estimated_distance_v21 / (111000 * np.cos(np.radians(CAMERA_LAT))),
        'distance_bin': 2,
        'altitude_bin': 2,
        'image_num_normalized': idx / len(test_images)
    }
    
    # ========================================================================
    # Predict with both models
    # ========================================================================
    X_v1 = pd.DataFrame([features_v1])[feature_cols_v1].fillna(0)
    X_v21 = pd.DataFrame([features_v21])[feature_cols_v21].fillna(0)
    
    # v1 predictions
    pred_lat_v1 = model_lat_v1.predict(X_v1)[0]
    pred_lon_v1 = model_lon_v1.predict(X_v1)[0]
    pred_alt_v1 = model_alt_v1.predict(X_v1)[0]
    
    # v21 predictions
    pred_lat_v21 = model_lat_v21.predict(X_v21)[0]
    pred_lon_v21 = model_lon_v21.predict(X_v21)[0]
    pred_alt_v21 = model_alt_v21.predict(X_v21)[0]
    
    # Ensemble
    final_lat = w1 * pred_lat_v1 + w2 * pred_lat_v21
    final_lon = w1 * pred_lon_v1 + w2 * pred_lon_v21
    final_alt = w1 * pred_alt_v1 + w2 * pred_alt_v21
    
    predictions.append({
        'image_name': img_name,
        'latitude': final_lat,
        'longitude': final_lon,
        'altitude': final_alt
    })

print(f"\n✅ Processed {len(predictions)} test images")

# ============================================================================
# Step 6: Save Predictions
# ============================================================================
print("\n" + "="*70)
print("Step 6: Save Predictions")
print("="*70)

# Create submission file
submission = pd.DataFrame(predictions)

# Save
submission.to_csv('test_predictions_ensemble.csv', index=False)

print(f"\n✅ Saved: test_predictions_ensemble.csv")
print(f"   Format: image_name, latitude, longitude, altitude")
print(f"   Rows: {len(submission)}")

# Display sample
print("\n📊 Sample predictions:")
print(submission.head(10).to_string(index=False))

# Statistics
print("\n" + "="*70)
print("📈 Prediction Statistics")
print("="*70)
print(f"   Latitude:  {submission['latitude'].min():.6f} - {submission['latitude'].max():.6f}")
print(f"   Longitude: {submission['longitude'].min():.6f} - {submission['longitude'].max():.6f}")
print(f"   Altitude:  {submission['altitude'].min():.2f} - {submission['altitude'].max():.2f} m")

print("\n" + "="*70)
print("✅ Complete! Ready for submission.")
print("="*70)
print(f"\nEnsemble Configuration:")
print(f"   - Validation Score: {ensemble_config['score']:.4f}")
print(f"   - Angle Error: {ensemble_config['angle_error']:.2f}°")
print(f"   - Height Error: {ensemble_config['height_error']:.2f} m")
print(f"   - Range Error: {ensemble_config['range_error']:.2f} m")
print(f"\n📁 Submit: test_predictions_ensemble.csv")
print("="*70)
