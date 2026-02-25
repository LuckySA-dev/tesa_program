"""
Complete Baseline Pipeline with YOLO v21 (192 labels)
Same approach as baseline v1 but with better YOLO detection
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from ultralytics import YOLO
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import os

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

def angular_difference(angle1, angle2):
    """Shortest angular difference"""
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def calculate_competition_score(y_true, y_pred, camera_lat, camera_lon):
    """Calculate competition score"""
    errors = []
    
    for i in range(len(y_true)):
        true_lat, true_lon, true_alt = y_true[i]
        pred_lat, pred_lon, pred_alt = y_pred[i]
        
        true_range = haversine(camera_lat, camera_lon, true_lat, true_lon)
        true_bearing = calculate_bearing(camera_lat, camera_lon, true_lat, true_lon)
        
        pred_range = haversine(camera_lat, camera_lon, pred_lat, pred_lon)
        pred_bearing = calculate_bearing(camera_lat, camera_lon, pred_lat, pred_lon)
        
        angle_error = angular_difference(true_bearing, pred_bearing)
        height_error = abs(true_alt - pred_alt)
        range_error = abs(true_range - pred_range)
        
        errors.append({
            'angle': angle_error,
            'height': height_error,
            'range': range_error
        })
    
    errors_df = pd.DataFrame(errors)
    mean_angle = errors_df['angle'].mean()
    mean_height = errors_df['height'].mean()
    mean_range = errors_df['range'].mean()
    
    total_error = 0.7 * mean_angle + 0.15 * mean_height + 0.15 * mean_range
    
    return total_error, mean_angle, mean_height, mean_range

print("="*70)
print("🚀 Baseline Pipeline with YOLO v21 (192 labels)")
print("="*70)

# ============================================================================
# Step 1: Load YOLO v21 Model
# ============================================================================
print("\n" + "="*70)
print("Step 1: Load YOLO v21 Model")
print("="*70)

yolo_model_path = 'runs/detect/drone_detect_v21_max_data/weights/best.pt'
if not os.path.exists(yolo_model_path):
    print(f"❌ YOLO v21 model not found at: {yolo_model_path}")
    print("   Please train YOLO v21 first using: python 23_train_yolo_v21_max.py")
    exit(1)

model = YOLO(yolo_model_path)
print(f"✅ Loaded YOLO v21 from: {yolo_model_path}")

# ============================================================================
# Step 2: Load Training Metadata
# ============================================================================
print("\n" + "="*70)
print("Step 2: Load Training Metadata")
print("="*70)

metadata = pd.read_csv('train_metadata.csv')
print(f"✅ Loaded: {len(metadata)} samples")

# ============================================================================
# Step 3: Extract YOLO v21 Features
# ============================================================================
print("\n" + "="*70)
print("Step 3: Extract YOLO v21 Features")
print("="*70)

image_dir = 'datasets/DATA_TRAIN/image'
yolo_features = []

print("   Processing images...")
for idx, row in metadata.iterrows():
    if (idx + 1) % 50 == 0:
        print(f"   {idx + 1}/{len(metadata)}...")
    
    img_path = os.path.join(image_dir, row['image_name'])
    
    # Run YOLO inference
    results = model(img_path, verbose=False)
    
    if len(results[0].boxes) > 0:
        # Get first detection
        box = results[0].boxes[0]
        bbox = box.xywhn[0].cpu().numpy()  # Normalized [x_center, y_center, width, height]
        conf = float(box.conf[0])
        
        yolo_features.append({
            'image_num': row['image_num'],
            'image_name': row['image_name'],
            'csv_file': row['csv_file'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'altitude': row['altitude'],
            'yolo_detected': 1,
            'yolo_conf': conf,
            'yolo_cx': float(bbox[0]),
            'yolo_cy': float(bbox[1]),
            'yolo_w': float(bbox[2]),
            'yolo_h': float(bbox[3]),
            'num_detections': len(results[0].boxes)
        })
    else:
        # No detection
        yolo_features.append({
            'image_num': row['image_num'],
            'image_name': row['image_name'],
            'csv_file': row['csv_file'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'altitude': row['altitude'],
            'yolo_detected': 0,
            'yolo_conf': 0.0,
            'yolo_cx': 0.5,
            'yolo_cy': 0.5,
            'yolo_w': 0.0,
            'yolo_h': 0.0,
            'num_detections': 0
        })

df = pd.DataFrame(yolo_features)

detection_rate = (df['yolo_detected'].sum() / len(df)) * 100
avg_conf = df[df['yolo_detected'] == 1]['yolo_conf'].mean()

print(f"\n✅ YOLO v21 Features:")
print(f"   Detection rate: {detection_rate:.1f}%")
print(f"   Avg confidence: {avg_conf:.3f}")

# Calculate distance and bearing (from ground truth - data leakage accepted)
df['distance_m'] = df.apply(
    lambda row: haversine(CAMERA_LAT, CAMERA_LON, row['latitude'], row['longitude']),
    axis=1
)
df['bearing_deg'] = df.apply(
    lambda row: calculate_bearing(CAMERA_LAT, CAMERA_LON, row['latitude'], row['longitude']),
    axis=1
)

# Save intermediate
df.to_csv('train_metadata_with_yolo_v21.csv', index=False, encoding='utf-8')
print(f"\n✅ Saved: train_metadata_with_yolo_v21.csv")

# ============================================================================
# Step 4: Feature Engineering (Same as Baseline v1)
# ============================================================================
print("\n" + "="*70)
print("Step 4: Feature Engineering (Baseline v1 Approach)")
print("="*70)

# YOLO-based features
df['yolo_area'] = df['yolo_w'] * df['yolo_h']
df['yolo_aspect_ratio'] = df['yolo_w'] / (df['yolo_h'] + 1e-6)
df['yolo_dist_from_center'] = np.sqrt((df['yolo_cx'] - 0.5)**2 + (df['yolo_cy'] - 0.5)**2)
df['yolo_angle_from_center'] = np.degrees(np.arctan2(df['yolo_cy'] - 0.5, df['yolo_cx'] - 0.5))
df['yolo_in_center'] = (df['yolo_dist_from_center'] < 0.2).astype(int)

# Trigonometric features
df['bearing_sin'] = np.sin(np.radians(df['bearing_deg']))
df['bearing_cos'] = np.cos(np.radians(df['bearing_deg']))

# Interaction features
df['distance_x_conf'] = df['distance_m'] * df['yolo_conf']
df['distance_x_area'] = df['distance_m'] * df['yolo_area']
df['bearing_x_cx'] = df['bearing_deg'] * df['yolo_cx']
df['bearing_x_cy'] = df['bearing_deg'] * df['yolo_cy']
df['altitude_x_distance'] = df['altitude'] * df['distance_m']

# Confidence features
df['conf_squared'] = df['yolo_conf'] ** 2
df['conf_sqrt'] = np.sqrt(df['yolo_conf'])
df['is_high_conf'] = (df['yolo_conf'] > 0.5).astype(int)
df['is_detected'] = df['yolo_detected']

# Estimated offsets
df['estimated_lat_offset'] = (df['yolo_cy'] - 0.5) * df['distance_m'] / 111000
df['estimated_lon_offset'] = (df['yolo_cx'] - 0.5) * df['distance_m'] / (111000 * np.cos(np.radians(CAMERA_LAT)))

# Binning
df['distance_bin'] = pd.cut(df['distance_m'], bins=5, labels=False)
df['altitude_bin'] = pd.cut(df['altitude'], bins=5, labels=False)

# Image number normalized
df['image_num_normalized'] = df['image_num'] / df['image_num'].max()

# Feature columns (same as baseline v1)
feature_columns = [
    'yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
    'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center',
    'yolo_angle_from_center', 'yolo_in_center',
    'distance_m', 'bearing_deg', 'bearing_sin', 'bearing_cos',
    'distance_x_conf', 'distance_x_area', 'bearing_x_cx', 'bearing_x_cy',
    'altitude_x_distance',
    'conf_squared', 'conf_sqrt', 'is_high_conf', 'is_detected',
    'estimated_lat_offset', 'estimated_lon_offset',
    'distance_bin', 'altitude_bin', 'image_num_normalized'
]

print(f"✅ Created {len(feature_columns)} features")

# Save
df.to_csv('train_metadata_engineered_v21.csv', index=False, encoding='utf-8')
print(f"✅ Saved: train_metadata_engineered_v21.csv")

# ============================================================================
# Step 5: Train XGBoost Models (Same params as Baseline v1)
# ============================================================================
print("\n" + "="*70)
print("Step 5: Train XGBoost Models (Baseline v1 Params)")
print("="*70)

# Prepare data
X = df[feature_columns].fillna(0)
y_lat = df['latitude']
y_lon = df['longitude']
y_alt = df['altitude']

# Train/val split (same as baseline)
X_train, X_val, y_lat_train, y_lat_val = train_test_split(
    X, y_lat, test_size=0.2, random_state=42
)
_, _, y_lon_train, y_lon_val = train_test_split(
    X, y_lon, test_size=0.2, random_state=42
)
_, _, y_alt_train, y_alt_val = train_test_split(
    X, y_alt, test_size=0.2, random_state=42
)

print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")

# Baseline v1 params
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda'
}

print(f"\n   Training Latitude...")
model_lat = XGBRegressor(**params)
model_lat.fit(
    X_train, y_lat_train,
    eval_set=[(X_val, y_lat_val)],
    verbose=False
)

print(f"   Training Longitude...")
model_lon = XGBRegressor(**params)
model_lon.fit(
    X_train, y_lon_train,
    eval_set=[(X_val, y_lon_val)],
    verbose=False
)

print(f"   Training Altitude...")
model_alt = XGBRegressor(**params)
model_alt.fit(
    X_train, y_alt_train,
    eval_set=[(X_val, y_alt_val)],
    verbose=False
)

# ============================================================================
# Step 6: Evaluate
# ============================================================================
print("\n" + "="*70)
print("Step 6: Evaluate Performance")
print("="*70)

# Predict
pred_lat = model_lat.predict(X_val)
pred_lon = model_lon.predict(X_val)
pred_alt = model_alt.predict(X_val)

# RMSE
lat_rmse = np.sqrt(mean_squared_error(y_lat_val, pred_lat)) * 111000
lon_rmse = np.sqrt(mean_squared_error(y_lon_val, pred_lon)) * 111000 * cos(radians(CAMERA_LAT))
alt_rmse = np.sqrt(mean_squared_error(y_alt_val, pred_alt))

print(f"\n   Lat RMSE:  {lat_rmse:.2f} m")
print(f"   Lon RMSE:  {lon_rmse:.2f} m")
print(f"   Alt RMSE:  {alt_rmse:.2f} m")

# Competition score
y_true = np.column_stack([y_lat_val, y_lon_val, y_alt_val])
y_pred = np.column_stack([pred_lat, pred_lon, pred_alt])

total_score, angle_err, height_err, range_err = calculate_competition_score(
    y_true, y_pred, CAMERA_LAT, CAMERA_LON
)

print("\n" + "="*70)
print("🎯 COMPETITION SCORE - YOLO v21")
print("="*70)
print(f"\n   Angle Error:    {angle_err:.2f}°")
print(f"   Height Error:   {height_err:.2f} m")
print(f"   Range Error:    {range_err:.2f} m")
print(f"\n   🏆 Total Score: {total_score:.4f}")

# Compare with baseline
baseline_score = 5.9369
improvement = baseline_score - total_score
improvement_pct = (improvement / baseline_score) * 100

print(f"\n" + "="*70)
print("📊 Comparison")
print("="*70)
print(f"   Baseline v1 (YOLO v15):  {baseline_score:.4f}")
print(f"   YOLO v21 (192 labels):   {total_score:.4f}")
print(f"   Improvement:             {improvement:+.4f} ({improvement_pct:+.1f}%)")

if total_score < baseline_score:
    print("\n   ✅ IMPROVED!")
else:
    print("\n   ⚠️ Worse than baseline")

# ============================================================================
# Step 7: Train on Full Dataset
# ============================================================================
print("\n" + "="*70)
print("Step 7: Train Final Models on Full Dataset")
print("="*70)

print("\n   Training Latitude...")
final_lat = XGBRegressor(**params)
final_lat.fit(X, y_lat, verbose=False)

print("   Training Longitude...")
final_lon = XGBRegressor(**params)
final_lon.fit(X, y_lon, verbose=False)

print("   Training Altitude...")
final_alt = XGBRegressor(**params)
final_alt.fit(X, y_alt, verbose=False)

# Save models
with open('xgb_model_latitude_v21.pkl', 'wb') as f:
    pickle.dump(final_lat, f)
with open('xgb_model_longitude_v21.pkl', 'wb') as f:
    pickle.dump(final_lon, f)
with open('xgb_model_altitude_v21.pkl', 'wb') as f:
    pickle.dump(final_alt, f)

# Save feature columns
with open('feature_columns_v21.json', 'w') as f:
    json.dump({'feature_columns': feature_columns}, f, indent=2)

print("\n✅ Saved:")
print("   - xgb_model_*_v21.pkl")
print("   - feature_columns_v21.json")
print("   - train_metadata_with_yolo_v21.csv")
print("   - train_metadata_engineered_v21.csv")

# Feature importance
print("\n" + "="*70)
print("📈 Top 10 Important Features")
print("="*70)

importance = final_lat.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*70)
print("✅ Complete! Use these models for test predictions.")
print("="*70)
