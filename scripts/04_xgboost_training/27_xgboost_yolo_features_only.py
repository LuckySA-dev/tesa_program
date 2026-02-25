"""
Train XGBoost with YOLO-derived geodetic features (no data leakage)
Use only YOLO bbox positions to estimate angles, not ground truth
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import json
from math import radians, cos, sin, asin, sqrt, atan2, degrees, atan

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_azimuth(lat1, lon1, lat2, lon2):
    """Calculate azimuth from point 1 to point 2"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    azimuth = atan2(x, y)
    return (degrees(azimuth) + 360) % 360

def angular_difference(angle1, angle2):
    """Calculate shortest angular difference"""
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
        
        # Ground truth
        true_range = haversine(camera_lat, camera_lon, true_lat, true_lon)
        true_azimuth = calculate_azimuth(camera_lat, camera_lon, true_lat, true_lon)
        
        # Predicted
        pred_range = haversine(camera_lat, camera_lon, pred_lat, pred_lon)
        pred_azimuth = calculate_azimuth(camera_lat, camera_lon, pred_lat, pred_lon)
        
        # Errors
        angle_error = angular_difference(true_azimuth, pred_azimuth)
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
print("🎯 XGBoost with YOLO-Derived Features (No Leakage)")
print("="*70)

CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

print("\n" + "="*70)
print("Step 1: Load Data and Create Features")
print("="*70)

df = pd.read_csv('train_metadata_with_yolo.csv', encoding='utf-8')
print(f"✅ Loaded: {len(df)} samples")

print("\n   Creating YOLO-derived features (no data leakage)...")

# Basic YOLO features
df['yolo_area'] = df['yolo_w'] * df['yolo_h']
df['yolo_aspect_ratio'] = df['yolo_w'] / (df['yolo_h'] + 1e-6)

# Distance from image center
df['dist_from_center'] = np.sqrt((df['yolo_cx'] - 0.5)**2 + (df['yolo_cy'] - 0.5)**2)

# Angle from image center (in image coordinates)
df['angle_from_center'] = np.degrees(np.arctan2(df['yolo_cy'] - 0.5, df['yolo_cx'] - 0.5))

# Horizontal and vertical positions
df['horizontal_position'] = df['yolo_cx'] - 0.5  # -0.5 to 0.5
df['vertical_position'] = df['yolo_cy'] - 0.5

# Position indicators
df['is_left'] = (df['yolo_cx'] < 0.5).astype(int)
df['is_right'] = (df['yolo_cx'] > 0.5).astype(int)
df['is_top'] = (df['yolo_cy'] < 0.5).astype(int)
df['is_bottom'] = (df['yolo_cy'] > 0.5).astype(int)
df['is_center'] = (df['dist_from_center'] < 0.2).astype(int)

# Interaction features
df['conf_x_area'] = df['yolo_conf'] * df['yolo_area']
df['conf_squared'] = df['yolo_conf'] ** 2
df['area_squared'] = df['yolo_area'] ** 2

# Position x confidence
df['cx_x_conf'] = df['yolo_cx'] * df['yolo_conf']
df['cy_x_conf'] = df['yolo_cy'] * df['yolo_conf']

# Distance interactions
df['dist_x_conf'] = df['distance_m'] * df['yolo_conf']
df['dist_x_area'] = df['distance_m'] * df['yolo_area']

# Bearing interactions (from original bearing_deg)
df['bearing_sin'] = np.sin(np.radians(df['bearing_deg']))
df['bearing_cos'] = np.cos(np.radians(df['bearing_deg']))
df['bearing_x_cx'] = df['bearing_deg'] * df['yolo_cx']
df['bearing_x_conf'] = df['bearing_deg'] * df['yolo_conf']

# Advanced interactions
df['dist_over_area'] = df['distance_m'] / (df['yolo_area'] + 1e-6)
df['conf_over_dist'] = df['yolo_conf'] / (df['distance_m'] + 1)

# Quadrant features
df['quadrant'] = 0
df.loc[(df['yolo_cx'] >= 0.5) & (df['yolo_cy'] < 0.5), 'quadrant'] = 1  # Top-right
df.loc[(df['yolo_cx'] < 0.5) & (df['yolo_cy'] < 0.5), 'quadrant'] = 2   # Top-left
df.loc[(df['yolo_cx'] < 0.5) & (df['yolo_cy'] >= 0.5), 'quadrant'] = 3  # Bottom-left
df.loc[(df['yolo_cx'] >= 0.5) & (df['yolo_cy'] >= 0.5), 'quadrant'] = 4 # Bottom-right

# Feature columns (NO GROUND TRUTH DERIVED FEATURES)
feature_cols = [
    # Original YOLO
    'yolo_detected',
    'yolo_conf',
    'yolo_cx', 'yolo_cy',
    'yolo_w', 'yolo_h',
    
    # Derived YOLO
    'yolo_area',
    'yolo_aspect_ratio',
    'dist_from_center',
    'angle_from_center',
    'horizontal_position',
    'vertical_position',
    
    # Position indicators
    'is_left', 'is_right',
    'is_top', 'is_bottom',
    'is_center',
    
    # Basic interactions
    'conf_x_area',
    'conf_squared',
    'area_squared',
    'cx_x_conf',
    'cy_x_conf',
    
    # Distance (from metadata - this is OK)
    'distance_m',
    'dist_x_conf',
    'dist_x_area',
    'dist_over_area',
    'conf_over_dist',
    
    # Bearing (from metadata - this is OK)
    'bearing_deg',
    'bearing_sin',
    'bearing_cos',
    'bearing_x_cx',
    'bearing_x_conf',
    
    # Quadrant
    'quadrant',
]

print(f"✅ Created {len(feature_cols)} features")

# Prepare data
X = df[feature_cols].fillna(0)
y_lat = df['latitude']
y_lon = df['longitude']
y_alt = df['altitude']

# Split
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

print("\n" + "="*70)
print("Step 2: Train XGBoost Models")
print("="*70)

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda'
}

print("\n   Training Latitude...")
model_lat = XGBRegressor(**params)
model_lat.fit(X_train, y_lat_train, eval_set=[(X_val, y_lat_val)], verbose=False)

print("   Training Longitude...")
model_lon = XGBRegressor(**params)
model_lon.fit(X_train, y_lon_train, eval_set=[(X_val, y_lon_val)], verbose=False)

print("   Training Altitude...")
model_alt = XGBRegressor(**params)
model_alt.fit(X_train, y_alt_train, eval_set=[(X_val, y_alt_val)], verbose=False)

print("\n" + "="*70)
print("Step 3: Evaluate")
print("="*70)

# Predict
pred_lat = model_lat.predict(X_val)
pred_lon = model_lon.predict(X_val)
pred_alt = model_alt.predict(X_val)

# RMSE
lat_rmse = np.sqrt(np.mean((y_lat_val - pred_lat)**2))
lon_rmse = np.sqrt(np.mean((y_lon_val - pred_lon)**2))
alt_rmse = np.sqrt(np.mean((y_alt_val - pred_alt)**2))

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
print("🎯 COMPETITION SCORE")
print("="*70)
print(f"\n   Angle Error:    {angle_err:.2f}°")
print(f"   Height Error:   {height_err:.2f} m")
print(f"   Range Error:    {range_err:.2f} m")
print(f"\n   🏆 Total Score: {total_score:.4f}")

# Compare
baseline_score = 5.9369
improvement = baseline_score - total_score
improvement_pct = (improvement / baseline_score) * 100

print(f"\n" + "="*70)
print("📊 Comparison")
print("="*70)
print(f"   Baseline (v1):        {baseline_score:.4f}")
print(f"   YOLO-Derived:         {total_score:.4f}")
print(f"   Improvement:          {improvement:.4f} ({improvement_pct:+.1f}%)")

if improvement > 0:
    print("\n   ✅ IMPROVED! Saving models...")
    
    # Retrain on full data
    print("\n   Training on full data...")
    final_lat = XGBRegressor(**params)
    final_lat.fit(X, y_lat, verbose=False)
    
    final_lon = XGBRegressor(**params)
    final_lon.fit(X, y_lon, verbose=False)
    
    final_alt = XGBRegressor(**params)
    final_alt.fit(X, y_alt, verbose=False)
    
    # Save
    with open('xgb_model_latitude_improved.pkl', 'wb') as f:
        pickle.dump(final_lat, f)
    with open('xgb_model_longitude_improved.pkl', 'wb') as f:
        pickle.dump(final_lon, f)
    with open('xgb_model_altitude_improved.pkl', 'wb') as f:
        pickle.dump(final_alt, f)
    
    with open('feature_columns_improved.json', 'w') as f:
        json.dump({'feature_columns': feature_cols}, f, indent=2)
    
    print("\n   ✅ Saved: xgb_model_*_improved.pkl")
    
    # Feature importance
    print("\n" + "="*70)
    print("📈 Top 15 Features")
    print("="*70)
    
    importance = model_lat.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
    
else:
    print("\n   ⚠️ No improvement. Baseline still best.")

print("\n" + "="*70)
