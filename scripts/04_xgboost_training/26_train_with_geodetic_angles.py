"""
Train XGBoost with improved geodetic angle calculations
Using more accurate azimuth/elevation angles based on spherical geometry
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
    """
    Calculate azimuth (bearing) from point 1 to point 2
    More accurate than simple bearing calculation
    Returns angle in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    azimuth = atan2(x, y)
    azimuth_deg = (degrees(azimuth) + 360) % 360
    
    return azimuth_deg

def calculate_elevation_angle(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Calculate elevation angle from point 1 to point 2
    Takes into account altitude difference and horizontal distance
    Returns angle in degrees (-90 to +90)
    """
    # Horizontal distance
    horizontal_dist = haversine(lat1, lon1, lat2, lon2)
    
    # Vertical distance (altitude difference)
    vertical_dist = alt2 - alt1
    
    # Elevation angle
    if horizontal_dist > 0:
        elevation = degrees(atan(vertical_dist / horizontal_dist))
    else:
        elevation = 90.0 if vertical_dist > 0 else -90.0
    
    return elevation

def calculate_3d_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Calculate 3D distance between two points (lat, lon, alt)
    """
    horizontal_dist = haversine(lat1, lon1, lat2, lon2)
    vertical_dist = alt2 - alt1
    
    dist_3d = sqrt(horizontal_dist**2 + vertical_dist**2)
    return dist_3d

def angular_difference(angle1, angle2):
    """
    Calculate shortest angular difference between two angles
    Returns value between 0 and 180 degrees
    """
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def calculate_competition_score(y_true, y_pred, camera_lat, camera_lon, camera_alt=0):
    """
    Calculate competition score using geodetic angles
    """
    errors = []
    
    for i in range(len(y_true)):
        true_lat, true_lon, true_alt = y_true[i]
        pred_lat, pred_lon, pred_alt = y_pred[i]
        
        # Ground truth metrics
        true_range = haversine(camera_lat, camera_lon, true_lat, true_lon)
        true_azimuth = calculate_azimuth(camera_lat, camera_lon, true_lat, true_lon)
        
        # Predicted metrics
        pred_range = haversine(camera_lat, camera_lon, pred_lat, pred_lon)
        pred_azimuth = calculate_azimuth(camera_lat, camera_lon, pred_lat, pred_lon)
        
        # Angle error (azimuth difference)
        angle_error = angular_difference(true_azimuth, pred_azimuth)
        
        # Height error
        height_error = abs(true_alt - pred_alt)
        
        # Range error
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
print("🌍 XGBoost with Geodetic Angle Features")
print("="*70)

# Camera position
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010
CAMERA_ALT = 0  # Assume ground level

print("\n" + "="*70)
print("Step 1: Load Data and Create Geodetic Features")
print("="*70)

# Load YOLO features from baseline (best performing)
df = pd.read_csv('train_metadata_with_yolo.csv', encoding='utf-8')
print(f"✅ Loaded: {len(df)} samples")

# Calculate geodetic features
print("\n   Calculating geodetic angles...")

# Azimuth (horizontal angle) from camera to drone
df['azimuth_deg'] = df.apply(
    lambda row: calculate_azimuth(
        CAMERA_LAT, CAMERA_LON, 
        row['latitude'], row['longitude']
    ), axis=1
)

# Elevation angle from camera to drone
df['elevation_deg'] = df.apply(
    lambda row: calculate_elevation_angle(
        CAMERA_LAT, CAMERA_LON, CAMERA_ALT,
        row['latitude'], row['longitude'], row['altitude']
    ), axis=1
)

# 3D distance from camera to drone
df['distance_3d'] = df.apply(
    lambda row: calculate_3d_distance(
        CAMERA_LAT, CAMERA_LON, CAMERA_ALT,
        row['latitude'], row['longitude'], row['altitude']
    ), axis=1
)

# Trigonometric features for azimuth
df['azimuth_sin'] = np.sin(np.radians(df['azimuth_deg']))
df['azimuth_cos'] = np.cos(np.radians(df['azimuth_deg']))

# Trigonometric features for elevation
df['elevation_sin'] = np.sin(np.radians(df['elevation_deg']))
df['elevation_cos'] = np.cos(np.radians(df['elevation_deg']))

# YOLO-based features (from baseline)
df['yolo_area'] = df['yolo_w'] * df['yolo_h']
df['yolo_aspect_ratio'] = df['yolo_w'] / (df['yolo_h'] + 1e-6)
df['yolo_center_x'] = df['yolo_cx']
df['yolo_center_y'] = df['yolo_cy']

# Distance from image center
df['dist_from_center'] = np.sqrt((df['yolo_center_x'] - 0.5)**2 + (df['yolo_center_y'] - 0.5)**2)

# Interaction features between YOLO and geodetic
df['conf_x_elevation'] = df['yolo_conf'] * df['elevation_deg']
df['area_x_distance3d'] = df['yolo_area'] * df['distance_3d'] / 1000  # Normalize
df['azimuth_x_centerx'] = df['azimuth_deg'] * df['yolo_center_x']
df['elevation_x_centery'] = df['elevation_deg'] * df['yolo_center_y']

# Angular velocity indicators (change per unit distance)
df['azimuth_per_meter'] = df['azimuth_deg'] / (df['distance_m'] + 1)
df['elevation_per_meter'] = df['elevation_deg'] / (df['distance_m'] + 1)

# Feature columns
feature_cols = [
    # YOLO bbox features
    'yolo_detected',
    'yolo_conf',
    'yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h',
    'yolo_area',
    'yolo_aspect_ratio',
    'dist_from_center',
    
    # Geodetic angles
    'azimuth_deg',
    'elevation_deg',
    'azimuth_sin', 'azimuth_cos',
    'elevation_sin', 'elevation_cos',
    
    # Distance features
    'distance_m',
    'distance_3d',
    
    # Interactions
    'conf_x_elevation',
    'area_x_distance3d',
    'azimuth_x_centerx',
    'elevation_x_centery',
    'azimuth_per_meter',
    'elevation_per_meter',
]

print(f"✅ Created {len(feature_cols)} features")

# Fill NaN values (for images without detection)
X = df[feature_cols].fillna(0)
y_lat = df['latitude']
y_lon = df['longitude']
y_alt = df['altitude']

# Train/val split
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

# XGBoost parameters (baseline)
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
model_lat.fit(
    X_train, y_lat_train,
    eval_set=[(X_val, y_lat_val)],
    verbose=False
)

print("   Training Longitude...")
model_lon = XGBRegressor(**params)
model_lon.fit(
    X_train, y_lon_train,
    eval_set=[(X_val, y_lon_val)],
    verbose=False
)

print("   Training Altitude...")
model_alt = XGBRegressor(**params)
model_alt.fit(
    X_train, y_alt_train,
    eval_set=[(X_val, y_alt_val)],
    verbose=False
)

print("\n" + "="*70)
print("Step 3: Evaluate Performance")
print("="*70)

# Predictions
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
    y_true, y_pred, CAMERA_LAT, CAMERA_LON, CAMERA_ALT
)

print("\n" + "="*70)
print("🎯 COMPETITION SCORE - Geodetic Angles")
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
print("📊 Comparison with Baseline")
print("="*70)
print(f"   Baseline (v1):     {baseline_score:.4f}")
print(f"   Geodetic Angles:   {total_score:.4f}")
print(f"   Improvement:       {improvement:.4f} ({improvement_pct:+.1f}%)")

if improvement > 0:
    print("\n   ✅ IMPROVED!")
    
    # Train on full dataset
    print("\n" + "="*70)
    print("Step 4: Train Final Models on Full Data")
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
    with open('xgb_model_latitude_geodetic.pkl', 'wb') as f:
        pickle.dump(final_lat, f)
    with open('xgb_model_longitude_geodetic.pkl', 'wb') as f:
        pickle.dump(final_lon, f)
    with open('xgb_model_altitude_geodetic.pkl', 'wb') as f:
        pickle.dump(final_alt, f)
    
    # Save feature columns
    with open('feature_columns_geodetic.json', 'w') as f:
        json.dump({'feature_columns': feature_cols}, f, indent=2)
    
    # Save enhanced metadata with geodetic features
    df.to_csv('train_metadata_geodetic.csv', index=False, encoding='utf-8')
    
    print("\n   ✅ Saved:")
    print("      - xgb_model_*_geodetic.pkl")
    print("      - feature_columns_geodetic.json")
    print("      - train_metadata_geodetic.csv")
    
    # Feature importance
    print("\n" + "="*70)
    print("📈 Top 10 Important Features")
    print("="*70)
    
    # Get feature importance from latitude model
    importance = model_lat.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
else:
    print("\n   ⚠️ No improvement. Baseline is still better.")

print("\n" + "="*70)
