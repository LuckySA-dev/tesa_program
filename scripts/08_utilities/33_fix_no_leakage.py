"""
แก้ปัญหา Data Leakage - ลบ distance_m และ bearing_deg ออกจาก features
ใช้เฉพาะ YOLO bbox features ที่ได้จากการ detect จริง
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import joblib
import os

CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360

def angular_difference(angle1, angle2):
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

print("="*70)
print("🔧 แก้ปัญหา Data Leakage - ลบ distance_m และ bearing_deg")
print("="*70)

# Load data with YOLO features
print("\n📂 Loading data...")
df = pd.read_csv('train_metadata_with_yolo_v21.csv', encoding='utf-8')
print(f"✅ Loaded {len(df)} samples")

# Features WITHOUT data leakage
print("\n🎯 Selecting features WITHOUT data leakage:")
print("   ❌ Removed: distance_m (calculated from ground truth)")
print("   ❌ Removed: bearing_deg (calculated from ground truth)")
print("   ✅ Keep only: YOLO bbox features")

# Original YOLO features (no leakage)
yolo_features = [
    'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
    'bbox_area', 'bbox_aspect_ratio', 'confidence',
    'bbox_center_x', 'bbox_center_y'
]

# Additional engineered features (no leakage)
engineered_features = [
    'bbox_x_offset',  # distance from image center
    'bbox_y_offset',
    'bbox_x_norm',    # normalized position
    'bbox_y_norm'
]

# Check which features exist
available_features = [f for f in yolo_features + engineered_features if f in df.columns]
print(f"\n✅ Available features ({len(available_features)}):")
for i, f in enumerate(available_features, 1):
    print(f"   {i:2d}. {f}")

# Add more features if they don't exist
if 'bbox_x_offset' not in df.columns:
    df['bbox_x_offset'] = df['bbox_center_x'] - 0.5  # offset from center
    df['bbox_y_offset'] = df['bbox_center_y'] - 0.5
    df['bbox_x_norm'] = df['bbox_x']  # already normalized
    df['bbox_y_norm'] = df['bbox_y']
    available_features = yolo_features + engineered_features

print(f"\n📊 Final feature count: {len(available_features)}")

# Prepare data
X = df[available_features].copy()
y_lat = df['latitude_deg']
y_lon = df['longitude_deg']
y_alt = df['altitude_m']

# Train-test split
X_train, X_test, y_lat_train, y_lat_test = train_test_split(
    X, y_lat, test_size=0.2, random_state=42
)
_, _, y_lon_train, y_lon_test = train_test_split(
    X, y_lon, test_size=0.2, random_state=42
)
_, _, y_alt_train, y_alt_test = train_test_split(
    X, y_alt, test_size=0.2, random_state=42
)

print(f"\n📊 Data split:")
print(f"   Training:   {len(X_train)} samples")
print(f"   Validation: {len(X_test)} samples")

# Train models
print("\n" + "="*70)
print("🏋️ Training XGBoost models (NO DATA LEAKAGE)...")
print("="*70)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda'
}

# Train Latitude model
print("\n1️⃣ Training Latitude model...")
model_lat = xgb.XGBRegressor(**params)
model_lat.fit(X_train, y_lat_train, verbose=False)
pred_lat = model_lat.predict(X_test)
r2_lat = r2_score(y_lat_test, pred_lat)
rmse_lat = np.sqrt(mean_squared_error(y_lat_test, pred_lat))
print(f"   ✅ R² = {r2_lat:.4f}, RMSE = {rmse_lat:.6f}°")

# Train Longitude model
print("\n2️⃣ Training Longitude model...")
model_lon = xgb.XGBRegressor(**params)
model_lon.fit(X_train, y_lon_train, verbose=False)
pred_lon = model_lon.predict(X_test)
r2_lon = r2_score(y_lon_test, pred_lon)
rmse_lon = np.sqrt(mean_squared_error(y_lon_test, pred_lon))
print(f"   ✅ R² = {r2_lon:.4f}, RMSE = {rmse_lon:.6f}°")

# Train Altitude model
print("\n3️⃣ Training Altitude model...")
model_alt = xgb.XGBRegressor(**params)
model_alt.fit(X_train, y_alt_train, verbose=False)
pred_alt = model_alt.predict(X_test)
r2_alt = r2_score(y_alt_test, pred_alt)
rmse_alt = np.sqrt(mean_squared_error(y_alt_test, pred_alt))
print(f"   ✅ R² = {r2_alt:.4f}, RMSE = {rmse_alt:.2f} m")

# Calculate competition score
print("\n" + "="*70)
print("📊 Competition Score Calculation")
print("="*70)

# Get ground truth for test set
test_indices = X_test.index
df_test = df.loc[test_indices].copy()

# Calculate errors
errors_angle = []
errors_height = []
errors_range = []

for idx in range(len(X_test)):
    true_lat = y_lat_test.iloc[idx]
    true_lon = y_lon_test.iloc[idx]
    true_alt = y_alt_test.iloc[idx]
    
    pred_lat_val = pred_lat[idx]
    pred_lon_val = pred_lon[idx]
    pred_alt_val = pred_alt[idx]
    
    # Angle error
    true_bearing = calculate_bearing(CAMERA_LAT, CAMERA_LON, true_lat, true_lon)
    pred_bearing = calculate_bearing(CAMERA_LAT, CAMERA_LON, pred_lat_val, pred_lon_val)
    angle_error = angular_difference(true_bearing, pred_bearing)
    errors_angle.append(angle_error)
    
    # Height error
    height_error = abs(true_alt - pred_alt_val)
    errors_height.append(height_error)
    
    # Range error
    true_dist = haversine(CAMERA_LAT, CAMERA_LON, true_lat, true_lon)
    pred_dist = haversine(CAMERA_LAT, CAMERA_LON, pred_lat_val, pred_lon_val)
    range_error = abs(true_dist - pred_dist)
    errors_range.append(range_error)

mean_angle = np.mean(errors_angle)
mean_height = np.mean(errors_height)
mean_range = np.mean(errors_range)

total_error = 0.7 * mean_angle + 0.15 * mean_height + 0.15 * mean_range

print(f"\nValidation Errors:")
print(f"  Angle Error:  {mean_angle:.2f}° (weight: 70%)")
print(f"  Height Error: {mean_height:.2f} m (weight: 15%)")
print(f"  Range Error:  {mean_range:.2f} m (weight: 15%)")
print(f"\n🎯 Total Error: {total_error:.4f}")

# Compare with baseline
baseline_error = 5.9369
improvement = ((baseline_error - total_error) / baseline_error) * 100

print(f"\n📊 Comparison:")
print(f"  Baseline (v1):      {baseline_error:.4f}")
print(f"  Ensemble (v1+v21):  5.2838")
print(f"  No Leakage (v33):   {total_error:.4f}")

if total_error < baseline_error:
    print(f"  ✅ Better than baseline by {improvement:.1f}%")
elif total_error < 5.2838:
    print(f"  ✅ Better than ensemble!")
else:
    print(f"  ❌ Worse than baseline by {-improvement:.1f}%")

# Save models
print("\n" + "="*70)
print("💾 Saving models...")
print("="*70)

os.makedirs('models_no_leakage', exist_ok=True)
joblib.dump(model_lat, 'models_no_leakage/xgb_model_lat_no_leakage.pkl')
joblib.dump(model_lon, 'models_no_leakage/xgb_model_lon_no_leakage.pkl')
joblib.dump(model_alt, 'models_no_leakage/xgb_model_alt_no_leakage.pkl')

# Save feature list
import json
with open('feature_columns_no_leakage.json', 'w') as f:
    json.dump(available_features, f, indent=2)

print(f"✅ Saved models to: models_no_leakage/")
print(f"✅ Saved features to: feature_columns_no_leakage.json")

# Feature importance
print("\n" + "="*70)
print("🔍 Top 10 Most Important Features")
print("="*70)

importance_scores = {}
for name, model in [('Latitude', model_lat), ('Longitude', model_lon), ('Altitude', model_alt)]:
    importance_scores[name] = model.feature_importances_

avg_importance = np.mean([importance_scores['Latitude'], 
                          importance_scores['Longitude'], 
                          importance_scores['Altitude']], axis=0)

feature_importance = list(zip(available_features, avg_importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(feature_importance[:10], 1):
    print(f"  {i:2d}. {feature:20s}: {importance:.4f}")

print("\n" + "="*70)
print("✅ Training Complete!")
print("="*70)
print(f"\n🎯 Final Score (No Data Leakage): {total_error:.4f}")
print(f"   Angle:  {mean_angle:.2f}° → {0.7*mean_angle:.2f} points")
print(f"   Height: {mean_height:.2f}m → {0.15*mean_height:.2f} points")
print(f"   Range:  {mean_range:.2f}m → {0.15*mean_range:.2f} points")

if total_error < 5.2838:
    print("\n🏆 BETTER THAN ENSEMBLE! Use this for submission!")
else:
    print(f"\n💡 Suggestion: Ensemble is still better ({5.2838:.4f})")
    print("   Consider hybrid approach or more feature engineering")
