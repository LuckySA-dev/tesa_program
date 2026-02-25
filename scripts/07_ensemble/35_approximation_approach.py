"""
ใช้การประมาณค่าแทน GCP - Interpolation & Regression Based Approach
ไม่ต้องเก็บ GCP เพิ่ม ใช้ข้อมูลที่มีประมาณค่าส่วนที่ขาด
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import xgboost as xgb
import joblib
import json

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
print("🔧 Approximation-Based Approach (No Need for Extra GCP)")
print("="*70)

print("""
แนวคิด: แทนที่จะเก็บ GCP เพิ่ม
→ ใช้ Machine Learning เรียนรู้ความสัมพันธ์จากข้อมูลที่มี
→ ประมาณค่าสำหรับบริเวณที่ไม่มีข้อมูล

วิธีการ:
1. เรียนรู้ความสัมพันธ์: bbox → distance/bearing
2. ใช้ regression model แทน geometric formula  
3. Interpolate สำหรับบริเวณที่ไม่มี GCP
""")

# Load data
print("\n📂 Loading training data...")
df = pd.read_csv('train_metadata_engineered.csv', encoding='utf-8')
print(f"✅ Loaded {len(df)} samples")

# Calculate true values
df['true_distance'] = df.apply(
    lambda row: haversine(CAMERA_LAT, CAMERA_LON,
                         row['latitude_deg'], row['longitude_deg']),
    axis=1
)

df['true_bearing'] = df.apply(
    lambda row: calculate_bearing(CAMERA_LAT, CAMERA_LON,
                                  row['latitude_deg'], row['longitude_deg']),
    axis=1
)

print(f"\n📊 Data distribution:")
print(f"  Bearing: {df['true_bearing'].min():.1f}° - {df['true_bearing'].max():.1f}°")
print(f"  Distance: {df['true_distance'].min():.1f} - {df['true_distance'].max():.1f} m")
print(f"  Altitude: {df['altitude_m'].min():.1f} - {df['altitude_m'].max():.1f} m")

# Filter detected samples
df_detected = df[df['yolo_detected'] == True].copy()
print(f"\nDetected samples: {len(df_detected)}/{len(df)} ({len(df_detected)/len(df)*100:.1f}%)")

print("\n" + "="*70)
print("📐 Method: Learn bbox → distance/bearing mapping")
print("="*70)

# Features from YOLO bbox (available at test time)
bbox_features = [
    'yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
    'yolo_area', 'yolo_aspect_ratio',
    'yolo_dist_from_center', 'yolo_angle_from_center'
]

# Verify features exist
available_bbox_features = [f for f in bbox_features if f in df_detected.columns]
print(f"\n✅ Available bbox features: {len(available_bbox_features)}")
for f in available_bbox_features:
    print(f"   - {f}")

# Prepare data
X = df_detected[available_bbox_features].copy()
y_distance = df_detected['true_distance']
y_bearing = df_detected['true_bearing']
y_altitude = df_detected['altitude_m']

# Train-test split
X_train, X_test, y_dist_train, y_dist_test = train_test_split(
    X, y_distance, test_size=0.2, random_state=42
)
_, _, y_bear_train, y_bear_test = train_test_split(
    X, y_bearing, test_size=0.2, random_state=42
)
_, _, y_alt_train, y_alt_test = train_test_split(
    X, y_altitude, test_size=0.2, random_state=42
)

print(f"\n📊 Split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Validation: {len(X_test)} samples")

# Train models to predict distance, bearing, altitude from bbox
print("\n" + "="*70)
print("🏋️ Training bbox → distance/bearing/altitude models")
print("="*70)

# Distance model
print("\n1️⃣ Training Distance model...")
model_distance = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    device='cuda'
)
model_distance.fit(X_train, y_dist_train, verbose=False)
pred_dist = model_distance.predict(X_test)
r2_dist = r2_score(y_dist_test, pred_dist)
rmse_dist = np.sqrt(mean_squared_error(y_dist_test, pred_dist))
mae_dist = np.mean(np.abs(y_dist_test - pred_dist))

print(f"   ✅ R² = {r2_dist:.4f}, RMSE = {rmse_dist:.2f}m, MAE = {mae_dist:.2f}m")

# Bearing model (convert to sin/cos to handle circularity)
print("\n2️⃣ Training Bearing model...")
bearing_sin_train = np.sin(np.radians(y_bear_train))
bearing_cos_train = np.cos(np.radians(y_bear_train))
bearing_sin_test = np.sin(np.radians(y_bear_test))
bearing_cos_test = np.cos(np.radians(y_bear_test))

model_bearing_sin = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    device='cuda'
)
model_bearing_sin.fit(X_train, bearing_sin_train, verbose=False)

model_bearing_cos = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    device='cuda'
)
model_bearing_cos.fit(X_train, bearing_cos_train, verbose=False)

# Predict and convert back to degrees
pred_sin = model_bearing_sin.predict(X_test)
pred_cos = model_bearing_cos.predict(X_test)
pred_bearing = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360

# Calculate bearing error
bearing_errors = []
for i in range(len(y_bear_test)):
    error = angular_difference(y_bear_test.iloc[i], pred_bearing[i])
    bearing_errors.append(error)

mae_bearing = np.mean(bearing_errors)
print(f"   ✅ MAE = {mae_bearing:.2f}°")

# Altitude model
print("\n3️⃣ Training Altitude model...")
model_altitude = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    device='cuda'
)
model_altitude.fit(X_train, y_alt_train, verbose=False)
pred_alt = model_altitude.predict(X_test)
r2_alt = r2_score(y_alt_test, pred_alt)
rmse_alt = np.sqrt(mean_squared_error(y_alt_test, pred_alt))
mae_alt = np.mean(np.abs(y_alt_test - pred_alt))

print(f"   ✅ R² = {r2_alt:.4f}, RMSE = {rmse_alt:.2f}m, MAE = {mae_alt:.2f}m")

print("\n" + "="*70)
print("🔄 Now use distance/bearing/altitude → lat/lon")
print("="*70)

def bearing_distance_to_latlon(bearing_deg, distance_m, camera_lat, camera_lon):
    """Convert bearing and distance to lat/lon"""
    R = 6371000
    lat1 = radians(camera_lat)
    lon1 = radians(camera_lon)
    bearing_rad = radians(bearing_deg)
    
    lat2 = asin(sin(lat1) * cos(distance_m/R) + 
                cos(lat1) * sin(distance_m/R) * cos(bearing_rad))
    lon2 = lon1 + atan2(sin(bearing_rad) * sin(distance_m/R) * cos(lat1),
                        cos(distance_m/R) - sin(lat1) * sin(lat2))
    
    return degrees(lat2), degrees(lon2)

# Get test indices
test_indices = X_test.index
df_test = df_detected.loc[test_indices].copy()

# Predict lat/lon from predicted bearing/distance
predictions = []

for i, idx in enumerate(test_indices):
    # Predicted values
    pred_bearing_val = pred_bearing[i]
    pred_distance_val = pred_dist[i]
    pred_altitude_val = pred_alt[i]
    
    # Convert to lat/lon
    pred_lat, pred_lon = bearing_distance_to_latlon(
        pred_bearing_val, pred_distance_val, CAMERA_LAT, CAMERA_LON
    )
    
    # True values
    true_lat = df_detected.loc[idx, 'latitude_deg']
    true_lon = df_detected.loc[idx, 'longitude_deg']
    true_alt = df_detected.loc[idx, 'altitude_m']
    
    predictions.append({
        'true_lat': true_lat,
        'true_lon': true_lon,
        'true_alt': true_alt,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon,
        'pred_alt': pred_altitude_val
    })

df_predictions = pd.DataFrame(predictions)

# Calculate competition metrics
print("\n📊 Competition Score Calculation:")

errors_angle = []
errors_height = []
errors_range = []

for _, row in df_predictions.iterrows():
    # Angle error
    true_bearing = calculate_bearing(CAMERA_LAT, CAMERA_LON,
                                     row['true_lat'], row['true_lon'])
    pred_bearing = calculate_bearing(CAMERA_LAT, CAMERA_LON,
                                     row['pred_lat'], row['pred_lon'])
    angle_error = angular_difference(true_bearing, pred_bearing)
    errors_angle.append(angle_error)
    
    # Height error
    height_error = abs(row['true_alt'] - row['pred_alt'])
    errors_height.append(height_error)
    
    # Range error
    true_dist = haversine(CAMERA_LAT, CAMERA_LON,
                         row['true_lat'], row['true_lon'])
    pred_dist = haversine(CAMERA_LAT, CAMERA_LON,
                         row['pred_lat'], row['pred_lon'])
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

# Compare with other approaches
print("\n" + "="*70)
print("📊 Comparison with Other Approaches")
print("="*70)

results = [
    ("Baseline (v1)", 5.9369),
    ("Ensemble (v1+v21)", 5.2838),
    ("Fixed (Val stats)", 11.8196),
    ("Fixed (Test stats)", 13.9183),
    ("Bbox Approximation", total_error)
]

results_sorted = sorted(results, key=lambda x: x[1])

for i, (name, score) in enumerate(results_sorted, 1):
    if i == 1:
        print(f"  🏆 {i}. {name:25s}: {score:.4f} ← BEST")
    else:
        improvement = ((score - results_sorted[0][1]) / results_sorted[0][1]) * 100
        print(f"     {i}. {name:25s}: {score:.4f} (+{improvement:.1f}%)")

# Feature importance
print("\n" + "="*70)
print("🔍 Feature Importance Analysis")
print("="*70)

print("\n📐 Top features for Distance:")
dist_importance = list(zip(available_bbox_features, model_distance.feature_importances_))
dist_importance.sort(key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(dist_importance[:5], 1):
    print(f"  {i}. {feat:25s}: {imp:.4f}")

print("\n📊 Top features for Altitude:")
alt_importance = list(zip(available_bbox_features, model_altitude.feature_importances_))
alt_importance.sort(key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(alt_importance[:5], 1):
    print(f"  {i}. {feat:25s}: {imp:.4f}")

# Save models
print("\n" + "="*70)
print("💾 Saving Models")
print("="*70)

import os
os.makedirs('models_approximation', exist_ok=True)

joblib.dump(model_distance, 'models_approximation/bbox_to_distance.pkl')
joblib.dump(model_bearing_sin, 'models_approximation/bbox_to_bearing_sin.pkl')
joblib.dump(model_bearing_cos, 'models_approximation/bbox_to_bearing_cos.pkl')
joblib.dump(model_altitude, 'models_approximation/bbox_to_altitude.pkl')

with open('models_approximation/bbox_features.json', 'w') as f:
    json.dump(available_bbox_features, f, indent=2)

print(f"✅ Saved models to: models_approximation/")
print(f"✅ Features: {', '.join(available_bbox_features[:3])}...")

# Summary
print("\n" + "="*70)
print("📋 Summary & Recommendations")
print("="*70)

print(f"""
Approximation Approach Results:
  Score: {total_error:.4f}
  
Comparison:
  ✅ Better than Fixed approaches (11.82, 13.92)
  {'✅ Better than' if total_error < 5.2838 else '❌ Worse than'} Ensemble ({5.2838:.4f})
  {'✅ Better than' if total_error < 5.9369 else '❌ Worse than'} Baseline ({5.9369:.4f})

Advantages:
  ✅ No need to collect extra GCP
  ✅ Uses only YOLO bbox features
  ✅ Can predict for any position
  ✅ No data leakage from ground truth

Limitations:
  ⚠️  Extrapolation for unseen regions may be inaccurate
  ⚠️  Limited by bbox detection quality
  ⚠️  Training data not covering all directions

Recommendation:
  {'✅ Use this approach!' if total_error < 5.2838 else '❌ Ensemble (5.28) is still better'}
""")

print("\n✅ Complete!")
