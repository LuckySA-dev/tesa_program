"""
Ensemble v1 + v21: Average predictions from both models
Goal: Combine baseline v1 (5.94) and YOLO v21 (6.41) for better performance
"""

import pandas as pd
import numpy as np
import pickle
import json
from math import radians, cos, sin, asin, sqrt, atan2, degrees
from sklearn.model_selection import train_test_split

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
print("🎯 Ensemble: Baseline v1 + YOLO v21")
print("="*70)

# ============================================================================
# Step 1: Load Models v1 (Baseline)
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

print("✅ Loaded v1 models")

# ============================================================================
# Step 2: Load Models v21 (YOLO v21)
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

print("✅ Loaded v21 models")

# ============================================================================
# Step 3: Load Data and Features
# ============================================================================
print("\n" + "="*70)
print("Step 3: Load Features")
print("="*70)

# Load v1 features
df_v1 = pd.read_csv('train_metadata_engineered.csv', encoding='utf-8')
with open('feature_columns.json', 'r') as f:
    feature_data_v1 = json.load(f)
    feature_cols_v1 = feature_data_v1['feature_columns']

X_v1 = df_v1[feature_cols_v1].fillna(0)
y_lat = df_v1['latitude_deg']
y_lon = df_v1['longitude_deg']
y_alt = df_v1['altitude_m']

print(f"✅ v1 features: {len(feature_cols_v1)}")

# Load v21 features
df_v21 = pd.read_csv('train_metadata_engineered_v21.csv', encoding='utf-8')
with open('feature_columns_v21.json', 'r') as f:
    feature_data_v21 = json.load(f)
    feature_cols_v21 = feature_data_v21['feature_columns']

X_v21 = df_v21[feature_cols_v21].fillna(0)

print(f"✅ v21 features: {len(feature_cols_v21)}")

# ============================================================================
# Step 4: Create Same Train/Val Split
# ============================================================================
print("\n" + "="*70)
print("Step 4: Train/Val Split")
print("="*70)

# Use same random_state as baseline
X_v1_train, X_v1_val, y_lat_train, y_lat_val = train_test_split(
    X_v1, y_lat, test_size=0.2, random_state=42
)
_, _, y_lon_train, y_lon_val = train_test_split(
    X_v1, y_lon, test_size=0.2, random_state=42
)
_, _, y_alt_train, y_alt_val = train_test_split(
    X_v1, y_alt, test_size=0.2, random_state=42
)

# v21 split (same indices)
X_v21_train, X_v21_val = train_test_split(
    X_v21, test_size=0.2, random_state=42
)

print(f"   Training: {len(X_v1_train)} samples")
print(f"   Validation: {len(X_v1_val)} samples")

# ============================================================================
# Step 5: Test Different Ensemble Weights
# ============================================================================
print("\n" + "="*70)
print("Step 5: Test Ensemble Weights")
print("="*70)

# Predict with v1
print("\n   Predicting with v1...")
pred_lat_v1 = model_lat_v1.predict(X_v1_val)
pred_lon_v1 = model_lon_v1.predict(X_v1_val)
pred_alt_v1 = model_alt_v1.predict(X_v1_val)

# Predict with v21
print("   Predicting with v21...")
pred_lat_v21 = model_lat_v21.predict(X_v21_val)
pred_lon_v21 = model_lon_v21.predict(X_v21_val)
pred_alt_v21 = model_alt_v21.predict(X_v21_val)

# Test different weights
weight_configs = [
    {'name': 'v1_only', 'w1': 1.0, 'w2': 0.0},
    {'name': 'v21_only', 'w1': 0.0, 'w2': 1.0},
    {'name': '90-10_v1', 'w1': 0.9, 'w2': 0.1},
    {'name': '80-20_v1', 'w1': 0.8, 'w2': 0.2},
    {'name': '70-30_v1', 'w1': 0.7, 'w2': 0.3},
    {'name': '60-40_v1', 'w1': 0.6, 'w2': 0.4},
    {'name': '50-50', 'w1': 0.5, 'w2': 0.5},
    {'name': '40-60_v21', 'w1': 0.4, 'w2': 0.6},
    {'name': '30-70_v21', 'w1': 0.3, 'w2': 0.7},
    {'name': '20-80_v21', 'w1': 0.2, 'w2': 0.8},
    {'name': '10-90_v21', 'w1': 0.1, 'w2': 0.9},
]

results = []

for config in weight_configs:
    w1 = config['w1']
    w2 = config['w2']
    
    # Ensemble predictions
    pred_lat = w1 * pred_lat_v1 + w2 * pred_lat_v21
    pred_lon = w1 * pred_lon_v1 + w2 * pred_lon_v21
    pred_alt = w1 * pred_alt_v1 + w2 * pred_alt_v21
    
    # Calculate score
    y_true = np.column_stack([y_lat_val, y_lon_val, y_alt_val])
    y_pred = np.column_stack([pred_lat, pred_lon, pred_alt])
    
    total_score, angle_err, height_err, range_err = calculate_competition_score(
        y_true, y_pred, CAMERA_LAT, CAMERA_LON
    )
    
    results.append({
        'name': config['name'],
        'w1': w1,
        'w2': w2,
        'score': total_score,
        'angle': angle_err,
        'height': height_err,
        'range': range_err
    })

# Display results
print("\n" + "="*70)
print("📊 ENSEMBLE RESULTS")
print("="*70)

results_df = pd.DataFrame([{
    'Config': r['name'],
    'v1_weight': f"{r['w1']:.1f}",
    'v21_weight': f"{r['w2']:.1f}",
    'Score': f"{r['score']:.4f}",
    'Angle°': f"{r['angle']:.2f}",
    'Height m': f"{r['height']:.2f}",
    'Range m': f"{r['range']:.2f}",
} for r in results])

# Sort by score
results_sorted = sorted(results, key=lambda x: x['score'])
results_df = pd.DataFrame([{
    'Config': r['name'],
    'v1_weight': f"{r['w1']:.1f}",
    'v21_weight': f"{r['w2']:.1f}",
    'Score': f"{r['score']:.4f}",
    'Angle°': f"{r['angle']:.2f}",
    'Height m': f"{r['height']:.2f}",
    'Range m': f"{r['range']:.2f}",
} for r in results_sorted])

print(results_df.to_string(index=False))

# Best ensemble
best = results_sorted[0]

print("\n" + "="*70)
print("🏆 BEST ENSEMBLE")
print("="*70)
print(f"Configuration:    {best['name']}")
print(f"v1 weight:        {best['w1']:.1f}")
print(f"v21 weight:       {best['w2']:.1f}")
print(f"Competition Score: {best['score']:.4f}")
print(f"Angle Error:      {best['angle']:.2f}°")
print(f"Height Error:     {best['height']:.2f} m")
print(f"Range Error:      {best['range']:.2f} m")

# Compare
baseline_score = 5.9369
improvement = baseline_score - best['score']
improvement_pct = (improvement / baseline_score) * 100

print(f"\n" + "="*70)
print("📈 vs Baseline v1")
print("="*70)
print(f"Baseline v1:      {baseline_score:.4f}")
print(f"Best Ensemble:    {best['score']:.4f}")
print(f"Improvement:      {improvement:+.4f} ({improvement_pct:+.1f}%)")

if improvement > 0:
    print("\n✅ IMPROVED! Ensemble is better than v1 alone!")
    
    # Save ensemble weights
    ensemble_config = {
        'v1_weight': best['w1'],
        'v21_weight': best['w2'],
        'score': best['score'],
        'angle_error': best['angle'],
        'height_error': best['height'],
        'range_error': best['range']
    }
    
    with open('ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print("\n✅ Saved: ensemble_config.json")
    print(f"   Use weights: v1={best['w1']:.1f}, v21={best['w2']:.1f}")
    
else:
    print("\n⚠️ No improvement. Baseline v1 is still best.")
    print("   Recommendation: Use baseline v1 for final predictions")

print("\n" + "="*70)
