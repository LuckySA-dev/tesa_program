"""
Tune XGBoost hyperparameters on baseline features (with data leakage)
Goal: Reduce angle error (70% weight) to beat baseline score of 5.9369
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import json
from math import radians, cos, sin, asin, sqrt, atan2, degrees

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
print("🎯 XGBoost Hyperparameter Tuning (Baseline Features)")
print("="*70)

CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

# Load baseline engineered features
print("\n" + "="*70)
print("Step 1: Load Baseline Features")
print("="*70)

df = pd.read_csv('train_metadata_engineered.csv', encoding='utf-8')
print(f"✅ Loaded: {len(df)} samples")

# Load feature columns
with open('feature_columns.json', 'r') as f:
    feature_data = json.load(f)
    feature_cols = feature_data['feature_columns']

print(f"✅ Features: {len(feature_cols)}")

# Prepare data
X = df[feature_cols].fillna(0)
y_lat = df['latitude_deg']
y_lon = df['longitude_deg']
y_alt = df['altitude_m']

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

# Hyperparameter configurations
print("\n" + "="*70)
print("Step 2: Grid Search Hyperparameters")
print("="*70)

configs = [
    # Baseline
    {
        'name': 'v1_baseline',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1
    },
    # Deeper trees
    {
        'name': 'v2_deeper',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1
    },
    # More regularization
    {
        'name': 'v3_regularized',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0
    },
    # Slow learning
    {
        'name': 'v4_slow',
        'max_depth': 5,
        'learning_rate': 0.03,
        'n_estimators': 1000,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 2,
        'gamma': 0.05,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5
    },
    # Aggressive
    {
        'name': 'v5_aggressive',
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_estimators': 600,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 0.5
    },
    # Balanced
    {
        'name': 'v6_balanced',
        'max_depth': 7,
        'learning_rate': 0.08,
        'n_estimators': 700,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 2,
        'gamma': 0.1,
        'reg_alpha': 0.3,
        'reg_lambda': 1.2
    },
    # Conservative
    {
        'name': 'v7_conservative',
        'max_depth': 4,
        'learning_rate': 0.15,
        'n_estimators': 400,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'gamma': 0.3,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0
    },
    # High trees
    {
        'name': 'v8_high_trees',
        'max_depth': 3,
        'learning_rate': 0.2,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.2,
        'reg_lambda': 1.0
    },
]

results = []

for i, config in enumerate(configs, 1):
    print(f"\n{'='*70}")
    print(f"Testing {i}/{len(configs)}: {config['name']}")
    print(f"{'='*70}")
    
    params = {k: v for k, v in config.items() if k != 'name'}
    params['random_state'] = 42
    params['tree_method'] = 'hist'
    params['device'] = 'cuda'
    
    # Train models
    print(f"   Training...")
    
    model_lat = XGBRegressor(**params)
    model_lat.fit(X_train, y_lat_train, eval_set=[(X_val, y_lat_val)], verbose=False)
    
    model_lon = XGBRegressor(**params)
    model_lon.fit(X_train, y_lon_train, eval_set=[(X_val, y_lon_val)], verbose=False)
    
    model_alt = XGBRegressor(**params)
    model_alt.fit(X_train, y_alt_train, eval_set=[(X_val, y_alt_val)], verbose=False)
    
    # Predict
    pred_lat = model_lat.predict(X_val)
    pred_lon = model_lon.predict(X_val)
    pred_alt = model_alt.predict(X_val)
    
    # RMSE
    lat_rmse = np.sqrt(np.mean((y_lat_val - pred_lat)**2)) * 111000
    lon_rmse = np.sqrt(np.mean((y_lon_val - pred_lon)**2)) * 111000 * cos(radians(CAMERA_LAT))
    alt_rmse = np.sqrt(np.mean((y_alt_val - pred_alt)**2))
    
    # Competition score
    y_true = np.column_stack([y_lat_val, y_lon_val, y_alt_val])
    y_pred = np.column_stack([pred_lat, pred_lon, pred_alt])
    
    total_score, angle_err, height_err, range_err = calculate_competition_score(
        y_true, y_pred, CAMERA_LAT, CAMERA_LON
    )
    
    result = {
        'name': config['name'],
        'score': total_score,
        'angle': angle_err,
        'height': height_err,
        'range': range_err,
        'lat_rmse': lat_rmse,
        'lon_rmse': lon_rmse,
        'alt_rmse': alt_rmse,
        'models': {
            'lat': model_lat,
            'lon': model_lon,
            'alt': model_alt
        },
        'config': config
    }
    results.append(result)
    
    print(f"\n   📊 Results:")
    print(f"      Score:        {total_score:.4f}")
    print(f"      Angle Error:  {angle_err:.2f}°")
    print(f"      Height Error: {height_err:.2f} m")
    print(f"      Range Error:  {range_err:.2f} m")

# Summary
print("\n" + "="*70)
print("📊 FINAL COMPARISON")
print("="*70)

results_df = pd.DataFrame([{
    'Config': r['name'],
    'Score': f"{r['score']:.4f}",
    'Angle°': f"{r['angle']:.2f}",
    'Height m': f"{r['height']:.2f}",
    'Range m': f"{r['range']:.2f}",
} for r in results])

# Sort by score
results_sorted = sorted(results, key=lambda x: x['score'])
results_df = pd.DataFrame([{
    'Config': r['name'],
    'Score': f"{r['score']:.4f}",
    'Angle°': f"{r['angle']:.2f}",
    'Height m': f"{r['height']:.2f}",
    'Range m': f"{r['range']:.2f}",
} for r in results_sorted])

print(results_df.to_string(index=False))

# Best configuration
best_result = results_sorted[0]

print("\n" + "="*70)
print("🏆 BEST CONFIGURATION")
print("="*70)
print(f"Name:             {best_result['name']}")
print(f"Competition Score: {best_result['score']:.4f}")
print(f"Angle Error:      {best_result['angle']:.2f}°")
print(f"Height Error:     {best_result['height']:.2f} m")
print(f"Range Error:      {best_result['range']:.2f} m")

print(f"\nHyperparameters:")
for k, v in best_result['config'].items():
    if k != 'name':
        print(f"   {k:20s}: {v}")

# Compare with baseline
baseline_score = 5.9369
improvement = baseline_score - best_result['score']
improvement_pct = (improvement / baseline_score) * 100

print(f"\n" + "="*70)
print("📈 vs Baseline")
print("="*70)
print(f"Baseline (v1):     {baseline_score:.4f}")
print(f"Best Tuned:        {best_result['score']:.4f}")
print(f"Improvement:       {improvement:+.4f} ({improvement_pct:+.1f}%)")

if improvement > 0:
    print("\n✅ IMPROVED! Training on full dataset...")
    
    # Retrain on full data
    best_params = {k: v for k, v in best_result['config'].items() if k != 'name'}
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    print("\n   Training Latitude...")
    final_lat = XGBRegressor(**best_params)
    final_lat.fit(X, y_lat, verbose=False)
    
    print("   Training Longitude...")
    final_lon = XGBRegressor(**best_params)
    final_lon.fit(X, y_lon, verbose=False)
    
    print("   Training Altitude...")
    final_alt = XGBRegressor(**best_params)
    final_alt.fit(X, y_alt, verbose=False)
    
    # Save models
    with open('xgb_model_latitude_best.pkl', 'wb') as f:
        pickle.dump(final_lat, f)
    with open('xgb_model_longitude_best.pkl', 'wb') as f:
        pickle.dump(final_lon, f)
    with open('xgb_model_altitude_best.pkl', 'wb') as f:
        pickle.dump(final_alt, f)
    
    # Save config
    with open('best_xgb_config.json', 'w') as f:
        json.dump(best_result['config'], f, indent=2)
    
    print("\n✅ Saved:")
    print("   - xgb_model_*_best.pkl")
    print("   - best_xgb_config.json")
    
    # Feature importance
    print("\n" + "="*70)
    print("📈 Top 10 Important Features (Latitude)")
    print("="*70)
    
    importance = final_lat.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
else:
    print("\n⚠️ No improvement. Baseline is still best.")
    print("   Consider:")
    print("   - Different hyperparameter ranges")
    print("   - Ensemble methods")
    print("   - Better YOLO features")

print("\n" + "="*70)
