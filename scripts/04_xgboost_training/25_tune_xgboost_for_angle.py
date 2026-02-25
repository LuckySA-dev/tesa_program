"""
Tune XGBoost hyperparameters to improve angle and range errors
Focus on reducing angle error (70% weight) and range error (15% weight)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import json
from math import radians, cos, sin, asin, sqrt, atan2, degrees

# Competition scoring function
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees between two lat/lon points"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    bearing_deg = (degrees(bearing) + 360) % 360
    return bearing_deg

def calculate_competition_score(y_true, y_pred):
    """Calculate competition score"""
    # Camera position
    camera_lat, camera_lon = 14.305029, 101.173010
    
    errors = []
    for i in range(len(y_true)):
        true_lat, true_lon, true_alt = y_true[i]
        pred_lat, pred_lon, pred_alt = y_pred[i]
        
        # Calculate ground truth metrics
        true_range = haversine(camera_lat, camera_lon, true_lat, true_lon)
        true_bearing = calculate_bearing(camera_lat, camera_lon, true_lat, true_lon)
        
        # Calculate predicted metrics
        pred_range = haversine(camera_lat, camera_lon, pred_lat, pred_lon)
        pred_bearing = calculate_bearing(camera_lat, camera_lon, pred_lat, pred_lon)
        
        # Calculate errors
        angle_error = abs(true_bearing - pred_bearing)
        if angle_error > 180:
            angle_error = 360 - angle_error
        
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
print("🎯 XGBoost Hyperparameter Tuning for Angle & Range")
print("="*70)

# Load engineered features from baseline (v1) - proven best
print("\n" + "="*70)
print("Step 1: Load Baseline Features (v1)")
print("="*70)

df = pd.read_csv('train_metadata_engineered.csv', encoding='utf-8')
print(f"✅ Loaded: {len(df)} samples")

# Load feature columns
with open('feature_columns.json', 'r') as f:
    feature_data = json.load(f)
    feature_cols = feature_data['feature_columns']

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

# Hyperparameter configurations to try
print("\n" + "="*70)
print("Step 2: Hyperparameter Grid Search")
print("="*70)

configs = [
    # Original baseline
    {
        'name': 'v1_baseline',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1
    },
    # Deeper trees for complex patterns
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
    # Regularized to prevent overfitting
    {
        'name': 'v3_regularized',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5
    },
    # More estimators, slower learning
    {
        'name': 'v4_slow_learning',
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 2,
        'gamma': 0.05,
        'reg_alpha': 0.05,
        'reg_lambda': 1.2
    },
    # Conservative: shallow + regularized
    {
        'name': 'v5_conservative',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 600,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.2,
        'reg_lambda': 2.0
    },
    # Aggressive: deep + many estimators
    {
        'name': 'v6_aggressive',
        'max_depth': 10,
        'learning_rate': 0.03,
        'n_estimators': 1200,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1
    },
]

results = []

for i, config in enumerate(configs, 1):
    print(f"\n{'='*70}")
    print(f"Testing {i}/{len(configs)}: {config['name']}")
    print(f"{'='*70}")
    
    params = {k: v for k, v in config.items() if k != 'name'}
    
    # Train models
    print(f"   Training Latitude...")
    model_lat = XGBRegressor(
        **params,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    model_lat.fit(
        X_train, y_lat_train,
        eval_set=[(X_val, y_lat_val)],
        verbose=False
    )
    
    print(f"   Training Longitude...")
    model_lon = XGBRegressor(
        **params,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    model_lon.fit(
        X_train, y_lon_train,
        eval_set=[(X_val, y_lon_val)],
        verbose=False
    )
    
    print(f"   Training Altitude...")
    model_alt = XGBRegressor(
        **params,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    model_alt.fit(
        X_train, y_alt_train,
        eval_set=[(X_val, y_alt_val)],
        verbose=False
    )
    
    # Predict
    pred_lat = model_lat.predict(X_val)
    pred_lon = model_lon.predict(X_val)
    pred_alt = model_alt.predict(X_val)
    
    # Calculate RMSE
    lat_rmse = np.sqrt(np.mean((y_lat_val - pred_lat)**2))
    lon_rmse = np.sqrt(np.mean((y_lon_val - pred_lon)**2))
    alt_rmse = np.sqrt(np.mean((y_alt_val - pred_alt)**2))
    
    # Calculate competition score
    y_true = np.column_stack([y_lat_val, y_lon_val, y_alt_val])
    y_pred = np.column_stack([pred_lat, pred_lon, pred_alt])
    
    total_score, angle_err, height_err, range_err = calculate_competition_score(y_true, y_pred)
    
    result = {
        'name': config['name'],
        'score': total_score,
        'angle': angle_err,
        'height': height_err,
        'range': range_err,
        'lat_rmse': lat_rmse,
        'lon_rmse': lon_rmse,
        'alt_rmse': alt_rmse,
        'config': config
    }
    results.append(result)
    
    print(f"\n   📊 Results:")
    print(f"      Competition Score: {total_score:.4f}")
    print(f"      Angle Error:  {angle_err:.2f}°")
    print(f"      Height Error: {height_err:.2f} m")
    print(f"      Range Error:  {range_err:.2f} m")
    print(f"      Lat RMSE:     {lat_rmse:.2f} m")
    print(f"      Lon RMSE:     {lon_rmse:.2f} m")
    print(f"      Alt RMSE:     {alt_rmse:.2f} m")

# Summary
print("\n" + "="*70)
print("📊 FINAL COMPARISON")
print("="*70)

results_df = pd.DataFrame([{
    'Config': r['name'],
    'Score': r['score'],
    'Angle°': r['angle'],
    'Height m': r['height'],
    'Range m': r['range'],
    'Lat RMSE': r['lat_rmse'],
    'Lon RMSE': r['lon_rmse'],
    'Alt RMSE': r['alt_rmse']
} for r in results])

results_df = results_df.sort_values('Score')
print(results_df.to_string(index=False))

# Best configuration
best_result = results[0]
for r in results:
    if r['score'] < best_result['score']:
        best_result = r

print("\n" + "="*70)
print("🏆 BEST CONFIGURATION")
print("="*70)
print(f"Name: {best_result['name']}")
print(f"Competition Score: {best_result['score']:.4f}")
print(f"Angle Error:  {best_result['angle']:.2f}°")
print(f"Height Error: {best_result['height']:.2f} m")
print(f"Range Error:  {best_result['range']:.2f} m")

print(f"\nHyperparameters:")
for k, v in best_result['config'].items():
    if k != 'name':
        print(f"   {k}: {v}")

# Compare with original baseline
baseline_score = 5.9369
improvement = baseline_score - best_result['score']
improvement_pct = (improvement / baseline_score) * 100

print(f"\n" + "="*70)
print("📈 Improvement vs Original Baseline")
print("="*70)
print(f"Original (v1):  {baseline_score:.4f}")
print(f"Best Tuned:     {best_result['score']:.4f}")
print(f"Improvement:    {improvement:.4f} ({improvement_pct:+.1f}%)")

if improvement > 0:
    print("\n✅ IMPROVED! Training best model on full data...")
    
    # Retrain best model on full data
    best_params = {k: v for k, v in best_result['config'].items() if k != 'name'}
    
    print("\n   Training Latitude...")
    final_lat = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    final_lat.fit(X, y_lat, verbose=False)
    
    print("   Training Longitude...")
    final_lon = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    final_lon.fit(X, y_lon, verbose=False)
    
    print("   Training Altitude...")
    final_alt = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    final_alt.fit(X, y_alt, verbose=False)
    
    # Save models
    with open('xgb_model_latitude_tuned.pkl', 'wb') as f:
        pickle.dump(final_lat, f)
    with open('xgb_model_longitude_tuned.pkl', 'wb') as f:
        pickle.dump(final_lon, f)
    with open('xgb_model_altitude_tuned.pkl', 'wb') as f:
        pickle.dump(final_alt, f)
    
    # Save config
    with open('best_xgb_config.json', 'w') as f:
        json.dump(best_result['config'], f, indent=2)
    
    print("\n✅ Saved:")
    print("   - xgb_model_*_tuned.pkl")
    print("   - best_xgb_config.json")
    
else:
    print("\n⚠️ No improvement found. Baseline v1 is still best.")

print("\n" + "="*70)
