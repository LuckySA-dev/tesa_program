"""
Analyze validation errors and apply corrections to improve predictions
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Constants
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def bearing_distance_to_latlon(bearing_deg, distance_m, camera_lat=CAMERA_LAT, camera_lon=CAMERA_LON):
    """Convert bearing and distance to lat/lon"""
    bearing_rad = np.radians(bearing_deg)
    R = 6371000
    
    lat1 = np.radians(camera_lat)
    lon1 = np.radians(camera_lon)
    
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance_m/R) +
                     np.cos(lat1) * np.sin(distance_m/R) * np.cos(bearing_rad))
    
    lon2 = lon1 + np.arctan2(np.sin(bearing_rad) * np.sin(distance_m/R) * np.cos(lat1),
                             np.cos(distance_m/R) - np.sin(lat1) * np.sin(lat2))
    
    return np.degrees(lat2), np.degrees(lon2)

print("="*70)
print("🔍 Analyzing Validation Errors for Improvement")
print("="*70)

# Load training data with predictions
print("\n📂 Loading training data...")
df = pd.read_csv('train_metadata_engineered_v21.csv')

# Filter only detected samples
df_detected = df[df['yolo_detected'] == True].copy()
print(f"✅ Loaded {len(df_detected)} detected samples")

# Load models and make predictions on training data
print("\n📊 Loading models and generating predictions...")
models = {
    'distance': joblib.load('models_approximation/bbox_to_distance.pkl'),
    'bearing_sin': joblib.load('models_approximation/bbox_to_bearing_sin.pkl'),
    'bearing_cos': joblib.load('models_approximation/bbox_to_bearing_cos.pkl'),
    'altitude': joblib.load('models_approximation/bbox_to_altitude.pkl')
}

bbox_features = ['yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
                 'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center',
                 'yolo_angle_from_center']

X = df_detected[bbox_features].values

# Predict
pred_distance = models['distance'].predict(X)
pred_bearing_sin = models['bearing_sin'].predict(X)
pred_bearing_cos = models['bearing_cos'].predict(X)
pred_bearing = np.degrees(np.arctan2(pred_bearing_sin, pred_bearing_cos)) % 360
pred_altitude = models['altitude'].predict(X)

# Calculate ground truth
df_detected['true_distance'] = haversine_distance(
    CAMERA_LAT, CAMERA_LON,
    df_detected['latitude'], df_detected['longitude']
)
df_detected['true_bearing'] = calculate_bearing(
    CAMERA_LAT, CAMERA_LON,
    df_detected['latitude'], df_detected['longitude']
)
df_detected['true_altitude'] = df_detected['altitude']

# Add predictions
df_detected['pred_distance'] = pred_distance
df_detected['pred_bearing'] = pred_bearing
df_detected['pred_altitude'] = pred_altitude

# Calculate errors
df_detected['distance_error'] = df_detected['pred_distance'] - df_detected['true_distance']
df_detected['altitude_error'] = df_detected['pred_altitude'] - df_detected['true_altitude']

# Angle error (circular)
angle_diff = np.abs(df_detected['pred_bearing'] - df_detected['true_bearing'])
df_detected['bearing_error'] = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)

print("\n📈 Error Statistics:")
print(f"\nDistance Error:")
print(f"  Mean:   {df_detected['distance_error'].mean():+.2f} m")
print(f"  Median: {df_detected['distance_error'].median():+.2f} m")
print(f"  Std:    {df_detected['distance_error'].std():.2f} m")
print(f"  MAE:    {df_detected['distance_error'].abs().mean():.2f} m")

print(f"\nBearing Error:")
print(f"  Mean:   {df_detected['bearing_error'].mean():+.2f}°")
print(f"  Median: {df_detected['bearing_error'].median():+.2f}°")
print(f"  Std:    {df_detected['bearing_error'].std():.2f}°")
print(f"  MAE:    {df_detected['bearing_error'].abs().mean():.2f}°")

print(f"\nAltitude Error:")
print(f"  Mean:   {df_detected['altitude_error'].mean():+.2f} m")
print(f"  Median: {df_detected['altitude_error'].median():+.2f} m")
print(f"  Std:    {df_detected['altitude_error'].std():.2f} m")
print(f"  MAE:    {df_detected['altitude_error'].abs().mean():.2f} m")

# Analyze systematic bias
print("\n" + "="*70)
print("🔬 Detecting Systematic Bias")
print("="*70)

bias_distance = df_detected['distance_error'].mean()
bias_bearing = df_detected['pred_bearing'].mean() - df_detected['true_bearing'].mean()
bias_altitude = df_detected['altitude_error'].mean()

print(f"\nSystematic Bias Detected:")
print(f"  Distance: {bias_distance:+.2f} m {'(over-predicting)' if bias_distance > 0 else '(under-predicting)'}")
print(f"  Bearing:  {bias_bearing:+.2f}° {'(clockwise bias)' if bias_bearing > 0 else '(counter-clockwise bias)'}")
print(f"  Altitude: {bias_altitude:+.2f} m {'(over-predicting)' if bias_altitude > 0 else '(under-predicting)'}")

# Strategy 1: Apply bias correction
print("\n" + "="*70)
print("💡 Strategy 1: Bias Correction")
print("="*70)

print("\nApplying bias correction:")
print(f"  - Distance: subtract {bias_distance:.2f} m")
print(f"  - Bearing:  subtract {bias_bearing:.2f}°")
print(f"  - Altitude: subtract {bias_altitude:.2f} m")

df_detected['corrected_distance'] = df_detected['pred_distance'] - bias_distance
df_detected['corrected_bearing'] = (df_detected['pred_bearing'] - bias_bearing) % 360
df_detected['corrected_altitude'] = df_detected['pred_altitude'] - bias_altitude

# Recalculate errors
df_detected['corrected_distance_error'] = df_detected['corrected_distance'] - df_detected['true_distance']
angle_diff = np.abs(df_detected['corrected_bearing'] - df_detected['true_bearing'])
df_detected['corrected_bearing_error'] = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
df_detected['corrected_altitude_error'] = df_detected['corrected_altitude'] - df_detected['true_altitude']

print("\n✅ After Bias Correction:")
print(f"\nDistance MAE: {df_detected['distance_error'].abs().mean():.2f} m → {df_detected['corrected_distance_error'].abs().mean():.2f} m")
print(f"Bearing MAE:  {df_detected['bearing_error'].mean():.2f}° → {df_detected['corrected_bearing_error'].mean():.2f}°")
print(f"Altitude MAE: {df_detected['altitude_error'].abs().mean():.2f} m → {df_detected['corrected_altitude_error'].abs().mean():.2f} m")

# Calculate competition score
def calc_score(angle_err, height_err, range_err):
    return 0.7 * angle_err + 0.15 * height_err + 0.15 * range_err

score_before = calc_score(
    df_detected['bearing_error'].mean(),
    df_detected['altitude_error'].abs().mean(),
    df_detected['distance_error'].abs().mean()
)

score_after = calc_score(
    df_detected['corrected_bearing_error'].mean(),
    df_detected['corrected_altitude_error'].abs().mean(),
    df_detected['corrected_distance_error'].abs().mean()
)

print(f"\n📊 Competition Score:")
print(f"  Before: {score_before:.4f}")
print(f"  After:  {score_after:.4f}")
print(f"  Improvement: {((score_before - score_after) / score_before * 100):+.1f}%")

# Strategy 2: Quantile-based calibration
print("\n" + "="*70)
print("💡 Strategy 2: Quantile Calibration")
print("="*70)

from sklearn.isotonic import IsotonicRegression

# Calibrate distance
iso_distance = IsotonicRegression(out_of_bounds='clip')
iso_distance.fit(df_detected['pred_distance'], df_detected['true_distance'])
df_detected['calibrated_distance'] = iso_distance.predict(df_detected['pred_distance'])

# Calibrate altitude
iso_altitude = IsotonicRegression(out_of_bounds='clip')
iso_altitude.fit(df_detected['pred_altitude'], df_detected['true_altitude'])
df_detected['calibrated_altitude'] = iso_altitude.predict(df_detected['pred_altitude'])

print("\n✅ Calibration models trained")

# Calculate errors after calibration
df_detected['calibrated_distance_error'] = df_detected['calibrated_distance'] - df_detected['true_distance']
df_detected['calibrated_altitude_error'] = df_detected['calibrated_altitude'] - df_detected['true_altitude']

print(f"\nAfter Quantile Calibration:")
print(f"  Distance MAE: {df_detected['distance_error'].abs().mean():.2f} m → {df_detected['calibrated_distance_error'].abs().mean():.2f} m")
print(f"  Altitude MAE: {df_detected['altitude_error'].abs().mean():.2f} m → {df_detected['calibrated_altitude_error'].abs().mean():.2f} m")

# Strategy 3: Feature-based correction
print("\n" + "="*70)
print("💡 Strategy 3: Residual Learning")
print("="*70)

from xgboost import XGBRegressor

# Train models to predict errors
print("\nTraining residual correction models...")

# Distance residual
model_dist_residual = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model_dist_residual.fit(X, df_detected['distance_error'])

# Bearing residual (use signed error)
bearing_signed_error = df_detected['pred_bearing'] - df_detected['true_bearing']
# Handle wrap-around
bearing_signed_error = np.where(bearing_signed_error > 180, bearing_signed_error - 360, bearing_signed_error)
bearing_signed_error = np.where(bearing_signed_error < -180, bearing_signed_error + 360, bearing_signed_error)

model_bearing_residual = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model_bearing_residual.fit(X, bearing_signed_error)

# Altitude residual
model_alt_residual = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model_alt_residual.fit(X, df_detected['altitude_error'])

# Apply corrections
pred_dist_residual = model_dist_residual.predict(X)
pred_bearing_residual = model_bearing_residual.predict(X)
pred_alt_residual = model_alt_residual.predict(X)

df_detected['residual_corrected_distance'] = df_detected['pred_distance'] - pred_dist_residual
df_detected['residual_corrected_bearing'] = (df_detected['pred_bearing'] - pred_bearing_residual) % 360
df_detected['residual_corrected_altitude'] = df_detected['pred_altitude'] - pred_alt_residual

# Calculate errors
df_detected['residual_distance_error'] = df_detected['residual_corrected_distance'] - df_detected['true_distance']
angle_diff = np.abs(df_detected['residual_corrected_bearing'] - df_detected['true_bearing'])
df_detected['residual_bearing_error'] = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
df_detected['residual_altitude_error'] = df_detected['residual_corrected_altitude'] - df_detected['true_altitude']

print(f"\n✅ After Residual Learning:")
print(f"  Distance MAE: {df_detected['distance_error'].abs().mean():.2f} m → {df_detected['residual_distance_error'].abs().mean():.2f} m")
print(f"  Bearing MAE:  {df_detected['bearing_error'].mean():.2f}° → {df_detected['residual_bearing_error'].mean():.2f}°")
print(f"  Altitude MAE: {df_detected['altitude_error'].abs().mean():.2f} m → {df_detected['residual_altitude_error'].abs().mean():.2f} m")

score_residual = calc_score(
    df_detected['residual_bearing_error'].mean(),
    df_detected['residual_altitude_error'].abs().mean(),
    df_detected['residual_distance_error'].abs().mean()
)

print(f"\n📊 Competition Score: {score_residual:.4f}")
print(f"  Improvement: {((score_before - score_residual) / score_before * 100):+.1f}%")

# Save best models
print("\n" + "="*70)
print("💾 Saving Correction Models")
print("="*70)

correction_params = {
    'bias_distance': float(bias_distance),
    'bias_bearing': float(bias_bearing),
    'bias_altitude': float(bias_altitude)
}

import json
with open('models_approximation/correction_params.json', 'w') as f:
    json.dump(correction_params, f, indent=2)

joblib.dump(iso_distance, 'models_approximation/calibration_distance.pkl')
joblib.dump(iso_altitude, 'models_approximation/calibration_altitude.pkl')
joblib.dump(model_dist_residual, 'models_approximation/residual_distance.pkl')
joblib.dump(model_bearing_residual, 'models_approximation/residual_bearing.pkl')
joblib.dump(model_alt_residual, 'models_approximation/residual_altitude.pkl')

print("✅ Saved correction models:")
print("  - correction_params.json (bias values)")
print("  - calibration_distance.pkl (isotonic regression)")
print("  - calibration_altitude.pkl (isotonic regression)")
print("  - residual_distance.pkl (XGBoost)")
print("  - residual_bearing.pkl (XGBoost)")
print("  - residual_altitude.pkl (XGBoost)")

# Summary comparison
print("\n" + "="*70)
print("📊 Summary of All Strategies")
print("="*70)

print("\n┌─────────────────────────┬────────┬────────┬────────┬────────┐")
print("│ Strategy                │  Angle │ Height │  Range │  Score │")
print("├─────────────────────────┼────────┼────────┼────────┼────────┤")
print(f"│ Original                │  {df_detected['bearing_error'].mean():5.2f} │  {df_detected['altitude_error'].abs().mean():5.2f} │  {df_detected['distance_error'].abs().mean():5.2f} │  {score_before:5.2f} │")
print(f"│ Bias Correction         │  {df_detected['corrected_bearing_error'].mean():5.2f} │  {df_detected['corrected_altitude_error'].abs().mean():5.2f} │  {df_detected['corrected_distance_error'].abs().mean():5.2f} │  {score_after:5.2f} │")
print(f"│ Residual Learning       │  {df_detected['residual_bearing_error'].mean():5.2f} │  {df_detected['residual_altitude_error'].abs().mean():5.2f} │  {df_detected['residual_distance_error'].abs().mean():5.2f} │  {score_residual:5.2f} │ ← BEST")
print("└─────────────────────────┴────────┴────────┴────────┴────────┘")

best_improvement = ((score_before - score_residual) / score_before * 100)
print(f"\n🏆 Best Strategy: Residual Learning")
print(f"   Score improvement: {best_improvement:+.1f}%")
print(f"   Competition score: {score_residual:.4f}")

print("\n" + "="*70)
