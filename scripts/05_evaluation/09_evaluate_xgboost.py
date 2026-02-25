"""
Phase 5: Evaluate XGBoost Models on Validation Set
Calculate competition metrics: angle error and altitude error
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points"""
    R = 6371000  # Earth radius in meters
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def calculate_angle_error(lat_true, lon_true, lat_pred, lon_pred):
    """Calculate angle error in degrees"""
    # Calculate bearing from camera to true position
    lat1, lon1 = 14.305029, 101.173010  # Camera position
    
    # True bearing
    dy_true = lat_true - lat1
    dx_true = (lon_true - lon1) * np.cos(np.radians(lat1))
    bearing_true = np.degrees(np.arctan2(dx_true, dy_true))
    
    # Predicted bearing
    dy_pred = lat_pred - lat1
    dx_pred = (lon_pred - lon1) * np.cos(np.radians(lat1))
    bearing_pred = np.degrees(np.arctan2(dx_pred, dy_pred))
    
    # Angular difference
    angle_diff = np.abs(bearing_true - bearing_pred)
    
    # Normalize to [-180, 180]
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    
    return angle_diff

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 5: Evaluate XGBoost Models")
    print("=" * 70)

    # Load data
    df = pd.read_csv('train_metadata_engineered.csv')
    print(f"\n✅ Loaded data: {df.shape}")

    # Load feature columns
    with open('feature_columns.json', 'r') as f:
        feature_info = json.load(f)
    
    feature_columns = feature_info['feature_columns']
    
    # Load models
    print("\n📦 Loading models...")
    with open('xgb_model_latitude.pkl', 'rb') as f:
        model_lat = pickle.load(f)
    print("✅ Loaded: xgb_model_latitude.pkl")
    
    with open('xgb_model_longitude.pkl', 'rb') as f:
        model_lon = pickle.load(f)
    print("✅ Loaded: xgb_model_longitude.pkl")
    
    with open('xgb_model_altitude.pkl', 'rb') as f:
        model_alt = pickle.load(f)
    print("✅ Loaded: xgb_model_altitude.pkl")

    # Prepare data
    X = df[feature_columns].values
    y_lat = df['latitude_deg'].values
    y_lon = df['longitude_deg'].values
    y_alt = df['altitude_m'].values

    # Split data (same split as training)
    X_train, X_val, y_lat_train, y_lat_val = train_test_split(
        X, y_lat, test_size=0.2, random_state=42
    )
    _, _, y_lon_train, y_lon_val = train_test_split(
        X, y_lon, test_size=0.2, random_state=42
    )
    _, _, y_alt_train, y_alt_val = train_test_split(
        X, y_alt, test_size=0.2, random_state=42
    )

    print(f"\n✅ Validation set: {X_val.shape[0]} samples")

    # Predict on validation set
    print("\n" + "=" * 70)
    print("Making Predictions on Validation Set")
    print("=" * 70)
    
    y_lat_pred = model_lat.predict(X_val)
    y_lon_pred = model_lon.predict(X_val)
    y_alt_pred = model_alt.predict(X_val)
    
    print(f"✅ Predicted {len(y_lat_pred)} samples")

    # Calculate competition metrics
    print("\n" + "=" * 70)
    print("📊 Competition Metrics")
    print("=" * 70)

    # 1. Calculate horizontal distance error
    horizontal_errors = haversine_distance(y_lat_val, y_lon_val, y_lat_pred, y_lon_pred)
    
    # 2. Calculate angle error
    angle_errors = calculate_angle_error(y_lat_val, y_lon_val, y_lat_pred, y_lon_pred)
    
    # 3. Calculate altitude error
    altitude_errors = np.abs(y_alt_val - y_alt_pred)

    # NEW Competition score: 0.7 * mean_angle_error + 0.15 * mean_altitude_error + 0.15 * mean_range_error
    mean_angle_error = np.mean(angle_errors)
    mean_altitude_error = np.mean(altitude_errors)
    mean_range_error = np.mean(horizontal_errors)
    
    # Calculate new total error
    total_error = 0.7 * mean_angle_error + 0.15 * mean_altitude_error + 0.15 * mean_range_error

    print(f"\n🎯 Validation Set Performance:")
    print(f"   Mean Horizontal Distance Error (Range): {mean_range_error:.2f} m")
    print(f"   Median Horizontal Distance Error: {np.median(horizontal_errors):.2f} m")
    print(f"   Max Horizontal Distance Error: {np.max(horizontal_errors):.2f} m")
    print(f"\n   Mean Angle Error: {mean_angle_error:.4f} degrees")
    print(f"   Median Angle Error: {np.median(angle_errors):.4f} degrees")
    print(f"   Max Angle Error: {np.max(angle_errors):.4f} degrees")
    print(f"\n   Mean Altitude Error (Height): {mean_altitude_error:.2f} m")
    print(f"   Median Altitude Error: {np.median(altitude_errors):.2f} m")
    print(f"   Max Altitude Error: {np.max(altitude_errors):.2f} m")
    print(f"\n" + "=" * 70)
    print(f"🏆 NEW COMPETITION SCORE (Validation)")
    print(f"   Formula: 0.7 × mean_angle_error + 0.15 × mean_height_error + 0.15 × mean_range_error")
    print(f"   Angle Error:  {mean_angle_error:.4f}° × 0.7  = {0.7 * mean_angle_error:.4f}")
    print(f"   Height Error: {mean_altitude_error:.2f} m × 0.15 = {0.15 * mean_altitude_error:.4f}")
    print(f"   Range Error:  {mean_range_error:.2f} m × 0.15 = {0.15 * mean_range_error:.4f}")
    print(f"   Total Score: {total_error:.4f}")
    print(f"=" * 70)

    # Detailed statistics
    print(f"\n📈 Error Distribution:")
    print(f"   Horizontal Distance:")
    print(f"      25th percentile: {np.percentile(horizontal_errors, 25):.2f} m")
    print(f"      50th percentile: {np.percentile(horizontal_errors, 50):.2f} m")
    print(f"      75th percentile: {np.percentile(horizontal_errors, 75):.2f} m")
    print(f"      95th percentile: {np.percentile(horizontal_errors, 95):.2f} m")
    
    print(f"\n   Angle Error:")
    print(f"      25th percentile: {np.percentile(angle_errors, 25):.4f} deg")
    print(f"      50th percentile: {np.percentile(angle_errors, 50):.4f} deg")
    print(f"      75th percentile: {np.percentile(angle_errors, 75):.4f} deg")
    print(f"      95th percentile: {np.percentile(angle_errors, 95):.4f} deg")
    
    print(f"\n   Altitude Error:")
    print(f"      25th percentile: {np.percentile(altitude_errors, 25):.2f} m")
    print(f"      50th percentile: {np.percentile(altitude_errors, 50):.2f} m")
    print(f"      75th percentile: {np.percentile(altitude_errors, 75):.2f} m")
    print(f"      95th percentile: {np.percentile(altitude_errors, 95):.2f} m")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Error distributions
    axes[0, 0].hist(horizontal_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(horizontal_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(horizontal_errors):.2f} m')
    axes[0, 0].axvline(np.median(horizontal_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(horizontal_errors):.2f} m')
    axes[0, 0].set_xlabel('Horizontal Distance Error (m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Horizontal Distance Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(angle_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(mean_angle_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle_error:.4f}°')
    axes[0, 1].axvline(np.median(angle_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(angle_errors):.4f}°')
    axes[0, 1].set_xlabel('Angle Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Angle Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(altitude_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(mean_altitude_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_altitude_error:.2f} m')
    axes[0, 2].axvline(np.median(altitude_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(altitude_errors):.2f} m')
    axes[0, 2].set_xlabel('Altitude Error (m)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Altitude Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Scatter plots (True vs Predicted)
    axes[1, 0].scatter(y_lat_val, y_lat_pred, alpha=0.5, s=50)
    axes[1, 0].plot([y_lat_val.min(), y_lat_val.max()], [y_lat_val.min(), y_lat_val.max()], 'r--', linewidth=2, label='Perfect')
    axes[1, 0].set_xlabel('True Latitude (degrees)')
    axes[1, 0].set_ylabel('Predicted Latitude (degrees)')
    axes[1, 0].set_title('Latitude: True vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(y_lon_val, y_lon_pred, alpha=0.5, s=50)
    axes[1, 1].plot([y_lon_val.min(), y_lon_val.max()], [y_lon_val.min(), y_lon_val.max()], 'r--', linewidth=2, label='Perfect')
    axes[1, 1].set_xlabel('True Longitude (degrees)')
    axes[1, 1].set_ylabel('Predicted Longitude (degrees)')
    axes[1, 1].set_title('Longitude: True vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(y_alt_val, y_alt_pred, alpha=0.5, s=50)
    axes[1, 2].plot([y_alt_val.min(), y_alt_val.max()], [y_alt_val.min(), y_alt_val.max()], 'r--', linewidth=2, label='Perfect')
    axes[1, 2].set_xlabel('True Altitude (m)')
    axes[1, 2].set_ylabel('Predicted Altitude (m)')
    axes[1, 2].set_title('Altitude: True vs Predicted')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgb_validation_results.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: xgb_validation_results.png")

    # Save validation results
    val_results = pd.DataFrame({
        'true_lat': y_lat_val,
        'true_lon': y_lon_val,
        'true_alt': y_alt_val,
        'pred_lat': y_lat_pred,
        'pred_lon': y_lon_pred,
        'pred_alt': y_alt_pred,
        'horizontal_error_m': horizontal_errors,
        'angle_error_deg': angle_errors,
        'altitude_error_m': altitude_errors
    })
    val_results.to_csv('validation_predictions.csv', index=False)
    print("✅ Saved: validation_predictions.csv")

    print("\n" + "=" * 70)
    print("✅ Phase 5 Complete: Evaluation")
    print("=" * 70)
    print(f"\n🎯 NEW Competition Score: {total_error:.4f}")
    print(f"   Breakdown:")
    print(f"      Angle Error:  {mean_angle_error:.4f}° (weight 0.7)")
    print(f"      Height Error: {mean_altitude_error:.2f} m (weight 0.15)")
    print(f"      Range Error:  {mean_range_error:.2f} m (weight 0.15)")
    print(f"\n   (Target: < 2.0 for good performance)")
    print(f"\n   Ready for Phase 6: Test Set Prediction")
