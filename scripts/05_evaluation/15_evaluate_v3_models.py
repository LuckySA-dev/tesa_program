"""
Phase 7.5: Evaluate v3 Models with Competition Scoring
Compare: Baseline (v1) vs Feature Selected (v3)
"""

import pandas as pd
import numpy as np
import pickle
import json

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def angular_difference(angle1, angle2):
    """Calculate minimum angular difference"""
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.5: Evaluate v3 Models")
    print("=" * 70)

    # Camera position
    camera_lat, camera_lon, camera_alt = 14.305029, 101.173010, 0.0

    # Load both datasets
    df_baseline = pd.read_csv('train_metadata_engineered.csv')
    df_enhanced = pd.read_csv('train_metadata_enhanced_v2.csv')
    print(f"\n✅ Loaded baseline data: {df_baseline.shape}")
    print(f"✅ Loaded enhanced data: {df_enhanced.shape}")

    # Load v3 models and selected features
    with open('xgb_model_latitude_v3.pkl', 'rb') as f:
        model_lat_v3 = pickle.load(f)
    with open('xgb_model_longitude_v3.pkl', 'rb') as f:
        model_lon_v3 = pickle.load(f)
    with open('xgb_model_altitude_v3.pkl', 'rb') as f:
        model_alt_v3 = pickle.load(f)
    
    with open('selected_features_v3.json', 'r') as f:
        selected_features = json.load(f)
    
    print(f"✅ Loaded v3 models")

    # Load baseline v1 models
    with open('xgb_model_latitude.pkl', 'rb') as f:
        model_lat_v1 = pickle.load(f)
    with open('xgb_model_longitude.pkl', 'rb') as f:
        model_lon_v1 = pickle.load(f)
    with open('xgb_model_altitude.pkl', 'rb') as f:
        model_alt_v1 = pickle.load(f)
    
    with open('feature_columns.json', 'r') as f:
        baseline_features = json.load(f)['feature_columns']
    
    print(f"✅ Loaded v1 baseline models")

    # Create validation split (same as training)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        range(len(df_baseline)), test_size=0.2, random_state=42
    )
    
    df_val_v1 = df_baseline.iloc[val_idx].copy()
    df_val_v3 = df_enhanced.iloc[val_idx].copy()
    print(f"\n✅ Validation set: {len(df_val_v1)} samples")

    # ========== BASELINE v1 PREDICTIONS ==========
    print("\n" + "=" * 70)
    print("Baseline v1 (28 features)")
    print("=" * 70)

    X_val_v1 = df_val_v1[baseline_features]
    pred_lat_v1 = model_lat_v1.predict(X_val_v1)
    pred_lon_v1 = model_lon_v1.predict(X_val_v1)
    pred_alt_v1 = model_alt_v1.predict(X_val_v1)

    # Calculate metrics
    pred_bearing_v1 = calculate_bearing(
        camera_lat, camera_lon, pred_lat_v1, pred_lon_v1
    )
    pred_range_v1 = haversine_distance(
        camera_lat, camera_lon, pred_lat_v1, pred_lon_v1
    )

    angle_error_v1 = angular_difference(
        pred_bearing_v1, df_val_v1['bearing_deg'].values
    )
    altitude_error_v1 = np.abs(pred_alt_v1 - df_val_v1['altitude_m'].values)
    range_error_v1 = np.abs(pred_range_v1 - df_val_v1['distance_m'].values)

    mean_angle_v1 = angle_error_v1.mean()
    mean_altitude_v1 = altitude_error_v1.mean()
    mean_range_v1 = range_error_v1.mean()

    score_v1 = 0.7 * mean_angle_v1 + 0.15 * mean_altitude_v1 + 0.15 * mean_range_v1

    print(f"\n📊 Baseline v1 Results:")
    print(f"   Mean Angle Error:    {mean_angle_v1:.2f}°")
    print(f"   Mean Altitude Error: {mean_altitude_v1:.2f} m")
    print(f"   Mean Range Error:    {mean_range_v1:.2f} m")
    print(f"\n   Competition Score: {score_v1:.4f}")
    print(f"      = 0.70 × {mean_angle_v1:.2f} + 0.15 × {mean_altitude_v1:.2f} + 0.15 × {mean_range_v1:.2f}")

    # ========== v3 PREDICTIONS ==========
    print("\n" + "=" * 70)
    print("v3 (30 selected features)")
    print("=" * 70)

    X_val_lat_v3 = df_val_v3[selected_features['lat_features']]
    X_val_lon_v3 = df_val_v3[selected_features['lon_features']]
    X_val_alt_v3 = df_val_v3[selected_features['alt_features']]

    pred_lat_v3 = model_lat_v3.predict(X_val_lat_v3)
    pred_lon_v3 = model_lon_v3.predict(X_val_lon_v3)
    pred_alt_v3 = model_alt_v3.predict(X_val_alt_v3)

    # Calculate metrics
    pred_bearing_v3 = calculate_bearing(
        camera_lat, camera_lon, pred_lat_v3, pred_lon_v3
    )
    pred_range_v3 = haversine_distance(
        camera_lat, camera_lon, pred_lat_v3, pred_lon_v3
    )

    angle_error_v3 = angular_difference(
        pred_bearing_v3, df_val_v3['bearing_deg'].values
    )
    altitude_error_v3 = np.abs(pred_alt_v3 - df_val_v3['altitude_m'].values)
    range_error_v3 = np.abs(pred_range_v3 - df_val_v3['distance_m'].values)

    mean_angle_v3 = angle_error_v3.mean()
    mean_altitude_v3 = altitude_error_v3.mean()
    mean_range_v3 = range_error_v3.mean()

    score_v3 = 0.7 * mean_angle_v3 + 0.15 * mean_altitude_v3 + 0.15 * mean_range_v3

    print(f"\n📊 v3 Results:")
    print(f"   Mean Angle Error:    {mean_angle_v3:.2f}°")
    print(f"   Mean Altitude Error: {mean_altitude_v3:.2f} m")
    print(f"   Mean Range Error:    {mean_range_v3:.2f} m")
    print(f"\n   Competition Score: {score_v3:.4f}")
    print(f"      = 0.70 × {mean_angle_v3:.2f} + 0.15 × {mean_altitude_v3:.2f} + 0.15 × {mean_range_v3:.2f}")

    # ========== COMPARISON ==========
    print("\n" + "=" * 70)
    print("📊 Comparison: v3 vs Baseline v1")
    print("=" * 70)

    angle_improve = mean_angle_v1 - mean_angle_v3
    altitude_improve = mean_altitude_v1 - mean_altitude_v3
    range_improve = mean_range_v1 - mean_range_v3
    score_improve = score_v1 - score_v3

    print(f"\n   Angle Error:    {mean_angle_v1:.2f}° → {mean_angle_v3:.2f}°")
    print(f"      Change: {angle_improve:+.2f}° ({angle_improve/mean_angle_v1*100:+.1f}%)")
    
    print(f"\n   Altitude Error: {mean_altitude_v1:.2f}m → {mean_altitude_v3:.2f}m")
    print(f"      Change: {altitude_improve:+.2f}m ({altitude_improve/mean_altitude_v1*100:+.1f}%)")
    
    print(f"\n   Range Error:    {mean_range_v1:.2f}m → {mean_range_v3:.2f}m")
    print(f"      Change: {range_improve:+.2f}m ({range_improve/mean_range_v1*100:+.1f}%)")
    
    print(f"\n   Competition Score: {score_v1:.4f} → {score_v3:.4f}")
    print(f"      Change: {score_improve:+.4f} ({score_improve/score_v1*100:+.1f}%)")

    if score_v3 < score_v1:
        print(f"\n   ✅ v3 is BETTER by {score_improve:.4f} points!")
    else:
        print(f"\n   ⚠️ v3 is WORSE by {-score_improve:.4f} points")

    # Save detailed comparison
    comparison_df = pd.DataFrame({
        'image_name': df_val_v1['image_name'].values,
        'latitude_deg': df_val_v1['latitude_deg'].values,
        'longitude_deg': df_val_v1['longitude_deg'].values,
        'altitude_m': df_val_v1['altitude_m'].values,
        'bearing_deg': df_val_v1['bearing_deg'].values,
        'distance_m': df_val_v1['distance_m'].values,
        'pred_lat_v1': pred_lat_v1,
        'pred_lon_v1': pred_lon_v1,
        'pred_alt_v1': pred_alt_v1,
        'angle_error_v1': angle_error_v1,
        'altitude_error_v1': altitude_error_v1,
        'range_error_v1': range_error_v1,
        'pred_lat_v3': pred_lat_v3,
        'pred_lon_v3': pred_lon_v3,
        'pred_alt_v3': pred_alt_v3,
        'angle_error_v3': angle_error_v3,
        'altitude_error_v3': altitude_error_v3,
        'range_error_v3': range_error_v3
    })

    comparison_df.to_csv('validation_comparison_v1_vs_v3.csv', index=False)
    print(f"\n✅ Saved detailed comparison: validation_comparison_v1_vs_v3.csv")

    print("\n" + "=" * 70)
    print("✅ Phase 7.5 Complete")
    print("=" * 70)
