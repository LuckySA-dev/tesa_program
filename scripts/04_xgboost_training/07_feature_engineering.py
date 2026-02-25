"""
Phase 3: Feature Engineering for XGBoost
Combine YOLO features with geospatial features and create additional features
"""

import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 3: Feature Engineering for XGBoost")
    print("=" * 70)

    # Load data with YOLO features
    df = pd.read_csv('train_metadata_with_yolo.csv')
    print(f"\n✅ Loaded data: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Rename columns for consistency
    df.rename(columns={
        'latitude': 'latitude_deg',
        'longitude': 'longitude_deg',
        'altitude': 'altitude_m'
    }, inplace=True)

    # Camera position (fixed)
    CAMERA_LAT = 14.305029
    CAMERA_LON = 101.173010
    CAMERA_ALT = 0  # Assume ground level

    print("\n" + "=" * 70)
    print("Creating Engineered Features")
    print("=" * 70)

    # 1. Image-based features
    print("\n1️⃣ Image-based features...")
    df['image_num'] = df['image_name'].str.extract(r'(\d+)').astype(int)
    df['image_num_normalized'] = df['image_num'] / df['image_num'].max()

    # 2. YOLO bbox features
    print("2️⃣ YOLO bbox features...")
    df['yolo_area'] = df['yolo_w'] * df['yolo_h']
    df['yolo_aspect_ratio'] = df['yolo_w'] / (df['yolo_h'] + 1e-8)
    
    # Distance from center of image
    df['yolo_dist_from_center'] = np.sqrt((df['yolo_cx'] - 0.5)**2 + (df['yolo_cy'] - 0.5)**2)
    
    # Angle from image center (in radians)
    df['yolo_angle_from_center'] = np.arctan2(df['yolo_cy'] - 0.5, df['yolo_cx'] - 0.5)
    
    # Position quadrant (0=center, 1-4=corners)
    df['yolo_in_center'] = ((df['yolo_cx'] > 0.3) & (df['yolo_cx'] < 0.7) & 
                             (df['yolo_cy'] > 0.3) & (df['yolo_cy'] < 0.7)).astype(int)

    # 3. Geospatial features (already calculated)
    print("3️⃣ Geospatial features...")
    df['bearing_rad'] = np.deg2rad(df['bearing_deg'])
    df['bearing_sin'] = np.sin(df['bearing_rad'])
    df['bearing_cos'] = np.cos(df['bearing_rad'])

    # 4. Interaction features (YOLO × Geospatial)
    print("4️⃣ Interaction features...")
    
    # Distance-related interactions
    df['distance_x_conf'] = df['distance_m'] * df['yolo_conf']
    df['distance_x_area'] = df['distance_m'] * df['yolo_area']
    
    # Bearing-related interactions
    df['bearing_x_cx'] = df['bearing_deg'] * df['yolo_cx']
    df['bearing_x_cy'] = df['bearing_deg'] * df['yolo_cy']
    
    # Altitude-related features
    df['altitude_diff_from_camera'] = df['altitude_m'] - CAMERA_ALT
    df['altitude_x_distance'] = df['altitude_m'] * df['distance_m']
    
    # 5. Confidence-based features
    print("5️⃣ Confidence-based features...")
    df['conf_squared'] = df['yolo_conf'] ** 2
    df['conf_sqrt'] = np.sqrt(df['yolo_conf'])
    df['is_high_conf'] = (df['yolo_conf'] > 0.5).astype(int)
    df['is_detected'] = df['yolo_detected'].astype(int)

    # 6. Position estimation features (pixel to degree approximation)
    print("6️⃣ Position estimation features...")
    
    # Assume image covers certain field of view
    # Rough approximation: offset from center relates to lat/lon offset
    df['estimated_lat_offset'] = (df['yolo_cy'] - 0.5) * df['distance_m'] / 111000  # meters to degrees
    df['estimated_lon_offset'] = (df['yolo_cx'] - 0.5) * df['distance_m'] / (111000 * np.cos(np.deg2rad(CAMERA_LAT)))
    
    df['estimated_lat'] = CAMERA_LAT + df['estimated_lat_offset']
    df['estimated_lon'] = CAMERA_LON + df['estimated_lon_offset']

    # 7. Statistical features
    print("7️⃣ Statistical features...")
    
    # Distance bins
    df['distance_bin'] = pd.cut(df['distance_m'], bins=5, labels=False)
    
    # Altitude bins
    df['altitude_bin'] = pd.cut(df['altitude_m'], bins=5, labels=False)

    print("\n" + "=" * 70)
    print("Feature Summary")
    print("=" * 70)
    print(f"Total features: {df.shape[1]}")
    print(f"\nNew features added:")
    
    new_features = [
        'image_num', 'image_num_normalized',
        'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center', 'yolo_angle_from_center', 'yolo_in_center',
        'bearing_rad', 'bearing_sin', 'bearing_cos',
        'distance_x_conf', 'distance_x_area', 'bearing_x_cx', 'bearing_x_cy',
        'altitude_diff_from_camera', 'altitude_x_distance',
        'conf_squared', 'conf_sqrt', 'is_high_conf', 'is_detected',
        'estimated_lat_offset', 'estimated_lon_offset', 'estimated_lat', 'estimated_lon',
        'distance_bin', 'altitude_bin'
    ]
    
    for i, feat in enumerate(new_features, 1):
        print(f"   {i:2d}. {feat}")

    # Save engineered features
    output_file = 'train_metadata_engineered.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved engineered features: {output_file}")
    print(f"   Shape: {df.shape}")

    # Feature statistics
    print("\n" + "=" * 70)
    print("Feature Statistics")
    print("=" * 70)
    
    print(f"\nYOLO Features:")
    print(f"   Detection rate: {df['yolo_detected'].mean():.1%}")
    print(f"   Mean confidence: {df['yolo_conf'].mean():.3f}")
    print(f"   Mean bbox area: {df['yolo_area'].mean():.6f}")
    print(f"   Mean aspect ratio: {df['yolo_aspect_ratio'].mean():.3f}")
    print(f"   In center: {df['yolo_in_center'].mean():.1%}")
    
    print(f"\nGeospatial Features:")
    print(f"   Distance range: {df['distance_m'].min():.1f} - {df['distance_m'].max():.1f} m")
    print(f"   Bearing range: {df['bearing_deg'].min():.1f} - {df['bearing_deg'].max():.1f} deg")
    print(f"   Altitude range: {df['altitude_m'].min():.1f} - {df['altitude_m'].max():.1f} m")

    print(f"\nTarget Variables:")
    print(f"   Latitude range: {df['latitude_deg'].min():.6f} - {df['latitude_deg'].max():.6f}")
    print(f"   Longitude range: {df['longitude_deg'].min():.6f} - {df['longitude_deg'].max():.6f}")
    print(f"   Altitude range: {df['altitude_m'].min():.1f} - {df['altitude_m'].max():.1f} m")

    # Check for missing values
    print(f"\n" + "=" * 70)
    print("Data Quality Check")
    print("=" * 70)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values!")

    # Prepare feature list for XGBoost
    feature_columns = [
        # YOLO features
        'yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
        'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center', 'yolo_angle_from_center', 'yolo_in_center',
        # Geospatial features
        'distance_m', 'bearing_deg', 'bearing_sin', 'bearing_cos',
        # Interaction features
        'distance_x_conf', 'distance_x_area', 'bearing_x_cx', 'bearing_x_cy',
        'altitude_x_distance',
        # Confidence features
        'conf_squared', 'conf_sqrt', 'is_high_conf', 'is_detected',
        # Position estimation
        'estimated_lat_offset', 'estimated_lon_offset',
        # Bins
        'distance_bin', 'altitude_bin',
        # Image features
        'image_num_normalized'
    ]
    
    target_columns = ['latitude_deg', 'longitude_deg', 'altitude_m']

    print(f"\n✅ Selected {len(feature_columns)} features for XGBoost")
    print(f"✅ Target variables: {target_columns}")

    # Save feature names for later use
    feature_info = {
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }
    
    import json
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"\n✅ Saved feature info: feature_columns.json")

    print("\n" + "=" * 70)
    print("✅ Phase 3 Complete: Feature Engineering")
    print("=" * 70)
    print(f"\nReady for Phase 4: XGBoost Training")
    print(f"   Input file: {output_file}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Samples: {len(df)}")
