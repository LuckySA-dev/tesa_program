"""
Phase 7.2: Enhanced Feature Engineering to Reduce Angle & Range Errors
Focus: Improve angle prediction (70% weight) and range prediction (15% weight)
"""

import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.2: Enhanced Feature Engineering")
    print("=" * 70)

    # Load data with YOLO features (using best YOLOv8n v15 model)
    df = pd.read_csv('train_metadata_with_yolo.csv')
    print(f"\n✅ Loaded data: {df.shape}")
    
    # Rename columns for consistency
    df.rename(columns={
        'latitude': 'latitude_deg',
        'longitude': 'longitude_deg',
        'altitude': 'altitude_m'
    }, inplace=True)

    # Camera position (fixed)
    CAMERA_LAT = 14.305029
    CAMERA_LON = 101.173010
    CAMERA_ALT = 0

    print("\n" + "=" * 70)
    print("Creating ENHANCED Features for Angle & Range Prediction")
    print("=" * 70)

    # ========== BASIC FEATURES ==========
    print("\n1️⃣ Basic features...")
    df['image_num'] = df['image_num'].astype(int)
    df['image_num_normalized'] = df['image_num'] / df['image_num'].max()

    # ========== YOLO FEATURES ==========
    print("2️⃣ YOLO bbox features...")
    df['yolo_area'] = df['yolo_w'] * df['yolo_h']
    df['yolo_aspect_ratio'] = df['yolo_w'] / (df['yolo_h'] + 1e-8)
    df['yolo_dist_from_center'] = np.sqrt((df['yolo_cx'] - 0.5)**2 + (df['yolo_cy'] - 0.5)**2)
    df['yolo_angle_from_center'] = np.arctan2(df['yolo_cy'] - 0.5, df['yolo_cx'] - 0.5)
    df['yolo_in_center'] = ((df['yolo_cx'] > 0.3) & (df['yolo_cx'] < 0.7) & 
                             (df['yolo_cy'] > 0.3) & (df['yolo_cy'] < 0.7)).astype(int)

    # ========== ENHANCED BEARING FEATURES (for Angle Error) ==========
    print("3️⃣ 🔥 ENHANCED Bearing features (Critical for Angle Error)...")
    
    # Original bearing features
    df['bearing_rad'] = np.deg2rad(df['bearing_deg'])
    df['bearing_sin'] = np.sin(df['bearing_rad'])
    df['bearing_cos'] = np.cos(df['bearing_rad'])
    
    # Multi-scale bearing features
    df['bearing_sin_2x'] = np.sin(2 * df['bearing_rad'])  # Double frequency
    df['bearing_cos_2x'] = np.cos(2 * df['bearing_rad'])
    df['bearing_sin_half'] = np.sin(0.5 * df['bearing_rad'])  # Half frequency
    df['bearing_cos_half'] = np.cos(0.5 * df['bearing_rad'])
    
    # Bearing quadrant (SW, W, NW zones for this camera angle)
    df['bearing_quadrant'] = pd.cut(df['bearing_deg'], 
                                      bins=[0, 200, 220, 240, 270, 360], 
                                      labels=[0, 1, 2, 3, 4]).astype(float)
    
    # Normalized bearing (0-1 scale)
    df['bearing_normalized'] = (df['bearing_deg'] - df['bearing_deg'].min()) / (df['bearing_deg'].max() - df['bearing_deg'].min())
    
    # Bearing deviation from center bearing
    center_bearing = df['bearing_deg'].median()
    df['bearing_deviation'] = np.abs(df['bearing_deg'] - center_bearing)
    df['bearing_deviation_squared'] = df['bearing_deviation'] ** 2

    # ========== ENHANCED POSITION FEATURES (for Angle & Range) ==========
    print("4️⃣ 🔥 ENHANCED Position features...")
    
    # Pixel position relative to image center (crucial for angle)
    df['pixel_offset_x'] = df['yolo_cx'] - 0.5
    df['pixel_offset_y'] = df['yolo_cy'] - 0.5
    df['pixel_offset_x_squared'] = df['pixel_offset_x'] ** 2
    df['pixel_offset_y_squared'] = df['pixel_offset_y'] ** 2
    
    # Pixel angle from center (in image space)
    df['pixel_angle'] = np.arctan2(df['pixel_offset_y'], df['pixel_offset_x'])
    df['pixel_angle_sin'] = np.sin(df['pixel_angle'])
    df['pixel_angle_cos'] = np.cos(df['pixel_angle'])
    
    # Distance-weighted position
    df['weighted_cx'] = df['yolo_cx'] * df['distance_m']
    df['weighted_cy'] = df['yolo_cy'] * df['distance_m']
    
    # Bearing × Position interactions (KEY for angle prediction)
    df['bearing_x_cx'] = df['bearing_deg'] * df['yolo_cx']
    df['bearing_x_cy'] = df['bearing_deg'] * df['yolo_cy']
    df['bearing_x_cx_squared'] = df['bearing_x_cx'] ** 2
    df['bearing_x_cy_squared'] = df['bearing_x_cy'] ** 2
    
    # Bearing × Pixel offset (strong angle predictor)
    df['bearing_x_offset_x'] = df['bearing_deg'] * df['pixel_offset_x']
    df['bearing_x_offset_y'] = df['bearing_deg'] * df['pixel_offset_y']

    # ========== ENHANCED DISTANCE FEATURES (for Range Error) ==========
    print("5️⃣ 🔥 ENHANCED Distance features (Critical for Range Error)...")
    
    # Distance transformations
    df['distance_log'] = np.log1p(df['distance_m'])
    df['distance_sqrt'] = np.sqrt(df['distance_m'])
    df['distance_squared'] = df['distance_m'] ** 2
    df['distance_inv'] = 1.0 / (df['distance_m'] + 1e-3)
    
    # Bbox size × Distance relationship (KEY for range)
    df['area_x_distance'] = df['yolo_area'] * df['distance_m']
    df['width_x_distance'] = df['yolo_w'] * df['distance_m']
    df['height_x_distance'] = df['yolo_h'] * df['distance_m']
    
    # Inverse relationship (farther = smaller bbox)
    df['distance_per_area'] = df['distance_m'] / (df['yolo_area'] + 1e-6)
    df['distance_per_width'] = df['distance_m'] / (df['yolo_w'] + 1e-6)
    df['distance_per_height'] = df['distance_m'] / (df['yolo_h'] + 1e-6)

    # ========== ALTITUDE INTERACTIONS ==========
    print("6️⃣ Altitude interaction features...")
    df['altitude_diff_from_camera'] = df['altitude_m'] - CAMERA_ALT
    df['altitude_x_distance'] = df['altitude_m'] * df['distance_m']
    df['altitude_sqrt'] = np.sqrt(df['altitude_m'])
    df['altitude_log'] = np.log1p(df['altitude_m'])

    # ========== CONFIDENCE FEATURES ==========
    print("7️⃣ Confidence-based features...")
    df['conf_squared'] = df['yolo_conf'] ** 2
    df['conf_sqrt'] = np.sqrt(df['yolo_conf'])
    df['conf_log'] = np.log1p(df['yolo_conf'])
    df['is_high_conf'] = (df['yolo_conf'] > 0.5).astype(int)
    df['is_detected'] = df['yolo_detected'].astype(int)
    
    # Confidence × Distance interaction
    df['conf_x_distance'] = df['yolo_conf'] * df['distance_m']
    df['conf_x_distance_inv'] = df['yolo_conf'] / (df['distance_m'] + 1e-3)

    # ========== GEOMETRIC FEATURES ==========
    print("8️⃣ Geometric relationship features...")
    
    # Estimated position from pixel location (rough approximation)
    df['estimated_lat_offset'] = (df['yolo_cy'] - 0.5) * df['distance_m'] / 111000
    df['estimated_lon_offset'] = (df['yolo_cx'] - 0.5) * df['distance_m'] / (111000 * np.cos(np.deg2rad(CAMERA_LAT)))
    df['estimated_lat'] = CAMERA_LAT + df['estimated_lat_offset']
    df['estimated_lon'] = CAMERA_LON + df['estimated_lon_offset']
    
    # Angular size (apparent size in image)
    df['angular_size_x'] = df['yolo_w'] * df['distance_m']
    df['angular_size_y'] = df['yolo_h'] * df['distance_m']

    # ========== STATISTICAL BINNING ==========
    print("9️⃣ Statistical binning features...")
    df['distance_bin'] = pd.cut(df['distance_m'], bins=5, labels=False)
    df['altitude_bin'] = pd.cut(df['altitude_m'], bins=5, labels=False)
    df['bearing_bin'] = pd.cut(df['bearing_deg'], bins=8, labels=False)  # 8 directions
    
    # One-hot encode bearing bins (important for angle prediction)
    for i in range(8):
        df[f'bearing_bin_{i}'] = (df['bearing_bin'] == i).astype(int)

    # ========== POLYNOMIAL INTERACTIONS ==========
    print("🔟 Polynomial interaction features...")
    
    # Distance × Bearing × Position (triple interaction)
    df['dist_bear_cx'] = df['distance_m'] * df['bearing_deg'] * df['yolo_cx']
    df['dist_bear_cy'] = df['distance_m'] * df['bearing_deg'] * df['yolo_cy']
    
    # Altitude × Bearing interaction
    df['alt_x_bearing'] = df['altitude_m'] * df['bearing_deg']
    df['alt_x_bearing_sin'] = df['altitude_m'] * df['bearing_sin']
    df['alt_x_bearing_cos'] = df['altitude_m'] * df['bearing_cos']

    print("\n" + "=" * 70)
    print("Feature Summary")
    print("=" * 70)
    print(f"Total features: {df.shape[1]}")
    
    # Save enhanced features
    output_file = 'train_metadata_enhanced_v2.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved enhanced features: {output_file}")
    print(f"   Shape: {df.shape}")

    # Prepare feature list for XGBoost (exclude metadata and targets)
    exclude_cols = ['image_num', 'image_name', 'csv_file', 'latitude_deg', 'longitude_deg', 
                    'altitude_m', 'yolo_detected', 'num_detections']
    
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    target_columns = ['latitude_deg', 'longitude_deg', 'altitude_m']

    print(f"\n✅ Selected {len(feature_columns)} features for XGBoost")
    print("\nKey feature groups added:")
    print("   🎯 Angle-focused: Multi-scale bearing, bearing×position interactions")
    print("   📏 Range-focused: Distance transforms, bbox×distance relationships")
    print("   🎲 Polynomial: Triple interactions (distance×bearing×position)")

    # Save feature names
    import json
    feature_info = {
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }
    
    with open('feature_columns_v2.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"\n✅ Saved feature info: feature_columns_v2.json")

    # Feature statistics
    print("\n" + "=" * 70)
    print("Feature Statistics")
    print("=" * 70)
    
    print(f"\nYOLO Features:")
    print(f"   Detection rate: {df['is_detected'].mean():.1%}")
    print(f"   Mean confidence: {df['yolo_conf'].mean():.3f}")
    
    print(f"\nGeospatial Features:")
    print(f"   Distance range: {df['distance_m'].min():.1f} - {df['distance_m'].max():.1f} m")
    print(f"   Bearing range: {df['bearing_deg'].min():.1f} - {df['bearing_deg'].max():.1f} deg")
    print(f"   Altitude range: {df['altitude_m'].min():.1f} - {df['altitude_m'].max():.1f} m")

    print("\n" + "=" * 70)
    print("✅ Phase 7.2 Complete: Enhanced Feature Engineering")
    print("=" * 70)
    print(f"\nReady for Phase 7.3: Retrain XGBoost with Enhanced Features")
    print(f"   Input file: {output_file}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Target: Reduce competition score from 6.64 to < 2.0")
