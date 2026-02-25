"""
Phase 7.9: Feature Engineering + Train XGBoost with YOLO v16
Complete pipeline with improved YOLO features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.9: Feature Engineering + XGBoost with YOLO v16")
    print("=" * 70)

    # Load YOLO v16 features
    df = pd.read_csv('train_metadata_with_yolo_v16.csv')
    print(f"\n✅ Loaded v16 features: {df.shape}")
    
    # Rename columns for consistency
    df = df.rename(columns={
        'latitude': 'latitude_deg',
        'longitude': 'longitude_deg',
        'altitude': 'altitude_m'
    })

    # Camera position
    camera_lat, camera_lon = 14.305029, 101.173010
    
    # Feature Engineering
    print("\n" + "=" * 70)
    print("Feature Engineering")
    print("=" * 70)
    
    # 1. Geospatial features (same as before)
    df['distance_m'] = np.sqrt(
        ((df['latitude_deg'] - camera_lat) * 111000) ** 2 +
        ((df['longitude_deg'] - camera_lon) * 111000 * np.cos(np.radians(camera_lat))) ** 2
    )
    
    # Bearing
    dlon = np.radians(df['longitude_deg'] - camera_lon)
    lat1 = np.radians(camera_lat)
    lat2 = np.radians(df['latitude_deg'])
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    df['bearing_deg'] = (np.degrees(bearing) + 360) % 360
    
    df['sin_bearing'] = np.sin(np.radians(df['bearing_deg']))
    df['cos_bearing'] = np.cos(np.radians(df['bearing_deg']))
    
    # Altitude features
    df['altitude_log'] = np.log1p(df['altitude_m'])
    df['altitude_sqrt'] = np.sqrt(df['altitude_m'])
    
    # 2. YOLO bbox features
    df['bbox_aspect_ratio'] = df['bbox_h'] / (df['bbox_w'] + 1e-6)
    df['bbox_center_distance'] = np.sqrt((df['bbox_x'] - 0.5)**2 + (df['bbox_y'] - 0.5)**2)
    
    # Offset from center
    df['offset_x'] = df['bbox_x'] - 0.5
    df['offset_y'] = df['bbox_y'] - 0.5
    
    # 3. Interaction features
    df['distance_x_conf'] = df['distance_m'] * df['confidence']
    df['distance_x_area'] = df['distance_m'] * df['bbox_area']
    df['altitude_x_conf'] = df['altitude_m'] * df['confidence']
    
    # Position-related
    df['bearing_x_offset_x'] = df['sin_bearing'] * df['offset_x']
    df['bearing_x_offset_y'] = df['cos_bearing'] * df['offset_y']
    
    # Size-distance relationship
    df['area_per_distance'] = df['bbox_area'] / (df['distance_m'] + 1)
    df['conf_x_area'] = df['confidence'] * df['bbox_area']
    
    # Pixel position features
    df['pixel_angle'] = np.arctan2(df['offset_y'], df['offset_x'])
    df['pixel_angle_sin'] = np.sin(df['pixel_angle'])
    df['pixel_angle_cos'] = np.cos(df['pixel_angle'])
    
    print(f"✅ Created {df.shape[1] - 13} engineered features")
    
    # Select features for XGBoost
    feature_columns = [
        # YOLO features
        'has_detection', 'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'bbox_area',
        'bbox_aspect_ratio', 'bbox_center_distance', 'offset_x', 'offset_y',
        
        # Geospatial
        'distance_m', 'sin_bearing', 'cos_bearing',
        'altitude_log', 'altitude_sqrt',
        
        # Interactions
        'distance_x_conf', 'distance_x_area', 'altitude_x_conf',
        'bearing_x_offset_x', 'bearing_x_offset_y',
        'area_per_distance', 'conf_x_area',
        'pixel_angle_sin', 'pixel_angle_cos'
    ]
    
    print(f"✅ Selected {len(feature_columns)} features for training")
    
    # Save engineered dataset
    df.to_csv('train_metadata_engineered_v16.csv', index=False)
    print(f"✅ Saved: train_metadata_engineered_v16.csv")
    
    # Prepare data
    X = df[feature_columns].values
    y_lat = df['latitude_deg'].values
    y_lon = df['longitude_deg'].values
    y_alt = df['altitude_m'].values
    
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
    
    print(f"\n✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # Train XGBoost
    print("\n" + "=" * 70)
    print("Training XGBoost Models")
    print("=" * 70)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    # 1. Latitude
    print("\n1️⃣ Training Latitude model...")
    model_lat = xgb.XGBRegressor(**params)
    model_lat.fit(
        X_train, y_lat_train,
        eval_set=[(X_train, y_lat_train), (X_val, y_lat_val)],
        verbose=False
    )
    
    y_lat_val_pred = model_lat.predict(X_val)
    lat_rmse = np.sqrt(mean_squared_error(y_lat_val, y_lat_val_pred)) * 111000
    lat_r2 = r2_score(y_lat_val, y_lat_val_pred)
    print(f"   Val RMSE: {lat_rmse:.2f} m")
    print(f"   Val R²:   {lat_r2:.4f}")
    
    # 2. Longitude
    print("\n2️⃣ Training Longitude model...")
    model_lon = xgb.XGBRegressor(**params)
    model_lon.fit(
        X_train, y_lon_train,
        eval_set=[(X_train, y_lon_train), (X_val, y_lon_val)],
        verbose=False
    )
    
    y_lon_val_pred = model_lon.predict(X_val)
    cos_lat = np.cos(np.radians(14.305))
    lon_rmse = np.sqrt(mean_squared_error(y_lon_val, y_lon_val_pred)) * 111000 * cos_lat
    lon_r2 = r2_score(y_lon_val, y_lon_val_pred)
    print(f"   Val RMSE: {lon_rmse:.2f} m")
    print(f"   Val R²:   {lon_r2:.4f}")
    
    # 3. Altitude
    print("\n3️⃣ Training Altitude model...")
    model_alt = xgb.XGBRegressor(**params)
    model_alt.fit(
        X_train, y_alt_train,
        eval_set=[(X_train, y_alt_train), (X_val, y_alt_val)],
        verbose=False
    )
    
    y_alt_val_pred = model_alt.predict(X_val)
    alt_rmse = np.sqrt(mean_squared_error(y_alt_val, y_alt_val_pred))
    alt_r2 = r2_score(y_alt_val, y_alt_val_pred)
    print(f"   Val RMSE: {alt_rmse:.2f} m")
    print(f"   Val R²:   {alt_r2:.4f}")
    
    # Save models
    with open('xgb_model_latitude_v16.pkl', 'wb') as f:
        pickle.dump(model_lat, f)
    with open('xgb_model_longitude_v16.pkl', 'wb') as f:
        pickle.dump(model_lon, f)
    with open('xgb_model_altitude_v16.pkl', 'wb') as f:
        pickle.dump(model_alt, f)
    
    # Save feature list
    import json
    with open('feature_columns_v16.json', 'w') as f:
        json.dump({'feature_columns': feature_columns}, f, indent=2)
    
    print("\n✅ Saved models: xgb_model_*_v16.pkl")
    
    # Compare with v1
    print("\n" + "=" * 70)
    print("📈 Comparison with Baseline v1")
    print("=" * 70)
    print("\n   Baseline v1 (YOLO v15, 50 labels):")
    print("      Lat RMSE:  12.07 m, R²: 0.77")
    print("      Lon RMSE:  8.64 m, R²: 0.94")
    print("      Alt RMSE:  1.01 m, R²: 0.99")
    
    print(f"\n   v16 (YOLO v16, 368 labels):")
    print(f"      Lat RMSE:  {lat_rmse:.2f} m, R²: {lat_r2:.2f}")
    print(f"      Lon RMSE:  {lon_rmse:.2f} m, R²: {lon_r2:.2f}")
    print(f"      Alt RMSE:  {alt_rmse:.2f} m, R²: {alt_r2:.2f}")
    
    lat_improve = (12.07 - lat_rmse) / 12.07 * 100
    lon_improve = (8.64 - lon_rmse) / 8.64 * 100
    alt_improve = (1.01 - alt_rmse) / 1.01 * 100
    
    print(f"\n   Improvement:")
    print(f"      Lat: {lat_improve:+.1f}%")
    print(f"      Lon: {lon_improve:+.1f}%")
    print(f"      Alt: {alt_improve:+.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ Phase 7.9 Complete!")
    print("=" * 70)
    print("\n🎯 Next: Evaluate competition score with v16 models")
