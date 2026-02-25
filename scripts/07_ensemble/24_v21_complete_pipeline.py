"""
Complete Pipeline: YOLO v21 → XGBoost → Competition Score
"""

import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def angular_difference(angle1, angle2):
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Complete Pipeline: YOLO v21 → XGBoost → Score")
    print("=" * 70)

    camera_lat, camera_lon = 14.305029, 101.173010

    # ========== STEP 1: Extract YOLO v21 Features ==========
    print("\n" + "=" * 70)
    print("Step 1: Extract YOLO v21 Features")
    print("=" * 70)
    
    model = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')
    df = pd.read_csv('train_metadata.csv')
    print(f"✅ Loaded: {df.shape[0]} images")
    
    image_dir = Path('datasets/DATA_TRAIN/image')
    yolo_features = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"   Processing: {idx+1}/{len(df)}...")
        
        img_path = image_dir / row['image_name']
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            yolo_features.append({
                'has_detection': 0, 'confidence': 0.0,
                'bbox_x': 0.5, 'bbox_y': 0.5, 'bbox_w': 0.0, 'bbox_h': 0.0, 'bbox_area': 0.0
            })
        else:
            boxes = results[0].boxes
            best_box = boxes[boxes.conf.argmax()]
            x, y, w, h = best_box.xywhn[0].cpu().numpy()
            yolo_features.append({
                'has_detection': 1, 'confidence': float(best_box.conf),
                'bbox_x': float(x), 'bbox_y': float(y),
                'bbox_w': float(w), 'bbox_h': float(h),
                'bbox_area': float(w * h)
            })
    
    yolo_df = pd.DataFrame(yolo_features)
    result_df = pd.concat([df, yolo_df], axis=1)
    
    detection_rate = yolo_df['has_detection'].mean()
    mean_conf = yolo_df[yolo_df['has_detection'] == 1]['confidence'].mean()
    print(f"\n✅ Features extracted: Detection {detection_rate:.1%}, Confidence {mean_conf:.3f}")
    
    # ========== STEP 2: Feature Engineering ==========
    print("\n" + "=" * 70)
    print("Step 2: Feature Engineering")
    print("=" * 70)
    
    result_df['distance_m'] = np.sqrt(
        ((result_df['latitude'] - camera_lat) * 111000) ** 2 +
        ((result_df['longitude'] - camera_lon) * 111000 * np.cos(np.radians(camera_lat))) ** 2
    )
    
    dlon = np.radians(result_df['longitude'] - camera_lon)
    lat1, lat2 = np.radians(camera_lat), np.radians(result_df['latitude'])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    result_df['bearing_deg'] = (np.degrees(np.arctan2(x, y)) + 360) % 360
    result_df['sin_bearing'] = np.sin(np.radians(result_df['bearing_deg']))
    result_df['cos_bearing'] = np.cos(np.radians(result_df['bearing_deg']))
    
    result_df['altitude_log'] = np.log1p(result_df['altitude'])
    result_df['altitude_sqrt'] = np.sqrt(result_df['altitude'])
    result_df['bbox_aspect_ratio'] = result_df['bbox_h'] / (result_df['bbox_w'] + 1e-6)
    result_df['bbox_center_distance'] = np.sqrt((result_df['bbox_x'] - 0.5)**2 + (result_df['bbox_y'] - 0.5)**2)
    result_df['offset_x'] = result_df['bbox_x'] - 0.5
    result_df['offset_y'] = result_df['bbox_y'] - 0.5
    result_df['distance_x_conf'] = result_df['distance_m'] * result_df['confidence']
    result_df['distance_x_area'] = result_df['distance_m'] * result_df['bbox_area']
    result_df['bearing_x_offset_x'] = result_df['sin_bearing'] * result_df['offset_x']
    result_df['bearing_x_offset_y'] = result_df['cos_bearing'] * result_df['offset_y']
    
    feature_columns = [
        'has_detection', 'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'bbox_area',
        'bbox_aspect_ratio', 'bbox_center_distance', 'offset_x', 'offset_y',
        'distance_m', 'sin_bearing', 'cos_bearing', 'altitude_log', 'altitude_sqrt',
        'distance_x_conf', 'distance_x_area', 'bearing_x_offset_x', 'bearing_x_offset_y'
    ]
    
    print(f"✅ Created {len(feature_columns)} features")
    
    # ========== STEP 3: Train XGBoost ==========
    print("\n" + "=" * 70)
    print("Step 3: Train XGBoost Models")
    print("=" * 70)
    
    X = result_df[feature_columns].values
    y_lat = result_df['latitude'].values
    y_lon = result_df['longitude'].values
    y_alt = result_df['altitude'].values
    
    X_train, X_val, y_lat_train, y_lat_val = train_test_split(X, y_lat, test_size=0.2, random_state=42)
    _, _, y_lon_train, y_lon_val = train_test_split(X, y_lon, test_size=0.2, random_state=42)
    _, _, y_alt_train, y_alt_val = train_test_split(X, y_alt, test_size=0.2, random_state=42)
    
    params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 500, 'early_stopping_rounds': 50}
    
    print("   Training Latitude...")
    model_lat = xgb.XGBRegressor(**params)
    model_lat.fit(X_train, y_lat_train, eval_set=[(X_val, y_lat_val)], verbose=False)
    
    print("   Training Longitude...")
    model_lon = xgb.XGBRegressor(**params)
    model_lon.fit(X_train, y_lon_train, eval_set=[(X_val, y_lon_val)], verbose=False)
    
    print("   Training Altitude...")
    model_alt = xgb.XGBRegressor(**params)
    model_alt.fit(X_train, y_alt_train, eval_set=[(X_val, y_alt_val)], verbose=False)
    
    # Metrics
    lat_rmse = np.sqrt(mean_squared_error(y_lat_val, model_lat.predict(X_val))) * 111000
    lon_rmse = np.sqrt(mean_squared_error(y_lon_val, model_lon.predict(X_val))) * 111000 * np.cos(np.radians(14.305))
    alt_rmse = np.sqrt(mean_squared_error(y_alt_val, model_alt.predict(X_val)))
    
    print(f"\n   ✅ Lat RMSE: {lat_rmse:.2f} m")
    print(f"   ✅ Lon RMSE: {lon_rmse:.2f} m")
    print(f"   ✅ Alt RMSE: {alt_rmse:.2f} m")
    
    # ========== STEP 4: Competition Score ==========
    print("\n" + "=" * 70)
    print("Step 4: Calculate Competition Score")
    print("=" * 70)
    
    _, val_idx = train_test_split(range(len(result_df)), test_size=0.2, random_state=42)
    df_val = result_df.iloc[val_idx]
    X_val_full = df_val[feature_columns].values
    
    pred_lat = model_lat.predict(X_val_full)
    pred_lon = model_lon.predict(X_val_full)
    pred_alt = model_alt.predict(X_val_full)
    
    pred_bearing = calculate_bearing(camera_lat, camera_lon, pred_lat, pred_lon)
    pred_range = haversine_distance(camera_lat, camera_lon, pred_lat, pred_lon)
    
    angle_err = angular_difference(pred_bearing, df_val['bearing_deg'].values)
    alt_err = np.abs(pred_alt - df_val['altitude'].values)
    range_err = np.abs(pred_range - df_val['distance_m'].values)
    
    score = 0.7 * angle_err.mean() + 0.15 * alt_err.mean() + 0.15 * range_err.mean()
    
    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS - YOLO v21 + XGBoost")
    print("=" * 70)
    print(f"\n   Angle Error:    {angle_err.mean():.2f}°")
    print(f"   Altitude Error: {alt_err.mean():.2f} m")
    print(f"   Range Error:    {range_err.mean():.2f} m")
    print(f"\n   🏆 Competition Score: {score:.4f}")
    print(f"\n   Baseline (v1):        5.9369")
    print(f"   Improvement:          {5.9369-score:+.4f} ({(5.9369-score)/5.9369*100:+.1f}%)")
    
    if score < 5.9369:
        print(f"\n   ✅ SUCCESS! v21 is BETTER by {5.9369-score:.4f} points!")
        if score < 2.0:
            print(f"   🎉🎉🎉 TARGET ACHIEVED! Score < 2.0!")
    else:
        print(f"\n   ⚠️ Need more optimization")
    
    print("\n" + "=" * 70)
    
    # Save models
    with open('xgb_model_latitude_v21.pkl', 'wb') as f:
        pickle.dump(model_lat, f)
    with open('xgb_model_longitude_v21.pkl', 'wb') as f:
        pickle.dump(model_lon, f)
    with open('xgb_model_altitude_v21.pkl', 'wb') as f:
        pickle.dump(model_alt, f)
    
    print("✅ Models saved: xgb_model_*_v21.pkl")
