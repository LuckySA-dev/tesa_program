"""
Generate predictions for ALL detected drones in each image
Output format: Multiple rows per image if multiple drones detected
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from ultralytics import YOLO
import cv2

# Constants
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

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

print("="*80)
print("🚁 Multi-Drone Prediction - ALL Drones")
print("="*80)

# Load models
print("\n📂 Loading models...")
models = {
    'distance': joblib.load('models_approximation/bbox_to_distance.pkl'),
    'bearing_sin': joblib.load('models_approximation/bbox_to_bearing_sin.pkl'),
    'bearing_cos': joblib.load('models_approximation/bbox_to_bearing_cos.pkl'),
    'altitude': joblib.load('models_approximation/bbox_to_altitude.pkl'),
    'dist_residual': joblib.load('models_approximation/residual_distance.pkl'),
    'bearing_residual': joblib.load('models_approximation/residual_bearing.pkl'),
    'alt_residual': joblib.load('models_approximation/residual_altitude.pkl')
}

with open('models_approximation/bbox_features.json', 'r') as f:
    bbox_features = json.load(f)

yolo_model = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')
print("✅ Models loaded")

# Find test images
test_dir = Path('datasets/DATA_TEST')
test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))
print(f"✅ Found {len(test_images)} test images")

def extract_bbox_features(box, img_width, img_height):
    """Extract features from a single bbox"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0])
    
    # Normalize coordinates
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    
    area = w * h
    aspect_ratio = w / h if h > 0 else 0
    dist_from_center = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    angle_from_center = np.degrees(np.arctan2(cy - 0.5, cx - 0.5))
    
    return {
        'yolo_cx': cx,
        'yolo_cy': cy,
        'yolo_w': w,
        'yolo_h': h,
        'yolo_conf': conf,
        'yolo_area': area,
        'yolo_aspect_ratio': aspect_ratio,
        'yolo_dist_from_center': dist_from_center,
        'yolo_angle_from_center': angle_from_center
    }

def predict_gps(features):
    """Predict GPS from bbox features"""
    X = np.array([[features[f] for f in bbox_features]])
    
    # Base predictions
    pred_distance_base = models['distance'].predict(X)[0]
    pred_bearing_sin = models['bearing_sin'].predict(X)[0]
    pred_bearing_cos = models['bearing_cos'].predict(X)[0]
    pred_bearing_base = np.degrees(np.arctan2(pred_bearing_sin, pred_bearing_cos)) % 360
    pred_altitude_base = models['altitude'].predict(X)[0]
    
    # Apply residual corrections
    residual_distance = models['dist_residual'].predict(X)[0]
    residual_bearing = models['bearing_residual'].predict(X)[0]
    residual_altitude = models['alt_residual'].predict(X)[0]
    
    pred_distance = pred_distance_base - residual_distance
    pred_bearing = (pred_bearing_base - residual_bearing) % 360
    pred_altitude = pred_altitude_base - residual_altitude
    
    # Convert to GPS
    lat, lon = bearing_distance_to_latlon(pred_bearing, pred_distance)
    
    return lat, lon, pred_altitude, features['yolo_conf'], features['yolo_area']

print("\n🔍 Processing ALL drones in test images...")
print("="*80)

all_predictions = []
image_stats = {
    '0_drones': 0,
    '1_drone': 0,
    '2_drones': 0,
    '3+_drones': 0
}

for img_path in test_images:
    img_name = img_path.name
    
    # Load image
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]
    
    # Run YOLO
    results = yolo_model(img_path, verbose=False)
    boxes = results[0].boxes
    
    num_drones = len(boxes)
    
    if num_drones == 0:
        image_stats['0_drones'] += 1
        # No detection - add one default prediction
        default_features = {
            'yolo_cx': 0.5, 'yolo_cy': 0.5, 'yolo_w': 0.1, 'yolo_h': 0.1,
            'yolo_conf': 0.0, 'yolo_area': 0.01, 'yolo_aspect_ratio': 1.0,
            'yolo_dist_from_center': 0.0, 'yolo_angle_from_center': 0.0
        }
        lat, lon, alt, conf, area = predict_gps(default_features)
        
        all_predictions.append({
            'ImageName': img_name,
            'DroneID': 1,
            'Latitude': lat,
            'Longitude': lon,
            'Altitude': alt,
            'Confidence': conf,
            'BboxArea': area
        })
    elif num_drones == 1:
        image_stats['1_drone'] += 1
    elif num_drones == 2:
        image_stats['2_drones'] += 1
    else:
        image_stats['3+_drones'] += 1
    
    if num_drones > 0:
        # Process ALL detected drones
        for drone_idx, box in enumerate(boxes, start=1):
            features = extract_bbox_features(box, img_width, img_height)
            lat, lon, alt, conf, area = predict_gps(features)
            
            all_predictions.append({
                'ImageName': img_name,
                'DroneID': drone_idx,
                'Latitude': lat,
                'Longitude': lon,
                'Altitude': alt,
                'Confidence': conf,
                'BboxArea': area
            })

print("✅ Processing complete")

# Create DataFrame
df_all = pd.DataFrame(all_predictions)

print("\n📊 Detection Statistics:")
print(f"  0 drones: {image_stats['0_drones']} images")
print(f"  1 drone:  {image_stats['1_drone']} images")
print(f"  2 drones: {image_stats['2_drones']} images")
print(f"  3+ drones: {image_stats['3+_drones']} images")
print(f"\nTotal predictions: {len(df_all)} (for {len(test_images)} images)")
print(f"Average drones per image: {len(df_all)/len(test_images):.2f}")

# Save full predictions with DroneID
print("\n💾 Saving predictions...")
df_all_sorted = df_all.sort_values(['ImageName', 'DroneID'])
df_all_sorted.to_csv('test_predictions_all_drones.csv', index=False, encoding='utf-8')
print(f"✅ Saved: test_predictions_all_drones.csv")

# Also save simplified format (without DroneID and metadata)
df_simple = df_all_sorted[['ImageName', 'Latitude', 'Longitude', 'Altitude']].copy()
df_simple.to_csv('test_predictions_all_drones_simple.csv', index=False, encoding='utf-8')
print(f"✅ Saved: test_predictions_all_drones_simple.csv")

# Show sample
print("\n📋 Sample predictions (first 20 rows):")
print(df_all_sorted.head(20).to_string(index=False))

print("\n" + "="*80)
print("📊 Summary Statistics")
print("="*80)

print(f"\nBy number of drones per image:")
images_with_multiple = df_all.groupby('ImageName').size()
print(f"  Images with 1 prediction:  {(images_with_multiple == 1).sum()}")
print(f"  Images with 2 predictions: {(images_with_multiple == 2).sum()}")
print(f"  Images with 3 predictions: {(images_with_multiple == 3).sum()}")
print(f"  Images with 4+ predictions: {(images_with_multiple >= 4).sum()}")

print(f"\nConfidence distribution:")
print(f"  Mean confidence: {df_all['Confidence'].mean():.3f}")
print(f"  Min confidence:  {df_all['Confidence'].min():.3f}")
print(f"  Max confidence:  {df_all['Confidence'].max():.3f}")

print(f"\nGPS coordinate ranges:")
print(f"  Latitude:  {df_all['Latitude'].min():.6f} - {df_all['Latitude'].max():.6f}")
print(f"  Longitude: {df_all['Longitude'].min():.6f} - {df_all['Longitude'].max():.6f}")
print(f"  Altitude:  {df_all['Altitude'].min():.2f} - {df_all['Altitude'].max():.2f} m")

print("\n" + "="*80)
print("⚠️  IMPORTANT NOTE")
print("="*80)
print("""
สร้าง 2 ไฟล์:

1. test_predictions_all_drones.csv (รายละเอียดเต็ม)
   - มี DroneID, Confidence, BboxArea
   - {len(df_all)} predictions สำหรับ {len(test_images)} images
   - ใช้สำหรับวิเคราะห์

2. test_predictions_all_drones_simple.csv (สำหรับส่งแข่ง?)
   - เฉพาะ ImageName, Latitude, Longitude, Altitude
   - หลายแถวต่อภาพถ้ามีหลายโดรน

⚠️  ตรวจสอบ format ที่โจทย์ต้องการ:
   - ถ้าต้องการ 1 แถวต่อภาพ: ใช้ไฟล์เดิม (test_predictions_residual.csv)
   - ถ้าต้องการทุกโดรน: ใช้ test_predictions_all_drones_simple.csv
   - ถ้ามี format พิเศษ (เช่น multiple columns): ต้องปรับ format
""")

print("\n" + "="*80)
print("✅ Complete!")
print("="*80)
