"""
Test different strategies for selecting the correct drone in multi-drone images
1. Highest confidence (current)
2. Largest bbox (closest to camera)
3. Closest to image center (main target)
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from ultralytics import YOLO

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
print("🧪 Testing Multi-Drone Selection Strategies")
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
    
    return lat, lon, pred_altitude

print("\n🔍 Processing test images with different strategies...")
print("="*80)

strategies = {
    'highest_conf': [],
    'largest_bbox': [],
    'closest_to_center': []
}

for img_path in test_images:
    img_name = img_path.name
    
    # Load image to get dimensions
    import cv2
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]
    
    # Run YOLO
    results = yolo_model(img_path, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        # No detection - use default
        default_features = {
            'yolo_cx': 0.5, 'yolo_cy': 0.5, 'yolo_w': 0.1, 'yolo_h': 0.1,
            'yolo_conf': 0.0, 'yolo_area': 0.01, 'yolo_aspect_ratio': 1.0,
            'yolo_dist_from_center': 0.0, 'yolo_angle_from_center': 0.0
        }
        lat, lon, alt = predict_gps(default_features)
        
        for strategy in strategies.values():
            strategy.append({
                'image_name': img_name,
                'latitude': lat,
                'longitude': lon,
                'altitude': alt
            })
    else:
        # Extract features for all boxes
        all_features = []
        for box in boxes:
            features = extract_bbox_features(box, img_width, img_height)
            all_features.append(features)
        
        # Strategy 1: Highest confidence (current method)
        conf_values = [f['yolo_conf'] for f in all_features]
        best_conf_idx = np.argmax(conf_values)
        lat, lon, alt = predict_gps(all_features[best_conf_idx])
        strategies['highest_conf'].append({
            'image_name': img_name,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt
        })
        
        # Strategy 2: Largest bbox (closest to camera)
        areas = [f['yolo_area'] for f in all_features]
        largest_idx = np.argmax(areas)
        lat, lon, alt = predict_gps(all_features[largest_idx])
        strategies['largest_bbox'].append({
            'image_name': img_name,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt
        })
        
        # Strategy 3: Closest to image center
        dists = [f['yolo_dist_from_center'] for f in all_features]
        closest_idx = np.argmin(dists)
        lat, lon, alt = predict_gps(all_features[closest_idx])
        strategies['closest_to_center'].append({
            'image_name': img_name,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt
        })

print("✅ Processing complete")

# Save predictions for each strategy
print("\n💾 Saving predictions...")
output_dir = Path('multi_drone_strategies')
output_dir.mkdir(exist_ok=True)

for strategy_name, predictions in strategies.items():
    df = pd.DataFrame(predictions)
    df = df.sort_values('image_name')
    df.columns = ['ImageName', 'Latitude', 'Longitude', 'Altitude']
    
    output_file = output_dir / f'test_predictions_{strategy_name}.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Saved: {output_file}")

# Compare statistics
print("\n" + "="*80)
print("📊 Comparison of Strategies")
print("="*80)

for strategy_name, predictions in strategies.items():
    df = pd.DataFrame(predictions)
    
    print(f"\n{strategy_name.upper().replace('_', ' ')}:")
    print(f"  Latitude:  {df['latitude'].mean():.6f} ± {df['latitude'].std():.6f}")
    print(f"  Longitude: {df['longitude'].mean():.6f} ± {df['longitude'].std():.6f}")
    print(f"  Altitude:  {df['altitude'].mean():.2f} ± {df['altitude'].std():.2f} m")

# Calculate differences
print("\n" + "="*80)
print("📏 Differences Between Strategies")
print("="*80)

df_conf = pd.DataFrame(strategies['highest_conf'])
df_bbox = pd.DataFrame(strategies['largest_bbox'])
df_center = pd.DataFrame(strategies['closest_to_center'])

# How many predictions are different?
diff_conf_bbox = ((df_conf['latitude'] != df_bbox['latitude']) | 
                  (df_conf['longitude'] != df_bbox['longitude'])).sum()
diff_conf_center = ((df_conf['latitude'] != df_center['latitude']) | 
                    (df_conf['longitude'] != df_center['longitude'])).sum()
diff_bbox_center = ((df_bbox['latitude'] != df_center['latitude']) | 
                    (df_bbox['longitude'] != df_center['longitude'])).sum()

print(f"\nNumber of different predictions:")
print(f"  Highest Conf vs Largest Bbox:  {diff_conf_bbox}/{len(df_conf)} ({diff_conf_bbox/len(df_conf)*100:.1f}%)")
print(f"  Highest Conf vs Closest Center: {diff_conf_center}/{len(df_conf)} ({diff_conf_center/len(df_conf)*100:.1f}%)")
print(f"  Largest Bbox vs Closest Center: {diff_bbox_center}/{len(df_conf)} ({diff_bbox_center/len(df_conf)*100:.1f}%)")

print("\n" + "="*80)
print("💡 Recommendation")
print("="*80)

print("""
ไม่มี ground truth สำหรับ test set ดังนั้นไม่สามารถเปรียบเทียบ accuracy ได้

แนะนำให้เลือกตาม logic:

1. **Highest Confidence** (ปัจจุบัน) ✅
   - YOLO มั่นใจสูงสุด = มักเป็นโดรนที่ชัดที่สุด
   - ใช้อยู่แล้วใน: test_predictions_residual.csv

2. **Largest Bbox** (ใกล้กล้องที่สุด)
   - ใกล้ = ใหญ่ = ชัด = น่าจะเป็นเป้าหมาย?
   - ไฟล์: multi_drone_strategies/test_predictions_largest_bbox.csv

3. **Closest to Center** (กึ่งกลางภาพ)
   - Photographer เล็งกล้องไปที่เป้าหมาย
   - ไฟล์: multi_drone_strategies/test_predictions_closest_to_center.csv

หากไม่แน่ใจ ลอง submit ทั้ง 3 แล้วดูว่าอันไหนได้คะแนนดีที่สุด!
หรือใช้ Ensemble (เฉลี่ย) ของทั้ง 3 วิธี
""")

print("\n" + "="*80)
print("✅ Analysis Complete!")
print("="*80)
