"""
Predict GPS for TOP-2 most confident drones per image
Strategy: Filter false positives by keeping only 2 highest confidence detections
"""
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

print("="*80)
print("🎯 Predicting TOP-2 Most Confident Drones (Filter False Positives)")
print("="*80)

# Load models
print("\n📦 Loading models...")
yolo = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

with open('models_approximation/bbox_to_distance.pkl', 'rb') as f:
    model_distance = pickle.load(f)
with open('models_approximation/bbox_to_bearing_sin.pkl', 'rb') as f:
    model_bearing_sin = pickle.load(f)
with open('models_approximation/bbox_to_bearing_cos.pkl', 'rb') as f:
    model_bearing_cos = pickle.load(f)
with open('models_approximation/bbox_to_altitude.pkl', 'rb') as f:
    model_altitude = pickle.load(f)

# Residual models
with open('models_approximation/residual_distance.pkl', 'rb') as f:
    residual_distance = pickle.load(f)
with open('models_approximation/residual_bearing.pkl', 'rb') as f:
    residual_bearing = pickle.load(f)
with open('models_approximation/residual_altitude.pkl', 'rb') as f:
    residual_altitude = pickle.load(f)

print("✅ All models loaded")

# Camera position (from training data)
camera_lat = 14.304750
camera_lon = 101.172718
camera_alt = 0.0

def extract_bbox_features(box, img_width, img_height):
    """Extract features from bounding box (9 features only)"""
    xyxy = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = xyxy
    
    # Center position (normalized)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    yolo_cx = center_x / img_width
    yolo_cy = center_y / img_height
    
    # Width and height (normalized)
    width = x2 - x1
    height = y2 - y1
    yolo_w = width / img_width
    yolo_h = height / img_height
    
    # Confidence
    yolo_conf = float(box.conf[0])
    
    # Area
    yolo_area = yolo_w * yolo_h
    
    # Aspect ratio
    yolo_aspect_ratio = yolo_w / yolo_h if yolo_h > 0 else 0
    
    # Distance from image center
    img_center_x = 0.5
    img_center_y = 0.5
    yolo_dist_from_center = np.sqrt((yolo_cx - img_center_x)**2 + (yolo_cy - img_center_y)**2)
    
    # Angle from image center
    yolo_angle_from_center = np.degrees(np.arctan2(yolo_cy - img_center_y, yolo_cx - img_center_x))
    
    return {
        'yolo_cx': yolo_cx,
        'yolo_cy': yolo_cy,
        'yolo_w': yolo_w,
        'yolo_h': yolo_h,
        'yolo_conf': yolo_conf,
        'yolo_area': yolo_area,
        'yolo_aspect_ratio': yolo_aspect_ratio,
        'yolo_dist_from_center': yolo_dist_from_center,
        'yolo_angle_from_center': yolo_angle_from_center,
        'bbox_area_pixels': width * height  # For sorting/display
    }

def predict_gps_with_residual(features_dict):
    """Predict GPS using base + residual models"""
    # Prepare features in correct order (9 features)
    feature_names = [
        'yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
        'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center',
        'yolo_angle_from_center'
    ]
    X = np.array([[features_dict[f] for f in feature_names]])
    
    # Base predictions
    distance = model_distance.predict(X)[0]
    bearing_sin = model_bearing_sin.predict(X)[0]
    bearing_cos = model_bearing_cos.predict(X)[0]
    altitude = model_altitude.predict(X)[0]
    
    bearing_rad = np.arctan2(bearing_sin, bearing_cos)
    
    # Residual corrections
    distance_residual = residual_distance.predict(X)[0]
    bearing_residual = residual_bearing.predict(X)[0]
    altitude_residual = residual_altitude.predict(X)[0]
    
    # Apply corrections
    distance_corrected = distance + distance_residual
    bearing_corrected = bearing_rad + np.radians(bearing_residual)
    altitude_corrected = altitude + altitude_residual
    
    # Convert to GPS
    bearing_deg = np.degrees(bearing_corrected) % 360
    
    distance_lat = distance_corrected * np.cos(bearing_corrected)
    distance_lon = distance_corrected * np.sin(bearing_corrected)
    
    lat_per_meter = 1 / 111320
    lon_per_meter = 1 / (111320 * np.cos(np.radians(camera_lat)))
    
    predicted_lat = camera_lat + (distance_lat * lat_per_meter)
    predicted_lon = camera_lon + (distance_lon * lon_per_meter)
    predicted_alt = camera_alt + altitude_corrected
    
    return predicted_lat, predicted_lon, predicted_alt, features_dict['yolo_conf'], features_dict['bbox_area_pixels']

# Process test images
test_dir = Path('datasets/DATA_TEST')
test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))

print(f"\n📸 Processing {len(test_images)} test images...")

all_predictions = []
detection_stats = {0: 0, 1: 0, 2: 0, '3+': 0}

for img_path in test_images:
    img_name = img_path.name
    
    # YOLO detection
    results = yolo(img_path, verbose=False, conf=0.25, iou=0.45, max_det=10)
    boxes = results[0].boxes
    
    img_height, img_width = results[0].orig_shape
    
    num_detections = len(boxes)
    
    # Count detections
    if num_detections == 0:
        detection_stats[0] += 1
    elif num_detections == 1:
        detection_stats[1] += 1
    elif num_detections == 2:
        detection_stats[2] += 1
    else:
        detection_stats['3+'] += 1
    
    # Get all detections with confidence
    detections = []
    for box in boxes:
        features = extract_bbox_features(box, img_width, img_height)
        lat, lon, alt, conf, area = predict_gps_with_residual(features)
        detections.append({
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'conf': conf,
            'area': area
        })
    
    # Sort by confidence and take TOP-2
    detections.sort(key=lambda x: x['conf'], reverse=True)
    top2_detections = detections[:2]  # Keep maximum 2 drones
    
    # Add to predictions
    for drone_idx, det in enumerate(top2_detections, start=1):
        all_predictions.append({
            'ImageName': img_name,
            'DroneID': drone_idx,
            'Latitude': det['lat'],
            'Longitude': det['lon'],
            'Altitude': det['alt'],
            'Confidence': det['conf'],
            'BboxArea': det['area']
        })

print(f"\n✅ Processing complete!")
print(f"\n📊 Detection Statistics:")
print(f"  0 drones: {detection_stats[0]} images")
print(f"  1 drone:  {detection_stats[1]} images")
print(f"  2 drones: {detection_stats[2]} images")
print(f"  3+ drones: {detection_stats['3+']} images (filtered to top-2)")

print(f"\n📝 Total predictions: {len(all_predictions)} (for {len(test_images)} images)")
print(f"   Average: {len(all_predictions)/len(test_images):.2f} drones per image")

# Create detailed CSV
df_detailed = pd.DataFrame(all_predictions)
df_detailed.to_csv('test_predictions_top2_confident_detailed.csv', index=False)
print(f"\n💾 Saved: test_predictions_top2_confident_detailed.csv")

# Create simple CSV (competition format)
df_simple = df_detailed[['ImageName', 'Latitude', 'Longitude', 'Altitude']].copy()
df_simple.to_csv('test_predictions_top2_confident.csv', index=False)
print(f"💾 Saved: test_predictions_top2_confident.csv")

print(f"\n📈 Statistics:")
print(f"  Confidence: {df_detailed['Confidence'].mean():.3f} ± {df_detailed['Confidence'].std():.3f}")
print(f"  Latitude:   {df_detailed['Latitude'].mean():.6f} ± {df_detailed['Latitude'].std():.6f}")
print(f"  Longitude:  {df_detailed['Longitude'].mean():.6f} ± {df_detailed['Longitude'].std():.6f}")
print(f"  Altitude:   {df_detailed['Altitude'].mean():.2f} ± {df_detailed['Altitude'].std():.2f}")

print(f"\n🎯 Strategy: TOP-2 Most Confident")
print(f"  ✅ Filters false positives (low confidence detections)")
print(f"  ✅ Ensures <= 2 drones per image")
print(f"  ✅ Keeps best detections only")

# Show sample
print(f"\n📋 Sample predictions (first 20 rows):")
print(df_detailed.head(20).to_string(index=False))

print("\n" + "="*80)
print("✅ Done! Use test_predictions_top2_confident.csv for submission")
print("="*80)
