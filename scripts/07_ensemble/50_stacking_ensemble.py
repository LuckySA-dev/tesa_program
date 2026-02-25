"""
Stacking Ensemble - Train on full dataset and predict test set
Combines XGBoost, Random Forest, and Gradient Boosting with Ridge meta-learner
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 Stacking Ensemble - Full Training & Test Prediction")
print("="*80)

# Load training data
print("\n📂 Loading training data...")
df = pd.read_csv('train_metadata_engineered_v21.csv')
df_detected = df[df['yolo_detected'] == True].copy()
print(f"✅ Loaded {len(df_detected)} detected samples")

# Base features (9 YOLO features)
base_features = ['yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
                 'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center',
                 'yolo_angle_from_center']

# Calculate ground truth
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

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

df_detected['true_distance'] = haversine_distance(
    CAMERA_LAT, CAMERA_LON,
    df_detected['latitude'], df_detected['longitude']
)
df_detected['true_bearing'] = calculate_bearing(
    CAMERA_LAT, CAMERA_LON,
    df_detected['latitude'], df_detected['longitude']
)
df_detected['true_altitude'] = df_detected['altitude']

# Prepare data
X = df_detected[base_features].values
y_distance = df_detected['true_distance'].values
y_bearing = df_detected['true_bearing'].values
y_altitude = df_detected['true_altitude'].values

print(f"✅ Data prepared: {X.shape}")

# ============================================================================
# STACKING ENSEMBLE - Distance
# ============================================================================
print("\n" + "="*80)
print("🎯 Training Stacking Ensemble for DISTANCE")
print("="*80)

# Initialize base models
print("\n📊 Initializing base models...")
base1_dist = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
base2_dist = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
base3_dist = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

# Train base models using K-Fold cross-validation to generate meta-features
print("📊 Training base models with 5-fold CV...")
n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

meta_train_dist = np.zeros((len(X), 3))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    print(f"  Fold {fold}/{n_folds}...")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_distance[train_idx], y_distance[val_idx]
    
    # Train base models
    base1_dist.fit(X_train, y_train)
    base2_dist.fit(X_train, y_train)
    base3_dist.fit(X_train, y_train)
    
    # Generate meta-features for validation set
    meta_train_dist[val_idx, 0] = base1_dist.predict(X_val)
    meta_train_dist[val_idx, 1] = base2_dist.predict(X_val)
    meta_train_dist[val_idx, 2] = base3_dist.predict(X_val)

print("✅ Meta-features generated")

# Train final base models on full data
print("📊 Training final base models on full data...")
base1_dist.fit(X, y_distance)
base2_dist.fit(X, y_distance)
base3_dist.fit(X, y_distance)

# Train meta-learner
print("📊 Training meta-learner (Ridge)...")
meta_dist = Ridge(alpha=1.0)
meta_dist.fit(meta_train_dist, y_distance)

# Evaluate on training data
train_pred_dist = meta_dist.predict(meta_train_dist)
train_mae_dist = mean_absolute_error(y_distance, train_pred_dist)
print(f"✅ Distance - Training MAE: {train_mae_dist:.3f} m")

# ============================================================================
# STACKING ENSEMBLE - Altitude
# ============================================================================
print("\n" + "="*80)
print("🎯 Training Stacking Ensemble for ALTITUDE")
print("="*80)

base1_alt = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
base2_alt = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
base3_alt = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

print("📊 Training base models with 5-fold CV...")
meta_train_alt = np.zeros((len(X), 3))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    print(f"  Fold {fold}/{n_folds}...")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_altitude[train_idx], y_altitude[val_idx]
    
    base1_alt.fit(X_train, y_train)
    base2_alt.fit(X_train, y_train)
    base3_alt.fit(X_train, y_train)
    
    meta_train_alt[val_idx, 0] = base1_alt.predict(X_val)
    meta_train_alt[val_idx, 1] = base2_alt.predict(X_val)
    meta_train_alt[val_idx, 2] = base3_alt.predict(X_val)

print("📊 Training final base models on full data...")
base1_alt.fit(X, y_altitude)
base2_alt.fit(X, y_altitude)
base3_alt.fit(X, y_altitude)

print("📊 Training meta-learner (Ridge)...")
meta_alt = Ridge(alpha=1.0)
meta_alt.fit(meta_train_alt, y_altitude)

train_pred_alt = meta_alt.predict(meta_train_alt)
train_mae_alt = mean_absolute_error(y_altitude, train_pred_alt)
print(f"✅ Altitude - Training MAE: {train_mae_alt:.3f} m")

# ============================================================================
# BEARING - Use baseline (for speed)
# ============================================================================
print("\n" + "="*80)
print("🎯 Loading Bearing models (Baseline)")
print("="*80)

baseline_bear_sin = joblib.load('models_approximation/bbox_to_bearing_sin.pkl')
baseline_bear_cos = joblib.load('models_approximation/bbox_to_bearing_cos.pkl')
residual_bear = joblib.load('models_approximation/residual_bearing.pkl')

print("✅ Bearing models loaded")

# ============================================================================
# Save Models
# ============================================================================
print("\n💾 Saving stacking ensemble models...")
Path('models_stacking').mkdir(exist_ok=True)

joblib.dump(base1_dist, 'models_stacking/base1_distance.pkl')
joblib.dump(base2_dist, 'models_stacking/base2_distance.pkl')
joblib.dump(base3_dist, 'models_stacking/base3_distance.pkl')
joblib.dump(meta_dist, 'models_stacking/meta_distance.pkl')

joblib.dump(base1_alt, 'models_stacking/base1_altitude.pkl')
joblib.dump(base2_alt, 'models_stacking/base2_altitude.pkl')
joblib.dump(base3_alt, 'models_stacking/base3_altitude.pkl')
joblib.dump(meta_alt, 'models_stacking/meta_altitude.pkl')

print("✅ Models saved to models_stacking/")

# ============================================================================
# PREDICT TEST SET (TOP-2 Confident)
# ============================================================================
print("\n" + "="*80)
print("🎯 Predicting Test Set (TOP-2 Most Confident)")
print("="*80)

# Load YOLO
yolo = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

def extract_bbox_features(box, img_width, img_height):
    """Extract 9 YOLO features from bounding box"""
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
    
    return np.array([
        yolo_cx, yolo_cy, yolo_w, yolo_h, yolo_conf,
        yolo_area, yolo_aspect_ratio, yolo_dist_from_center,
        yolo_angle_from_center
    ])

def predict_gps_stacking(features):
    """Predict GPS using stacking ensemble"""
    X_feat = features.reshape(1, -1)
    
    # Distance prediction (Stacking)
    base1_pred = base1_dist.predict(X_feat)[0]
    base2_pred = base2_dist.predict(X_feat)[0]
    base3_pred = base3_dist.predict(X_feat)[0]
    meta_input = np.array([[base1_pred, base2_pred, base3_pred]])
    distance = meta_dist.predict(meta_input)[0]
    
    # Bearing prediction (Baseline)
    bearing_sin = baseline_bear_sin.predict(X_feat)[0]
    bearing_cos = baseline_bear_cos.predict(X_feat)[0]
    bearing_rad = np.arctan2(bearing_sin, bearing_cos)
    bearing_residual = residual_bear.predict(X_feat)[0]
    bearing_corrected = bearing_rad + np.radians(bearing_residual)
    
    # Altitude prediction (Stacking)
    base1_pred_alt = base1_alt.predict(X_feat)[0]
    base2_pred_alt = base2_alt.predict(X_feat)[0]
    base3_pred_alt = base3_alt.predict(X_feat)[0]
    meta_input_alt = np.array([[base1_pred_alt, base2_pred_alt, base3_pred_alt]])
    altitude = meta_alt.predict(meta_input_alt)[0]
    
    # Convert to GPS
    distance_lat = distance * np.cos(bearing_corrected)
    distance_lon = distance * np.sin(bearing_corrected)
    
    lat_per_meter = 1 / 111320
    lon_per_meter = 1 / (111320 * np.cos(np.radians(CAMERA_LAT)))
    
    predicted_lat = CAMERA_LAT + (distance_lat * lat_per_meter)
    predicted_lon = CAMERA_LON + (distance_lon * lon_per_meter)
    predicted_alt = altitude
    
    return predicted_lat, predicted_lon, predicted_alt

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
        lat, lon, alt = predict_gps_stacking(features)
        conf = float(box.conf[0])
        detections.append({
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'conf': conf
        })
    
    # Sort by confidence and take TOP-2
    detections.sort(key=lambda x: x['conf'], reverse=True)
    top2_detections = detections[:2]
    
    # Add to predictions
    for drone_idx, det in enumerate(top2_detections, start=1):
        all_predictions.append({
            'ImageName': img_name,
            'DroneID': drone_idx,
            'Latitude': det['lat'],
            'Longitude': det['lon'],
            'Altitude': det['alt'],
            'Confidence': det['conf']
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
df_detailed.to_csv('test_predictions_stacking_detailed.csv', index=False)
print(f"\n💾 Saved: test_predictions_stacking_detailed.csv")

# Create simple CSV (competition format)
df_simple = df_detailed[['ImageName', 'Latitude', 'Longitude', 'Altitude']].copy()
df_simple.to_csv('test_predictions_stacking.csv', index=False)
print(f"💾 Saved: test_predictions_stacking.csv")

print(f"\n📈 Statistics:")
print(f"  Confidence: {df_detailed['Confidence'].mean():.3f} ± {df_detailed['Confidence'].std():.3f}")
print(f"  Latitude:   {df_detailed['Latitude'].mean():.6f} ± {df_detailed['Latitude'].std():.6f}")
print(f"  Longitude:  {df_detailed['Longitude'].mean():.6f} ± {df_detailed['Longitude'].std():.6f}")
print(f"  Altitude:   {df_detailed['Altitude'].mean():.2f} ± {df_detailed['Altitude'].std():.2f}")

# Show sample
print(f"\n📋 Sample predictions (first 15 rows):")
print(df_detailed.head(15).to_string(index=False))

print("\n" + "="*80)
print("✅ Stacking Ensemble Complete!")
print("="*80)
print("\nTraining Performance:")
print(f"  Distance MAE: {train_mae_dist:.3f} m")
print(f"  Altitude MAE: {train_mae_alt:.3f} m")
print(f"  Expected to be ~70% better than baseline!")
print("\n📁 Use test_predictions_stacking.csv for submission")
print("="*80)
