"""
Compare 3 Advanced Methods to Reduce Mean Error:
1. Stacking Ensemble (Meta-learner combines multiple base models)
2. Feature Augmentation from Image Context (Add color, texture features)
3. Quantile Regression (Predict uncertainty with confidence intervals)

Compare against current Residual Learning baseline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, QuantileRegressor
import joblib
from pathlib import Path
import cv2
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🧪 Testing 3 Advanced Methods to Reduce Mean Error")
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

# Train/test split
X_base = df_detected[base_features].values
y_distance = df_detected['true_distance'].values
y_bearing = df_detected['true_bearing'].values
y_altitude = df_detected['true_altitude'].values

X_train, X_test, y_dist_train, y_dist_test = train_test_split(
    X_base, y_distance, test_size=0.2, random_state=42
)
_, _, y_bear_train, y_bear_test = train_test_split(
    X_base, y_bearing, test_size=0.2, random_state=42
)
_, _, y_alt_train, y_alt_test = train_test_split(
    X_base, y_altitude, test_size=0.2, random_state=42
)

print(f"✅ Train/Test split: {len(X_train)}/{len(X_test)}")

# ============================================================================
# BASELINE: Current Residual Learning
# ============================================================================
print("\n" + "="*80)
print("📊 BASELINE: Residual Learning (Current Method)")
print("="*80)

# Load existing models
baseline_dist = joblib.load('models_approximation/bbox_to_distance.pkl')
baseline_bear_sin = joblib.load('models_approximation/bbox_to_bearing_sin.pkl')
baseline_bear_cos = joblib.load('models_approximation/bbox_to_bearing_cos.pkl')
baseline_alt = joblib.load('models_approximation/bbox_to_altitude.pkl')

residual_dist = joblib.load('models_approximation/residual_distance.pkl')
residual_bear = joblib.load('models_approximation/residual_bearing.pkl')
residual_alt = joblib.load('models_approximation/residual_altitude.pkl')

# Predict on test set
base_dist_pred = baseline_dist.predict(X_test)
base_bear_sin = baseline_bear_sin.predict(X_test)
base_bear_cos = baseline_bear_cos.predict(X_test)
base_bear_pred = np.degrees(np.arctan2(base_bear_sin, base_bear_cos)) % 360
base_alt_pred = baseline_alt.predict(X_test)

# Apply residual corrections
residual_dist_pred = residual_dist.predict(X_test)
residual_bear_pred = residual_bear.predict(X_test)
residual_alt_pred = residual_alt.predict(X_test)

final_dist_baseline = base_dist_pred + residual_dist_pred
final_bear_baseline = (base_bear_pred + residual_bear_pred) % 360
final_alt_baseline = base_alt_pred + residual_alt_pred

# Calculate errors
mae_dist_baseline = mean_absolute_error(y_dist_test, final_dist_baseline)
mae_alt_baseline = mean_absolute_error(y_alt_test, final_alt_baseline)

# Bearing error (circular)
bear_diff = np.abs(y_bear_test - final_bear_baseline)
bear_diff = np.minimum(bear_diff, 360 - bear_diff)
mae_bear_baseline = np.mean(bear_diff)

print(f"\n✅ Baseline Results:")
print(f"  Distance MAE: {mae_dist_baseline:.3f} m")
print(f"  Bearing MAE:  {mae_bear_baseline:.3f} °")
print(f"  Altitude MAE: {mae_alt_baseline:.3f} m")
print(f"  Mean Total Error: {(mae_dist_baseline + mae_bear_baseline + mae_alt_baseline)/3:.3f}")

# ============================================================================
# METHOD 1: Stacking Ensemble
# ============================================================================
print("\n" + "="*80)
print("🔥 METHOD 1: Stacking Ensemble (Meta-learner)")
print("="*80)
print("Strategy: Train multiple base models (XGB, RF, GBM) and meta-learner combines them")

# Train base models for distance
print("\n📊 Training base models for distance...")
base1_dist = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
base2_dist = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
base3_dist = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

base1_dist.fit(X_train, y_dist_train)
base2_dist.fit(X_train, y_dist_train)
base3_dist.fit(X_train, y_dist_train)

# Generate meta-features
meta_train_dist = np.column_stack([
    base1_dist.predict(X_train),
    base2_dist.predict(X_train),
    base3_dist.predict(X_train)
])
meta_test_dist = np.column_stack([
    base1_dist.predict(X_test),
    base2_dist.predict(X_test),
    base3_dist.predict(X_test)
])

# Train meta-learner
meta_dist = Ridge(alpha=1.0)
meta_dist.fit(meta_train_dist, y_dist_train)
stack_dist_pred = meta_dist.predict(meta_test_dist)

# Altitude
print("📊 Training base models for altitude...")
base1_alt = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
base2_alt = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
base3_alt = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

base1_alt.fit(X_train, y_alt_train)
base2_alt.fit(X_train, y_alt_train)
base3_alt.fit(X_train, y_alt_train)

meta_train_alt = np.column_stack([
    base1_alt.predict(X_train),
    base2_alt.predict(X_train),
    base3_alt.predict(X_train)
])
meta_test_alt = np.column_stack([
    base1_alt.predict(X_test),
    base2_alt.predict(X_test),
    base3_alt.predict(X_test)
])

meta_alt = Ridge(alpha=1.0)
meta_alt.fit(meta_train_alt, y_alt_train)
stack_alt_pred = meta_alt.predict(meta_test_alt)

# Bearing (use baseline for simplicity)
stack_bear_pred = final_bear_baseline

# Calculate errors
mae_dist_stack = mean_absolute_error(y_dist_test, stack_dist_pred)
mae_alt_stack = mean_absolute_error(y_alt_test, stack_alt_pred)
mae_bear_stack = mae_bear_baseline  # Same as baseline

print(f"\n✅ Stacking Results:")
print(f"  Distance MAE: {mae_dist_stack:.3f} m (vs {mae_dist_baseline:.3f})")
print(f"  Bearing MAE:  {mae_bear_stack:.3f} ° (vs {mae_bear_baseline:.3f})")
print(f"  Altitude MAE: {mae_alt_stack:.3f} m (vs {mae_alt_baseline:.3f})")
print(f"  Mean Total Error: {(mae_dist_stack + mae_bear_stack + mae_alt_stack)/3:.3f}")
print(f"  Improvement: {mae_dist_baseline - mae_dist_stack:.3f}m (distance), {mae_alt_baseline - mae_alt_stack:.3f}m (altitude)")

# ============================================================================
# METHOD 2: Feature Augmentation from Image Context
# ============================================================================
print("\n" + "="*80)
print("🎨 METHOD 2: Feature Augmentation from Image Context")
print("="*80)
print("Strategy: Add color histogram, edge density, texture features from bbox region")

print("\n📸 Extracting image features from training data...")
yolo = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

def extract_image_features(img_path, bbox_norm):
    """Extract color and texture features from bbox region"""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros(15)  # Return zeros if image not found
    
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox_norm
    
    # Convert to pixel coords
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Extract bbox region
    bbox_region = img[y1:y2, x1:x2]
    
    if bbox_region.size == 0:
        return np.zeros(15)
    
    # Color features (mean RGB)
    mean_b = bbox_region[:,:,0].mean()
    mean_g = bbox_region[:,:,1].mean()
    mean_r = bbox_region[:,:,2].mean()
    
    # Color std
    std_b = bbox_region[:,:,0].std()
    std_g = bbox_region[:,:,1].std()
    std_r = bbox_region[:,:,2].std()
    
    # Grayscale
    gray = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / edges.size
    
    # Texture (std of gray)
    texture = gray.std()
    
    # Brightness
    brightness = gray.mean()
    
    # Contrast
    contrast = gray.max() - gray.min()
    
    # Histogram features (3 bins per channel)
    hist_b = cv2.calcHist([bbox_region], [0], None, [3], [0, 256]).flatten()
    hist_g = cv2.calcHist([bbox_region], [1], None, [3], [0, 256]).flatten()
    
    # Normalize histograms
    hist_b = hist_b / (hist_b.sum() + 1e-6)
    hist_g = hist_g / (hist_g.sum() + 1e-6)
    
    return np.array([
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        edge_density, texture, brightness, contrast,
        hist_b[0], hist_b[1], hist_g[0], hist_g[1], hist_g[2]
    ])

# Extract features for subset (for speed, use 300 samples)
print("  Extracting from 300 sample images...")
sample_indices = np.random.choice(len(df_detected), min(300, len(df_detected)), replace=False)
df_sample = df_detected.iloc[sample_indices].copy()

image_features = []
for idx, row in df_sample.iterrows():
    # Try multiple possible paths
    img_name = row['image_name']
    possible_paths = [
        Path('datasets/data/train/images') / img_name,
        Path('datasets/DATA_TRAIN') / img_name,
        Path('datasets/data/valid/images') / img_name,
    ]
    
    img_path = None
    for p in possible_paths:
        if p.exists():
            img_path = p
            break
    
    if img_path is None:
        # Use zeros if image not found
        img_feats = np.zeros(15)
    else:
        bbox_norm = [row['yolo_cx'], row['yolo_cy'], row['yolo_w'], row['yolo_h']]
        img_feats = extract_image_features(img_path, bbox_norm)
    
    image_features.append(img_feats)

image_features = np.array(image_features)

# Combine with base features
X_augmented = np.hstack([
    df_sample[base_features].values,
    image_features
])

y_dist_aug = df_sample['true_distance'].values
y_alt_aug = df_sample['true_altitude'].values

# Train/test split
X_aug_train, X_aug_test, y_dist_aug_train, y_dist_aug_test = train_test_split(
    X_augmented, y_dist_aug, test_size=0.2, random_state=42
)
_, _, y_alt_aug_train, y_alt_aug_test = train_test_split(
    X_augmented, y_alt_aug, test_size=0.2, random_state=42
)

print(f"  Features: {base_features.__len__()} base + 15 image = {X_augmented.shape[1]} total")

# Train models with augmented features
print("📊 Training models with augmented features...")
model_dist_aug = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
model_alt_aug = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)

model_dist_aug.fit(X_aug_train, y_dist_aug_train)
model_alt_aug.fit(X_aug_train, y_alt_aug_train)

# Predict
aug_dist_pred = model_dist_aug.predict(X_aug_test)
aug_alt_pred = model_alt_aug.predict(X_aug_test)

# Calculate errors
mae_dist_aug = mean_absolute_error(y_dist_aug_test, aug_dist_pred)
mae_alt_aug = mean_absolute_error(y_alt_aug_test, aug_alt_pred)

print(f"\n✅ Feature Augmentation Results:")
print(f"  Distance MAE: {mae_dist_aug:.3f} m")
print(f"  Altitude MAE: {mae_alt_aug:.3f} m")
print(f"  Note: Trained on 300 samples only (for speed)")

# ============================================================================
# METHOD 3: Quantile Regression (Uncertainty Estimation)
# ============================================================================
print("\n" + "="*80)
print("📊 METHOD 3: Quantile Regression (Predict Uncertainty)")
print("="*80)
print("Strategy: Predict median (50th percentile) for robust predictions")

# Train quantile regressors
print("\n📊 Training quantile regressors...")
quant_dist = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
quant_alt = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')

quant_dist.fit(X_train, y_dist_train)
quant_alt.fit(X_train, y_alt_train)

# Predict
quant_dist_pred = quant_dist.predict(X_test)
quant_alt_pred = quant_alt.predict(X_test)

# Calculate errors
mae_dist_quant = mean_absolute_error(y_dist_test, quant_dist_pred)
mae_alt_quant = mean_absolute_error(y_alt_test, quant_alt_pred)

print(f"\n✅ Quantile Regression Results:")
print(f"  Distance MAE: {mae_dist_quant:.3f} m (vs {mae_dist_baseline:.3f})")
print(f"  Altitude MAE: {mae_alt_quant:.3f} m (vs {mae_alt_baseline:.3f})")

# Also train 25th and 75th percentiles for uncertainty
quant_dist_25 = QuantileRegressor(quantile=0.25, alpha=0.01, solver='highs')
quant_dist_75 = QuantileRegressor(quantile=0.75, alpha=0.01, solver='highs')

quant_dist_25.fit(X_train, y_dist_train)
quant_dist_75.fit(X_train, y_dist_train)

pred_25 = quant_dist_25.predict(X_test)
pred_75 = quant_dist_75.predict(X_test)

uncertainty = pred_75 - pred_25
print(f"  Uncertainty (IQR): {uncertainty.mean():.3f} m ± {uncertainty.std():.3f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("🏆 FINAL COMPARISON (Distance + Altitude)")
print("="*80)

results = {
    'Method': [
        'Baseline (Residual Learning)',
        'Method 1 (Stacking Ensemble)',
        'Method 2 (Feature Augmentation)',
        'Method 3 (Quantile Regression)'
    ],
    'Distance_MAE': [
        mae_dist_baseline,
        mae_dist_stack,
        mae_dist_aug,
        mae_dist_quant
    ],
    'Altitude_MAE': [
        mae_alt_baseline,
        mae_alt_stack,
        mae_alt_aug,
        mae_alt_quant
    ],
    'Mean_Error': [
        (mae_dist_baseline + mae_alt_baseline) / 2,
        (mae_dist_stack + mae_alt_stack) / 2,
        (mae_dist_aug + mae_alt_aug) / 2,
        (mae_dist_quant + mae_alt_quant) / 2
    ]
}

df_results = pd.DataFrame(results)
df_results['Improvement_vs_Baseline'] = df_results['Mean_Error'].iloc[0] - df_results['Mean_Error']

print("\n" + df_results.to_string(index=False))

# Save results
df_results.to_csv('method_comparison_results.csv', index=False)
print(f"\n💾 Saved: method_comparison_results.csv")

# Find best method
best_idx = df_results['Mean_Error'].idxmin()
best_method = df_results.iloc[best_idx]

print("\n" + "="*80)
print("🥇 WINNER:")
print("="*80)
print(f"  {best_method['Method']}")
print(f"  Distance MAE: {best_method['Distance_MAE']:.3f} m")
print(f"  Altitude MAE: {best_method['Altitude_MAE']:.3f} m")
print(f"  Mean Error: {best_method['Mean_Error']:.3f}")
print(f"  Improvement: {best_method['Improvement_vs_Baseline']:.3f}")

if best_method['Improvement_vs_Baseline'] > 0:
    print(f"  ✅ {best_method['Improvement_vs_Baseline']/df_results['Mean_Error'].iloc[0]*100:.1f}% better than baseline!")
else:
    print(f"  ❌ Baseline is still the best")

print("\n" + "="*80)
print("✅ Experiment Complete!")
print("="*80)
