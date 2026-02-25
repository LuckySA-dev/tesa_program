"""
ตรวจสอบ Training Data ว่าใช้เป็น GCP ได้หรือไม่
และเลือก GCP samples ที่เหมาะสม
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import json

CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360

def get_bearing_category(bearing):
    """แบ่งตามทิศ 8 ทิศ"""
    if bearing < 22.5 or bearing >= 337.5:
        return "N"
    elif 22.5 <= bearing < 67.5:
        return "NE"
    elif 67.5 <= bearing < 112.5:
        return "E"
    elif 112.5 <= bearing < 157.5:
        return "SE"
    elif 157.5 <= bearing < 202.5:
        return "S"
    elif 202.5 <= bearing < 247.5:
        return "SW"
    elif 247.5 <= bearing < 292.5:
        return "W"
    else:
        return "NW"

def get_distance_category(distance):
    """แบ่งตามระยะ"""
    if distance < 50:
        return "Very Close"
    elif distance < 100:
        return "Close"
    elif distance < 150:
        return "Medium"
    else:
        return "Far"

def get_altitude_category(altitude):
    """แบ่งตามความสูง"""
    if altitude < 55:
        return "Low"
    elif altitude < 70:
        return "Medium"
    else:
        return "High"

print("="*70)
print("📍 GCP Selection for Camera Geometry Calibration")
print("="*70)

# Load training data
print("\n📂 Loading training data...")
df = pd.read_csv('train_metadata_engineered.csv', encoding='utf-8')
print(f"✅ Loaded {len(df)} training samples")

# Calculate bearing and distance for all samples
df['distance'] = df.apply(
    lambda row: haversine(CAMERA_LAT, CAMERA_LON, 
                         row['latitude_deg'], row['longitude_deg']),
    axis=1
)

df['bearing'] = df.apply(
    lambda row: calculate_bearing(CAMERA_LAT, CAMERA_LON,
                                 row['latitude_deg'], row['longitude_deg']),
    axis=1
)

# Categorize
df['bearing_cat'] = df['bearing'].apply(get_bearing_category)
df['distance_cat'] = df['distance'].apply(get_distance_category)
df['altitude_cat'] = df['altitude_m'].apply(get_altitude_category)

# Overall statistics
print("\n" + "="*70)
print("📊 Training Data Distribution")
print("="*70)

print(f"\nBearing range: {df['bearing'].min():.1f}° - {df['bearing'].max():.1f}°")
print(f"Distance range: {df['distance'].min():.1f} - {df['distance'].max():.1f} m")
print(f"Altitude range: {df['altitude_m'].min():.1f} - {df['altitude_m'].max():.1f} m")

print("\n📐 Bearing Distribution (8 directions):")
bearing_counts = df['bearing_cat'].value_counts().sort_index()
for cat in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    count = bearing_counts.get(cat, 0)
    pct = (count / len(df)) * 100
    bar = "█" * int(count / 5)
    print(f"  {cat:3s}: {count:3d} ({pct:5.1f}%) {bar}")

print("\n📏 Distance Distribution:")
dist_counts = df['distance_cat'].value_counts().sort_index()
for cat in ['Very Close', 'Close', 'Medium', 'Far']:
    count = dist_counts.get(cat, 0)
    pct = (count / len(df)) * 100
    bar = "█" * int(count / 5)
    print(f"  {cat:12s}: {count:3d} ({pct:5.1f}%) {bar}")

print("\n📊 Altitude Distribution:")
alt_counts = df['altitude_cat'].value_counts().sort_index()
for cat in ['Low', 'Medium', 'High']:
    count = alt_counts.get(cat, 0)
    pct = (count / len(df)) * 100
    bar = "█" * int(count / 5)
    print(f"  {cat:8s}: {count:3d} ({pct:5.1f}%) {bar}")

# Check YOLO detection quality
print("\n" + "="*70)
print("🎯 YOLO Detection Quality")
print("="*70)

detected = df[df['yolo_detected'] == True]
print(f"\nTotal detected: {len(detected)}/{len(df)} ({len(detected)/len(df)*100:.1f}%)")
print(f"Average confidence: {detected['yolo_conf'].mean():.3f}")
print(f"Confidence range: {detected['yolo_conf'].min():.3f} - {detected['yolo_conf'].max():.3f}")

# High quality samples (high confidence)
high_quality = df[(df['yolo_detected'] == True) & (df['yolo_conf'] > 0.5)]
print(f"\nHigh quality (conf > 0.5): {len(high_quality)} samples")

# Select GCP samples
print("\n" + "="*70)
print("🎯 GCP Selection Strategy")
print("="*70)

# Strategy 1: Stratified selection (cover all combinations)
gcp_samples = []

# For each bearing category
for bearing_cat in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    # For each distance category
    for dist_cat in ['Very Close', 'Close', 'Medium', 'Far']:
        # Get samples in this category
        candidates = high_quality[
            (high_quality['bearing_cat'] == bearing_cat) &
            (high_quality['distance_cat'] == dist_cat)
        ]
        
        if len(candidates) > 0:
            # Select 1-2 best samples (highest confidence)
            n_select = min(2, len(candidates))
            selected = candidates.nlargest(n_select, 'yolo_conf')
            gcp_samples.extend(selected.index.tolist())

print(f"\n✅ Selected {len(gcp_samples)} GCP samples (stratified)")

# Check coverage
df_gcp = df.loc[gcp_samples]

print("\n📐 GCP Coverage by Bearing:")
gcp_bearing_counts = df_gcp['bearing_cat'].value_counts().sort_index()
for cat in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    count = gcp_bearing_counts.get(cat, 0)
    total = bearing_counts.get(cat, 0)
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {cat:3s}: {count:2d} GCPs (out of {total:3d} total)")

print("\n📏 GCP Coverage by Distance:")
gcp_dist_counts = df_gcp['distance_cat'].value_counts().sort_index()
for cat in ['Very Close', 'Close', 'Medium', 'Far']:
    count = gcp_dist_counts.get(cat, 0)
    total = dist_counts.get(cat, 0)
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {cat:12s}: {count:2d} GCPs (out of {total:3d} total)")

print("\n📊 GCP Coverage by Altitude:")
gcp_alt_counts = df_gcp['altitude_cat'].value_counts().sort_index()
for cat in ['Low', 'Medium', 'High']:
    count = gcp_alt_counts.get(cat, 0)
    total = alt_counts.get(cat, 0)
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {cat:8s}: {count:2d} GCPs (out of {total:3d} total)")

# Additional GCPs for test region (bearing ~237°, distance ~131m)
print("\n" + "="*70)
print("🎯 Special GCPs for Test Region")
print("="*70)

test_region = high_quality[
    (high_quality['bearing'] >= 230) & (high_quality['bearing'] <= 245) &
    (high_quality['distance'] >= 120) & (high_quality['distance'] <= 140)
]

print(f"\nSamples in test region (bearing 230-245°, distance 120-140m):")
print(f"  Found: {len(test_region)} samples")

if len(test_region) > 0:
    # Add top 5 to GCP list
    n_add = min(5, len(test_region))
    test_gcps = test_region.nlargest(n_add, 'yolo_conf')
    additional_gcps = [idx for idx in test_gcps.index if idx not in gcp_samples]
    gcp_samples.extend(additional_gcps)
    print(f"  Added: {len(additional_gcps)} additional GCPs for test region")
else:
    print(f"  ⚠️  No samples in exact test region!")
    # Find closest
    high_quality_copy = high_quality.copy()
    high_quality_copy['dist_to_test'] = abs(high_quality_copy['bearing'] - 237) + abs(high_quality_copy['distance'] - 131) / 10
    closest = high_quality_copy.nsmallest(5, 'dist_to_test')
    additional_gcps = [idx for idx in closest.index if idx not in gcp_samples]
    gcp_samples.extend(additional_gcps)
    print(f"  Added: {len(additional_gcps)} closest GCPs")

# Final GCP list
df_gcp_final = df.loc[gcp_samples]

print("\n" + "="*70)
print("📋 Final GCP Summary")
print("="*70)

print(f"\nTotal GCPs: {len(gcp_samples)}")
print(f"Average confidence: {df_gcp_final['yolo_conf'].mean():.3f}")
print(f"\nBearing coverage: {df_gcp_final['bearing'].min():.1f}° - {df_gcp_final['bearing'].max():.1f}°")
print(f"Distance coverage: {df_gcp_final['distance'].min():.1f} - {df_gcp_final['distance'].max():.1f} m")
print(f"Altitude coverage: {df_gcp_final['altitude_m'].min():.1f} - {df_gcp_final['altitude_m'].max():.1f} m")

# Save GCP list
gcp_info = {
    'num_gcps': len(gcp_samples),
    'gcp_indices': gcp_samples,
    'statistics': {
        'bearing_range': [float(df_gcp_final['bearing'].min()), float(df_gcp_final['bearing'].max())],
        'distance_range': [float(df_gcp_final['distance'].min()), float(df_gcp_final['distance'].max())],
        'altitude_range': [float(df_gcp_final['altitude_m'].min()), float(df_gcp_final['altitude_m'].max())],
        'avg_confidence': float(df_gcp_final['yolo_conf'].mean())
    }
}

with open('gcp_samples.json', 'w') as f:
    json.dump(gcp_info, f, indent=2)

# Save GCP data to CSV
df_gcp_final.to_csv('gcp_samples.csv', index=False, encoding='utf-8')

print(f"\n✅ Saved GCP list to: gcp_samples.json")
print(f"✅ Saved GCP data to: gcp_samples.csv")

# Recommendations
print("\n" + "="*70)
print("💡 Recommendations")
print("="*70)

print(f"\n📍 GCP Quality Assessment:")
if len(gcp_samples) >= 25:
    print(f"  ✅ Excellent: {len(gcp_samples)} GCPs (recommended: 25-30)")
elif len(gcp_samples) >= 15:
    print(f"  ✅ Good: {len(gcp_samples)} GCPs (minimum: 15-20)")
else:
    print(f"  ⚠️  Low: {len(gcp_samples)} GCPs (need at least 15-20)")

print(f"\n🎯 Coverage Assessment:")
missing_bearings = []
for cat in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    if gcp_bearing_counts.get(cat, 0) == 0:
        missing_bearings.append(cat)

if len(missing_bearings) == 0:
    print(f"  ✅ Full bearing coverage (all 8 directions)")
else:
    print(f"  ⚠️  Missing bearings: {', '.join(missing_bearings)}")

# Check test region coverage
test_coverage = len(test_region) > 0 or len(additional_gcps) > 0
if test_coverage:
    print(f"  ✅ Test region covered (bearing ~237°, distance ~131m)")
else:
    print(f"  ⚠️  Test region not well covered")

print(f"\n🚀 Next Steps:")
print(f"  1. Review gcp_samples.csv")
print(f"  2. Verify GPS coordinates are accurate")
print(f"  3. Run camera calibration with these GCPs")
print(f"  4. Train geometry-based model")

print("\n✅ Complete!")
