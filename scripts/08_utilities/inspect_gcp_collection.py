"""
วิเคราะห์ว่าต้องไปถ่ายภาพโดรนที่ไหนเพิ่ม เพื่อให้ได้ GCP ครบ
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import matplotlib.pyplot as plt

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

def bearing_distance_to_latlon(bearing_deg, distance_m, camera_lat, camera_lon):
    """แปลง bearing และ distance เป็น lat/lon"""
    R = 6371000
    lat1 = radians(camera_lat)
    lon1 = radians(camera_lon)
    bearing_rad = radians(bearing_deg)
    
    lat2 = asin(sin(lat1) * cos(distance_m/R) + 
                cos(lat1) * sin(distance_m/R) * cos(bearing_rad))
    lon2 = lon1 + atan2(sin(bearing_rad) * sin(distance_m/R) * cos(lat1),
                        cos(distance_m/R) - sin(lat1) * sin(lat2))
    
    return degrees(lat2), degrees(lon2)

print("="*70)
print("📍 วิธีหา GCP เพิ่มเติมสำหรับ Camera Geometry Calibration")
print("="*70)

# Load existing data
df = pd.read_csv('train_metadata_engineered.csv', encoding='utf-8')
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

print(f"\n📊 Current Data Coverage:")
print(f"  Samples: {len(df)}")
print(f"  Bearing range: {df['bearing'].min():.1f}° - {df['bearing'].max():.1f}°")
print(f"  Distance range: {df['distance'].min():.1f} - {df['distance'].max():.1f} m")

# Define target grid
print("\n" + "="*70)
print("🎯 Target GCP Grid")
print("="*70)

# 8 directions × 4 distances = 32 target points
bearings = [0, 45, 90, 135, 180, 225, 270, 315]  # N, NE, E, SE, S, SW, W, NW
bearing_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
distances = [30, 70, 110, 150]  # Very Close, Close, Medium, Far
altitudes = [50, 65, 80]  # Low, Medium, High

print(f"\n📐 Recommended GCP Grid:")
print(f"  Bearings: {len(bearings)} directions (every 45°)")
print(f"  Distances: {len(distances)} ranges")
print(f"  Altitudes: {len(altitudes)} levels")
print(f"  Total targets: {len(bearings) * len(distances)} positions")

# Check which targets are covered
print("\n" + "="*70)
print("📋 Coverage Analysis")
print("="*70)

target_points = []
for bearing in bearings:
    for distance in distances:
        # Find nearest sample in training data
        bearing_diff = np.minimum(
            abs(df['bearing'] - bearing),
            360 - abs(df['bearing'] - bearing)
        )
        distance_diff = abs(df['distance'] - distance)
        
        # Combined distance (normalize to same scale)
        combined_dist = bearing_diff / 10 + distance_diff
        
        if len(df) > 0:
            nearest_idx = combined_dist.idxmin()
            nearest_bearing_diff = bearing_diff[nearest_idx]
            nearest_distance_diff = distance_diff[nearest_idx]
            
            # Consider "covered" if within 20° and 30m
            covered = (nearest_bearing_diff < 20) and (nearest_distance_diff < 30)
        else:
            covered = False
            nearest_bearing_diff = 999
            nearest_distance_diff = 999
        
        bearing_idx = bearings.index(bearing)
        bearing_name = bearing_names[bearing_idx]
        
        target_points.append({
            'bearing': bearing,
            'bearing_name': bearing_name,
            'distance': distance,
            'covered': covered,
            'nearest_bearing_diff': nearest_bearing_diff,
            'nearest_distance_diff': nearest_distance_diff
        })

df_targets = pd.DataFrame(target_points)

# Summary by direction
print("\n📊 Coverage by Direction:")
print("-"*70)
print(f"{'Dir':3s} {'Bearing':>7s} | {'30m':^6s} {'70m':^6s} {'110m':^6s} {'150m':^6s} | {'Total':>5s}")
print("-"*70)

for bearing, name in zip(bearings, bearing_names):
    targets = df_targets[df_targets['bearing'] == bearing]
    coverage = []
    for dist in distances:
        target = targets[targets['distance'] == dist].iloc[0]
        if target['covered']:
            coverage.append('✅')
        else:
            coverage.append('❌')
    
    total_covered = sum(1 for c in coverage if c == '✅')
    total_str = f"{total_covered}/{len(distances)}"
    
    print(f"{name:3s} {bearing:3d}°    | {coverage[0]:^6s} {coverage[1]:^6s} {coverage[2]:^6s} {coverage[3]:^6s} | {total_str:>5s}")

total_targets = len(df_targets)
covered_targets = df_targets['covered'].sum()
coverage_pct = (covered_targets / total_targets) * 100

print("-"*70)
print(f"Total Coverage: {covered_targets}/{total_targets} ({coverage_pct:.1f}%)")

# List missing targets
print("\n" + "="*70)
print("❌ Missing GCP Targets (Need to Collect)")
print("="*70)

missing = df_targets[~df_targets['covered']].copy()
print(f"\nTotal missing: {len(missing)} positions\n")

# Calculate GPS coordinates for missing targets
print("📍 GPS Coordinates to Collect:")
print("-"*70)
print(f"{'No.':>3s} {'Dir':3s} {'Bearing':>7s} {'Distance':>8s} {'Latitude':>11s} {'Longitude':>12s} | {'Nearest Gap':>15s}")
print("-"*70)

for i, (idx, row) in enumerate(missing.iterrows(), 1):
    # Calculate target GPS
    target_lat, target_lon = bearing_distance_to_latlon(
        row['bearing'], row['distance'], CAMERA_LAT, CAMERA_LON
    )
    
    gap_info = f"±{row['nearest_bearing_diff']:.0f}° ±{row['nearest_distance_diff']:.0f}m"
    
    print(f"{i:3d} {row['bearing_name']:3s} {row['bearing']:3.0f}° {row['distance']:5.0f}m "
          f"{target_lat:11.6f} {target_lon:12.6f} | {gap_info:>15s}")

# Priority recommendations
print("\n" + "="*70)
print("🎯 Priority Collection Recommendations")
print("="*70)

# Priority 1: Test region
test_targets = missing[
    (missing['bearing'].between(210, 250)) &
    (missing['distance'].between(100, 150))
]

print(f"\n🔴 Priority 1: Test Region (CRITICAL)")
print(f"   Need {len(test_targets)} points near bearing 237°, distance 131m")
if len(test_targets) > 0:
    print(f"   Specific targets:")
    for _, row in test_targets.iterrows():
        target_lat, target_lon = bearing_distance_to_latlon(
            row['bearing'], row['distance'], CAMERA_LAT, CAMERA_LON
        )
        print(f"   - {row['bearing_name']} {row['bearing']:.0f}° at {row['distance']:.0f}m "
              f"→ {target_lat:.6f}, {target_lon:.6f}")

# Priority 2: Missing directions
missing_dirs = missing['bearing_name'].unique()
current_dirs = df_targets[df_targets['covered']]['bearing_name'].unique()
completely_missing_dirs = [d for d in bearing_names if d not in current_dirs]

if len(completely_missing_dirs) > 0:
    print(f"\n🟠 Priority 2: Missing Directions")
    print(f"   Completely missing: {', '.join(completely_missing_dirs)}")
    print(f"   Need at least 2-3 points per direction")

# Priority 3: Distance coverage
for dist in distances:
    targets_at_dist = missing[missing['distance'] == dist]
    if len(targets_at_dist) > len(bearings) * 0.5:  # If >50% missing
        print(f"\n🟡 Priority 3: Distance {dist}m range")
        print(f"   Missing {len(targets_at_dist)}/{len(bearings)} directions")

# Create visual guide
print("\n" + "="*70)
print("🗺️ Visual Collection Guide")
print("="*70)

print("""
         N (0°)
          ↑
          |
   NW  ←  •  →  NE
  315°    |    45°
          |
    W ← • • • → E
   270°  CAMERA  90°
          |
    SW  • | •  SE
   225°   |    135°
          ↓
         S (180°)

Target Distances:
  • Inner circle:  30m (Very Close)
  • 2nd circle:    70m (Close)
  • 3rd circle:   110m (Medium) ← Test here!
  • Outer circle: 150m (Far)

Altitudes to capture:
  ✈️ Low:    50m
  ✈️ Medium: 65m
  ✈️ High:   80m
""")

# Export collection list
collection_list = []
for _, row in missing.iterrows():
    target_lat, target_lon = bearing_distance_to_latlon(
        row['bearing'], row['distance'], CAMERA_LAT, CAMERA_LON
    )
    
    for alt in altitudes:
        collection_list.append({
            'direction': row['bearing_name'],
            'bearing_deg': row['bearing'],
            'distance_m': row['distance'],
            'altitude_m': alt,
            'target_latitude': target_lat,
            'target_longitude': target_lon,
            'priority': 1 if (row['bearing'] >= 210 and row['bearing'] <= 250 and 
                            row['distance'] >= 100 and row['distance'] <= 150) else 2
        })

df_collection = pd.DataFrame(collection_list)
df_collection = df_collection.sort_values(['priority', 'bearing_deg', 'distance_m', 'altitude_m'])
df_collection.to_csv('gcp_collection_targets.csv', index=False, encoding='utf-8')

print(f"\n✅ Saved collection targets to: gcp_collection_targets.csv")
print(f"   Total targets: {len(df_collection)} (combinations of position + altitude)")

# Summary
print("\n" + "="*70)
print("📋 Summary & Next Steps")
print("="*70)

print(f"""
Current Status:
  ✅ Have data: {covered_targets}/{total_targets} positions ({coverage_pct:.1f}%)
  ❌ Missing:    {len(missing)} positions ({100-coverage_pct:.1f}%)

To get complete GCP coverage:
  
  1. 🔴 PRIORITY 1: Collect {len(test_targets)} points in test region
     - Bearing: 210-250° (SW-S direction)
     - Distance: 100-150m
     - Altitude: 50m, 65m, 80m each
     
  2. 🟠 PRIORITY 2: Collect missing directions
     - Completely missing: {', '.join(completely_missing_dirs) if completely_missing_dirs else 'None'}
     - Total: ~{len(missing_dirs) * 3} points needed
     
  3. 🟡 PRIORITY 3: Fill remaining gaps
     - Complete the grid systematically
     - Total: {len(missing)} positions × 3 altitudes = {len(df_collection)} images

Minimum Recommended:
  - At least 8 directions × 3 distances = 24 positions
  - Each position × 3 altitudes = 72 images minimum
  - Current: {covered_targets} positions covered

How to Collect:
  1. Use drone with GPS
  2. Fly to coordinates in gcp_collection_targets.csv
  3. Hover at specified altitude
  4. Take photo facing camera
  5. Record GPS coordinates
  6. Repeat for all targets
""")

print("\n✅ Complete!")
