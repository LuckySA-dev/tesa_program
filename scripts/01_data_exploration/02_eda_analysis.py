"""
Task 1.3 & 1.4: Exploratory Data Analysis (EDA) + Geospatial Analysis
Comprehensive analysis of drone flight data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import math
import folium
from folium import plugins

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 70)
print("Task 1.3 & 1.4: EDA + Geospatial Analysis")
print("=" * 70)

# Load data
train_df = pd.read_csv('train_metadata.csv')
print(f"\n✅ Loaded: {len(train_df)} records")

# Camera position
CAM_LAT = 14.305029
CAM_LON = 101.173010

# ========== Geospatial Calculations ==========
print("\n" + "=" * 70)
print("🌍 Geospatial Analysis")
print("=" * 70)

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing (azimuth) from point1 to point2 in degrees (0-360)"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

# Calculate distance from camera to each drone position
print("\nCalculating distances and bearings from camera...")
train_df['distance_m'] = train_df.apply(
    lambda row: geodesic((CAM_LAT, CAM_LON), (row['latitude'], row['longitude'])).meters,
    axis=1
)

# Calculate bearing (azimuth angle)
train_df['bearing_deg'] = train_df.apply(
    lambda row: calculate_bearing(CAM_LAT, CAM_LON, row['latitude'], row['longitude']),
    axis=1
)

print(f"\nDistance Statistics:")
print(f"   Min: {train_df['distance_m'].min():.2f} m")
print(f"   Max: {train_df['distance_m'].max():.2f} m")
print(f"   Mean: {train_df['distance_m'].mean():.2f} m")
print(f"   Std: {train_df['distance_m'].std():.2f} m")

print(f"\nBearing Statistics:")
print(f"   Min: {train_df['bearing_deg'].min():.2f}°")
print(f"   Max: {train_df['bearing_deg'].max():.2f}°")
print(f"   Mean: {train_df['bearing_deg'].mean():.2f}°")
print(f"   Std: {train_df['bearing_deg'].std():.2f}°")

# ========== Visualizations ==========
print("\n" + "=" * 70)
print("📊 Creating Visualizations...")
print("=" * 70)

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Latitude distribution
ax1 = plt.subplot(3, 4, 1)
train_df['latitude'].hist(bins=30, edgecolor='black', ax=ax1)
ax1.set_title('Latitude Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Latitude')
ax1.set_ylabel('Frequency')

# 2. Longitude distribution
ax2 = plt.subplot(3, 4, 2)
train_df['longitude'].hist(bins=30, edgecolor='black', ax=ax2, color='orange')
ax2.set_title('Longitude Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Frequency')

# 3. Altitude distribution
ax3 = plt.subplot(3, 4, 3)
train_df['altitude'].hist(bins=30, edgecolor='black', ax=ax3, color='green')
ax3.set_title('Altitude Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Altitude (m)')
ax3.set_ylabel('Frequency')

# 4. Distance distribution
ax4 = plt.subplot(3, 4, 4)
train_df['distance_m'].hist(bins=30, edgecolor='black', ax=ax4, color='red')
ax4.set_title('Distance from Camera', fontsize=12, fontweight='bold')
ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Frequency')

# 5. Bearing distribution (circular)
ax5 = plt.subplot(3, 4, 5)
train_df['bearing_deg'].hist(bins=36, edgecolor='black', ax=ax5, color='purple')
ax5.set_title('Bearing Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Bearing (degrees)')
ax5.set_ylabel('Frequency')

# 6. Lat-Lon trajectory (2D path)
ax6 = plt.subplot(3, 4, 6)
scatter = ax6.scatter(train_df['longitude'], train_df['latitude'], 
                     c=train_df['image_num'], cmap='viridis', s=20, alpha=0.6)
ax6.plot(CAM_LON, CAM_LAT, 'r*', markersize=15, label='Camera')
ax6.set_title('Flight Trajectory (2D)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
ax6.legend()
plt.colorbar(scatter, ax=ax6, label='Image Number')

# 7. Altitude vs Distance
ax7 = plt.subplot(3, 4, 7)
ax7.scatter(train_df['distance_m'], train_df['altitude'], alpha=0.5, c='green')
ax7.set_title('Altitude vs Distance', fontsize=12, fontweight='bold')
ax7.set_xlabel('Distance from Camera (m)')
ax7.set_ylabel('Altitude (m)')

# 8. Altitude over time (sequence)
ax8 = plt.subplot(3, 4, 8)
ax8.plot(train_df['image_num'], train_df['altitude'], alpha=0.7, linewidth=0.5)
ax8.set_title('Altitude Timeline', fontsize=12, fontweight='bold')
ax8.set_xlabel('Image Number')
ax8.set_ylabel('Altitude (m)')

# 9. Distance over time
ax9 = plt.subplot(3, 4, 9)
ax9.plot(train_df['image_num'], train_df['distance_m'], alpha=0.7, linewidth=0.5, color='orange')
ax9.set_title('Distance Timeline', fontsize=12, fontweight='bold')
ax9.set_xlabel('Image Number')
ax9.set_ylabel('Distance (m)')

# 10. Bearing over time
ax10 = plt.subplot(3, 4, 10)
ax10.plot(train_df['image_num'], train_df['bearing_deg'], alpha=0.7, linewidth=0.5, color='red')
ax10.set_title('Bearing Timeline', fontsize=12, fontweight='bold')
ax10.set_xlabel('Image Number')
ax10.set_ylabel('Bearing (degrees)')

# 11. Correlation heatmap
ax11 = plt.subplot(3, 4, 11)
corr_data = train_df[['latitude', 'longitude', 'altitude', 'distance_m', 'bearing_deg']].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax11, cbar_kws={'shrink': 0.8})
ax11.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# 12. Polar plot (bearing)
ax12 = plt.subplot(3, 4, 12, projection='polar')
bearings_rad = np.radians(train_df['bearing_deg'].values)
ax12.scatter(bearings_rad, train_df['distance_m'], alpha=0.5, s=10, c='blue')
ax12.set_title('Polar View (Bearing vs Distance)', fontsize=12, fontweight='bold')
ax12.set_theta_zero_location('N')
ax12.set_theta_direction(-1)

plt.tight_layout()
plt.savefig('01_eda_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: 01_eda_analysis.png")

# ========== Interactive Map ==========
print("\n📍 Creating interactive map...")

# Create map centered on camera
map_center = [CAM_LAT, CAM_LON]
m = folium.Map(location=map_center, zoom_start=15, tiles='OpenStreetMap')

# Add camera marker
folium.Marker(
    [CAM_LAT, CAM_LON],
    popup='<b>Camera Position</b>',
    tooltip='Camera',
    icon=folium.Icon(color='red', icon='camera', prefix='fa')
).add_to(m)

# Add drone positions (sample every 10th point for clarity)
for idx, row in train_df[::10].iterrows():
    folium.CircleMarker(
        [row['latitude'], row['longitude']],
        radius=3,
        popup=f"<b>Image {row['image_num']}</b><br>"
              f"Alt: {row['altitude']:.2f}m<br>"
              f"Dist: {row['distance_m']:.2f}m<br>"
              f"Bearing: {row['bearing_deg']:.2f}°",
        color='blue',
        fill=True,
        fillOpacity=0.6
    ).add_to(m)

# Add flight path line
coordinates = train_df[['latitude', 'longitude']].values.tolist()
folium.PolyLine(
    coordinates,
    color='blue',
    weight=2,
    opacity=0.5,
    popup='Flight Path'
).add_to(m)

# Add heat map
heat_data = [[row['latitude'], row['longitude']] for idx, row in train_df.iterrows()]
plugins.HeatMap(heat_data, radius=15).add_to(m)

# Save map
m.save('01_flight_trajectory_map.html')
print("✅ Saved: 01_flight_trajectory_map.html")

# ========== Flight Pattern Analysis ==========
print("\n" + "=" * 70)
print("🛸 Flight Pattern Analysis")
print("=" * 70)

# Analyze flight pattern
bearing_std = train_df['bearing_deg'].std()
bearing_range = train_df['bearing_deg'].max() - train_df['bearing_deg'].min()

print(f"\nPattern Indicators:")
print(f"   Bearing Range: {bearing_range:.2f}°")
print(f"   Bearing Std Dev: {bearing_std:.2f}°")

if bearing_range > 180:
    pattern = "Circular/Arc Flight Pattern"
elif bearing_std < 5:
    pattern = "Linear/Straight Flight Pattern"
else:
    pattern = "Mixed/Complex Flight Pattern"

print(f"\n   Detected Pattern: {pattern}")

# Altitude behavior
alt_trend = np.polyfit(train_df['image_num'], train_df['altitude'], 1)[0]
print(f"\n   Altitude Trend: {'Ascending' if alt_trend > 0 else 'Descending'} "
      f"({alt_trend:.4f} m per image)")

# Save enhanced metadata
train_df.to_csv('train_metadata_enhanced.csv', index=False)
print(f"\n💾 Saved enhanced metadata with distance/bearing to: train_metadata_enhanced.csv")

print("\n" + "=" * 70)
print("✅ Task 1.3 & 1.4 Complete!")
print("=" * 70)
print(f"\n📊 Generated Files:")
print(f"   1. 01_eda_analysis.png - Statistical visualizations")
print(f"   2. 01_flight_trajectory_map.html - Interactive map")
print(f"   3. train_metadata_enhanced.csv - Data with geospatial features")
print(f"\n🚀 Next: Phase 2 - YOLO Model Training")
