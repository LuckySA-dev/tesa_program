"""
Quick evaluation of v16 competition score
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split

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

camera_lat, camera_lon = 14.305029, 101.173010

# Load data
df = pd.read_csv('train_metadata_engineered_v16.csv')
_, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
df_val = df.iloc[val_idx]

# Load models
with open('xgb_model_latitude_v16.pkl', 'rb') as f:
    model_lat = pickle.load(f)
with open('xgb_model_longitude_v16.pkl', 'rb') as f:
    model_lon = pickle.load(f)
with open('xgb_model_altitude_v16.pkl', 'rb') as f:
    model_alt = pickle.load(f)

with open('feature_columns_v16.json', 'r') as f:
    features = json.load(f)['feature_columns']

# Predict
X_val = df_val[features]
pred_lat = model_lat.predict(X_val)
pred_lon = model_lon.predict(X_val)
pred_alt = model_alt.predict(X_val)

# Calculate errors
pred_bearing = calculate_bearing(camera_lat, camera_lon, pred_lat, pred_lon)
pred_range = haversine_distance(camera_lat, camera_lon, pred_lat, pred_lon)

angle_err = angular_difference(pred_bearing, df_val['bearing_deg'].values)
alt_err = np.abs(pred_alt - df_val['altitude_m'].values)
range_err = np.abs(pred_range - df_val['distance_m'].values)

score = 0.7 * angle_err.mean() + 0.15 * alt_err.mean() + 0.15 * range_err.mean()

print(f'\n📊 v16 Competition Score:')
print(f'   Angle Error:  {angle_err.mean():.2f}°')
print(f'   Alt Error:    {alt_err.mean():.2f} m')
print(f'   Range Error:  {range_err.mean():.2f} m')
print(f'   Total Score:  {score:.4f}')
print(f'\n   Baseline v1:  5.9369')
print(f'   Improvement:  {5.9369-score:+.4f} ({(5.9369-score)/5.9369*100:+.1f}%)')
