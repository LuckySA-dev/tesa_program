"""
Test Neural Network with Custom Competition Loss
Properly trained to optimize competition metric
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time

# Constants
CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees"""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad)*np.sin(lat2_rad) - np.sin(lat1_rad)*np.cos(lat2_rad)*np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def angle_difference(angle1, angle2):
    """Calculate smallest difference between angles"""
    diff = np.abs(angle1 - angle2)
    return np.where(diff > 180, 360 - diff, diff)

def competition_score(angle_error, height_error, range_error):
    """Calculate competition score"""
    return 0.7 * angle_error + 0.15 * height_error + 0.15 * range_error

print("="*80)
print("🧠 Neural Network with Custom Competition Loss")
print("="*80)

# Load data
print("\n📂 Loading training data...")
df = pd.read_csv('train_metadata_engineered_v21.csv')
df_detected = df[df['yolo_detected'] == True].copy()

df_detected['true_distance'] = haversine_distance(
    CAMERA_LAT, CAMERA_LON, df_detected['latitude'], df_detected['longitude']
)
df_detected['true_bearing'] = calculate_bearing(
    CAMERA_LAT, CAMERA_LON, df_detected['latitude'], df_detected['longitude']
)
df_detected['true_altitude'] = df_detected['altitude']

bbox_features = ['yolo_cx', 'yolo_cy', 'yolo_w', 'yolo_h', 'yolo_conf',
                 'yolo_area', 'yolo_aspect_ratio', 'yolo_dist_from_center',
                 'yolo_angle_from_center']

X = df_detected[bbox_features].values
y_bearing = df_detected['true_bearing'].values
y_distance = df_detected['true_distance'].values
y_altitude = df_detected['true_altitude'].values

#Split data
X_train, X_val, bearing_train, bearing_val = train_test_split(X, y_bearing, test_size=0.2, random_state=42)
_, _, distance_train, distance_val = train_test_split(X, y_distance, test_size=0.2, random_state=42)
_, _, altitude_train, altitude_val = train_test_split(X, y_altitude, test_size=0.2, random_state=42)

print(f"✅ Loaded {len(df_detected)} samples")
print(f"   Train: {len(X_train)}, Val: {len(X_val)}")

# Neural Network
class DronePositionNet(nn.Module):
    def __init__(self, input_dim=9):
        super(DronePositionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 3)  # bearing, distance, altitude
        )
        
    def forward(self, x):
        return self.network(x)

def custom_loss(pred, target):
    """Competition-metric loss function"""
    pred_bearing = pred[:, 0]
    pred_distance = pred[:, 1] 
    pred_altitude = pred[:, 2]
    
    true_bearing = target[:, 0]
    true_distance = target[:, 1]
    true_altitude = target[:, 2]
    
    # Angle error - use MSE on sin/cos to handle circularity
    pred_bearing_rad = torch.deg2rad(pred_bearing)
    true_bearing_rad = torch.deg2rad(true_bearing)
    
    angle_loss = (torch.sin(pred_bearing_rad) - torch.sin(true_bearing_rad))**2 + \
                 (torch.cos(pred_bearing_rad) - torch.cos(true_bearing_rad))**2
    
    # Distance and altitude - use Huber loss (robust to outliers)
    distance_loss = torch.nn.functional.huber_loss(pred_distance, true_distance, delta=10.0)
    altitude_loss = torch.nn.functional.huber_loss(pred_altitude, true_altitude, delta=5.0)
    
    # Weighted by competition formula
    total_loss = 0.7 * angle_loss.mean() + 0.15 * distance_loss + 0.15 * altitude_loss
    
    return total_loss

# Prepare tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)

y_train = np.column_stack([bearing_train, distance_train, altitude_train])
y_val = np.column_stack([bearing_val, distance_val, altitude_val])

y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor = torch.FloatTensor(y_val)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🔧 Device: {device}")

model = DronePositionNet(input_dim=9).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

print("\n📊 Training Neural Network...")
num_epochs = 200
batch_size = 32
best_val_score = float('inf')
patience = 30
patience_counter = 0

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    indices = torch.randperm(len(X_train_tensor))
    for i in range(0, len(X_train_tensor), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_X = X_train_tensor[batch_indices].to(device)
        batch_y = y_train_tensor[batch_indices].to(device)
        
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = custom_loss(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * len(batch_indices)
    
    train_loss /= len(X_train_tensor)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor.to(device)).cpu().numpy()
        
        pred_bearing_val = val_pred[:, 0]
        pred_distance_val = val_pred[:, 1]
        pred_altitude_val = val_pred[:, 2]
        
        angle_err = angle_difference(pred_bearing_val, bearing_val)
        range_err = np.abs(pred_distance_val - distance_val)
        height_err = np.abs(pred_altitude_val - altitude_val)
        
        val_score = competition_score(angle_err.mean(), height_err.mean(), range_err.mean())
    
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Score: {val_score:.4f} "
              f"(Angle: {angle_err.mean():.2f}° Height: {height_err.mean():.2f}m Range: {range_err.mean():.2f}m)")
    
    if val_score < best_val_score:
        best_val_score = val_score
        patience_counter = 0
        torch.save(model.state_dict(), 'models_approximation/nn_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

train_time = time.time() - start_time

# Load best model
model.load_state_dict(torch.load('models_approximation/nn_best.pth', weights_only=True))
model.eval()

with torch.no_grad():
    val_pred = model(X_val_tensor.to(device)).cpu().numpy()

pred_bearing_nn = val_pred[:, 0]
pred_distance_nn = val_pred[:, 1]
pred_altitude_nn = val_pred[:, 2]

angle_err_nn = angle_difference(pred_bearing_nn, bearing_val)
range_err_nn = np.abs(pred_distance_nn - distance_val)
height_err_nn = np.abs(pred_altitude_nn - altitude_val)

score_nn = competition_score(angle_err_nn.mean(), height_err_nn.mean(), range_err_nn.mean())

print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)

print(f"\n✅ Neural Network (Custom Loss):")
print(f"   Training time: {train_time:.1f}s")
print(f"   Angle Error:  {angle_err_nn.mean():.2f}°")
print(f"   Height Error: {height_err_nn.mean():.2f} m")
print(f"   Range Error:  {range_err_nn.mean():.2f} m")
print(f"   Competition Score: {score_nn:.4f}")

# Compare with baseline (Residual Learning)
import joblib

models = {
    'distance': joblib.load('models_approximation/bbox_to_distance.pkl'),
    'bearing_sin': joblib.load('models_approximation/bbox_to_bearing_sin.pkl'),
    'bearing_cos': joblib.load('models_approximation/bbox_to_bearing_cos.pkl'),
    'altitude': joblib.load('models_approximation/bbox_to_altitude.pkl'),
    'dist_residual': joblib.load('models_approximation/residual_distance.pkl'),
    'bearing_residual': joblib.load('models_approximation/residual_bearing.pkl'),
    'alt_residual': joblib.load('models_approximation/residual_altitude.pkl')
}

pred_distance_base = models['distance'].predict(X_val)
pred_bearing_sin = models['bearing_sin'].predict(X_val)
pred_bearing_cos = models['bearing_cos'].predict(X_val)
pred_bearing_base = np.degrees(np.arctan2(pred_bearing_sin, pred_bearing_cos)) % 360
pred_altitude_base = models['altitude'].predict(X_val)

residual_distance = models['dist_residual'].predict(X_val)
residual_bearing = models['bearing_residual'].predict(X_val)
residual_altitude = models['alt_residual'].predict(X_val)

pred_bearing_baseline = (pred_bearing_base - residual_bearing) % 360
pred_distance_baseline = pred_distance_base - residual_distance
pred_altitude_baseline = pred_altitude_base - residual_altitude

angle_err_baseline = angle_difference(pred_bearing_baseline, bearing_val)
range_err_baseline = np.abs(pred_distance_baseline - distance_val)
height_err_baseline = np.abs(pred_altitude_baseline - altitude_val)

score_baseline = competition_score(angle_err_baseline.mean(), height_err_baseline.mean(), range_err_baseline.mean())

print(f"\n📊 Baseline (Residual Learning):")
print(f"   Angle Error:  {angle_err_baseline.mean():.2f}°")
print(f"   Height Error: {height_err_baseline.mean():.2f} m")
print(f"   Range Error:  {range_err_baseline.mean():.2f} m")
print(f"   Competition Score: {score_baseline:.4f}")

print("\n" + "="*80)
print("📈 COMPARISON")
print("="*80)

print("\n┌──────────────────────────┬────────┬────────┬────────┬────────┐")
print("│ Method                   │  Angle │ Height │  Range │  Score │")
print("├──────────────────────────┼────────┼────────┼────────┼────────┤")
print(f"│ Baseline (Residual)      │  {angle_err_baseline.mean():5.2f} │  {height_err_baseline.mean():5.2f} │  {range_err_baseline.mean():5.2f} │  {score_baseline:5.2f} │")
print(f"│ Neural Net (Custom Loss) │  {angle_err_nn.mean():5.2f} │  {height_err_nn.mean():5.2f} │  {range_err_nn.mean():5.2f} │  {score_nn:5.2f} │")
print("└──────────────────────────┴────────┴────────┴────────┴────────┘")

if score_nn < score_baseline:
    improvement = ((score_baseline - score_nn) / score_baseline * 100)
    print(f"\n🏆 Neural Network is BETTER by {improvement:.1f}%!")
    print(f"   Use this for test set predictions")
else:
    diff = ((score_nn - score_baseline) / score_baseline * 100)
    print(f"\n⚠️  Neural Network is worse by {diff:.1f}%")
    print(f"   Baseline (Residual Learning) is still best")
    print(f"   Stick with existing predictions")

print("\n" + "="*80)
