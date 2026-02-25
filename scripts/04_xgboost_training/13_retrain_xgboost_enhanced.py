"""
Phase 7.3: Retrain XGBoost with Enhanced Features
Goal: Reduce competition score from 6.64 to < 2.0
Focus: Angle Error (70% weight) and Range Error (15% weight)
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.3: Retrain XGBoost with Enhanced Features")
    print("=" * 70)

    # Load enhanced features
    df = pd.read_csv('train_metadata_enhanced_v2.csv')
    print(f"\n✅ Loaded data: {df.shape}")

    # Load feature columns
    with open('feature_columns_v2.json', 'r') as f:
        feature_info = json.load(f)
    
    feature_columns = feature_info['feature_columns']
    target_columns = feature_info['target_columns']
    
    print(f"✅ Features: {len(feature_columns)}")
    print(f"✅ Targets: {target_columns}")

    # Prepare data
    X = df[feature_columns].values
    y_lat = df['latitude_deg'].values
    y_lon = df['longitude_deg'].values
    y_alt = df['altitude_m'].values

    # Split data (same random state for consistency)
    X_train, X_val, y_lat_train, y_lat_val = train_test_split(
        X, y_lat, test_size=0.2, random_state=42
    )
    _, _, y_lon_train, y_lon_val = train_test_split(
        X, y_lon, test_size=0.2, random_state=42
    )
    _, _, y_alt_train, y_alt_val = train_test_split(
        X, y_alt, test_size=0.2, random_state=42
    )

    print(f"\n✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Validation set: {X_val.shape[0]} samples")

    # Enhanced XGBoost parameters (more capacity for more features)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,                # Increased from 6
        'learning_rate': 0.05,         # Lower for better convergence
        'n_estimators': 1000,          # More iterations
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,     # Additional regularization
        'min_child_weight': 3,
        'gamma': 0.1,                  # Min loss reduction
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 1.0,             # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 100,
        'eval_metric': 'rmse'
    }

    print("\n" + "=" * 70)
    print("Training Enhanced XGBoost Models")
    print("=" * 70)
    print(f"\nEnhanced Parameters:")
    print(f"   max_depth: {params['max_depth']}")
    print(f"   learning_rate: {params['learning_rate']}")
    print(f"   n_estimators: {params['n_estimators']}")
    print(f"   Regularization: L1={params['reg_alpha']}, L2={params['reg_lambda']}")

    models = {}
    results = {}

    # 1. Train Latitude Model
    print("\n" + "=" * 70)
    print("1️⃣ Training Enhanced Latitude Model")
    print("=" * 70)
    
    model_lat = xgb.XGBRegressor(**params)
    model_lat.fit(
        X_train, y_lat_train,
        eval_set=[(X_train, y_lat_train), (X_val, y_lat_val)],
        verbose=50
    )
    
    y_lat_train_pred = model_lat.predict(X_train)
    y_lat_val_pred = model_lat.predict(X_val)
    
    lat_train_rmse = np.sqrt(mean_squared_error(y_lat_train, y_lat_train_pred))
    lat_val_rmse = np.sqrt(mean_squared_error(y_lat_val, y_lat_val_pred))
    lat_train_mae = mean_absolute_error(y_lat_train, y_lat_train_pred)
    lat_val_mae = mean_absolute_error(y_lat_val, y_lat_val_pred)
    lat_train_r2 = r2_score(y_lat_train, y_lat_train_pred)
    lat_val_r2 = r2_score(y_lat_val, y_lat_val_pred)
    
    print(f"\n📊 Latitude Results:")
    print(f"   Train RMSE: {lat_train_rmse:.8f} degrees ({lat_train_rmse*111000:.2f} m)")
    print(f"   Val RMSE:   {lat_val_rmse:.8f} degrees ({lat_val_rmse*111000:.2f} m)")
    print(f"   Train MAE:  {lat_train_mae:.8f} degrees ({lat_train_mae*111000:.2f} m)")
    print(f"   Val MAE:    {lat_val_mae:.8f} degrees ({lat_val_mae*111000:.2f} m)")
    print(f"   Train R²:   {lat_train_r2:.4f}")
    print(f"   Val R²:     {lat_val_r2:.4f}")
    
    # Compare with previous
    print(f"\n   📈 Previous Val RMSE: 12.07 m")
    print(f"   📈 Current Val RMSE:  {lat_val_rmse*111000:.2f} m")
    print(f"   {'✅ Improved!' if lat_val_rmse*111000 < 12.07 else '⚠️ No improvement'}")
    
    models['latitude'] = model_lat

    # 2. Train Longitude Model
    print("\n" + "=" * 70)
    print("2️⃣ Training Enhanced Longitude Model")
    print("=" * 70)
    
    model_lon = xgb.XGBRegressor(**params)
    model_lon.fit(
        X_train, y_lon_train,
        eval_set=[(X_train, y_lon_train), (X_val, y_lon_val)],
        verbose=50
    )
    
    y_lon_train_pred = model_lon.predict(X_train)
    y_lon_val_pred = model_lon.predict(X_val)
    
    cos_lat = np.cos(np.deg2rad(14.305))
    
    lon_train_rmse = np.sqrt(mean_squared_error(y_lon_train, y_lon_train_pred))
    lon_val_rmse = np.sqrt(mean_squared_error(y_lon_val, y_lon_val_pred))
    lon_train_mae = mean_absolute_error(y_lon_train, y_lon_train_pred)
    lon_val_mae = mean_absolute_error(y_lon_val, y_lon_val_pred)
    lon_train_r2 = r2_score(y_lon_train, y_lon_train_pred)
    lon_val_r2 = r2_score(y_lon_val, y_lon_val_pred)
    
    print(f"\n📊 Longitude Results:")
    print(f"   Train RMSE: {lon_train_rmse:.8f} degrees ({lon_train_rmse*111000*cos_lat:.2f} m)")
    print(f"   Val RMSE:   {lon_val_rmse:.8f} degrees ({lon_val_rmse*111000*cos_lat:.2f} m)")
    print(f"   Train MAE:  {lon_train_mae:.8f} degrees ({lon_train_mae*111000*cos_lat:.2f} m)")
    print(f"   Val MAE:    {lon_val_mae:.8f} degrees ({lon_val_mae*111000*cos_lat:.2f} m)")
    print(f"   Train R²:   {lon_train_r2:.4f}")
    print(f"   Val R²:     {lon_val_r2:.4f}")
    
    print(f"\n   📈 Previous Val RMSE: 8.64 m")
    print(f"   📈 Current Val RMSE:  {lon_val_rmse*111000*cos_lat:.2f} m")
    print(f"   {'✅ Improved!' if lon_val_rmse*111000*cos_lat < 8.64 else '⚠️ No improvement'}")
    
    models['longitude'] = model_lon

    # 3. Train Altitude Model
    print("\n" + "=" * 70)
    print("3️⃣ Training Enhanced Altitude Model")
    print("=" * 70)
    
    model_alt = xgb.XGBRegressor(**params)
    model_alt.fit(
        X_train, y_alt_train,
        eval_set=[(X_train, y_alt_train), (X_val, y_alt_val)],
        verbose=50
    )
    
    y_alt_train_pred = model_alt.predict(X_train)
    y_alt_val_pred = model_alt.predict(X_val)
    
    alt_train_rmse = np.sqrt(mean_squared_error(y_alt_train, y_alt_train_pred))
    alt_val_rmse = np.sqrt(mean_squared_error(y_alt_val, y_alt_val_pred))
    alt_train_mae = mean_absolute_error(y_alt_train, y_alt_train_pred)
    alt_val_mae = mean_absolute_error(y_alt_val, y_alt_val_pred)
    alt_train_r2 = r2_score(y_alt_train, y_alt_train_pred)
    alt_val_r2 = r2_score(y_alt_val, y_alt_val_pred)
    
    print(f"\n📊 Altitude Results:")
    print(f"   Train RMSE: {alt_train_rmse:.2f} m")
    print(f"   Val RMSE:   {alt_val_rmse:.2f} m")
    print(f"   Train MAE:  {alt_train_mae:.2f} m")
    print(f"   Val MAE:    {alt_val_mae:.2f} m")
    print(f"   Train R²:   {alt_train_r2:.4f}")
    print(f"   Val R²:     {alt_val_r2:.4f}")
    
    print(f"\n   📈 Previous Val RMSE: 1.01 m")
    print(f"   📈 Current Val RMSE:  {alt_val_rmse:.2f} m")
    print(f"   {'✅ Improved!' if alt_val_rmse < 1.01 else '⚠️ No improvement'}")
    
    models['altitude'] = model_alt

    # Save models
    print("\n" + "=" * 70)
    print("Saving Enhanced Models")
    print("=" * 70)
    
    with open('xgb_model_latitude_v2.pkl', 'wb') as f:
        pickle.dump(model_lat, f)
    print("✅ Saved: xgb_model_latitude_v2.pkl")
    
    with open('xgb_model_longitude_v2.pkl', 'wb') as f:
        pickle.dump(model_lon, f)
    print("✅ Saved: xgb_model_longitude_v2.pkl")
    
    with open('xgb_model_altitude_v2.pkl', 'wb') as f:
        pickle.dump(model_alt, f)
    print("✅ Saved: xgb_model_altitude_v2.pkl")

    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance (Top 15 per model)")
    print("=" * 70)
    
    for name, model in [('Latitude', model_lat), ('Longitude', model_lon), ('Altitude', model_alt)]:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n{name} Top 15:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"   {row['feature']:40s}: {row['importance']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Phase 7.3 Complete: Enhanced XGBoost Training")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Latitude Val RMSE:  {lat_val_rmse*111000:.2f} m (was 12.07 m)")
    print(f"   Longitude Val RMSE: {lon_val_rmse*111000*cos_lat:.2f} m (was 8.64 m)")
    print(f"   Altitude Val RMSE:  {alt_val_rmse:.2f} m (was 1.01 m)")
    print(f"\n   Models saved: 3 files (v2)")
    print(f"   Ready for Phase 7.4: Evaluate NEW Competition Score")
