"""
Phase 4: Train XGBoost Models for Coordinate Prediction
Train 3 separate models: Latitude, Longitude, Altitude
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
    print("Phase 4: Train XGBoost Models")
    print("=" * 70)

    # Load engineered features
    df = pd.read_csv('train_metadata_engineered.csv')
    print(f"\n✅ Loaded data: {df.shape}")

    # Load feature columns
    with open('feature_columns.json', 'r') as f:
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

    # Split data (80% train, 20% validation)
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

    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse'
    }

    print("\n" + "=" * 70)
    print("Training XGBoost Models")
    print("=" * 70)
    print(f"\nParameters: {params}")

    models = {}
    results = {}

    # 1. Train Latitude Model
    print("\n" + "=" * 70)
    print("1️⃣ Training Latitude Model")
    print("=" * 70)
    
    model_lat = xgb.XGBRegressor(**params)
    model_lat.fit(
        X_train, y_lat_train,
        eval_set=[(X_train, y_lat_train), (X_val, y_lat_val)],
        verbose=50
    )
    
    # Predict
    y_lat_train_pred = model_lat.predict(X_train)
    y_lat_val_pred = model_lat.predict(X_val)
    
    # Metrics
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
    
    models['latitude'] = model_lat
    results['latitude'] = {
        'train_rmse': lat_train_rmse,
        'val_rmse': lat_val_rmse,
        'train_mae': lat_train_mae,
        'val_mae': lat_val_mae,
        'train_r2': lat_train_r2,
        'val_r2': lat_val_r2
    }

    # 2. Train Longitude Model
    print("\n" + "=" * 70)
    print("2️⃣ Training Longitude Model")
    print("=" * 70)
    
    model_lon = xgb.XGBRegressor(**params)
    model_lon.fit(
        X_train, y_lon_train,
        eval_set=[(X_train, y_lon_train), (X_val, y_lon_val)],
        verbose=50
    )
    
    # Predict
    y_lon_train_pred = model_lon.predict(X_train)
    y_lon_val_pred = model_lon.predict(X_val)
    
    # Metrics
    lon_train_rmse = np.sqrt(mean_squared_error(y_lon_train, y_lon_train_pred))
    lon_val_rmse = np.sqrt(mean_squared_error(y_lon_val, y_lon_val_pred))
    lon_train_mae = mean_absolute_error(y_lon_train, y_lon_train_pred)
    lon_val_mae = mean_absolute_error(y_lon_val, y_lon_val_pred)
    lon_train_r2 = r2_score(y_lon_train, y_lon_train_pred)
    lon_val_r2 = r2_score(y_lon_val, y_lon_val_pred)
    
    # Convert to meters (approximate at this latitude)
    cos_lat = np.cos(np.deg2rad(14.305))
    
    print(f"\n📊 Longitude Results:")
    print(f"   Train RMSE: {lon_train_rmse:.8f} degrees ({lon_train_rmse*111000*cos_lat:.2f} m)")
    print(f"   Val RMSE:   {lon_val_rmse:.8f} degrees ({lon_val_rmse*111000*cos_lat:.2f} m)")
    print(f"   Train MAE:  {lon_train_mae:.8f} degrees ({lon_train_mae*111000*cos_lat:.2f} m)")
    print(f"   Val MAE:    {lon_val_mae:.8f} degrees ({lon_val_mae*111000*cos_lat:.2f} m)")
    print(f"   Train R²:   {lon_train_r2:.4f}")
    print(f"   Val R²:     {lon_val_r2:.4f}")
    
    models['longitude'] = model_lon
    results['longitude'] = {
        'train_rmse': lon_train_rmse,
        'val_rmse': lon_val_rmse,
        'train_mae': lon_train_mae,
        'val_mae': lon_val_mae,
        'train_r2': lon_train_r2,
        'val_r2': lon_val_r2
    }

    # 3. Train Altitude Model
    print("\n" + "=" * 70)
    print("3️⃣ Training Altitude Model")
    print("=" * 70)
    
    model_alt = xgb.XGBRegressor(**params)
    model_alt.fit(
        X_train, y_alt_train,
        eval_set=[(X_train, y_alt_train), (X_val, y_alt_val)],
        verbose=50
    )
    
    # Predict
    y_alt_train_pred = model_alt.predict(X_train)
    y_alt_val_pred = model_alt.predict(X_val)
    
    # Metrics
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
    
    models['altitude'] = model_alt
    results['altitude'] = {
        'train_rmse': alt_train_rmse,
        'val_rmse': alt_val_rmse,
        'train_mae': alt_train_mae,
        'val_mae': alt_val_mae,
        'train_r2': alt_train_r2,
        'val_r2': alt_val_r2
    }

    # Save models
    print("\n" + "=" * 70)
    print("Saving Models")
    print("=" * 70)
    
    with open('xgb_model_latitude.pkl', 'wb') as f:
        pickle.dump(model_lat, f)
    print("✅ Saved: xgb_model_latitude.pkl")
    
    with open('xgb_model_longitude.pkl', 'wb') as f:
        pickle.dump(model_lon, f)
    print("✅ Saved: xgb_model_longitude.pkl")
    
    with open('xgb_model_altitude.pkl', 'wb') as f:
        pickle.dump(model_alt, f)
    print("✅ Saved: xgb_model_altitude.pkl")

    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance (Top 10)")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, model) in enumerate([('Latitude', model_lat), ('Longitude', model_lon), ('Altitude', model_alt)]):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n{name} Top 10:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        # Plot
        top_10 = feature_importance.head(10)
        axes[idx].barh(range(len(top_10)), top_10['importance'].values)
        axes[idx].set_yticks(range(len(top_10)))
        axes[idx].set_yticklabels(top_10['feature'].values)
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{name} Feature Importance')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: xgb_feature_importance.png")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Phase 4 Complete: XGBoost Training")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Latitude Val RMSE:  {lat_val_rmse*111000:.2f} m")
    print(f"   Longitude Val RMSE: {lon_val_rmse*111000*cos_lat:.2f} m")
    print(f"   Altitude Val RMSE:  {alt_val_rmse:.2f} m")
    print(f"\n   Models saved: 3 files")
    print(f"   Ready for Phase 5: Evaluation")
