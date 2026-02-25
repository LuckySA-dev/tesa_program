"""
Phase 7.4: Feature Selection + Optimized XGBoost Training
Fix: Select only most important features to avoid overfitting
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import pickle

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.4: Feature Selection + Optimized Training")
    print("=" * 70)

    # Load enhanced features
    df = pd.read_csv('train_metadata_enhanced_v2.csv')
    print(f"\n✅ Loaded data: {df.shape}")

    # Load feature columns
    with open('feature_columns_v2.json', 'r') as f:
        feature_info = json.load(f)
    
    all_features = feature_info['feature_columns']
    print(f"✅ Total features available: {len(all_features)}")

    # Prepare targets
    y_lat = df['latitude_deg'].values
    y_lon = df['longitude_deg'].values
    y_alt = df['altitude_m'].values

    # ========== FEATURE SELECTION ==========
    print("\n" + "=" * 70)
    print("Step 1: Feature Selection (Top 30 features per target)")
    print("=" * 70)

    X_all = df[all_features].values

    # Select top 30 features for each target using F-test
    selector_lat = SelectKBest(f_regression, k=30)
    selector_lon = SelectKBest(f_regression, k=30)
    selector_alt = SelectKBest(f_regression, k=30)

    selector_lat.fit(X_all, y_lat)
    selector_lon.fit(X_all, y_lon)
    selector_alt.fit(X_all, y_alt)

    # Get selected feature indices
    lat_features_idx = selector_lat.get_support(indices=True)
    lon_features_idx = selector_lon.get_support(indices=True)
    alt_features_idx = selector_alt.get_support(indices=True)

    lat_features = [all_features[i] for i in lat_features_idx]
    lon_features = [all_features[i] for i in lon_features_idx]
    alt_features = [all_features[i] for i in alt_features_idx]

    print(f"\n✅ Selected features:")
    print(f"   Latitude:  {len(lat_features)} features")
    print(f"   Longitude: {len(lon_features)} features")
    print(f"   Altitude:  {len(alt_features)} features")

    # Print top 10 selected features per target
    lat_scores = [(all_features[i], selector_lat.scores_[i]) for i in lat_features_idx]
    lat_scores_sorted = sorted(lat_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Top 10 Latitude features:")
    for feat, score in lat_scores_sorted[:10]:
        print(f"   {feat:40s}: {score:.2f}")

    lon_scores = [(all_features[i], selector_lon.scores_[i]) for i in lon_features_idx]
    lon_scores_sorted = sorted(lon_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Top 10 Longitude features:")
    for feat, score in lon_scores_sorted[:10]:
        print(f"   {feat:40s}: {score:.2f}")

    # ========== PREPARE DATA WITH SELECTED FEATURES ==========
    X_lat = df[lat_features].values
    X_lon = df[lon_features].values
    X_alt = df[alt_features].values

    # Split data
    X_lat_train, X_lat_val, y_lat_train, y_lat_val = train_test_split(
        X_lat, y_lat, test_size=0.2, random_state=42
    )
    X_lon_train, X_lon_val, y_lon_train, y_lon_val = train_test_split(
        X_lon, y_lon, test_size=0.2, random_state=42
    )
    X_alt_train, X_alt_val, y_alt_train, y_alt_val = train_test_split(
        X_alt, y_alt, test_size=0.2, random_state=42
    )

    print(f"\n✅ Train set: {X_lat_train.shape[0]} samples")
    print(f"✅ Validation set: {X_lat_val.shape[0]} samples")

    # Optimized XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }

    print("\n" + "=" * 70)
    print("Step 2: Train XGBoost with Selected Features")
    print("=" * 70)

    # 1. Latitude Model
    print("\n1️⃣ Training Latitude Model (30 features)...")
    model_lat = xgb.XGBRegressor(**params)
    model_lat.fit(
        X_lat_train, y_lat_train,
        eval_set=[(X_lat_train, y_lat_train), (X_lat_val, y_lat_val)],
        verbose=50
    )
    
    y_lat_val_pred = model_lat.predict(X_lat_val)
    lat_val_rmse = np.sqrt(mean_squared_error(y_lat_val, y_lat_val_pred))
    lat_val_r2 = r2_score(y_lat_val, y_lat_val_pred)
    
    print(f"\n   Val RMSE: {lat_val_rmse*111000:.2f} m")
    print(f"   Val R²:   {lat_val_r2:.4f}")

    # 2. Longitude Model
    print("\n2️⃣ Training Longitude Model (30 features)...")
    model_lon = xgb.XGBRegressor(**params)
    model_lon.fit(
        X_lon_train, y_lon_train,
        eval_set=[(X_lon_train, y_lon_train), (X_lon_val, y_lon_val)],
        verbose=50
    )
    
    y_lon_val_pred = model_lon.predict(X_lon_val)
    cos_lat = np.cos(np.deg2rad(14.305))
    lon_val_rmse = np.sqrt(mean_squared_error(y_lon_val, y_lon_val_pred))
    lon_val_r2 = r2_score(y_lon_val, y_lon_val_pred)
    
    print(f"\n   Val RMSE: {lon_val_rmse*111000*cos_lat:.2f} m")
    print(f"   Val R²:   {lon_val_r2:.4f}")

    # 3. Altitude Model
    print("\n3️⃣ Training Altitude Model (30 features)...")
    model_alt = xgb.XGBRegressor(**params)
    model_alt.fit(
        X_alt_train, y_alt_train,
        eval_set=[(X_alt_train, y_alt_train), (X_alt_val, y_alt_val)],
        verbose=50
    )
    
    y_alt_val_pred = model_alt.predict(X_alt_val)
    alt_val_rmse = np.sqrt(mean_squared_error(y_alt_val, y_alt_val_pred))
    alt_val_r2 = r2_score(y_alt_val, y_alt_val_pred)
    
    print(f"\n   Val RMSE: {alt_val_rmse:.2f} m")
    print(f"   Val R²:   {alt_val_r2:.4f}")

    # Save models and selected features
    print("\n" + "=" * 70)
    print("Saving Models and Feature Lists")
    print("=" * 70)
    
    with open('xgb_model_latitude_v3.pkl', 'wb') as f:
        pickle.dump(model_lat, f)
    with open('xgb_model_longitude_v3.pkl', 'wb') as f:
        pickle.dump(model_lon, f)
    with open('xgb_model_altitude_v3.pkl', 'wb') as f:
        pickle.dump(model_alt, f)
    
    # Save selected features
    selected_features = {
        'lat_features': lat_features,
        'lon_features': lon_features,
        'alt_features': alt_features
    }
    
    with open('selected_features_v3.json', 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    print("✅ Saved models: xgb_model_*_v3.pkl")
    print("✅ Saved features: selected_features_v3.json")

    print("\n" + "=" * 70)
    print("✅ Phase 7.4 Complete")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Latitude Val RMSE:  {lat_val_rmse*111000:.2f} m (target: < 12 m)")
    print(f"   Longitude Val RMSE: {lon_val_rmse*111000*cos_lat:.2f} m (target: < 9 m)")
    print(f"   Altitude Val RMSE:  {alt_val_rmse:.2f} m (target: < 1 m)")
    print(f"\n   Ready for Phase 7.5: Evaluate Competition Score")
