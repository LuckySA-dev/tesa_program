"""
Task 1.2: Data Loading & Validation
Load all CSV files and validate data completeness
"""

import pandas as pd
import glob
import os
from pathlib import Path

print("=" * 60)
print("Task 1.2: Data Loading & Validation")
print("=" * 60)

# Paths
DATA_TRAIN_CSV = "datasets/DATA_TRAIN/csv"
DATA_TRAIN_IMAGE = "datasets/DATA_TRAIN/image"

# Camera position (fixed)
CAM_LAT = 14.305029
CAM_LON = 101.173010

print(f"\n📂 Loading CSV files from: {DATA_TRAIN_CSV}")
print("-" * 60)

# Load all CSV files
csv_files = sorted(glob.glob(f"{DATA_TRAIN_CSV}/*.csv"))
print(f"Found {len(csv_files)} CSV files")

# Load data
all_data = []
missing_files = []
errors = []

for csv_file in csv_files:
    try:
        # Extract image number from filename (e.g., img_0001.csv -> 1)
        filename = os.path.basename(csv_file)
        img_num = int(filename.split('_')[1].split('.')[0])
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Validate columns
        required_cols = ['Latitude', 'Longitude', 'Altitude']
        if not all(col in df.columns for col in required_cols):
            errors.append(f"{filename}: Missing required columns")
            continue
        
        # Get values (assume single row per CSV)
        if len(df) > 0:
            row_data = {
                'image_num': img_num,
                'image_name': f"img_{img_num:04d}.jpg",
                'csv_file': filename,
                'latitude': df['Latitude'].values[0],
                'longitude': df['Longitude'].values[0],
                'altitude': df['Altitude'].values[0]
            }
            all_data.append(row_data)
        else:
            errors.append(f"{filename}: Empty CSV file")
            
    except Exception as e:
        errors.append(f"{filename}: {str(e)}")

# Create DataFrame
train_metadata = pd.DataFrame(all_data)

print(f"\n✅ Successfully loaded: {len(train_metadata)} records")
if errors:
    print(f"⚠️  Errors encountered: {len(errors)}")
    for err in errors[:5]:  # Show first 5 errors
        print(f"   - {err}")

# Sort by image number
train_metadata = train_metadata.sort_values('image_num').reset_index(drop=True)

print("\n" + "=" * 60)
print("📊 Data Statistics")
print("=" * 60)

print(f"\n1. Dataset Overview:")
print(f"   - Total records: {len(train_metadata)}")
print(f"   - Image numbers: {train_metadata['image_num'].min()} to {train_metadata['image_num'].max()}")
print(f"   - Expected: 438 records")
print(f"   - Complete: {'✅ Yes' if len(train_metadata) == 438 else '❌ No (missing data)'}")

print(f"\n2. Coordinate Ranges:")
print(f"   Latitude:")
print(f"      Min: {train_metadata['latitude'].min():.6f}")
print(f"      Max: {train_metadata['latitude'].max():.6f}")
print(f"      Mean: {train_metadata['latitude'].mean():.6f}")
print(f"      Std: {train_metadata['latitude'].std():.6f}")

print(f"\n   Longitude:")
print(f"      Min: {train_metadata['longitude'].min():.6f}")
print(f"      Max: {train_metadata['longitude'].max():.6f}")
print(f"      Mean: {train_metadata['longitude'].mean():.6f}")
print(f"      Std: {train_metadata['longitude'].std():.6f}")

print(f"\n   Altitude (meters):")
print(f"      Min: {train_metadata['altitude'].min():.2f}")
print(f"      Max: {train_metadata['altitude'].max():.2f}")
print(f"      Mean: {train_metadata['altitude'].mean():.2f}")
print(f"      Std: {train_metadata['altitude'].std():.2f}")

print(f"\n3. Missing Values:")
missing_lat = train_metadata['latitude'].isna().sum()
missing_lon = train_metadata['longitude'].isna().sum()
missing_alt = train_metadata['altitude'].isna().sum()

print(f"   - Latitude: {missing_lat} ({'✅ None' if missing_lat == 0 else '⚠️ Found'})")
print(f"   - Longitude: {missing_lon} ({'✅ None' if missing_lon == 0 else '⚠️ Found'})")
print(f"   - Altitude: {missing_alt} ({'✅ None' if missing_alt == 0 else '⚠️ Found'})")

print(f"\n4. Outlier Detection:")
# Simple outlier detection using IQR
def detect_outliers(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper)).sum()
    print(f"   - {name}: {outliers} potential outliers")
    return outliers

detect_outliers(train_metadata['latitude'], 'Latitude')
detect_outliers(train_metadata['longitude'], 'Longitude')
detect_outliers(train_metadata['altitude'], 'Altitude')

print(f"\n5. Sample Data (first 5 records):")
print(train_metadata[['image_name', 'latitude', 'longitude', 'altitude']].head())

# Save metadata
output_file = 'train_metadata.csv'
train_metadata.to_csv(output_file, index=False)
print(f"\n💾 Saved metadata to: {output_file}")

print("\n" + "=" * 60)
print("✅ Task 1.2 Complete!")
print("=" * 60)
print("\n🚀 Next: Task 1.3 - Exploratory Data Analysis (EDA)")
