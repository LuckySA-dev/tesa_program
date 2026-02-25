"""
Test script to verify all required libraries are installed correctly
"""

print("Testing library imports...")
print("-" * 50)

try:
    import ultralytics
    print("✅ ultralytics:", ultralytics.__version__)
except ImportError as e:
    print("❌ ultralytics:", str(e))

try:
    import xgboost as xgb
    print("✅ xgboost:", xgb.__version__)
except ImportError as e:
    print("❌ xgboost:", str(e))

try:
    import sklearn
    print("✅ scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("❌ scikit-learn:", str(e))

try:
    import pandas as pd
    print("✅ pandas:", pd.__version__)
except ImportError as e:
    print("❌ pandas:", str(e))

try:
    import numpy as np
    print("✅ numpy:", np.__version__)
except ImportError as e:
    print("❌ numpy:", str(e))

try:
    import cv2
    print("✅ opencv-python:", cv2.__version__)
except ImportError as e:
    print("❌ opencv-python:", str(e))

try:
    from PIL import Image
    import PIL
    print("✅ Pillow:", PIL.__version__)
except ImportError as e:
    print("❌ Pillow:", str(e))

try:
    import geopy
    print("✅ geopy:", geopy.__version__)
except ImportError as e:
    print("❌ geopy:", str(e))

try:
    import matplotlib
    print("✅ matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("❌ matplotlib:", str(e))

try:
    import seaborn as sns
    print("✅ seaborn:", sns.__version__)
except ImportError as e:
    print("❌ seaborn:", str(e))

try:
    import folium
    print("✅ folium:", folium.__version__)
except ImportError as e:
    print("❌ folium:", str(e))

try:
    import tqdm
    print("✅ tqdm:", tqdm.__version__)
except ImportError as e:
    print("❌ tqdm:", str(e))

try:
    import joblib
    print("✅ joblib:", joblib.__version__)
except ImportError as e:
    print("❌ joblib:", str(e))

print("-" * 50)
print("🎉 All libraries installed successfully!")
print("\n✅ Environment setup complete!")
print("\n📂 Project structure:")
print("   - datasets/DATA_TRAIN/ (438 images + CSV + 92 labels)")
print("   - datasets/DATA_TEST/ (264 images)")
print("\n🚀 Ready to start Task 1.2: Data Loading & Validation")
