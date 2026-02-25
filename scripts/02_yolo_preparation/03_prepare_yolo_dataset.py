"""
Phase 2 - Task 2.1: Prepare YOLO Dataset
Organize data for YOLOv8-OBB training
"""

import os
import shutil
from pathlib import Path

print("=" * 70)
print("Phase 2 - Task 2.1: Prepare YOLO Dataset")
print("=" * 70)

# Base paths
BASE_DIR = Path("datasets/DATA_TRAIN")
YOLO_DIR = Path("yolo_dataset")

# Create YOLO dataset structure
print("\n📁 Creating YOLO dataset structure...")
YOLO_DIR.mkdir(exist_ok=True)
(YOLO_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "valid" / "images").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "valid" / "labels").mkdir(parents=True, exist_ok=True)

print("✅ Directory structure created")

# Copy train images and labels
print("\n📦 Copying training images and labels...")
train_images = list((BASE_DIR / "train").glob("*.jpg"))
train_labels = list((BASE_DIR / "labels").glob("*.txt"))

print(f"Found {len(train_images)} training images")
print(f"Found {len(train_labels)} training labels")

for img_file in train_images:
    shutil.copy(img_file, YOLO_DIR / "train" / "images" / img_file.name)

for label_file in train_labels:
    shutil.copy(label_file, YOLO_DIR / "train" / "labels" / label_file.name)

print(f"✅ Copied {len(train_images)} training images")
print(f"✅ Copied {len(train_labels)} training labels")

# Copy valid images and labels
print("\n📦 Copying validation images and labels...")
valid_images = list((BASE_DIR / "valid").glob("*.jpg"))
valid_labels = list((BASE_DIR / "labelsvalid").glob("*.txt"))

print(f"Found {len(valid_images)} validation images")
print(f"Found {len(valid_labels)} validation labels")

for img_file in valid_images:
    shutil.copy(img_file, YOLO_DIR / "valid" / "images" / img_file.name)

for label_file in valid_labels:
    shutil.copy(label_file, YOLO_DIR / "valid" / "labels" / label_file.name)

print(f"✅ Copied {len(valid_images)} validation images")
print(f"✅ Copied {len(valid_labels)} validation labels")

# Verify label format
print("\n🔍 Verifying label format...")
sample_label = list((YOLO_DIR / "train" / "labels").glob("*.txt"))[0]
with open(sample_label, 'r') as f:
    content = f.read().strip()
    parts = content.split()
    print(f"\nSample label: {sample_label.name}")
    print(f"Content: {content}")
    print(f"Format: class={parts[0]}, cx={parts[1]}, cy={parts[2]}, w={parts[3]}, h={parts[4]}, angle={parts[5] if len(parts) > 5 else 'N/A'}")
    
    if len(parts) == 6:
        print("✅ Format: Oriented Bounding Box (OBB) - class cx cy w h angle")
    elif len(parts) == 5:
        print("✅ Format: Standard Bounding Box - class cx cy w h")
    else:
        print("⚠️ Unknown format")

print("\n" + "=" * 70)
print("📊 Dataset Summary")
print("=" * 70)
print(f"\nTrain Set:")
print(f"  Images: {len(list((YOLO_DIR / 'train' / 'images').glob('*.jpg')))}")
print(f"  Labels: {len(list((YOLO_DIR / 'train' / 'labels').glob('*.txt')))}")

print(f"\nValidation Set:")
print(f"  Images: {len(list((YOLO_DIR / 'valid' / 'images').glob('*.jpg')))}")
print(f"  Labels: {len(list((YOLO_DIR / 'valid' / 'labels').glob('*.txt')))}")

print(f"\n✅ Task 2.1 Complete!")
print(f"\n🚀 Next: Task 2.2 - Create data.yaml and train YOLOv8-OBB model")
