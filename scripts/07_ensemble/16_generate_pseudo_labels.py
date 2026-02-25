"""
Phase 7.6: Generate Pseudo Labels for Unannotated Images
Strategy: Use best YOLO model (v15) to predict on remaining 346 images
Then use high-confidence predictions as pseudo-labels for re-training
"""

from ultralytics import YOLO
import os
from pathlib import Path
import shutil
import pandas as pd

if __name__ == '__main__':
    print("=" * 70)
    print("Phase 7.6: Generate Pseudo Labels")
    print("=" * 70)

    # Load best YOLO model
    model_path = "runs/detect/drone_detect_v15/weights/best.pt"
    print(f"\n📦 Loading best YOLO model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded")

    # Get all images with CSV but without labels
    DATA_TRAIN = Path("datasets/DATA_TRAIN")
    csv_files = list((DATA_TRAIN / "csv").glob("*.csv"))
    image_dir = DATA_TRAIN / "image"
    existing_labels_dir = DATA_TRAIN / "labels"
    
    print(f"\n✅ Found {len(csv_files)} CSV files")
    
    # Check which images already have labels
    existing_labels = {f.stem for f in existing_labels_dir.glob("*.txt")}
    print(f"✅ Already annotated: {len(existing_labels)} images")
    
    # Find images without labels
    all_image_names = {f.stem for f in csv_files}
    images_without_labels = all_image_names - existing_labels
    print(f"✅ Images without labels: {len(images_without_labels)} images")
    
    # Create output directory for pseudo labels
    pseudo_label_dir = DATA_TRAIN / "pseudo_labels"
    pseudo_label_dir.mkdir(exist_ok=True)
    print(f"\n📁 Created directory: {pseudo_label_dir}")

    # Run inference on images without labels
    print("\n" + "=" * 70)
    print("Generating Pseudo Labels")
    print("=" * 70)
    print(f"Processing {len(images_without_labels)} images...")
    print("Using confidence threshold: 0.25 (moderate confidence)")
    print()

    high_conf_count = 0
    medium_conf_count = 0
    low_conf_count = 0
    no_detection_count = 0

    for img_name in sorted(images_without_labels):
        img_path = image_dir / f"{img_name}.jpg"
        
        if not img_path.exists():
            continue
        
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=0.25,  # Lower threshold to get more detections
            iou=0.45,
            verbose=False
        )
        
        result = results[0]
        
        # Check if detection exists
        if len(result.boxes) == 0:
            no_detection_count += 1
            continue
        
        # Get highest confidence detection
        confidences = result.boxes.conf.cpu().numpy()
        max_conf_idx = confidences.argmax()
        max_conf = confidences[max_conf_idx]
        
        # Only use detections with reasonable confidence
        if max_conf >= 0.5:
            # High confidence - definitely use
            high_conf_count += 1
            use_label = True
        elif max_conf >= 0.35:
            # Medium confidence - use
            medium_conf_count += 1
            use_label = True
        elif max_conf >= 0.25:
            # Low confidence - use but mark
            low_conf_count += 1
            use_label = True
        else:
            use_label = False
        
        if use_label:
            # Get bbox coordinates (normalized)
            box = result.boxes[max_conf_idx]
            x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
            
            # Save in YOLO format: class x_center y_center width height
            label_path = pseudo_label_dir / f"{img_name}.txt"
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"\n📊 Pseudo Label Statistics:")
    print(f"   High confidence (≥0.50):   {high_conf_count} images")
    print(f"   Medium confidence (≥0.35):  {medium_conf_count} images")
    print(f"   Low confidence (≥0.25):     {low_conf_count} images")
    print(f"   No detection:               {no_detection_count} images")
    print(f"   Total pseudo labels:        {high_conf_count + medium_conf_count + low_conf_count}")

    # Create augmented dataset combining real + pseudo labels
    print("\n" + "=" * 70)
    print("Creating Augmented YOLO Dataset")
    print("=" * 70)

    YOLO_AUG_DIR = Path("yolo_dataset_augmented")
    YOLO_AUG_DIR.mkdir(exist_ok=True)
    (YOLO_AUG_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_AUG_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (YOLO_AUG_DIR / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_AUG_DIR / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    # Copy original training set
    print("\n1️⃣ Copying original training set...")
    orig_train_images = list(Path("yolo_dataset/train/images").glob("*.jpg"))
    orig_train_labels = list(Path("yolo_dataset/train/labels").glob("*.txt"))
    
    for img in orig_train_images:
        shutil.copy(img, YOLO_AUG_DIR / "train" / "images" / img.name)
    for lbl in orig_train_labels:
        shutil.copy(lbl, YOLO_AUG_DIR / "train" / "labels" / lbl.name)
    
    print(f"   ✅ Copied {len(orig_train_images)} original training images")

    # Add pseudo-labeled images
    print("\n2️⃣ Adding pseudo-labeled images...")
    pseudo_labels = list(pseudo_label_dir.glob("*.txt"))
    added_count = 0
    
    for label_file in pseudo_labels:
        img_name = label_file.stem
        img_path = image_dir / f"{img_name}.jpg"
        
        if img_path.exists():
            # Copy image and label
            shutil.copy(img_path, YOLO_AUG_DIR / "train" / "images" / img_path.name)
            shutil.copy(label_file, YOLO_AUG_DIR / "train" / "labels" / label_file.name)
            added_count += 1
    
    print(f"   ✅ Added {added_count} pseudo-labeled images")

    # Copy validation set (unchanged)
    print("\n3️⃣ Copying validation set...")
    valid_images = list(Path("yolo_dataset/valid/images").glob("*.jpg"))
    valid_labels = list(Path("yolo_dataset/valid/labels").glob("*.txt"))
    
    for img in valid_images:
        shutil.copy(img, YOLO_AUG_DIR / "valid" / "images" / img.name)
    for lbl in valid_labels:
        shutil.copy(lbl, YOLO_AUG_DIR / "valid" / "labels" / lbl.name)
    
    print(f"   ✅ Copied {len(valid_images)} validation images")

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Dataset Summary")
    print("=" * 70)
    print(f"\n   Original Dataset:")
    print(f"      Train: {len(orig_train_images)} images")
    print(f"      Valid: {len(valid_images)} images")
    print(f"      Total: {len(orig_train_images) + len(valid_images)} images")
    
    new_train_count = len(orig_train_images) + added_count
    print(f"\n   Augmented Dataset:")
    print(f"      Train: {new_train_count} images ({added_count} added, {added_count/len(orig_train_images)*100:.0f}% increase)")
    print(f"      Valid: {len(valid_images)} images")
    print(f"      Total: {new_train_count + len(valid_images)} images")

    # Create new data.yaml
    print("\n" + "=" * 70)
    print("Creating data_augmented.yaml")
    print("=" * 70)

    yaml_content = f"""# Augmented YOLO dataset configuration
# Original labels + Pseudo labels from YOLO predictions

path: {YOLO_AUG_DIR.absolute()}
train: train/images
val: valid/images

# Classes
names:
  0: drone

# Dataset statistics
# Original training images: {len(orig_train_images)}
# Pseudo-labeled images: {added_count}
# Total training images: {new_train_count}
# Validation images: {len(valid_images)}
"""

    with open("data_augmented.yaml", "w", encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("✅ Created data_augmented.yaml")

    print("\n" + "=" * 70)
    print("✅ Phase 7.6 Complete!")
    print("=" * 70)
    print(f"\n🎯 Result: Training set increased from {len(orig_train_images)} to {new_train_count} images!")
    print(f"   Improvement: +{added_count} images ({added_count/len(orig_train_images)*100:.0f}% increase)")
    print(f"\n📝 Next steps:")
    print(f"   1. Train YOLO with augmented dataset: python 17_train_yolo_augmented.py")
    print(f"   2. Expected improvements:")
    print(f"      - Better generalization (more diverse samples)")
    print(f"      - Reduced overfitting")
    print(f"      - Improved mAP50 (target: >0.50)")
