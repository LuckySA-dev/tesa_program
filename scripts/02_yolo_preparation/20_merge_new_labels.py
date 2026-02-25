"""
Merge more_label_1 into yolo_dataset and retrain YOLO
"""

import os
import shutil
import glob
from pathlib import Path

if __name__ == '__main__':
    print("=" * 70)
    print("Merging New Labels into YOLO Dataset")
    print("=" * 70)

    # Source paths
    new_labels_dir = Path('more_label_1/labels')
    new_images_dir = Path('more_label_1/train')
    
    # Destination paths
    dest_train_labels = Path('yolo_dataset/train/labels')
    dest_train_images = Path('yolo_dataset/train/images')
    
    # Check source
    new_labels = list(new_labels_dir.glob('*.txt'))
    new_images_png = list(new_images_dir.glob('*.png'))
    new_images_jpg = list(new_images_dir.glob('*.jpg'))
    new_images = new_images_png + new_images_jpg
    
    print(f"\n📦 Source (more_label_1):")
    print(f"   Labels: {len(new_labels)}")
    print(f"   Images: {len(new_images)}")
    
    # Check current dataset
    current_train_labels = list(dest_train_labels.glob('*.txt'))
    current_train_images = list(dest_train_images.glob('*.png')) + list(dest_train_images.glob('*.jpg'))
    
    print(f"\n📊 Current dataset:")
    print(f"   Train labels: {len(current_train_labels)}")
    print(f"   Train images: {len(current_train_images)}")
    
    # Copy new labels
    print(f"\n📋 Copying labels...")
    copied_labels = 0
    for label_file in new_labels:
        dest_file = dest_train_labels / label_file.name
        if not dest_file.exists():
            shutil.copy2(label_file, dest_file)
            copied_labels += 1
    
    print(f"   ✅ Copied {copied_labels} new labels")
    
    # Copy new images
    print(f"\n🖼️  Copying images...")
    copied_images = 0
    
    for img_file in new_images:
        # Try to match with label filename
        label_name = img_file.stem + '.txt'
        if (new_labels_dir / label_name).exists():
            dest_file = dest_train_images / img_file.name
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
                copied_images += 1
    
    print(f"   ✅ Copied {copied_images} new images")
    
    # Verify final dataset
    final_train_labels = len(list(dest_train_labels.glob('*.txt')))
    final_train_images = len(list(dest_train_images.glob('*.png'))) + len(list(dest_train_images.glob('*.jpg')))
    valid_labels = len(list(Path('yolo_dataset/valid/labels').glob('*.txt')))
    
    print("\n" + "=" * 70)
    print("📊 Final Dataset Summary")
    print("=" * 70)
    print(f"\n   Train:")
    print(f"      Before: {len(current_train_labels)} labels, {len(current_train_images)} images")
    print(f"      After:  {final_train_labels} labels, {final_train_images} images")
    print(f"      Added:  +{final_train_labels - len(current_train_labels)} labels")
    
    print(f"\n   Valid:")
    print(f"      Labels: {valid_labels}")
    
    print(f"\n   Total:")
    print(f"      {final_train_labels + valid_labels} labels")
    print(f"      Improvement: {(final_train_labels - 50)/50*100:.0f}% more training data!")
    
    print("\n" + "=" * 70)
    print("✅ Merge Complete!")
    print("=" * 70)
    print("\n🎯 Next step: Retrain YOLO with expanded dataset")
    print("   Run: python 04_train_yolo_obb.py")
    print("   (or create new training script with project name 'drone_detect_v20')")
