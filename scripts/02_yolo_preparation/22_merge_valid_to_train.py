"""
Merge validation set into training set for maximum data usage
Total: 150 train + 42 valid = 192 training images
"""

import shutil
import glob
from pathlib import Path

if __name__ == '__main__':
    print("=" * 70)
    print("Merging Validation Set into Training Set")
    print("=" * 70)

    # Paths
    valid_images = Path('yolo_dataset/valid/images')
    valid_labels = Path('yolo_dataset/valid/labels')
    train_images = Path('yolo_dataset/train/images')
    train_labels = Path('yolo_dataset/train/labels')

    # Count before
    train_imgs_before = len(list(train_images.glob('*.*')))
    train_lbls_before = len(list(train_labels.glob('*.txt')))
    valid_imgs = len(list(valid_images.glob('*.*')))
    valid_lbls = len(list(valid_labels.glob('*.txt')))

    print(f"\n📊 Current Status:")
    print(f"   Train: {train_imgs_before} images, {train_lbls_before} labels")
    print(f"   Valid: {valid_imgs} images, {valid_lbls} labels")

    # Copy validation images to train
    print(f"\n🖼️  Copying validation images to train...")
    copied_imgs = 0
    for img_file in valid_images.glob('*.*'):
        dest = train_images / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
            copied_imgs += 1

    # Copy validation labels to train
    print(f"📋 Copying validation labels to train...")
    copied_lbls = 0
    for lbl_file in valid_labels.glob('*.txt'):
        dest = train_labels / lbl_file.name
        if not dest.exists():
            shutil.copy2(lbl_file, dest)
            copied_lbls += 1

    print(f"\n✅ Copied: {copied_imgs} images, {copied_lbls} labels")

    # Create minimal validation set (keep 10 samples for validation)
    print(f"\n⚠️  Creating minimal validation set (10 samples)...")
    
    # Keep first 10 files in validation
    all_valid_imgs = list(valid_images.glob('*.*'))
    all_valid_lbls = list(valid_labels.glob('*.txt'))
    
    # Remove extras from validation (keep only 10)
    kept_valid = 10
    for img_file in all_valid_imgs[kept_valid:]:
        img_file.unlink()
    for lbl_file in all_valid_lbls[kept_valid:]:
        lbl_file.unlink()

    # Count after
    train_imgs_after = len(list(train_images.glob('*.*')))
    train_lbls_after = len(list(train_labels.glob('*.txt')))
    valid_imgs_after = len(list(valid_images.glob('*.*')))
    valid_lbls_after = len(list(valid_labels.glob('*.txt')))

    print("\n" + "=" * 70)
    print("📊 Final Dataset")
    print("=" * 70)
    print(f"\n   Training Set:")
    print(f"      Before: {train_imgs_before} images")
    print(f"      After:  {train_imgs_after} images (+{train_imgs_after - train_imgs_before})")
    print(f"      Total:  {train_lbls_after} labels")
    
    print(f"\n   Validation Set:")
    print(f"      Before: {valid_imgs} images")
    print(f"      After:  {valid_imgs_after} images (minimal for validation)")
    
    print(f"\n   📈 Improvement: +{(train_imgs_after - train_imgs_before)/train_imgs_before*100:.0f}% training data!")

    print("\n" + "=" * 70)
    print("✅ Merge Complete!")
    print("=" * 70)
    print("\n🎯 Next: Retrain YOLO with 192 training images")
    print("   Total usable data: 192 images (was 150)")
