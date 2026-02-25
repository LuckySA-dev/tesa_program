"""
Phase 2 - Task 2.1B: Fix YOLO Label Class Index
Convert class from 1 to 0 (YOLO format requirement)
"""

from pathlib import Path
import re

print("=" * 70)
print("Phase 2 - Task 2.1B: Fix YOLO Label Class Index")
print("=" * 70)

YOLO_DIR = Path("yolo_dataset")

# Process train labels
print("\n🔧 Fixing train labels...")
train_labels = list((YOLO_DIR / "train" / "labels").glob("*.txt"))
fixed_train = 0

for label_file in train_labels:
    with open(label_file, 'r') as f:
        content = f.read().strip()
    
    # Change class from 1 to 0
    if content.startswith('1 '):
        new_content = '0 ' + content[2:]
        with open(label_file, 'w') as f:
            f.write(new_content + '\n')
        fixed_train += 1

print(f"✅ Fixed {fixed_train}/{len(train_labels)} train labels")

# Process valid labels
print("\n🔧 Fixing validation labels...")
valid_labels = list((YOLO_DIR / "valid" / "labels").glob("*.txt"))
fixed_valid = 0

for label_file in valid_labels:
    with open(label_file, 'r') as f:
        content = f.read().strip()
    
    # Change class from 1 to 0
    if content.startswith('1 '):
        new_content = '0 ' + content[2:]
        with open(label_file, 'w') as f:
            f.write(new_content + '\n')
        fixed_valid += 1

print(f"✅ Fixed {fixed_valid}/{len(valid_labels)} validation labels")

# Verify fix
print("\n🔍 Verifying fixes...")
sample_train = list((YOLO_DIR / "train" / "labels").glob("*.txt"))[0]
sample_valid = list((YOLO_DIR / "valid" / "labels").glob("*.txt"))[0]

with open(sample_train, 'r') as f:
    train_content = f.read().strip()
with open(sample_valid, 'r') as f:
    valid_content = f.read().strip()

print(f"\nSample train label: {sample_train.name}")
print(f"  Content: {train_content}")
print(f"  Class: {train_content.split()[0]}")

print(f"\nSample valid label: {sample_valid.name}")
print(f"  Content: {valid_content}")
print(f"  Class: {valid_content.split()[0]}")

if train_content.startswith('0 ') and valid_content.startswith('0 '):
    print("\n✅ All labels fixed successfully!")
else:
    print("\n⚠️ Some labels may still need fixing")

print("\n" + "=" * 70)
print("✅ Task 2.1B Complete!")
print("🚀 Ready to train YOLO-OBB model")
