"""
Phase 2 - Task 2.1C: Convert OBB to Standard YOLO Format
Remove angle column (6th value) from all labels
"""

from pathlib import Path

print("=" * 70)
print("Phase 2 - Task 2.1C: Convert OBB to Standard YOLO")
print("=" * 70)

YOLO_DIR = Path("yolo_dataset")

# Process train labels
print("\n🔧 Converting train labels (removing angle column)...")
train_labels = list((YOLO_DIR / "train" / "labels").glob("*.txt"))
converted_train = 0

for label_file in train_labels:
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 6:  # OBB format: class cx cy w h angle
            # Keep only: class cx cy w h
            new_line = ' '.join(parts[:5])
            new_lines.append(new_line)
            converted_train += 1
        elif len(parts) == 5:  # Already standard format
            new_lines.append(line.strip())
        else:
            print(f"⚠️ Unexpected format in {label_file.name}: {line.strip()}")
    
    with open(label_file, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

print(f"✅ Converted {converted_train} train labels")

# Process valid labels
print("\n🔧 Converting validation labels (removing angle column)...")
valid_labels = list((YOLO_DIR / "valid" / "labels").glob("*.txt"))
converted_valid = 0

for label_file in valid_labels:
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 6:  # OBB format: class cx cy w h angle
            # Keep only: class cx cy w h
            new_line = ' '.join(parts[:5])
            new_lines.append(new_line)
            converted_valid += 1
        elif len(parts) == 5:  # Already standard format
            new_lines.append(line.strip())
        else:
            print(f"⚠️ Unexpected format in {label_file.name}: {line.strip()}")
    
    with open(label_file, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

print(f"✅ Converted {converted_valid} validation labels")

# Verify conversion
print("\n🔍 Verifying conversion...")
sample_train = list((YOLO_DIR / "train" / "labels").glob("*.txt"))[0]
sample_valid = list((YOLO_DIR / "valid" / "labels").glob("*.txt"))[0]

with open(sample_train, 'r') as f:
    train_lines = f.readlines()
with open(sample_valid, 'r') as f:
    valid_lines = f.readlines()

print(f"\nSample train label: {sample_train.name}")
for line in train_lines[:2]:  # Show first 2 lines
    parts = line.strip().split()
    print(f"  {line.strip()} → {len(parts)} columns")

print(f"\nSample valid label: {sample_valid.name}")
for line in valid_lines[:2]:  # Show first 2 lines
    parts = line.strip().split()
    print(f"  {line.strip()} → {len(parts)} columns")

# Check all are 5 columns
all_valid = True
for label_file in list((YOLO_DIR / "train" / "labels").glob("*.txt")) + list((YOLO_DIR / "valid" / "labels").glob("*.txt")):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 and len(parts) > 0:
                print(f"❌ {label_file.name}: {len(parts)} columns - {line.strip()}")
                all_valid = False

if all_valid:
    print("\n✅ All labels converted successfully! (5 columns: class cx cy w h)")
else:
    print("\n⚠️ Some labels still have issues")

print("\n" + "=" * 70)
print("✅ Task 2.1C Complete!")
print("🚀 Ready to train standard YOLO model")
