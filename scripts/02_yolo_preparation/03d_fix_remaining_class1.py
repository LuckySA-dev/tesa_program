"""Fix any remaining class=1 labels"""
from pathlib import Path

YOLO_DIR = Path("yolo_dataset")

fixed = 0
for label_file in list((YOLO_DIR / "train" / "labels").glob("*.txt")) + list((YOLO_DIR / "valid" / "labels").glob("*.txt")):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.strip().startswith('1 '):
            new_lines.append('0 ' + line.strip()[2:])
            fixed += 1
        else:
            new_lines.append(line.strip())
    
    with open(label_file, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

print(f"Fixed {fixed} labels with class=1")
