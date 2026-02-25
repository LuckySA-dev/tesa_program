"""
Quick check: What track IDs appear with IOU=0.3
"""
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO

model = YOLO('runs/detect/drone_detect_v21_max_data/weights/best.pt')

cap = cv2.VideoCapture('P3_VIDEO.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_track_ids = set()
track_count = {}
track_first = {}
track_last = {}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(
        frame,
        conf=0.10,
        iou=0.3,  # Same as pipeline
        persist=True,
        tracker='bytetrack.yaml',
        verbose=False
    )
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for tid in track_ids:
            all_track_ids.add(tid)
            track_count[tid] = track_count.get(tid, 0) + 1
            
            if tid not in track_first:
                track_first[tid] = frame_idx
            track_last[tid] = frame_idx
    
    frame_idx += 1
    if frame_idx % 200 == 0:
        print(f"Frame {frame_idx}/{total_frames} - IDs so far: {sorted(all_track_ids)}")

cap.release()

print(f"\n📊 Track IDs with IOU=0.3:")
print(f"   Total: {len(all_track_ids)}")
print(f"   IDs: {sorted(all_track_ids)}")
print()
print("-" * 80)
print(f"{'ID':<5} {'First':<10} {'Last':<10} {'Count':<10}")
print("-" * 80)
for tid in sorted(all_track_ids):
    print(f"{tid:<5} {track_first[tid]:<10} {track_last[tid]:<10} {track_count[tid]:<10}")
print("-" * 80)
