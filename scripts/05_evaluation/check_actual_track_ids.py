"""
Check what track IDs actually appear in the video
"""
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO

def check_track_ids():
    """Check all track IDs that appear"""
    
    print("="*70)
    print("Checking Actual Track IDs")
    print("="*70)
    
    # Load model
    model_path = 'runs/detect/drone_detect_v21_max_data/weights/best.pt'
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Load video
    video_path = 'P3_VIDEO.mp4'
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {video_path} ({total_frames} frames)")
    
    # Track all IDs
    all_track_ids = set()
    track_first_seen = {}
    track_last_seen = {}
    track_count = {}
    
    print(f"\n🎯 Running ByteTrack...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track
        results = model.track(
            frame,
            conf=0.10,
            iou=0.6,
            persist=True,
            tracker='bytetrack.yaml',
            verbose=False
        )
        
        # Extract track IDs
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for tid in track_ids:
                all_track_ids.add(tid)
                
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx
                
                track_count[tid] = track_count.get(tid, 0) + 1
        
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"   Processed {frame_idx}/{total_frames} frames... ({len(all_track_ids)} IDs so far)")
    
    cap.release()
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Total unique track IDs: {len(all_track_ids)}")
    print(f"   Track IDs: {sorted(all_track_ids)}")
    print()
    
    print("Track Details:")
    print("-" * 80)
    print(f"{'ID':<5} {'First Frame':<15} {'Last Frame':<15} {'Count':<10}")
    print("-" * 80)
    
    for tid in sorted(all_track_ids):
        print(f"{tid:<5} {track_first_seen[tid]:<15} {track_last_seen[tid]:<15} {track_count[tid]:<10}")
    
    print("-" * 80)
    
    return all_track_ids


if __name__ == "__main__":
    track_ids = check_track_ids()
    print("\n✅ Check complete!")
