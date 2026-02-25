"""
Analyze Track Patterns - Spatial and Temporal Analysis
Find which tracks belong to which drone
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
import yaml

def analyze_tracks():
    """Analyze track patterns to determine correct merging rules"""
    
    print("="*70)
    print("Track Pattern Analysis")
    print("="*70)
    
    # Load model
    model_path = 'runs/detect/drone_detect_v21_max_data/weights/best.pt'
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Load video
    video_path = 'P3_VIDEO.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {video_path}")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    
    # Track with ByteTrack
    print(f"\n🎯 Running ByteTrack with track_buffer=180...")
    
    # Load custom config
    with open('configs/botsort_custom.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)
    
    # Store track data: {track_id: [(frame_idx, bbox_center_x, bbox_center_y, width, height), ...]}
    track_data = {}
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
        
        # Extract track info
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                if tid not in track_data:
                    track_data[tid] = []
                
                track_data[tid].append((frame_idx, cx, cy, w, h))
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"   Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    
    # Analyze results
    print(f"\n📊 Analysis Results:")
    print(f"   Total unique track IDs: {len(track_data)}")
    print()
    
    # Print track summary
    print("Track Summary:")
    print("-" * 100)
    print(f"{'ID':<5} {'First Frame':<12} {'Last Frame':<12} {'Duration':<10} {'Count':<8} {'Avg Position (x,y)':<25} {'Avg Size (w,h)'}")
    print("-" * 100)
    
    track_summary = []
    for tid in sorted(track_data.keys()):
        frames = [d[0] for d in track_data[tid]]
        first_frame = min(frames)
        last_frame = max(frames)
        duration = last_frame - first_frame + 1
        count = len(frames)
        
        avg_cx = np.mean([d[1] for d in track_data[tid]])
        avg_cy = np.mean([d[2] for d in track_data[tid]])
        avg_w = np.mean([d[3] for d in track_data[tid]])
        avg_h = np.mean([d[4] for d in track_data[tid]])
        
        print(f"{tid:<5} {first_frame:<12} {last_frame:<12} {duration:<10} {count:<8} "
              f"({avg_cx:6.1f}, {avg_cy:6.1f})       ({avg_w:5.1f}, {avg_h:5.1f})")
        
        track_summary.append({
            'id': tid,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'duration': duration,
            'count': count,
            'avg_cx': avg_cx,
            'avg_cy': avg_cy,
            'avg_w': avg_w,
            'avg_h': avg_h
        })
    
    print("-" * 100)
    
    # Temporal overlap analysis
    print("\n🔍 Temporal Overlap Analysis:")
    print("   (Tracks that appear at the same time = different drones)")
    print()
    
    for i, t1 in enumerate(track_summary):
        for j, t2 in enumerate(track_summary):
            if i >= j:
                continue
            
            # Check if tracks overlap in time
            overlap_start = max(t1['first_frame'], t2['first_frame'])
            overlap_end = min(t1['last_frame'], t2['last_frame'])
            
            if overlap_start <= overlap_end:
                overlap_frames = overlap_end - overlap_start + 1
                print(f"   Track {t1['id']} & Track {t2['id']}: OVERLAP {overlap_frames} frames "
                      f"(frames {overlap_start}-{overlap_end}) -> DIFFERENT DRONES")
            else:
                print(f"   Track {t1['id']} & Track {t2['id']}: NO OVERLAP -> Could be SAME DRONE")
    
    # Spatial distance analysis
    print("\n📏 Spatial Distance Analysis:")
    print("   (Average position difference between tracks)")
    print()
    
    for i, t1 in enumerate(track_summary):
        for j, t2 in enumerate(track_summary):
            if i >= j:
                continue
            
            dx = abs(t1['avg_cx'] - t2['avg_cx'])
            dy = abs(t1['avg_cy'] - t2['avg_cy'])
            distance = np.sqrt(dx**2 + dy**2)
            
            print(f"   Track {t1['id']} & Track {t2['id']}: {distance:.1f} pixels apart")
            if distance < 300:
                print(f"      -> CLOSE: Likely SAME DRONE (different time periods)")
            else:
                print(f"      -> FAR: Likely DIFFERENT DRONES")
    
    # Suggest merge rules
    print("\n💡 Suggested Merge Rules:")
    print("   Based on temporal and spatial analysis:")
    print()
    
    # Find groups
    # Rule: Tracks that DON'T overlap in time AND are spatially close = same drone
    groups = {}
    for tid in sorted(track_data.keys()):
        groups[tid] = [tid]
    
    for i, t1 in enumerate(track_summary):
        for j, t2 in enumerate(track_summary):
            if i >= j:
                continue
            
            # Check temporal overlap
            overlap_start = max(t1['first_frame'], t2['first_frame'])
            overlap_end = min(t1['last_frame'], t2['last_frame'])
            no_overlap = overlap_start > overlap_end
            
            # Check spatial distance
            dx = abs(t1['avg_cx'] - t2['avg_cx'])
            dy = abs(t1['avg_cy'] - t2['avg_cy'])
            distance = np.sqrt(dx**2 + dy**2)
            spatially_close = distance < 300
            
            # If no overlap AND spatially close -> same drone
            if no_overlap and spatially_close:
                # Merge groups
                tid1 = t1['id']
                tid2 = t2['id']
                if tid1 in groups:
                    groups[tid1].append(tid2)
                    if tid2 in groups:
                        del groups[tid2]
    
    # Print suggested groups
    drone_num = 1
    merge_rules = {}
    for master_id, group_ids in sorted(groups.items()):
        if len(group_ids) > 1 or master_id == min(track_data.keys()):
            print(f"   Drone {drone_num}: Tracks {group_ids} -> Track ID {master_id}")
            for tid in group_ids:
                merge_rules[tid] = master_id
            drone_num += 1
    
    print(f"\n   Merge rules dict: {merge_rules}")
    
    # Save detailed track data for visualization
    output_dir = Path('outputs/problem_3/track_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'track_data.npy', track_data)
    print(f"\n💾 Saved track data to: {output_dir / 'track_data.npy'}")
    
    return track_data, merge_rules


if __name__ == "__main__":
    track_data, merge_rules = analyze_tracks()
    print("\n✅ Analysis complete!")
