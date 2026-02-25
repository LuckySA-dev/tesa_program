"""
Analyze specific frame range to understand track ID switching
Focus on frames 1380-1420 where track ID changes from 1 to 4
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO

def analyze_frame_range(video_path, model_path, tracker_config, start_frame, end_frame):
    """
    Analyze tracking behavior in specific frame range
    """
    print("="*70)
    print(f"Analyzing Frames {start_frame} to {end_frame}")
    print("="*70)
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"\n📹 Video FPS: {fps:.2f}")
    print(f"⏱️  Time range: {start_frame/fps:.1f}s - {end_frame/fps:.1f}s")
    print(f"\n{'Frame':<8} {'Time(s)':<8} {'Track IDs':<20} {'Confidences':<30} {'Bbox Centers'}")
    print("-"*100)
    
    # Track history
    track_history = {}
    
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run tracking
        results = model.track(
            frame,
            conf=0.10,
            iou=0.6,
            tracker=tracker_config,
            persist=True,
            verbose=False
        )
        
        # Extract tracking info
        track_ids = []
        confidences = []
        centers = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                tid = int(boxes.id[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                
                # Calculate center
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                track_ids.append(tid)
                confidences.append(conf)
                centers.append((int(cx), int(cy)))
                
                # Record in history
                if tid not in track_history:
                    track_history[tid] = {'frames': [], 'confs': [], 'centers': []}
                
                track_history[tid]['frames'].append(frame_idx)
                track_history[tid]['confs'].append(conf)
                track_history[tid]['centers'].append((int(cx), int(cy)))
        
        # Print frame info
        time_sec = frame_idx / fps
        ids_str = str(track_ids) if track_ids else "None"
        confs_str = ", ".join([f"{c:.3f}" for c in confidences]) if confidences else "N/A"
        centers_str = ", ".join([f"({c[0]},{c[1]})" for c in centers]) if centers else "N/A"
        
        print(f"{frame_idx:<8} {time_sec:<8.1f} {ids_str:<20} {confs_str:<30} {centers_str}")
        
        frame_idx += 1
    
    cap.release()
    
    # Print track summary
    print("\n" + "="*70)
    print("Track History Summary:")
    print("="*70)
    
    for tid in sorted(track_history.keys()):
        frames = track_history[tid]['frames']
        confs = track_history[tid]['confs']
        centers = track_history[tid]['centers']
        
        print(f"\nTrack ID {tid}:")
        print(f"  Frames: {min(frames)} - {max(frames)} ({len(frames)} frames)")
        print(f"  Time: {min(frames)/fps:.1f}s - {max(frames)/fps:.1f}s")
        print(f"  Avg Confidence: {np.mean(confs):.3f} (min: {min(confs):.3f}, max: {max(confs):.3f})")
        print(f"  Position change: ({centers[0][0]},{centers[0][1]}) → ({centers[-1][0]},{centers[-1][1]})")
        
        # Check for gaps
        frame_nums = sorted(frames)
        gaps = []
        for i in range(len(frame_nums)-1):
            gap = frame_nums[i+1] - frame_nums[i]
            if gap > 1:
                gaps.append((frame_nums[i], frame_nums[i+1], gap-1))
        
        if gaps:
            print(f"  ⚠️  Gaps detected: {len(gaps)}")
            for g in gaps[:3]:  # Show first 3 gaps
                print(f"     Gap between frame {g[0]} and {g[1]} ({g[2]} frames missing)")
        else:
            print(f"  ✅ Continuous (no gaps)")
    
    # Analyze ID switching
    print("\n" + "="*70)
    print("ID Switching Analysis:")
    print("="*70)
    
    # Find frames where both IDs appear
    all_frames = set()
    for tid in track_history:
        all_frames.update(track_history[tid]['frames'])
    
    for frame in sorted(all_frames):
        ids_in_frame = [tid for tid in track_history if frame in track_history[tid]['frames']]
        if len(ids_in_frame) > 1:
            print(f"Frame {frame} ({frame/fps:.1f}s): Multiple IDs detected: {ids_in_frame}")
    
    # Check transition points
    for tid in sorted(track_history.keys()):
        frames = sorted(track_history[tid]['frames'])
        if frames:
            print(f"\nTrack {tid} transitions:")
            print(f"  First appearance: Frame {frames[0]} ({frames[0]/fps:.1f}s)")
            print(f"  Last appearance: Frame {frames[-1]} ({frames[-1]/fps:.1f}s)")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    VIDEO_PATH = 'P3_VIDEO.mp4'
    MODEL_PATH = 'runs/detect/drone_detect_v21_max_data/weights/best.pt'
    TRACKER_CONFIG = 'configs/botsort_custom.yaml'
    
    # Analyze wider range to catch ID switching (54-57s = frames 1350-1425)
    START_FRAME = 1320  # 52.8s - Earlier to see Track 4
    END_FRAME = 1450    # 58.0s - Later to see full transition
    
    analyze_frame_range(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        tracker_config=TRACKER_CONFIG,
        start_frame=START_FRAME,
        end_frame=END_FRAME
    )
