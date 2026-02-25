"""
Track Merging Script - Merge fragmented tracks into 2 main drone tracks
Based on analysis:
- Track 1, 2, 3 → Drone 1 (merge into ID 1)
- Track 4, 5 → Drone 2 (merge into ID 4)
"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detector import DroneDetector
from src.tracker import ByteTracker
from src.localizer import ApproximationLocalizer
from src.visualizer import DroneVisualizer
from tqdm import tqdm
import time


class TrackMerger:
    """Merge multiple track IDs into consolidated tracks"""
    
    def __init__(self, merge_rules):
        """
        Args:
            merge_rules (dict): Mapping of old_track_id -> new_track_id
                               e.g., {2: 1, 3: 1, 5: 4}
        """
        self.merge_rules = merge_rules
        
    def merge_track_id(self, track_id):
        """Convert track ID according to merge rules"""
        return self.merge_rules.get(track_id, track_id)


def process_video_with_merged_tracks(
    video_path,
    output_path,
    model_path,
    tracker_config,
    merge_rules,
    conf_threshold=0.15,
    iou_threshold=0.6,
    start_frame=0,
    end_frame=None
):
    """
    Process video and apply track merging
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model_path: YOLO model path
        tracker_config: Tracker configuration file
        merge_rules: Dictionary mapping old IDs to new IDs
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IOU threshold
        start_frame: Starting frame
        end_frame: Ending frame (None = all)
    """
    
    print("=" * 70)
    print("Track Merging Pipeline")
    print("=" * 70)
    
    # Initialize components
    print("\n📦 Loading components...")
    detector = DroneDetector(model_path, conf_threshold, iou_threshold)
    tracker = ByteTracker(model=detector.model, custom_config=tracker_config)
    localizer = ApproximationLocalizer()
    visualizer = DroneVisualizer()
    merger = TrackMerger(merge_rules)
    
    print(f"✅ Merge rules: {merge_rules}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame is None:
        end_frame = total_frames
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Processing: Frame {start_frame} to {end_frame}")
    
    # Setup video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Processing loop
    print(f"\n🎯 Processing with track merging...")
    
    frame_idx = start_frame
    processed_frames = 0
    frames_with_detections = 0
    total_detections = 0
    merged_track_ids = set()
    processing_times = []
    
    pbar = tqdm(total=end_frame - start_frame, desc="Processing")
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Detection and tracking - Use YOLO track directly
        tracked_results = detector.model.track(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker=tracker_config,
            persist=True,
            verbose=False
        )
        
        # Prepare merged results
        merged_results = []
        
        if tracked_results and len(tracked_results) > 0:
            frames_with_detections += 1
            
            for result in tracked_results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get original track ID
                        original_track_id = int(boxes.id[i].item()) if boxes.id is not None else -1
                        
                        if original_track_id == -1:
                            continue
                        
                        # Merge track ID
                        merged_track_id = merger.merge_track_id(original_track_id)
                        merged_track_ids.add(merged_track_id)
                        total_detections += 1
                        
                        # Get bbox and confidence
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].item())
                        
                        # GPS prediction
                        gps = localizer.predict(
                            bbox=bbox,
                            frame_shape=frame.shape,
                            confidence=conf,
                            track_id=merged_track_id
                        )
                        
                        merged_results.append({
                            'track_id': merged_track_id,
                            'bbox': bbox,
                            'confidence': conf,
                            'gps': gps
                        })
        
        # Visualize with merged tracks
        frame_vis = frame.copy()
        for result in merged_results:
            frame_vis = visualizer.draw_full_annotation(
                frame_vis,
                result['bbox'],
                result['track_id'],
                result['gps']
            )
        
        # Draw frame info
        frame_vis = visualizer.draw_frame_info(
            frame_vis,
            frame_idx,
            fps,
            len(merged_results)
        )
        
        out.write(frame_vis)
        
        processing_time = (time.time() - start_time) * 1000
        processing_times.append(processing_time)
        
        processed_frames += 1
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Statistics
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1000.0 / avg_processing_time if avg_processing_time > 0 else 0
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"\n📊 Statistics:")
    print(f"   Total Frames Processed: {processed_frames}")
    print(f"   Frames with Detections: {frames_with_detections} ({frames_with_detections/processed_frames*100:.1f}%)")
    print(f"   Total Detections: {total_detections:,}")
    print(f"   Unique Track IDs (Merged): {len(merged_track_ids)} {sorted(merged_track_ids)}")
    print(f"   Average Processing Time: {avg_processing_time:.1f} ms/frame ({avg_fps:.1f} FPS)")
    print(f"\n💾 Output:")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print("=" * 70)
    print("\n✅ Track merging completed successfully!")


if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = 'P3_VIDEO.mp4'
    OUTPUT_PATH = 'outputs/problem_3/P3_OUTPUT_V21_BOTSORT_MERGED_2TRACKS.mp4'
    MODEL_PATH = 'runs/detect/drone_detect_v21_max_data/weights/best.pt'
    TRACKER_CONFIG = 'configs/botsort_custom.yaml'
    
    # Merge rules based on track analysis
    # Track 1, 2, 3 → Drone 1 (keep as ID 1)
    # Track 4, 5 → Drone 2 (keep as ID 4)
    MERGE_RULES = {
        1: 1,  # Keep Track 1
        2: 1,  # Merge Track 2 → 1
        3: 1,  # Merge Track 3 → 1
        4: 4,  # Keep Track 4
        5: 4,  # Merge Track 5 → 4
    }
    
    # Process full video
    START_FRAME = 0
    END_FRAME = None  # Process all frames
    
    process_video_with_merged_tracks(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        model_path=MODEL_PATH,
        tracker_config=TRACKER_CONFIG,
        merge_rules=MERGE_RULES,
        conf_threshold=0.15,
        iou_threshold=0.6,
        start_frame=START_FRAME,
        end_frame=END_FRAME
    )
