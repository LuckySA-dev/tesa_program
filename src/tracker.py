"""
Tracker Module - Multi-object tracking for drones
Supports ByteTrack, BoT-SORT, and DeepSORT
"""
import cv2
import numpy as np
from pathlib import Path
from hw import sprint

class ByteTracker:
    """
    Simple wrapper for YOLO's built-in ByteTrack/BoT-SORT
    Easy to use and well-optimized
    """
    
    def __init__(self, model, tracker_type='botsort', persist=True, custom_config=None):
        """
        Initialize tracker
        
        Args:
            model: YOLO model instance
            tracker_type: 'bytetrack' or 'botsort'
            persist: Keep track IDs across frames
            custom_config: Path to custom tracker config file
        """
        self.model = model
        self.tracker_type = tracker_type
        self.persist = persist
        self.custom_config = custom_config
        
        sprint(f"Initialized {tracker_type.upper()} tracker")
        print(f"   - Persist: {persist}")
        if custom_config:
            print(f"   - Custom config: {custom_config}")
    
    def track(self, frame, conf=0.15, iou=0.6, device=None):
        """
        Track objects in frame
        
        Args:
            frame: Input frame
            conf: Confidence threshold
            iou: IOU threshold
            device: Compute device ('cpu' or 'cuda')
            
        Returns:
            dict with:
                - boxes: [x1, y1, x2, y2]
                - track_ids: track IDs
                - confidences: confidence scores
        """
        # Use custom config if provided
        tracker_config = self.custom_config if self.custom_config else f'{self.tracker_type}.yaml'
        
        # Build kwargs
        kwargs = dict(
            conf=conf,
            iou=iou,
            tracker=tracker_config,
            persist=self.persist,
            verbose=False,
        )
        if device is not None:
            kwargs["device"] = device
        
        # Run tracking
        results = self.model.track(frame, **kwargs)
        
        # Extract results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
        else:
            boxes = np.array([])
            track_ids = np.array([])
            confidences = np.array([])
        
        return {
            'boxes': boxes,
            'track_ids': track_ids,
            'confidences': confidences,
            'num_tracks': len(track_ids)
        }


class DeepSORTTracker:
    """
    DeepSORT tracker with Re-ID feature
    Better for identical objects (2 identical drones)
    """
    
    def __init__(self, max_age=30, n_init=3, nn_budget=100):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Max frames to keep lost tracks
            n_init: Min detections before confirming track
            nn_budget: Max samples for Re-ID
        """
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nn_budget=nn_budget,
                embedder="mobilenet",
                embedder_gpu=True,
                half=True
            )
            
            sprint(f"🎯 Initialized DeepSORT tracker")
            print(f"   - Max age: {max_age}")
            print(f"   - N init: {n_init}")
            print(f"   - NN budget: {nn_budget}")
            
        except ImportError:
            sprint("❌ deep-sort-realtime not installed!")
            print("   Run: pip install deep-sort-realtime")
            raise
    
    def track(self, frame, detections):
        """
        Track objects using DeepSORT
        
        Args:
            frame: Input frame
            detections: Detection results from detector
                       {'boxes': [...], 'confidences': [...]}
        
        Returns:
            dict with tracking results
        """
        # Prepare detections for DeepSORT
        # Format: ([x1, y1, w, h], confidence, class)
        deep_sort_detections = []
        
        for box, conf in zip(detections['boxes'], detections['confidences']):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            deep_sort_detections.append(([x1, y1, w, h], conf, 'drone'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)
        
        # Extract confirmed tracks
        boxes = []
        track_ids = []
        confidences = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  # [x1, y1, x2, y2]
            
            boxes.append(bbox)
            track_ids.append(track_id)
            # Note: DeepSORT doesn't keep detection confidence
            confidences.append(1.0)
        
        return {
            'boxes': np.array(boxes),
            'track_ids': np.array(track_ids, dtype=int),
            'confidences': np.array(confidences),
            'num_tracks': len(track_ids)
        }


def visualize_tracks(frame, tracks, colors=None):
    """
    Visualize tracking results
    
    Args:
        frame: Input frame
        tracks: Tracking results
        colors: Dict of {track_id: (B,G,R)}
    """
    frame_viz = frame.copy()
    
    if colors is None:
        # Default colors
        colors = {
            1: (0, 0, 255),    # Red
            2: (0, 255, 0),    # Green
            3: (255, 0, 0),    # Blue
            4: (0, 255, 255),  # Yellow
        }
    
    boxes = tracks['boxes']
    track_ids = tracks['track_ids']
    confidences = tracks['confidences']
    
    for box, track_id, conf in zip(boxes, track_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        
        # Get color for this track
        color = colors.get(track_id, colors[track_id % len(colors)])
        
        # Draw box
        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 3)
        
        # Draw track ID
        label = f"ID: {track_id}"
        cv2.putText(frame_viz, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame_viz


def test_tracking():
    """Test tracking on sample video - use stream mode for persistence"""
    from ultralytics import YOLO
    
    print("="*70)
    print("Testing Tracking Methods")
    print("="*70)
    
    # Load model
    model = YOLO(str(Path(__file__).resolve().parent.parent / 'models' / 'tomorbest.pt'))
    
    sprint(f"🎯 Using BoT-SORT tracker with persist=True")
    
    # Load video
    video_path = str(Path(__file__).resolve().parent.parent / 'videos' / 'P3_VIDEO.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Process frames
    output_dir = Path('outputs/problem_3/tracking_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process continuously with stream mode
    sprint(f"\n📹 Processing video with tracking (stream mode)...")
    
    track_history = {}  # {track_id: [frame_indices]}
    save_frames = [0, 50, 200, 250, 500, 1000, 1100, 1500, 1800]  # Frames to save
    
    # Use generator mode for proper persistence
    results_generator = model.track(
        source=video_path,
        conf=0.15,
        iou=0.6,
        tracker='bytetrack.yaml',  # Use default bytetrack
        persist=True,
        stream=True,
        verbose=False
    )
    
    frame_idx = 0
    for result in results_generator:
        # Extract tracking results
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            # Record track IDs
            for track_id in track_ids:
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(frame_idx)
            
            # Save sample frames
            if frame_idx in save_frames:
                frame_viz = result.plot()  # Use YOLO's built-in visualization
                
                # Add info
                info = f"Frame {frame_idx} ({frame_idx/fps:.1f}s): {len(track_ids)} tracks (IDs: {track_ids})"
                cv2.putText(frame_viz, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                output_path = output_dir / f"track_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(output_path), frame_viz)
                print(f"  Frame {frame_idx:4d} ({frame_idx/fps:5.1f}s): {len(track_ids)} tracks - IDs: {track_ids}")
        
        frame_idx += 1
        
        # Stop after testing enough frames
        if frame_idx > 1850:
            break
    
    # Print track summary
    print("\n" + "="*70)
    sprint("📊 Track Summary:")
    print("="*70)
    for track_id, frames in sorted(track_history.items()):
        duration = (max(frames) - min(frames)) / fps
        print(f"Track ID {track_id}: {len(frames):4d} frames | "
              f"{min(frames)/fps:5.1f}s - {max(frames)/fps:5.1f}s | "
              f"duration: {duration:.1f}s")
    
    sprint(f"\n✅ Tracking test completed!")
    sprint(f"💾 Results saved to: {output_dir}/")


if __name__ == "__main__":
    test_tracking()
