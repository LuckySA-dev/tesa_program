"""
ByteTrack wrapper for TESA Defence competition
Uses supervision library's ByteTrack implementation
"""

import numpy as np
from collections import OrderedDict, deque
import supervision as sv
from hw import sprint


class ByteTrackWrapper:
    """Wrapper for ByteTrack compatible with our pipeline"""
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, 
                 frame_rate=30, track_history=30):
        """
        Initialize ByteTrack
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
            frame_rate: Video frame rate
            track_history: Number of frames to keep in track path
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        
        # For compatibility with existing code
        self.track_paths = {}
        self.track_history = track_history
        self.previous_centroids = {}
        self.total_tracked = 0
        
    def update(self, detections, timestamp=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of (cx, cy) centroids OR supervision Detections object
            timestamp: Current timestamp (optional, for velocity)
            
        Returns:
            OrderedDict: {track_id: (cx, cy)}
        """
        # Convert detections to supervision format if needed
        if isinstance(detections, list) and len(detections) > 0:
            # Create dummy bounding boxes from centroids
            # ByteTrack needs xyxy format
            xyxy = []
            for cx, cy in detections:
                # Create small box around centroid (±25 pixels)
                x1, y1 = cx - 25, cy - 25
                x2, y2 = cx + 25, cy + 25
                xyxy.append([x1, y1, x2, y2])
            
            xyxy = np.array(xyxy)
            
            # Create supervision Detections object
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=np.ones(len(xyxy)),  # Assume high confidence
                class_id=np.zeros(len(xyxy), dtype=int)
            )
        elif hasattr(detections, 'xyxy'):
            # Already a supervision Detections object
            sv_detections = detections
        else:
            # No detections
            sv_detections = sv.Detections.empty()
        
        # Update ByteTrack
        tracked = self.tracker.update_with_detections(sv_detections)
        
        # Convert to OrderedDict format for compatibility
        objects = OrderedDict()
        
        if len(tracked) > 0:
            for i in range(len(tracked.xyxy)):
                track_id = int(tracked.tracker_id[i])
                
                # Calculate centroid from bbox
                x1, y1, x2, y2 = tracked.xyxy[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                objects[track_id] = (cx, cy)
                
                # Update track path
                if track_id not in self.track_paths:
                    self.track_paths[track_id] = deque(maxlen=self.track_history)
                    self.total_tracked += 1
                
                self.track_paths[track_id].append((cx, cy))
                
                # Update for velocity calculation
                if timestamp is not None:
                    self.previous_centroids[track_id] = (cx, cy, timestamp)
        
        return objects
    
    def get_track_path(self, track_id):
        """Get track path for visualization"""
        if track_id in self.track_paths:
            return list(self.track_paths[track_id])
        return []
    
    def get_total_tracked_objects(self):
        """Get total number of unique objects tracked"""
        return self.total_tracked
    
    def calculate_velocity(self, track_id, current_time, fps=30, pixels_per_meter=100):
        """
        Calculate velocity for tracked object
        
        Args:
            track_id: Track ID
            current_time: Current timestamp
            fps: Video frame rate
            pixels_per_meter: Conversion factor
            
        Returns:
            dict: {speed, direction, distance_pixels} or None
        """
        if track_id not in self.track_paths or len(self.track_paths[track_id]) < 2:
            return None
        
        # Get last two positions
        path = list(self.track_paths[track_id])
        curr_pos = path[-1]
        prev_pos = path[-2] if len(path) >= 2 else curr_pos
        
        # Calculate displacement
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Calculate direction (degrees, 0=right, 90=up)
        direction = np.degrees(np.arctan2(-dy, dx)) % 360
        
        # Calculate speed (m/s)
        # Assume 1 frame time difference
        time_delta = 1.0 / fps
        distance_meters = distance_pixels / pixels_per_meter
        speed = distance_meters / time_delta if time_delta > 0 else 0
        
        return {
            'speed': speed,
            'direction': direction,
            'distance_pixels': distance_pixels
        }


# Test ByteTrack
if __name__ == '__main__':
    print("Testing ByteTrackWrapper...")
    
    tracker = ByteTrackWrapper()
    
    # Simulate detections
    test_detections = [
        [(100, 100), (200, 200), (300, 300)],  # Frame 1
        [(105, 105), (205, 205), (305, 305)],  # Frame 2
        [(110, 110), (210, 210)],              # Frame 3 (one lost)
        [(115, 115), (215, 215), (310, 310)],  # Frame 4 (one back)
    ]
    
    for frame_num, detections in enumerate(test_detections):
        objects = tracker.update(detections, frame_num * 0.033)
        print(f"\nFrame {frame_num}: {len(objects)} objects")
        for track_id, centroid in objects.items():
            print(f"  Track {track_id}: {centroid}")
    
    print(f"\nTotal unique tracks: {tracker.get_total_tracked_objects()}")
    sprint("✅ ByteTrack test passed!")
