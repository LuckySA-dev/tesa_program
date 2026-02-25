"""
Visualizer Module - Draw tracking results and coordinates on frames
"""
import cv2
import numpy as np
from hw import sprint

class DroneVisualizer:
    """
    Visualize drone tracking and localization results
    """
    
    def __init__(self, colors=None, track_history_length=100):
        """
        Initialize visualizer
        
        Args:
            colors: Dict of {track_id: (B,G,R)} colors
            track_history_length: Number of points to keep in tracking path (default: 50)
        """
        # Default colors for different tracks (BGR format)
        if colors is None:
            self.colors = {
                1: (0, 0, 255),      # Red
                2: (0, 255, 0),      # Green
                3: (255, 0, 0),      # Blue
                4: (0, 255, 255),    # Yellow
                5: (255, 0, 255),    # Magenta
                6: (255, 255, 0),    # Cyan
            }
        else:
            self.colors = colors
        
        # Tracking path settings
        self.track_history = {}  # {track_id: [(x, y), (x, y), ...]}
        self.track_history_length = track_history_length  # Max points to keep
        self.path_thickness = 2
        self.point_radius = 3
        
        # Text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.5
        self.text_thickness = 1
        self.text_color = (255, 255, 255)  # White
        self.bg_alpha = 0.6  # Background transparency
        
        # Box settings
        self.box_thickness = 2
    
    def update_tracking_path(self, track_id, bbox):
        """
        Update tracking path with new bbox center position
        
        Args:
            track_id: Track ID
            bbox: [x1, y1, x2, y2]
        """
        # Calculate bbox center
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Initialize history if needed
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        # Add new point
        self.track_history[track_id].append((center_x, center_y))
        
        # Limit history length
        if len(self.track_history[track_id]) > self.track_history_length:
            self.track_history[track_id] = self.track_history[track_id][-self.track_history_length:]
    
    def draw_tracking_path(self, frame, track_id, color=None):
        """
        Draw tracking path (trajectory) for a track
        
        Args:
            frame: Input frame
            track_id: Track ID
            color: Optional color override
        """
        if track_id not in self.track_history:
            return frame
        
        if color is None:
            color = self.get_color(track_id)
        
        points = self.track_history[track_id]
        
        if len(points) < 2:
            return frame
        
        # Draw lines connecting points
        for i in range(1, len(points)):
            # Calculate alpha based on position (older = more transparent)
            alpha = i / len(points)
            
            # Interpolate color with black for fade effect
            line_color = tuple(int(c * alpha) for c in color)
            
            # Draw line
            cv2.line(frame, points[i-1], points[i], line_color, self.path_thickness, cv2.LINE_AA)
        
        # Draw points
        for i, point in enumerate(points):
            alpha = i / len(points)
            point_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, point, self.point_radius, point_color, -1, cv2.LINE_AA)
        
        # Draw current position (brightest)
        if len(points) > 0:
            cv2.circle(frame, points[-1], self.point_radius + 2, color, -1, cv2.LINE_AA)
        
        return frame
    
    def get_color(self, track_id):
        """Get color for a track ID"""
        if track_id in self.colors:
            return self.colors[track_id]
        else:
            # Cycle through colors
            color_list = list(self.colors.values())
            return color_list[track_id % len(color_list)]
    
    def draw_bbox(self, frame, bbox, track_id, color=None):
        """
        Draw bounding box
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            track_id: Track ID
            color: Optional color override
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        if color is None:
            color = self.get_color(track_id)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        
        return frame
    
    def draw_track_id(self, frame, bbox, track_id, color=None):
        """
        Draw track ID label near bbox
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            track_id: Track ID
            color: Optional color override
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        if color is None:
            color = self.get_color(track_id)
        
        # Label
        label = f"ID: {track_id}"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, self.font, self.text_scale * 1.2, self.text_thickness + 1
        )
        
        # Draw background
        cv2.rectangle(frame,
                     (x1, y1 - label_h - baseline - 5),
                     (x1 + label_w + 5, y1),
                     color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                   self.font, self.text_scale * 1.2, (0, 0, 0), 
                   self.text_thickness + 1, cv2.LINE_AA)
        
        return frame
    
    def draw_coordinates(self, frame, bbox, coords, track_id):
        """
        Draw coordinate information at top-left of bbox
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            coords: Dict with 'lat', 'lon', 'alt'
            track_id: Track ID
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Prepare text lines
        lines = [
            f"track_id: {track_id}",
            f"lat: {coords['lat']:.5f}",
            f"lon: {coords['lon']:.5f}",
            f"alt: {coords['alt']:.2f}"
        ]
        
        # Calculate text box dimensions
        line_height = 18
        max_width = 0
        
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, self.font, self.text_scale, self.text_thickness)
            max_width = max(max_width, w)
        
        # Position (top-left of bbox, moved up)
        text_x = x1
        text_y_start = y1 - 10 - (len(lines) * line_height)
        
        # Make sure text doesn't go off screen
        if text_y_start < 0:
            text_y_start = y2 + 20  # Move below bbox instead
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (text_x - 2, text_y_start - 5),
                     (text_x + max_width + 10, text_y_start + len(lines) * line_height + 5),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0)
        
        # Draw text lines
        for i, line in enumerate(lines):
            y_pos = text_y_start + (i + 1) * line_height
            cv2.putText(frame, line, (text_x, y_pos),
                       self.font, self.text_scale, self.text_color,
                       self.text_thickness, cv2.LINE_AA)
        
        return frame
    
    def draw_full_annotation(self, frame, bbox, track_id, coords):
        """
        Draw complete annotation: bbox + track ID + coordinates + tracking path
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            track_id: Track ID
            coords: Dict with 'lat', 'lon', 'alt'
        """
        color = self.get_color(track_id)
        
        # Update tracking path
        self.update_tracking_path(track_id, bbox)
        
        # Draw tracking path first (so it appears behind bbox)
        frame = self.draw_tracking_path(frame, track_id, color)
        
        # Draw bounding box
        frame = self.draw_bbox(frame, bbox, track_id, color)
        
        # Draw track ID
        frame = self.draw_track_id(frame, bbox, track_id, color)
        
        # Draw coordinates
        frame = self.draw_coordinates(frame, bbox, coords, track_id)
        
        return frame
    
    def draw_info_panel(self, frame, detections_info):
        """
        Draw information panel at top-left corner with all drone data
        
        Args:
            frame: Input frame
            detections_info: List of dicts with keys: track_id, coords
                           coords has: lat, lon, alt
        """
        if not detections_info:
            return frame
        
        # Panel settings (กระชับขึ้น)
        panel_x = 10
        panel_y = 10
        line_height = 20  # ลดจาก 25 -> 20
        padding = 8       # ลดจาก 10 -> 8
        font_scale = 0.45 # ลดจาก 0.5 -> 0.45
        
        # Header (กระชับขึ้น)
        header = " ID |  Latitude  | Longitude  | Alt(m)"
        separator = "-" * 45  # ลดจาก 62 -> 45
        
        # Prepare data lines
        data_lines = []
        for info in sorted(detections_info, key=lambda x: x['track_id']):
            track_id = info['track_id']
            coords = info['coords']
            
            # Format coordinates (กระชับขึ้น)
            lat_str = f"{coords['lat']:10.6f}"   # ลดจาก 11.7 -> 10.6
            lon_str = f"{coords['lon']:10.6f}"   # ลดจาก 11.7 -> 10.6
            alt_str = f"{coords['alt']:6.1f}"    # ลดจาก 7.2 -> 6.1
            
            line = f" {track_id:<2} | {lat_str} | {lon_str} | {alt_str}"
            data_lines.append((line, track_id))
        
        # Calculate panel dimensions
        max_width = max(
            cv2.getTextSize(header, self.font, font_scale, 1)[0][0],
            cv2.getTextSize(separator, self.font, font_scale, 1)[0][0]
        )
        
        panel_height = (len(data_lines) + 3) * line_height + 2 * padding
        panel_width = max_width + 2 * padding
        
        # Draw semi-transparent background (โปร่งแสงมากขึ้น)
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # เปลี่ยนจาก 0.75 -> 0.4
        
        # Draw border (บางลง)
        cv2.rectangle(frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (80, 80, 80), 1)  # ความหนา 2 -> 1, สีอ่อนลง 100 -> 80
        
        # Draw header
        y_pos = panel_y + padding + 15  # ลดจาก 20 -> 15
        cv2.putText(frame, header, (panel_x + padding, y_pos),
                   self.font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_pos += line_height
        cv2.putText(frame, separator, (panel_x + padding, y_pos),
                   self.font, font_scale, (120, 120, 120), 1, cv2.LINE_AA)  # สีอ่อนลง
        
        # Draw data lines with color-coded IDs
        y_pos += line_height
        for line, track_id in data_lines:
            color = self.get_color(track_id)
            cv2.putText(frame, line, (panel_x + padding, y_pos),
                       self.font, font_scale, color, 1, cv2.LINE_AA)
            y_pos += line_height
        
        return frame
    
    def draw_frame_info(self, frame, frame_idx, fps, num_drones):
        """
        Draw frame information
        
        Args:
            frame: Input frame
            frame_idx: Frame number
            fps: Video FPS
            num_drones: Number of detected drones
        """
        height = frame.shape[0]
        
        # Info text
        time_sec = frame_idx / fps if fps > 0 else 0
        info = f"Frame: {frame_idx} | Time: {time_sec:.1f}s | Drones: {num_drones}"
        
        # Position at bottom
        y_pos = height - 20
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            info, self.font, 0.6, 2
        )
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y_pos - text_h - baseline - 5),
                     (15 + text_w, y_pos + baseline), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Text
        cv2.putText(frame, info, (10, y_pos),
                   self.font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame


def test_visualizer():
    """Test visualizer on sample frame"""
    print("="*70)
    print("Testing Drone Visualizer")
    print("="*70)
    
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    
    # Initialize visualizer
    viz = DroneVisualizer()
    
    # Load sample frame
    cap = cv2.VideoCapture(str(project_root / 'videos' / 'P3_VIDEO.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if not ret:
        sprint("❌ Cannot load frame")
        return
    
    sprint(f"✅ Loaded frame 1000")
    
    # Sample detections with coordinates
    detections = [
        {
            'bbox': [500, 200, 600, 300],
            'track_id': 1,
            'coords': {'lat': 14.304850, 'lon': 101.172803, 'alt': 45.5}
        },
        {
            'bbox': [1200, 600, 1300, 700],
            'track_id': 2,
            'coords': {'lat': 14.304920, 'lon': 101.172650, 'alt': 48.2}
        }
    ]
    
    # Draw annotations
    for det in detections:
        frame = viz.draw_full_annotation(
            frame, det['bbox'], det['track_id'], det['coords']
        )
    
    # Draw frame info
    frame = viz.draw_frame_info(frame, 1000, fps, len(detections))
    
    # Save result
    from pathlib import Path
    output_dir = Path('outputs/problem_3/visualizer_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'visualization_sample.jpg'
    cv2.imwrite(str(output_path), frame)
    
    sprint(f"💾 Saved visualization to: {output_path}")
    sprint("✅ Visualizer test completed!")


if __name__ == "__main__":
    test_visualizer()
