"""
Localizer Module - Predict drone coordinates (lat, lon, alt)
Uses Approximation models (bbox-based predictions)
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
import json
from hw import sprint, get_device


def _force_xgb_cpu(model):
    """Force XGBoost model to predict on CPU to avoid device mismatch."""
    try:
        booster = getattr(model, "get_booster", None)
        if booster:
            booster().set_param({"device": "cpu"})
        else:
            model.set_param({"device": "cpu"})
    except Exception:
        pass
    return model

class ApproximationLocalizer:
    """
    Localization using approximation approach
    Predicts distance, bearing, altitude from bbox → convert to lat/lon
    """
    
    def __init__(self):
        """Initialize localizer with approximation models"""
        sprint(f"Loading Approximation models...")
        
        # Resolve path relative to project root
        project_root = Path(__file__).resolve().parent.parent
        model_dir = project_root / 'models' / 'models_approximation'
        
        # Load bbox-based models
        self.model_distance = _force_xgb_cpu(pickle.load(open(model_dir / 'bbox_to_distance.pkl', 'rb')))
        self.model_bearing_sin = _force_xgb_cpu(pickle.load(open(model_dir / 'bbox_to_bearing_sin.pkl', 'rb')))
        self.model_bearing_cos = _force_xgb_cpu(pickle.load(open(model_dir / 'bbox_to_bearing_cos.pkl', 'rb')))
        self.model_altitude = _force_xgb_cpu(pickle.load(open(model_dir / 'bbox_to_altitude.pkl', 'rb')))
        
        # Load correction parameters
        with open(model_dir / 'correction_params.json', 'r') as f:
            self.correction_params = json.load(f)
        
        # Load feature columns
        with open(model_dir / 'bbox_features.json', 'r') as f:
            self.feature_columns = json.load(f)
        
        sprint(f"Models loaded successfully")
        print(f"   - Features: {self.feature_columns}")
        
        # Camera location (from training data)
        self.camera_lat = 14.3048539  # Approximate
        self.camera_lon = 101.1728033
        
        # Coordinate smoothing
        self.track_history = {}
        self.smooth_window = 5
    
    def extract_bbox_features(self, bbox, frame_shape, confidence):
        """
        Extract YOLO bbox features
        
        Args:
            bbox: [x1, y1, x2, y2]
            frame_shape: (height, width)
            confidence: detection confidence
            
        Returns:
            Feature array matching bbox_features.json
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Calculate YOLO format features
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Normalize to image dimensions
        yolo_cx = cx / width
        yolo_cy = cy / height
        yolo_w = w / width
        yolo_h = h / height
        
        # Additional features
        yolo_area = yolo_w * yolo_h
        yolo_aspect_ratio = yolo_w / yolo_h if yolo_h > 0 else 0
        
        # Distance and angle from image center
        center_x = 0.5
        center_y = 0.5
        dx = yolo_cx - center_x
        dy = yolo_cy - center_y
        
        yolo_dist_from_center = np.sqrt(dx**2 + dy**2)
        yolo_angle_from_center = np.arctan2(dy, dx)
        
        # Return features in correct order
        features = {
            'yolo_cx': yolo_cx,
            'yolo_cy': yolo_cy,
            'yolo_w': yolo_w,
            'yolo_h': yolo_h,
            'yolo_conf': confidence,
            'yolo_area': yolo_area,
            'yolo_aspect_ratio': yolo_aspect_ratio,
            'yolo_dist_from_center': yolo_dist_from_center,
            'yolo_angle_from_center': yolo_angle_from_center,
        }
        
        # Convert to array in correct order
        feature_array = np.array([features[col] for col in self.feature_columns])
        
        return feature_array
    
    def predict(self, bbox, frame_shape, confidence, track_id=None):
        """
        Predict coordinates for a detection
        
        Args:
            bbox: [x1, y1, x2, y2]
            frame_shape: (height, width, channels)
            confidence: detection confidence
            track_id: optional track ID for smoothing
            
        Returns:
            dict with lat, lon, alt
        """
        # Extract features
        X = self.extract_bbox_features(bbox, frame_shape, confidence).reshape(1, -1)
        
        # Predict distance, bearing, altitude
        distance_m = float(self.model_distance.predict(X)[0])
        bearing_sin = float(self.model_bearing_sin.predict(X)[0])
        bearing_cos = float(self.model_bearing_cos.predict(X)[0])
        altitude_m = float(self.model_altitude.predict(X)[0])
        
        # Convert sin/cos back to bearing angle
        bearing_rad = np.arctan2(bearing_sin, bearing_cos)
        bearing_deg = np.degrees(bearing_rad)
        
        # Convert distance + bearing to lat/lon offset
        # Approximate conversion (1 degree ≈ 111km)
        lat_offset = (distance_m * np.cos(bearing_rad)) / 111000
        lon_offset = (distance_m * np.sin(bearing_rad)) / (111000 * np.cos(np.radians(self.camera_lat)))
        
        lat = self.camera_lat + lat_offset
        lon = self.camera_lon + lon_offset
        alt = altitude_m
        
        # Apply smoothing if track_id provided
        if track_id is not None:
            lat, lon, alt = self.smooth_coordinates(track_id, (lat, lon, alt))
        
        return {
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'distance_m': distance_m,
            'bearing_deg': bearing_deg
        }
    
    def smooth_coordinates(self, track_id, coords):
        """Apply moving average smoothing"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        self.track_history[track_id].append(coords)
        
        if len(self.track_history[track_id]) > self.smooth_window:
            self.track_history[track_id] = self.track_history[track_id][-self.smooth_window:]
        
        history = np.array(self.track_history[track_id])
        smoothed = np.mean(history, axis=0)
        
        return tuple(smoothed)
    
    def predict_batch(self, detections, frame_shape):
        """
        Predict coordinates for multiple detections
        
        Args:
            detections: List of (bbox, confidence, track_id)
            frame_shape: Frame shape
            
        Returns:
            List of coordinate predictions
        """
        results = []
        for bbox, conf, track_id in detections:
            coords = self.predict(bbox, frame_shape, conf, track_id)
            results.append(coords)
        return results


def test_localizer():
    """Test localizer on sample detections"""
    print("="*70)
    print("Testing Approximation Localizer")
    print("="*70)
    
    # Initialize localizer
    localizer = ApproximationLocalizer()
    
    # Load video for frame shape
    cap = cv2.VideoCapture(str(Path(__file__).resolve().parent.parent / 'videos' / 'P3_VIDEO.mp4'))
    ret, frame = cap.read()
    frame_shape = frame.shape
    cap.release()
    
    sprint(f"\n[INFO] Frame shape: {frame_shape}")
    
    # Test with sample bounding boxes
    # (x1, y1, x2, y2, confidence)
    test_boxes = [
        ([100, 100, 200, 200], 0.5, 1),    # Top-left
        ([800, 400, 900, 500], 0.7, 2),    # Center
        ([1600, 800, 1700, 900], 0.6, 3),  # Bottom-right
    ]
    
    sprint(f"\n[INFO] Testing predictions:")
    print("-" * 70)
    
    for bbox, conf, track_id in test_boxes:
        coords = localizer.predict(bbox, frame_shape, conf, track_id)
        
        x1, y1, x2, y2 = bbox
        print(f"Track {track_id}: bbox=[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] conf={conf:.2f}")
        print(f"  → lat={coords['lat']:.6f}, lon={coords['lon']:.6f}, alt={coords['alt']:.2f}")
    
    # Test smoothing
    sprint(f"\n[INFO] Testing coordinate smoothing (Track 1):")
    print("-" * 70)
    
    bbox = [100, 100, 200, 200]
    for i in range(5):
        # Simulate slight bbox changes
        bbox_noisy = [b + np.random.randint(-5, 5) for b in bbox]
        coords = localizer.predict(bbox_noisy, frame_shape, 0.5, track_id=1)
        print(f"Frame {i+1}: lat={coords['lat']:.6f}, lon={coords['lon']:.6f}, alt={coords['alt']:.2f}")
    
    sprint("\n[OK] Localizer test completed!")


if __name__ == "__main__":
    test_localizer()
