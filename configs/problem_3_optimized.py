"""
Optimized Detection Settings for P3 Video
Based on threshold analysis results
"""

# Optimal detection parameters for P3_VIDEO.mp4
# Based on threshold and NMS analysis
DETECTION_CONFIG = {
    'conf_threshold': 0.15,  # Optimal: catches both drones (98% detection rate)
    'iou_threshold': 0.6,    # NMS threshold (7/8 frames = 2 drones)
    'max_det': 10,           # Maximum detections per frame
    'agnostic_nms': True,    # Class-agnostic NMS
    
    # Expected number of drones
    'expected_drones': 2,    # Video has 2 drones
    'max_drones': 4,         # Alert if more than this
}

# Tracking parameters (for next step)
TRACKING_CONFIG = {
    'method': 'botsort',  # or 'bytetrack', 'deepsort'
    'persist': True,
    'track_high_thresh': 0.4,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.5,
    'track_buffer': 30,
    'match_thresh': 0.8,
}

# Visualization
VISUALIZATION_CONFIG = {
    'colors': {
        1: (0, 0, 255),    # Red - Drone 1
        2: (0, 255, 0),    # Green - Drone 2
        3: (255, 0, 0),    # Blue - Drone 3 (if any)
        4: (0, 255, 255),  # Yellow - Drone 4 (if any)
    },
    'box_thickness': 2,
    'text_scale': 0.6,
    'text_color': (255, 255, 255),
    'bg_alpha': 0.6,
}

print("📋 Optimized Configuration Loaded")
print(f"Detection threshold: {DETECTION_CONFIG['conf_threshold']}")
print(f"Expected drones: {DETECTION_CONFIG['expected_drones']}")
print(f"Tracking method: {TRACKING_CONFIG['method']}")
