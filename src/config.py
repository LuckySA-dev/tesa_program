"""
Configuration for TESA Defence System
======================================

Central configuration file for all system parameters including:
- Model settings
- Tracking parameters
- Visualization settings
- Hardware calibration
- API configuration

Author: TESA Defence Team
Date: November 8, 2025
"""

import os

import cv2
from hw import sprint

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # YOLO-OBB Model
    'detection_model': 'yolov8n-obb.pt',  # Nano for Raspberry Pi 5
    # Alternative models:
    # 'yolov8s-obb.pt' - Small (better accuracy, slower)
    # 'yolov8m-obb.pt' - Medium (development only)
    
    # Detection thresholds
    'confidence_threshold': 0.25,  # 0.0-1.0, lower = more detections
    'iou_threshold': 0.45,         # NMS threshold
    'max_detections': 100,          # Max objects per frame
    
    # Device settings
    'device': 'auto',  # 'auto', 'cuda', or 'cpu'
    'half_precision': False,  # FP16 for speed (set True for deployment)
}

# ============================================================================
# TRACKING CONFIGURATION
# ============================================================================

TRACKING_CONFIG = {
    # Centroid Tracker parameters
    'max_disappeared': 30,      # Max frames object can disappear
    'max_distance': 100,        # Max pixel distance to match objects
    'track_history': 30,        # Frames to keep in path history
    
    # Velocity calculation
    'pixels_per_meter': 100,    # Calibration: pixels to meters
    # Note: Adjust based on camera height and FOV
    # Example: If 100 pixels = 1 meter at ground level
    
    # Smoothing
    'enable_smoothing': True,
    'smoothing_window': 5,      # Frames for moving average
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    # Colors for tracked objects (BGR format)
    'track_colors': [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 128, 0),    # Orange
        (0, 128, 255),    # Light Blue
        (128, 255, 0),    # Light Green
    ],
    
    # Drawing settings
    'bbox_thickness': 2,
    'path_thickness': 2,
    'text_font': cv2.FONT_HERSHEY_SIMPLEX,
    'text_scale': 0.7,
    'text_thickness': 2,
    
    # Path visualization
    'path_fade_effect': True,   # Fade older path points
    'path_max_length': 30,      # Max points to draw
    
    # Overlay
    'show_fps': True,
    'show_drone_count': True,
    'show_velocity': True,
    'show_direction': True,
    'overlay_transparency': 0.4,
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

VIDEO_CONFIG = {
    # Input
    'default_input': 0,  # 0 for webcam, or path to video
    
    # Output
    'output_codec': 'mp4v',  # Video codec
    'output_extension': '.mp4',
    'default_output_dir': 'output',
    
    # Processing
    'frame_skip': 0,  # Process every Nth frame (0 = process all)
    'resize_width': None,  # Resize to width (None = no resize)
    'resize_height': None,  # Resize to height
    
    # Display
    'display_window_name': 'TESA Defence - Drone Tracking System',
    'display_wait_key': 1,  # ms to wait between frames
}

# ============================================================================
# DATA LOGGING
# ============================================================================

LOGGING_CONFIG = {
    # CSV output
    'default_log_filename': 'p1_tracking_log.csv',
    'save_interval': 30,  # Save every N frames
    
    # Log fields
    'log_fields': [
        'frame',
        'timestamp',
        'object_id',
        'center_x',
        'center_y',
        'speed_ms',
        'direction_deg',
        'distance_pixels',
    ],
    
    # Additional logging
    'log_detections': True,
    'log_statistics': True,
}

# ============================================================================
# API CONFIGURATION (for satellite communication)
# ============================================================================

API_CONFIG = {
    # Protocol: rest | mqtt | websocket | grpc | mock
    'protocol': os.environ.get('API_PROTOCOL', 'rest'),

    # Endpoint
    'api_url': os.environ.get('API_URL', 'https://api.tesa.or.th/drone'),
    'api_key': os.environ.get('API_KEY', ''),
    'timeout': int(os.environ.get('API_TIMEOUT', '10')),
    'retries': int(os.environ.get('API_RETRIES', '2')),

    # First alarm
    'first_alarm_threshold': 1,  # Send alarm after N drones detected
    'alarm_cooldown': 30,  # seconds between alarms

    # Data format
    'send_image': True,
    'image_quality': 85,  # JPEG quality 0-100
    'image_max_size': (640, 480),  # Resize before sending

    # Batch sending
    'batch_interval': 5,  # Send batch every N seconds
    'max_batch_size': 100,  # Max records per batch

    # MQTT-specific
    'mqtt_topic': os.environ.get('MQTT_TOPIC', 'tesa/drone'),
    'mqtt_qos': int(os.environ.get('MQTT_QOS', '1')),
}

# ============================================================================
# HARDWARE CONFIGURATION (Raspberry Pi 5)
# ============================================================================

HARDWARE_CONFIG = {
    # Raspberry Pi 5 specific
    'platform': 'raspberry_pi_5',
    'enable_threading': True,
    'num_threads': 4,
    
    # Camera settings
    'camera_index': 0,
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 30,
    
    # Performance optimization
    'enable_gpu': False,  # No AI Board
    'opencv_threads': 4,
    'opencv_optimization': True,
    
    # Temperature monitoring
    'enable_thermal_monitoring': True,
    'max_temperature': 80,  # Celsius
    'throttle_on_overheat': True,
}

# ============================================================================
# DRONE TYPES (for classification)
# ============================================================================

DRONE_TYPES = {
    0: 'DJI_Mavic',
    1: 'DJI_Phantom',
    2: 'DJI_Inspire',
    3: 'Generic_Drone',
    4: 'Fixed_Wing',
    5: 'Racing_Drone',
    # Add more as needed
}

# ============================================================================
# ALERT RULES
# ============================================================================

ALERT_RULES = {
    # Detection thresholds
    'min_drones_for_alert': 1,
    'min_confidence_for_alert': 0.5,
    
    # Behavior detection
    'max_speed_threshold': 20.0,  # m/s
    'min_altitude_estimate': 10,  # meters (if available)
    
    # Zone rules
    'restricted_zones': [
        # Define restricted areas as (x1, y1, x2, y2)
        # Example: (100, 100, 500, 500)
    ],
    
    # Alert levels
    'alert_levels': {
        'low': {'min_drones': 1, 'color': (0, 255, 255)},     # Yellow
        'medium': {'min_drones': 3, 'color': (0, 165, 255)},  # Orange
        'high': {'min_drones': 5, 'color': (0, 0, 255)},      # Red
    }
}

# ============================================================================
# PATHS
# ============================================================================

PATHS = {
    'models_dir': 'models',
    'data_dir': 'data',
    'output_dir': 'output',
    'logs_dir': 'logs',
    'videos_dir': 'videos',
    'images_dir': 'images',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path():
    """Get full path to detection model"""
    import os
    return os.path.join(PATHS['models_dir'], MODEL_CONFIG['detection_model'])

def get_output_path(filename):
    """Get full path for output file"""
    import os
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    return os.path.join(PATHS['output_dir'], filename)

def get_log_path(filename=None):
    """Get full path for log file"""
    import os
    os.makedirs(PATHS['logs_dir'], exist_ok=True)
    if filename is None:
        filename = LOGGING_CONFIG['default_log_filename']
    return os.path.join(PATHS['logs_dir'], filename)

def print_config():
    """Print current configuration"""
    print("="*60)
    print("TESA DEFENCE SYSTEM CONFIGURATION")
    print("="*60)
    sprint(f"\n📦 Model:")
    print(f"   • Detection: {MODEL_CONFIG['detection_model']}")
    print(f"   • Confidence: {MODEL_CONFIG['confidence_threshold']}")
    print(f"   • Device: {MODEL_CONFIG['device']}")
    
    sprint(f"\n🎯 Tracking:")
    print(f"   • Max disappeared: {TRACKING_CONFIG['max_disappeared']} frames")
    print(f"   • Max distance: {TRACKING_CONFIG['max_distance']} pixels")
    print(f"   • Track history: {TRACKING_CONFIG['track_history']} frames")
    
    sprint(f"\n🎨 Visualization:")
    print(f"   • Colors: {len(VISUALIZATION_CONFIG['track_colors'])} unique")
    print(f"   • Show FPS: {VISUALIZATION_CONFIG['show_fps']}")
    print(f"   • Show velocity: {VISUALIZATION_CONFIG['show_velocity']}")
    
    sprint(f"\n💾 Logging:")
    print(f"   • Log file: {LOGGING_CONFIG['default_log_filename']}")
    print(f"   • Save interval: {LOGGING_CONFIG['save_interval']} frames")
    
    sprint(f"\n🖥️  Hardware:")
    print(f"   • Platform: {HARDWARE_CONFIG['platform']}")
    print(f"   • Camera: {HARDWARE_CONFIG['camera_width']}x{HARDWARE_CONFIG['camera_height']} @ {HARDWARE_CONFIG['camera_fps']} FPS")
    print(f"   • GPU: {HARDWARE_CONFIG['enable_gpu']}")
    
    print("="*60 + "\n")

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration values"""
    errors = []
    
    # Check confidence threshold
    if not 0.0 <= MODEL_CONFIG['confidence_threshold'] <= 1.0:
        errors.append("confidence_threshold must be between 0.0 and 1.0")
    
    # Check tracking parameters
    if TRACKING_CONFIG['max_disappeared'] < 1:
        errors.append("max_disappeared must be >= 1")
    
    if TRACKING_CONFIG['max_distance'] < 1:
        errors.append("max_distance must be >= 1")
    
    # Check paths exist
    import os
    for path_key, path_val in PATHS.items():
        if not os.path.exists(path_val):
            sprint(f"⚠️  Warning: {path_key} directory not found: {path_val}")
            print(f"   Creating: {path_val}")
            os.makedirs(path_val, exist_ok=True)
    
    if errors:
        sprint("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    sprint("✅ Configuration validated successfully")
    return True


if __name__ == '__main__':
    # Test configuration
    print_config()
    validate_config()
