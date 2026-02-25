"""
Detector Module - YOLO Detection Wrapper
Handles drone detection with optimal settings
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from hw import get_device, sprint


class DroneDetector:
    """
    YOLO-based drone detector optimized for P3_VIDEO.mp4
    """
    
    def __init__(self, model_path=None, 
                 conf_threshold=0.15, 
                 iou_threshold=0.3):  # ลดจาก 0.6 -> 0.3
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold (0.15 optimal for this video)
            iou_threshold: IOU threshold for NMS (0.3 = stricter, less overlapping boxes)
        """
        # Resolve default model path relative to project root
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent
            model_path = str(project_root / 'models' / 'tomorbest.pt')
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = get_device()
        
        sprint(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        sprint(f"Model loaded | Device: {self.device} | Conf: {conf_threshold} | IOU: {iou_threshold}")
    
    def detect(self, frame, verbose=False):
        """
        Detect drones in a single frame
        
        Args:
            frame: Input frame (numpy array)
            verbose: Print detection info
            
        Returns:
            dict with:
                - boxes: numpy array of [x1, y1, x2, y2]
                - confidences: numpy array of confidence scores
                - num_detections: int
        """
        # Run detection
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Extract results
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            num_detections = len(boxes)
        else:
            boxes = np.array([])
            confidences = np.array([])
            num_detections = 0
        
        if verbose:
            print(f"Detected {num_detections} drone(s)", end="")
            if num_detections > 0:
                print(f" (conf: {confidences.min():.3f}-{confidences.max():.3f})")
            else:
                print()
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'num_detections': num_detections
        }
    
    def detect_batch(self, frames):
        """
        Detect drones in multiple frames
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection results
        """
        return [self.detect(frame) for frame in frames]
    
    def visualize_detection(self, frame, detections, show_conf=True):
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            detections: Detection results from detect()
            show_conf: Whether to show confidence scores
            
        Returns:
            Frame with visualizations
        """
        frame_viz = frame.copy()
        
        boxes = detections['boxes']
        confidences = detections['confidences']
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # Color based on confidence
            if conf >= 0.4:
                color = (0, 255, 0)  # Green - high
            elif conf >= 0.2:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 165, 255)  # Orange - low
            
            # Draw box
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_conf:
                label = f"Drone #{i+1}: {conf:.3f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Background
                cv2.rectangle(frame_viz,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Text
                cv2.putText(frame_viz, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_viz


def test_detector():
    """Test detector on sample video"""
    print("="*70)
    print("Testing Drone Detector")
    print("="*70)
    
    project_root = Path(__file__).resolve().parent.parent
    
    # Initialize detector
    detector = DroneDetector()
    
    # Load video
    cap = cv2.VideoCapture(str(project_root / 'videos' / 'P3_VIDEO.mp4'))
    
    # Test on sample frames
    test_frames_idx = [0, 200, 1000, 1500]
    
    for frame_idx in test_frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Detect
        detections = detector.detect(frame, verbose=True)
        
        # Visualize
        frame_viz = detector.visualize_detection(frame, detections)
        
        # Save
        output_dir = Path('outputs/problem_3/detector_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"detection_frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_path), frame_viz)
        
        print(f"  Saved: {output_path}")
    
    cap.release()
    sprint("\n[OK] Detector test completed!")


if __name__ == "__main__":
    test_detector()
