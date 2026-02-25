"""
Test YOLO Detection on P3_VIDEO.mp4
Quick test to see if drone detection works
"""
import cv2
from ultralytics import YOLO
import os

def test_detection_on_video():
    # Load YOLO model
    model = YOLO('models/tomorbest.pt')
    
    # Open video
    video_path = 'P3_VIDEO.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info:")
    print(f"  - Total Frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {total_frames/fps:.2f} seconds")
    print()
    
    # Test on first 10 frames
    print("🔍 Testing detection on first 10 frames...")
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.3, verbose=False)
        
        # Count detections
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        
        print(f"Frame {i+1:3d}: {num_detections} drone(s) detected", end="")
        
        if num_detections > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            print(f" (conf: {confidences.max():.3f})")
        else:
            print()
    
    cap.release()
    print("\n✅ Detection test completed!")

if __name__ == "__main__":
    test_detection_on_video()
