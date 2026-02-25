"""
Test YOLO Detection with Visualization on P3_VIDEO.mp4
Shows detection results on sample frames
"""
import cv2
from ultralytics import YOLO
import os
from pathlib import Path

def test_detection_with_visualization():
    # Load YOLO model
    print("📦 Loading YOLO model...")
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video Info:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total Frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {total_frames/fps:.2f} seconds")
    print()
    
    # Create output directory
    output_dir = Path('outputs/problem_3/detection_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on frames at different time points
    test_frames = [0, 100, 500, 1000, 1500, 1800]  # Sample frames
    
    print(f"🔍 Testing detection on {len(test_frames)} sample frames...")
    print(f"💾 Saving visualizations to: {output_dir}/")
    print()
    
    detection_stats = {
        'total_tested': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'confidences': []
    }
    
    for frame_idx in test_frames:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Cannot read frame {frame_idx}")
            continue
        
        # Run detection
        results = model(frame, conf=0.3, verbose=False)
        
        # Get detections
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        detection_stats['total_tested'] += 1
        
        if num_detections > 0:
            detection_stats['frames_with_detections'] += 1
            detection_stats['total_detections'] += num_detections
            
            # Get boxes and confidences
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_stats['confidences'].extend(confidences.tolist())
            
            # Draw detections
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"Drone: {conf:.3f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Background for text
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (0, 255, 0), -1)
                
                # Text
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            status = f"✅ {num_detections} drone(s) (conf: {confidences.max():.3f})"
        else:
            status = "❌ No detection"
            
            # Add "No Detection" text on frame
            cv2.putText(frame, "No Drone Detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Add frame info
        frame_info = f"Frame: {frame_idx}/{total_frames} ({frame_idx/fps:.2f}s)"
        cv2.putText(frame, frame_info, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame
        output_path = output_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        
        print(f"Frame {frame_idx:4d} ({frame_idx/fps:6.2f}s): {status}")
    
    cap.release()
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 Detection Statistics:")
    print("="*60)
    print(f"Total frames tested: {detection_stats['total_tested']}")
    print(f"Frames with detections: {detection_stats['frames_with_detections']}")
    print(f"Detection rate: {detection_stats['frames_with_detections']/detection_stats['total_tested']*100:.1f}%")
    print(f"Total detections: {detection_stats['total_detections']}")
    
    if detection_stats['confidences']:
        import numpy as np
        confs = np.array(detection_stats['confidences'])
        print(f"\nConfidence scores:")
        print(f"  - Min:  {confs.min():.3f}")
        print(f"  - Max:  {confs.max():.3f}")
        print(f"  - Mean: {confs.mean():.3f}")
        print(f"  - Std:  {confs.std():.3f}")
    
    print(f"\n✅ Test completed! Check visualizations in: {output_dir}/")
    print(f"   Total {len(test_frames)} images saved")

if __name__ == "__main__":
    test_detection_with_visualization()
