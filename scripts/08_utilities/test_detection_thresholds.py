"""
Test Detection with Different Confidence Thresholds
Find optimal threshold for drone detection
"""
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def test_multiple_thresholds():
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
    
    print(f"\n📹 Video: {total_frames} frames, {fps} FPS, {total_frames/fps:.1f}s\n")
    
    # Test different confidence thresholds
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    # Sample frames evenly throughout video
    sample_interval = total_frames // 50  # Test 50 frames
    test_frames = list(range(0, total_frames, sample_interval))
    
    print(f"🔍 Testing {len(test_frames)} frames with different confidence thresholds...")
    print(f"Thresholds: {thresholds}\n")
    
    results_by_threshold = {}
    
    for threshold in thresholds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        detections_count = []
        frames_with_detection = 0
        all_confidences = []
        
        for frame_idx in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Run detection
            results = model(frame, conf=threshold, verbose=False)
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                num_det = len(results[0].boxes)
                detections_count.append(num_det)
                frames_with_detection += 1
                
                confs = results[0].boxes.conf.cpu().numpy()
                all_confidences.extend(confs.tolist())
            else:
                detections_count.append(0)
        
        detection_rate = (frames_with_detection / len(test_frames)) * 100
        avg_drones = np.mean(detections_count) if detections_count else 0
        max_drones = max(detections_count) if detections_count else 0
        
        results_by_threshold[threshold] = {
            'detection_rate': detection_rate,
            'frames_with_detection': frames_with_detection,
            'avg_drones_per_frame': avg_drones,
            'max_drones': max_drones,
            'total_detections': sum(detections_count),
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0
        }
        
        print(f"Conf={threshold:.2f}: {detection_rate:5.1f}% detection rate, "
              f"avg {avg_drones:.2f} drones/frame, max {max_drones} drones, "
              f"total {sum(detections_count)} detections")
    
    cap.release()
    
    # Find best threshold
    print("\n" + "="*70)
    print("📊 Recommendation:")
    print("="*70)
    
    best_threshold = max(results_by_threshold.items(), 
                        key=lambda x: x[1]['detection_rate'])
    
    print(f"Best threshold: {best_threshold[0]:.2f}")
    print(f"  - Detection rate: {best_threshold[1]['detection_rate']:.1f}%")
    print(f"  - Avg drones per frame: {best_threshold[1]['avg_drones_per_frame']:.2f}")
    print(f"  - Max drones: {best_threshold[1]['max_drones']}")
    print(f"  - Total detections: {best_threshold[1]['total_detections']}")
    
    # Detailed analysis on specific frames with low threshold
    print("\n" + "="*70)
    print("🔎 Detailed Frame Analysis (conf=0.15):")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    output_dir = Path('outputs/problem_3/detection_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check every 100th frame
    detailed_frames = list(range(0, total_frames, 100))
    
    for frame_idx in detailed_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Run detection with low threshold
        results = model(frame, conf=0.15, verbose=False)
        
        num_det = len(results[0].boxes) if results[0].boxes is not None else 0
        
        if num_det > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Draw all detections
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                
                # Color based on confidence
                if conf >= 0.3:
                    color = (0, 255, 0)  # Green - high conf
                elif conf >= 0.2:
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 165, 255)  # Orange - low
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"#{i+1} conf:{conf:.3f}"
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Frame info
            info = f"Frame {frame_idx} ({frame_idx/fps:.1f}s): {num_det} drones"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save
            output_path = output_dir / f"analysis_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            
            print(f"Frame {frame_idx:4d} ({frame_idx/fps:5.1f}s): {num_det} drone(s) - "
                  f"conf: {confidences.min():.3f}-{confidences.max():.3f}")
    
    cap.release()
    print(f"\n💾 Detailed analysis saved to: {output_dir}/")

if __name__ == "__main__":
    test_multiple_thresholds()
