"""
Test NMS (Non-Maximum Suppression) to filter false positives
Compare detection results with different IOU thresholds
"""
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def test_nms_settings():
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
    
    print(f"\n📹 Video: {total_frames} frames, {fps} FPS\n")
    
    # Test different IOU thresholds for NMS
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    conf_threshold = 0.15  # Use optimal conf from previous test
    
    # Sample frames
    test_frames = [0, 200, 400, 700, 1000, 1100, 1500, 1800]
    
    print(f"🔍 Testing NMS with IOU thresholds: {iou_thresholds}")
    print(f"Confidence threshold: {conf_threshold}\n")
    
    results_by_iou = {}
    
    for iou in iou_thresholds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        total_detections = 0
        frames_with_2_drones = 0
        frames_with_more = 0
        
        for frame_idx in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Run detection with specific IOU threshold
            results = model(frame, conf=conf_threshold, iou=iou, verbose=False)
            
            if results[0].boxes is not None:
                num_det = len(results[0].boxes)
                total_detections += num_det
                
                if num_det == 2:
                    frames_with_2_drones += 1
                elif num_det > 2:
                    frames_with_more += 1
        
        avg_det = total_detections / len(test_frames)
        
        results_by_iou[iou] = {
            'total': total_detections,
            'avg': avg_det,
            'exact_2': frames_with_2_drones,
            'more_than_2': frames_with_more
        }
        
        print(f"IOU={iou:.1f}: avg {avg_det:.2f} det/frame, "
              f"{frames_with_2_drones}/{len(test_frames)} frames with 2 drones, "
              f"{frames_with_more} frames with >2 drones")
    
    cap.release()
    
    # Detailed comparison with visualization
    print("\n" + "="*70)
    print("📊 Detailed Comparison on Key Frames:")
    print("="*70)
    
    output_dir = Path('outputs/problem_3/nms_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare best IOU settings visually
    best_ious = [0.3, 0.5, 0.7]  # Low, medium, high
    
    for frame_idx in [200, 1000, 1500]:  # Frames with 2 drones
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        print(f"\nFrame {frame_idx} ({frame_idx/fps:.1f}s):")
        
        # Create comparison image
        comparison_frames = []
        
        for iou in best_ious:
            frame_copy = frame.copy()
            
            # Run detection
            results = model(frame_copy, conf=conf_threshold, iou=iou, verbose=False)
            
            num_det = 0
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                num_det = len(boxes)
                
                # Draw detections
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Different color for each detection
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    color = colors[i % len(colors)]
                    
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
                    
                    # Label
                    label = f"#{i+1} {conf:.3f}"
                    cv2.putText(frame_copy, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add title
            title = f"IOU={iou:.1f} | Detections: {num_det}"
            cv2.putText(frame_copy, title, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            comparison_frames.append(frame_copy)
            
            print(f"  IOU={iou:.1f}: {num_det} detections", end="")
            if num_det > 0:
                print(f" (conf: {confidences.min():.3f}-{confidences.max():.3f})")
            else:
                print()
        
        # Stack images horizontally (resize if needed)
        h, w = comparison_frames[0].shape[:2]
        new_w = w // 2  # Reduce size for stacking
        new_h = h // 2
        
        resized = [cv2.resize(f, (new_w, new_h)) for f in comparison_frames]
        
        # Stack in 2 rows
        row1 = np.hstack(resized[:2])
        row2 = np.hstack([resized[2], np.zeros_like(resized[2])])
        stacked = np.vstack([row1, row2])
        
        # Save comparison
        output_path = output_dir / f"nms_comparison_frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_path), stacked)
    
    # Recommendation
    print("\n" + "="*70)
    print("💡 Recommendation:")
    print("="*70)
    
    # Find IOU with most consistent 2-drone detection
    best_iou = max(results_by_iou.items(), 
                   key=lambda x: x[1]['exact_2'])
    
    print(f"Best IOU threshold: {best_iou[0]:.1f}")
    print(f"  - Frames with exactly 2 drones: {best_iou[1]['exact_2']}/{len(test_frames)}")
    print(f"  - Frames with >2 drones: {best_iou[1]['more_than_2']}")
    print(f"  - Average detections: {best_iou[1]['avg']:.2f}")
    
    print(f"\n✅ Optimal settings:")
    print(f"  - Confidence: {conf_threshold}")
    print(f"  - IOU (NMS): {best_iou[0]:.1f}")
    
    print(f"\n💾 Visual comparisons saved to: {output_dir}/")

if __name__ == "__main__":
    test_nms_settings()
