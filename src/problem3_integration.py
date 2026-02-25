"""
Problem 3: Full Integration Pipeline
Complete system: YOLO Detection → ByteTrack → Regression → Submission CSV
"""

import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from ultralytics import YOLO
import torch
from byte_track_wrapper import ByteTrackWrapper
from problem2_inference import RegressionPipeline
from hw import get_device, sprint


class CompleteDroneSystem:
    """Complete drone detection, tracking, and localization system"""
    
    def __init__(self, model_path='yolov8n-obb.pt', 
                 regression_model='xgboost',
                 device='auto'):
        """
        Initialize complete system
        
        Args:
            model_path: Path to YOLO-OBB model
            regression_model: 'random_forest' or 'xgboost'
            device: 'auto', 'cuda', or 'cpu'
        """
        # Device
        self.device = get_device(force=device)
        sprint(f"Device: {self.device}")
        
        # Load YOLO
        sprint(f"Loading YOLO-OBB: {model_path}")
        self.model = YOLO(model_path)
        
        # ByteTrack will be initialized in process_video with correct FPS
        sprint(f"Initializing ByteTrack...")
        self.tracker = None
        
        # Load regression models
        sprint(f"Loading regression models ({regression_model})...")
        self.regression = RegressionPipeline(model_type=regression_model)
        
        # Storage
        self.all_predictions = []
        
    def process_video(self, video_path, video_id=None, conf_threshold=0.55,
                     display=False, save_video=None):
        """
        Process complete video with full pipeline
        
        Args:
            video_path: Path to video
            video_id: Video identifier for submission
            conf_threshold: Detection confidence threshold
            display: Show video while processing
            save_video: Path to save annotated video
            
        Returns:
            DataFrame with all predictions
        """
        # Video ID
        if video_id is None:
            video_id = Path(video_path).name
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize ByteTrack with correct FPS
        if self.tracker is None:
            self.tracker = ByteTrackWrapper(
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=fps,  # Use video's FPS
                track_history=30
            )
        
        print(f"\n{'='*70}")
        sprint(f"PROCESSING VIDEO: {video_id}")
        print(f"{'='*70}")
        sprint(f"Resolution: {width}x{height} @ {fps} FPS")
        sprint(f"Total frames: {total_frames}")
        sprint(f"Confidence: {conf_threshold}")
        print(f"{'='*70}\n")
        
        # Video writer
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        
        # Processing
        frame_count = 0
        start_time = time.time()
        detections_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Step 1: YOLO Detection
                results = self.model(
                    frame,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Extract detections
                centroids = []
                bboxes = []
                
                if results[0].obb is not None:
                    # OBB model output
                    for obb in results[0].obb:
                        xywhr = obb.xywhr[0].cpu().numpy()
                        cx, cy = int(xywhr[0]), int(xywhr[1])
                        w, h = int(xywhr[2]), int(xywhr[3])
                        theta = np.degrees(xywhr[4])
                        conf = float(obb.conf[0])
                        
                        centroids.append((cx, cy))
                        bboxes.append({
                            'cx': cx,
                            'cy': cy,
                            'w': w,
                            'h': h,
                            'theta': theta,
                            'conf': conf,
                            'points': obb.xyxyxyxy[0].cpu().numpy().astype(int)
                        })
                elif results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Regular detection model fallback (no OBB)
                    for box in results[0].boxes:
                        xywh = box.xywh[0].cpu().numpy()
                        cx, cy = int(xywh[0]), int(xywh[1])
                        w, h = int(xywh[2]), int(xywh[3])
                        theta = 0.0
                        conf = float(box.conf[0])
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        points = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                        
                        centroids.append((cx, cy))
                        bboxes.append({
                            'cx': cx,
                            'cy': cy,
                            'w': w,
                            'h': h,
                            'theta': theta,
                            'conf': conf,
                            'points': points
                        })
                
                # Step 2: ByteTrack Tracking
                objects = self.tracker.update(centroids, current_time)
                
                # Step 3: Regression Prediction
                for object_id, centroid in objects.items():
                    # Find matching bbox
                    matched_bbox = None
                    min_dist = float('inf')
                    
                    for bbox in bboxes:
                        dist = np.sqrt((bbox['cx'] - centroid[0])**2 + 
                                     (bbox['cy'] - centroid[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            matched_bbox = bbox
                    
                    if matched_bbox and min_dist < 50:
                        # Predict range, azimuth, elevation
                        pred = self.regression.predict(
                            matched_bbox['cx'],
                            matched_bbox['cy'],
                            matched_bbox['w'],
                            matched_bbox['h']
                        )
                        
                        # Store prediction
                        self.all_predictions.append({
                            'video_id': video_id,
                            'frame': frame_count,
                            'cx': matched_bbox['cx'],
                            'cy': matched_bbox['cy'],
                            'range_m_pred': round(pred['range_m'], 1),
                            'azimuth_deg_pred': round(pred['azimuth_deg'], 1),
                            'elevation_deg_pred': round(pred['elevation_deg'], 1)
                        })
                        
                        detections_count += 1
                
                # Draw annotations
                annotated = self._draw_frame(frame, objects, bboxes, frame_count, 
                                            total_frames, current_time - start_time)
                
                # Display
                if display:
                    cv2.imshow('Complete Drone System', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        sprint("\nStopped by user")
                        break
                
                # Save frame
                if writer:
                    writer.write(annotated)
                
                # Progress
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    sprint(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {fps_actual:.1f} | Objects: {len(objects)} | "
                          f"Predictions: {detections_count}")
        
        except KeyboardInterrupt:
            sprint("\nInterrupted (Ctrl+C)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Summary
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            sprint(f"PROCESSING COMPLETE")
            print(f"{'='*70}")
            sprint(f"Statistics:")
            print(f"   • Frames processed: {frame_count}")
            print(f"   • Time elapsed: {elapsed:.1f}s")
            print(f"   • Average FPS: {frame_count/elapsed:.1f}")
            print(f"   • Total predictions: {detections_count}")
            print(f"   • Unique tracks: {self.tracker.get_total_tracked_objects()}")
            print(f"{'='*70}\n")
        
        # Return predictions DataFrame
        predictions_df = pd.DataFrame(self.all_predictions)
        return predictions_df
    
    def _draw_frame(self, frame, objects, bboxes, frame_num, total_frames, elapsed):
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        # Draw detections
        for bbox in bboxes:
            points = bbox['points']
            cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
            cv2.circle(annotated, (bbox['cx'], bbox['cy']), 5, (0, 255, 0), -1)
        
        # Draw tracked objects
        for object_id, centroid in objects.items():
            # ID label
            text = f"ID: {object_id}"
            cv2.putText(annotated, text, (centroid[0] - 20, centroid[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Track path
            path = self.tracker.get_track_path(object_id)
            if len(path) > 1:
                for i in range(1, len(path)):
                    cv2.line(annotated, path[i-1], path[i], (255, 0, 255), 2)
        
        # Info overlay
        cv2.putText(annotated, f'Frame: {frame_num}/{total_frames}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f'Objects: {len(objects)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f'FPS: {frame_num/elapsed if elapsed > 0 else 0:.1f}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, 'TESA Defence System', (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        return annotated
    
    def save_submission(self, output_path):
        """Save predictions in submission format"""
        if not self.all_predictions:
            sprint("No predictions to save")
            return
        
        df = pd.DataFrame(self.all_predictions)
        df = df[['video_id', 'frame', 'cx', 'cy', 'range_m_pred', 
                'azimuth_deg_pred', 'elevation_deg_pred']]
        df.to_csv(output_path, index=False)
        
        sprint(f"Saved submission: {output_path}")
        print(f"   • Total predictions: {len(df)}")
        print(f"   • Format: video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred")


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Drone Detection System')
    parser.add_argument('--video', type=str, required=True, 
                       help='Input video path')
    parser.add_argument('--video-id', type=str, default=None,
                       help='Video ID for submission (default: filename)')
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                       help='Output submission CSV')
    parser.add_argument('--conf', type=float, default=0.55,
                       help='Detection confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='YOLO model path')
    parser.add_argument('--regression', type=str, default='xgboost',
                       choices=['random_forest', 'xgboost'],
                       help='Regression model type')
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save annotated video')
    
    args = parser.parse_args()
    
    print("="*70)
    sprint("TESA DEFENCE - COMPLETE SYSTEM")
    print("="*70)
    print("Components:")
    print("  1. YOLO-OBB Detection")
    print("  2. ByteTrack Tracking")
    print("  3. XGBoost/RandomForest Regression")
    print("="*70)
    
    # Initialize system
    system = CompleteDroneSystem(
        model_path=args.model,
        regression_model=args.regression
    )
    
    # Process video
    predictions = system.process_video(
        video_path=args.video,
        video_id=args.video_id,
        conf_threshold=args.conf,
        display=args.display,
        save_video=args.save_video
    )
    
    # Save submission
    system.save_submission(args.output)
    
    # Final summary
    print(f"\n{'='*70}")
    sprint("SYSTEM COMPLETE")
    print("="*70)
    sprint(f"Output Summary:")
    if len(predictions) > 0:
        print(f"   - Range: {predictions['range_m_pred'].min():.1f} - {predictions['range_m_pred'].max():.1f} m")
        print(f"   - Azimuth: {predictions['azimuth_deg_pred'].min():.1f} - {predictions['azimuth_deg_pred'].max():.1f} deg")
        print(f"   - Elevation: {predictions['elevation_deg_pred'].min():.1f} - {predictions['elevation_deg_pred'].max():.1f} deg")
    sprint(f"\nReady for submission: {args.output}")
    print("="*70)
