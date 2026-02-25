"""
Evaluate Coordinate Prediction Accuracy
Compare predicted coordinates with ground truth (if available)
Analyze coordinate stability and consistency
"""
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from detector import DroneDetector
from tracker import ByteTracker
from localizer import ApproximationLocalizer


def evaluate_coordinate_predictions(video_path, model_path, conf=0.15, iou=0.6, sample_interval=25):
    """
    Evaluate coordinate prediction quality
    
    Args:
        video_path: Path to video
        model_path: Path to YOLO model
        conf: Confidence threshold
        iou: IOU threshold
        sample_interval: Sample every N frames
    """
    
    print("="*70)
    print("Coordinate Prediction Accuracy Evaluation")
    print("="*70)
    
    # Initialize components
    print("📦 Loading components...")
    detector = DroneDetector(model_path=model_path, conf_threshold=conf, iou_threshold=iou)
    tracker = ByteTracker(model=detector.model, tracker_type='bytetrack', persist=True)
    localizer = ApproximationLocalizer()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {video_path}")
    print(f"   Frames: {total_frames}, FPS: {fps:.2f}")
    print(f"   Sampling interval: every {sample_interval} frames")
    
    # Collect predictions
    predictions = defaultdict(lambda: {
        'frames': [],
        'lats': [],
        'lons': [],
        'alts': [],
        'distances': [],
        'bearings': [],
        'bbox_sizes': [],
        'confidences': []
    })
    
    frame_idx = 0
    print("\n🎯 Processing video...")
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_interval == 0:
            # Track
            track_results = tracker.track(frame, conf=conf, iou=iou)
            
            # Process detections
            if track_results['num_tracks'] > 0:
                for i in range(track_results['num_tracks']):
                    bbox = track_results['boxes'][i]
                    track_id = int(track_results['track_ids'][i])
                    confidence = float(track_results['confidences'][i])
                    
                    # Predict coordinates
                    coords = localizer.predict(
                        bbox=bbox,
                        frame_shape=frame.shape,
                        confidence=confidence,
                        track_id=track_id
                    )
                    
                    # Calculate bbox size
                    x1, y1, x2, y2 = bbox
                    bbox_size = (x2 - x1) * (y2 - y1)
                    
                    # Store prediction
                    predictions[track_id]['frames'].append(frame_idx)
                    predictions[track_id]['lats'].append(coords['lat'])
                    predictions[track_id]['lons'].append(coords['lon'])
                    predictions[track_id]['alts'].append(coords['alt'])
                    predictions[track_id]['distances'].append(coords['distance_m'])
                    predictions[track_id]['bearings'].append(coords['bearing_deg'])
                    predictions[track_id]['bbox_sizes'].append(bbox_size)
                    predictions[track_id]['confidences'].append(confidence)
            
            if frame_idx % 500 == 0:
                print(f"   Processed frame {frame_idx}/{total_frames}...", end='\r')
        
        frame_idx += 1
    
    cap.release()
    print(f"   Processed frame {total_frames}/{total_frames}... Done!   ")
    
    # Analyze predictions
    print(f"\n📊 Collected predictions for {len(predictions)} tracks")
    
    # Sort by number of predictions
    sorted_tracks = sorted(predictions.items(), 
                          key=lambda x: len(x[1]['frames']), 
                          reverse=True)
    
    # Analyze top tracks
    analyze_coordinate_stability(sorted_tracks[:5], fps, sample_interval)
    
    # Visualize coordinate distributions
    visualize_coordinate_distributions(sorted_tracks[:3])
    
    # Analyze coordinate smoothness
    analyze_coordinate_smoothness(sorted_tracks[:3], fps, sample_interval)
    
    # Export predictions to CSV
    export_predictions_to_csv(sorted_tracks[:5])
    
    return predictions, sorted_tracks


def analyze_coordinate_stability(sorted_tracks, fps, sample_interval):
    """Analyze stability of coordinate predictions"""
    
    print("\n" + "="*70)
    print("Coordinate Stability Analysis:")
    print("-"*70)
    
    for tid, data in sorted_tracks:
        num_samples = len(data['frames'])
        duration = (max(data['frames']) - min(data['frames'])) / fps
        
        print(f"\nTrack {tid} ({num_samples} samples, {duration:.1f}s):")
        
        # Convert to numpy arrays
        lats = np.array(data['lats'])
        lons = np.array(data['lons'])
        alts = np.array(data['alts'])
        dists = np.array(data['distances'])
        
        # Calculate statistics
        print(f"  Latitude:")
        print(f"    Range: {lats.min():.6f} to {lats.max():.6f}")
        print(f"    Mean: {lats.mean():.6f} ± {lats.std():.6f}")
        print(f"    Variation: {(lats.max() - lats.min()) * 111000:.2f} meters")
        
        print(f"  Longitude:")
        print(f"    Range: {lons.min():.6f} to {lons.max():.6f}")
        print(f"    Mean: {lons.mean():.6f} ± {lons.std():.6f}")
        print(f"    Variation: {(lons.max() - lons.min()) * 111000:.2f} meters")
        
        print(f"  Altitude:")
        print(f"    Range: {alts.min():.2f} to {alts.max():.2f} m")
        print(f"    Mean: {alts.mean():.2f} ± {alts.std():.2f} m")
        
        print(f"  Distance from camera:")
        print(f"    Range: {dists.min():.2f} to {dists.max():.2f} m")
        print(f"    Mean: {dists.mean():.2f} ± {dists.std():.2f} m")


def visualize_coordinate_distributions(sorted_tracks):
    """Visualize coordinate distributions over time"""
    
    output_dir = Path('outputs/problem_3/coordinate_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['red', 'blue', 'green']
    
    for idx, (tid, data) in enumerate(sorted_tracks):
        frames = np.array(data['frames'])
        lats = np.array(data['lats'])
        lons = np.array(data['lons'])
        alts = np.array(data['alts'])
        dists = np.array(data['distances'])
        
        color = colors[idx % len(colors)]
        time_sec = frames / 25.0
        
        # Latitude over time
        axes[0, 0].plot(time_sec, lats, 'o-', markersize=3, alpha=0.7, 
                       color=color, label=f'Track {tid}')
        
        # Longitude over time
        axes[0, 1].plot(time_sec, lons, 'o-', markersize=3, alpha=0.7,
                       color=color, label=f'Track {tid}')
        
        # Altitude over time
        axes[1, 0].plot(time_sec, alts, 'o-', markersize=3, alpha=0.7,
                       color=color, label=f'Track {tid}')
        
        # Distance over time
        axes[1, 1].plot(time_sec, dists, 'o-', markersize=3, alpha=0.7,
                       color=color, label=f'Track {tid}')
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Latitude (°)')
    axes[0, 0].set_title('Latitude Over Time', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Longitude (°)')
    axes[0, 1].set_title('Longitude Over Time', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Altitude (m)')
    axes[1, 0].set_title('Altitude Over Time', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Distance (m)')
    axes[1, 1].set_title('Distance from Camera Over Time', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'coordinate_timeseries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Coordinate timeseries saved to: {output_path}")
    plt.close()


def analyze_coordinate_smoothness(sorted_tracks, fps, sample_interval):
    """Analyze smoothness of coordinate predictions (jitter analysis)"""
    
    output_dir = Path('outputs/problem_3/coordinate_evaluation')
    
    print("\n" + "="*70)
    print("Coordinate Smoothness Analysis (Jitter):")
    print("-"*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['red', 'blue', 'green']
    
    for idx, (tid, data) in enumerate(sorted_tracks):
        lats = np.array(data['lats'])
        lons = np.array(data['lons'])
        alts = np.array(data['alts'])
        frames = np.array(data['frames'])
        
        if len(lats) < 2:
            continue
        
        color = colors[idx % len(colors)]
        
        # Calculate frame-to-frame changes
        lat_diff = np.abs(np.diff(lats)) * 111000  # Convert to meters
        lon_diff = np.abs(np.diff(lons)) * 111000
        alt_diff = np.abs(np.diff(alts))
        
        time_sec = frames[1:] / fps
        
        print(f"\nTrack {tid}:")
        print(f"  Lat jitter: {lat_diff.mean():.2f} ± {lat_diff.std():.2f} m/sample")
        print(f"  Lon jitter: {lon_diff.mean():.2f} ± {lon_diff.std():.2f} m/sample")
        print(f"  Alt jitter: {alt_diff.mean():.2f} ± {alt_diff.std():.2f} m/sample")
        
        # Plot jitter
        axes[0].plot(time_sec, lat_diff, 'o-', markersize=2, alpha=0.7,
                    color=color, label=f'Track {tid}')
        axes[1].plot(time_sec, lon_diff, 'o-', markersize=2, alpha=0.7,
                    color=color, label=f'Track {tid}')
        axes[2].plot(time_sec, alt_diff, 'o-', markersize=2, alpha=0.7,
                    color=color, label=f'Track {tid}')
    
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Latitude Change (m)')
    axes[0].set_title('Latitude Jitter', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Longitude Change (m)')
    axes[1].set_title('Longitude Jitter', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Altitude Change (m)')
    axes[2].set_title('Altitude Jitter', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'coordinate_jitter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Jitter analysis saved to: {output_path}")
    plt.close()


def export_predictions_to_csv(sorted_tracks):
    """Export predictions to CSV for further analysis"""
    
    output_dir = Path('outputs/problem_3/coordinate_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for tid, data in sorted_tracks:
        df = pd.DataFrame({
            'frame': data['frames'],
            'time_sec': np.array(data['frames']) / 25.0,
            'latitude': data['lats'],
            'longitude': data['lons'],
            'altitude_m': data['alts'],
            'distance_m': data['distances'],
            'bearing_deg': data['bearings'],
            'bbox_size_px': data['bbox_sizes'],
            'confidence': data['confidences']
        })
        
        output_path = output_dir / f'predictions_track_{tid}.csv'
        df.to_csv(output_path, index=False)
        print(f"💾 Exported Track {tid} predictions to: {output_path}")
    
    print(f"\n✅ Exported {len(sorted_tracks)} track predictions to CSV")


if __name__ == "__main__":
    predictions, sorted_tracks = evaluate_coordinate_predictions(
        video_path='P3_VIDEO.mp4',
        model_path='models/tomorbest.pt',
        conf=0.15,
        iou=0.6,
        sample_interval=5  # Sample every 5 frames (0.2 second)
    )
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"✅ Evaluated {len(predictions)} tracks")
    print(f"✅ Top track: {sorted_tracks[0][0]} with {len(sorted_tracks[0][1]['frames'])} samples")
    print("✅ Generated visualizations and CSV exports")
    print("="*70)
    print("✅ Coordinate evaluation complete!")
    print("="*70)
