"""
Analyze Track ID Stability
Check if track IDs are consistent throughout the video
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))


def analyze_track_stability(video_path, model_path, conf=0.15, iou=0.6, tracker='botsort.yaml'):
    """Analyze track ID stability across video"""
    
    print("="*70)
    print("Track ID Stability Analysis")
    print("="*70)
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Track history
    track_data = defaultdict(lambda: {
        'frames': [],
        'centers': [],
        'confidences': [],
        'sizes': []
    })
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"📹 Processing video: {video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Settings: conf={conf}, iou={iou}")
    print(f"   Tracker: {tracker}")
    print("\n🎯 Running tracking analysis...")
    
    # Process video
    results = model.track(
        source=video_path,
        tracker=tracker,
        persist=True,
        conf=conf,
        iou=iou,
        stream=True,
        verbose=False
    )
    
    frame_idx = 0
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for bbox, tid, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                size = (x2 - x1) * (y2 - y1)
                
                track_data[tid]['frames'].append(frame_idx)
                track_data[tid]['centers'].append((center_x, center_y))
                track_data[tid]['confidences'].append(conf)
                track_data[tid]['sizes'].append(size)
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"   Processed {frame_idx}/{total_frames} frames...", end='\r')
    
    print(f"   Processed {frame_idx}/{total_frames} frames... Done!   ")
    
    # Analyze tracks
    print(f"\n📊 Found {len(track_data)} unique track IDs")
    print("\n" + "="*70)
    
    # Sort by duration
    sorted_tracks = sorted(track_data.items(), 
                          key=lambda x: len(x[1]['frames']), 
                          reverse=True)
    
    # Print top tracks
    print("\nTop 15 Longest Tracks:")
    print("-"*70)
    print(f"{'Track ID':<10} {'Frames':<10} {'Duration (s)':<15} {'Avg Conf':<12} {'Frame Range'}")
    print("-"*70)
    
    main_tracks = []
    
    for tid, data in sorted_tracks[:15]:
        num_frames = len(data['frames'])
        duration = num_frames / fps
        avg_conf = np.mean(data['confidences'])
        frame_range = f"{min(data['frames'])}-{max(data['frames'])}"
        
        print(f"{tid:<10} {num_frames:<10} {duration:<15.2f} {avg_conf:<12.3f} {frame_range}")
        
        # Consider tracks with >100 frames as main tracks
        if num_frames > 100:
            main_tracks.append(tid)
    
    # Analyze gaps
    print("\n" + "="*70)
    print("Track Continuity Analysis (Top 5 Longest Tracks):")
    print("-"*70)
    
    for tid, data in sorted_tracks[:5]:
        frames = sorted(data['frames'])
        gaps = []
        
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1]
            if gap > 1:
                gaps.append(gap)
        
        num_frames = len(frames)
        duration = num_frames / fps
        
        print(f"\nTrack {tid} ({num_frames} frames, {duration:.1f}s):")
        
        if gaps:
            avg_gap = np.mean(gaps)
            max_gap = max(gaps)
            num_gaps = len(gaps)
            print(f"  ⚠️  {num_gaps} interruptions detected")
            print(f"  Average gap: {avg_gap:.1f} frames ({avg_gap/fps:.2f}s)")
            print(f"  Max gap: {max_gap} frames ({max_gap/fps:.2f}s)")
        else:
            print(f"  ✅ Continuous (no gaps)")
    
    # Visualize track timeline
    visualize_track_timeline(track_data, sorted_tracks[:15], fps)
    
    # Movement analysis
    analyze_movement_patterns(track_data, sorted_tracks[:2], fps)
    
    # Track overlap analysis
    analyze_track_overlap(track_data, sorted_tracks[:5], fps)
    
    return track_data, main_tracks, sorted_tracks


def visualize_track_timeline(track_data, top_tracks, fps):
    """Visualize when each track appears"""
    
    output_dir = Path('outputs/problem_3/track_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_tracks)))
    
    for idx, (tid, data) in enumerate(top_tracks):
        frames = np.array(data['frames'])
        y_pos = idx
        
        # Plot as scatter points
        ax.scatter(frames/fps, [y_pos]*len(frames), 
                  s=10, alpha=0.7, color=colors[idx], label=f'Track {tid}')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Track ID (ordered by duration)', fontsize=12)
    ax.set_title('Track ID Timeline - When Each Track Appears', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(top_tracks)))
    ax.set_yticklabels([f'Track {tid}' for tid, _ in top_tracks])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    plt.tight_layout()
    output_path = output_dir / 'track_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Timeline saved to: {output_path}")
    plt.close()


def analyze_movement_patterns(track_data, top_tracks, fps):
    """Analyze movement patterns of main tracks"""
    
    output_dir = Path('outputs/problem_3/track_analysis')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, (tid, data) in enumerate(top_tracks):
        centers = np.array(data['centers'])
        frames = np.array(data['frames'])
        color = colors[idx % len(colors)]
        
        # Plot trajectory
        axes[0].plot(centers[:, 0], centers[:, 1], 
                    'o-', markersize=2, alpha=0.5, color=color, linewidth=1,
                    label=f'Track {tid} ({len(frames)} frames)')
        
        # Mark start and end
        if len(centers) > 0:
            axes[0].plot(centers[0, 0], centers[0, 1], 'o', 
                        markersize=10, color=color, markeredgecolor='black', markeredgewidth=2)
            axes[0].plot(centers[-1, 0], centers[-1, 1], 's', 
                        markersize=10, color=color, markeredgecolor='black', markeredgewidth=2)
        
        # Plot speed over time
        if len(centers) > 1:
            # Calculate frame-to-frame distance
            distances = np.sqrt(np.sum(np.diff(centers, axis=0)**2, axis=1))
            # Convert to speed (pixels per frame)
            frame_gaps = np.diff(frames)
            frame_gaps[frame_gaps == 0] = 1  # Avoid division by zero
            speeds = distances / frame_gaps
            
            axes[1].plot(frames[1:]/fps, speeds, 
                        alpha=0.6, linewidth=1.5, color=color, label=f'Track {tid}')
    
    axes[0].set_xlabel('X Position (pixels)', fontsize=11)
    axes[0].set_ylabel('Y Position (pixels)', fontsize=11)
    axes[0].set_title('Drone Trajectories (○=start, □=end)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()  # Image coordinates
    
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_ylabel('Speed (pixels/frame)', fontsize=11)
    axes[1].set_title('Movement Speed Over Time', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'movement_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📈 Movement analysis saved to: {output_path}")
    plt.close()


def analyze_track_overlap(track_data, top_tracks, fps):
    """Analyze temporal overlap between tracks"""
    
    output_dir = Path('outputs/problem_3/track_analysis')
    
    print("\n" + "="*70)
    print("Track Overlap Analysis:")
    print("-"*70)
    
    # Check which tracks appear simultaneously
    overlap_matrix = np.zeros((len(top_tracks), len(top_tracks)))
    
    for i, (tid1, data1) in enumerate(top_tracks):
        frames1 = set(data1['frames'])
        for j, (tid2, data2) in enumerate(top_tracks):
            if i != j:
                frames2 = set(data2['frames'])
                overlap = len(frames1 & frames2)
                overlap_matrix[i, j] = overlap
    
    # Print overlap info
    for i, (tid1, data1) in enumerate(top_tracks):
        print(f"\nTrack {tid1}:")
        for j, (tid2, data2) in enumerate(top_tracks):
            if i != j and overlap_matrix[i, j] > 0:
                overlap_pct = overlap_matrix[i, j] / len(data1['frames']) * 100
                print(f"  Overlaps with Track {tid2}: {int(overlap_matrix[i, j])} frames ({overlap_pct:.1f}%)")
    
    # Visualize overlap matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    track_labels = [f'T{tid}' for tid, _ in top_tracks]
    
    im = ax.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(top_tracks)))
    ax.set_yticks(range(len(top_tracks)))
    ax.set_xticklabels(track_labels)
    ax.set_yticklabels(track_labels)
    ax.set_xlabel('Track ID')
    ax.set_ylabel('Track ID')
    ax.set_title('Track Overlap Matrix (frames)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(top_tracks)):
        for j in range(len(top_tracks)):
            if overlap_matrix[i, j] > 0:
                text = ax.text(j, i, int(overlap_matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Overlapping Frames')
    plt.tight_layout()
    
    output_path = output_dir / 'track_overlap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Overlap matrix saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    track_data, main_tracks, sorted_tracks = analyze_track_stability(
        video_path='P3_VIDEO.mp4',
        model_path='runs/detect/drone_detect_v21_max_data/weights/best.pt',
        conf=0.15,
        iou=0.6,
        tracker='configs/botsort_custom.yaml'
    )
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"✅ Total unique tracks: {len(track_data)}")
    print(f"✅ Main tracks (>100 frames): {len(main_tracks)}")
    if main_tracks:
        print(f"✅ Main track IDs: {main_tracks[:10]}")
    
    # Recommend primary tracks
    print(f"\n🎯 Recommended primary drone tracks:")
    for tid, data in sorted_tracks[:2]:
        num_frames = len(data['frames'])
        duration = num_frames / 25
        print(f"   Track {tid}: {num_frames} frames ({duration:.1f}s)")
    
    print("="*70)
    print("✅ Analysis complete!")
    print("="*70)
