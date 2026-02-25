"""
Output Quality Check
Verify output video quality, file size, and frame integrity
"""
import cv2
import numpy as np
from pathlib import Path
import json


def check_output_quality(video_path, reference_video_path=None):
    """
    Check output video quality
    
    Args:
        video_path: Path to output video
        reference_video_path: Optional path to original video for comparison
    """
    
    print("="*70)
    print("Output Video Quality Check")
    print("="*70)
    
    # Check if file exists
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    duration_sec = total_frames / fps if fps > 0 else 0
    
    print(f"\n📹 Video Information:")
    print(f"   File: {video_path.name}")
    print(f"   Path: {video_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration_sec:.2f} seconds")
    print(f"   Codec (FourCC): {fourcc_to_string(fourcc)}")
    
    # Check file size constraint
    print(f"\n📦 File Size Check:")
    if file_size_mb < 200:
        print(f"   ✅ PASS: {file_size_mb:.2f} MB < 200 MB limit")
    else:
        print(f"   ❌ FAIL: {file_size_mb:.2f} MB > 200 MB limit")
    
    # Sample frames for quality check
    print(f"\n🎬 Sampling Frames for Quality Check...")
    sample_indices = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
    sample_indices = [i for i in sample_indices if i < total_frames]
    
    frame_qualities = []
    corrupt_frames = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            corrupt_frames.append(idx)
            print(f"   ⚠️  Frame {idx}: Cannot read")
        else:
            # Check if frame is mostly black
            mean_brightness = np.mean(frame)
            std_brightness = np.std(frame)
            
            frame_qualities.append({
                'frame': idx,
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'shape': frame.shape
            })
            
            status = "✅" if mean_brightness > 10 else "⚠️"
            print(f"   {status} Frame {idx}: brightness={mean_brightness:.1f}, std={std_brightness:.1f}")
    
    cap.release()
    
    # Frame integrity check
    print(f"\n🔍 Frame Integrity:")
    if corrupt_frames:
        print(f"   ⚠️  Found {len(corrupt_frames)} corrupt frames: {corrupt_frames}")
    else:
        print(f"   ✅ All sampled frames readable")
    
    # Quality assessment
    if frame_qualities:
        avg_brightness = np.mean([f['mean_brightness'] for f in frame_qualities])
        avg_std = np.mean([f['std_brightness'] for f in frame_qualities])
        
        print(f"\n📊 Overall Quality Metrics:")
        print(f"   Average Brightness: {avg_brightness:.2f}")
        print(f"   Average Std Dev: {avg_std:.2f}")
        
        if avg_brightness > 10 and avg_std > 5:
            print(f"   ✅ Quality: GOOD")
        elif avg_brightness > 5:
            print(f"   ⚠️  Quality: MODERATE (low contrast)")
        else:
            print(f"   ❌ Quality: POOR (too dark)")
    
    # Compare with reference if provided
    if reference_video_path:
        compare_with_reference(video_path, reference_video_path)
    
    # Summary
    results = {
        'file_path': str(video_path),
        'file_size_mb': file_size_mb,
        'size_check_pass': file_size_mb < 200,
        'resolution': f"{width}x{height}",
        'fps': fps,
        'total_frames': total_frames,
        'duration_sec': duration_sec,
        'corrupt_frames': len(corrupt_frames),
        'avg_brightness': avg_brightness if frame_qualities else 0,
        'avg_std': avg_std if frame_qualities else 0
    }
    
    return results


def compare_with_reference(output_path, reference_path):
    """Compare output video with reference video"""
    
    print(f"\n🔄 Comparing with Reference Video:")
    print(f"   Reference: {reference_path}")
    
    ref_cap = cv2.VideoCapture(reference_path)
    out_cap = cv2.VideoCapture(str(output_path))
    
    if not ref_cap.isOpened() or not out_cap.isOpened():
        print(f"   ❌ Cannot open videos for comparison")
        return
    
    ref_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
    out_fps = out_cap.get(cv2.CAP_PROP_FPS)
    
    ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Frames: {out_frames} vs {ref_frames} (reference)")
    print(f"   FPS: {out_fps:.2f} vs {ref_fps:.2f} (reference)")
    print(f"   Resolution: {out_width}x{out_height} vs {ref_width}x{ref_height} (reference)")
    
    if abs(out_frames - ref_frames) <= 50:  # Allow small difference
        print(f"   ✅ Frame count matches (±50)")
    else:
        print(f"   ⚠️  Frame count differs by {abs(out_frames - ref_frames)}")
    
    if out_fps == ref_fps:
        print(f"   ✅ FPS matches")
    else:
        print(f"   ⚠️  FPS differs")
    
    if out_width == ref_width and out_height == ref_height:
        print(f"   ✅ Resolution matches")
    else:
        print(f"   ⚠️  Resolution differs")
    
    ref_cap.release()
    out_cap.release()


def fourcc_to_string(fourcc):
    """Convert FourCC code to string"""
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])


def check_annotation_presence(video_path, sample_frames=10):
    """Check if annotations are present in the video"""
    
    print(f"\n🎨 Checking for Annotations:")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames throughout video
    sample_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
    
    frames_with_annotations = 0
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret and frame is not None:
            # Simple check: look for colored pixels that might be bboxes
            # Bboxes typically have pure colors (red, green, blue, etc.)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Check for high saturation (colored annotations)
            high_sat = np.sum(hsv[:, :, 1] > 200)
            
            if high_sat > 100:  # If there are enough colored pixels
                frames_with_annotations += 1
    
    cap.release()
    
    annotation_rate = frames_with_annotations / len(sample_indices) * 100
    print(f"   Sampled {len(sample_indices)} frames")
    print(f"   Frames with likely annotations: {frames_with_annotations} ({annotation_rate:.1f}%)")
    
    if annotation_rate > 50:
        print(f"   ✅ Annotations appear to be present")
    else:
        print(f"   ⚠️  Low annotation detection - verify manually")
    
    return annotation_rate


def save_quality_report(results, output_path):
    """Save quality check results to JSON"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Quality report saved to: {output_path}")


if __name__ == "__main__":
    # Check main output video
    output_video = 'outputs/problem_3/P3_OUTPUT_FULL.mp4'
    reference_video = 'P3_VIDEO.mp4'
    
    print("Checking output video quality...\n")
    
    results = check_output_quality(
        video_path=output_video,
        reference_video_path=reference_video
    )
    
    if results:
        # Check for annotations
        annotation_rate = check_annotation_presence(output_video, sample_frames=20)
        results['annotation_rate'] = annotation_rate
        
        # Save report
        save_quality_report(
            results,
            'outputs/problem_3/quality_report.json'
        )
        
        print("\n" + "="*70)
        print("Quality Check Summary:")
        print("="*70)
        print(f"✅ File Size: {results['file_size_mb']:.2f} MB (< 200 MB: {results['size_check_pass']})")
        print(f"✅ Resolution: {results['resolution']}")
        print(f"✅ FPS: {results['fps']:.2f}")
        print(f"✅ Duration: {results['duration_sec']:.2f} seconds")
        print(f"✅ Corrupt Frames: {results['corrupt_frames']}")
        print(f"✅ Annotation Rate: {annotation_rate:.1f}%")
        print("="*70)
        print("✅ Quality check complete!")
        print("="*70)
