"""
Raspberry Pi 5 Optimized Problem 1: Detection + Tracking
Optimized specifically for Raspberry Pi 5 hardware
"""

import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import argparse
import platform
import psutil

from byte_track_wrapper import ByteTrackWrapper
from hw import sprint


class RaspberryPiOptimizer:
    """Raspberry Pi 5 specific optimizations"""
    
    def __init__(self):
        self.is_pi = self._detect_raspberry_pi()
        self.optimizations = []
        
    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            if Path('/proc/cpuinfo').exists():
                with open('/proc/cpuinfo', 'r') as f:
                    if 'Raspberry Pi' in f.read():
                        return True
        except:
            pass
        
        # Check ARM architecture
        machine = platform.machine()
        if machine in ['aarch64', 'armv7l', 'armv8']:
            return True
        
        return False
    
    def apply_optimizations(self):
        """Apply Raspberry Pi specific optimizations"""
        if self.is_pi:
            sprint("\n🔧 Applying Raspberry Pi 5 optimizations...")
            
            # Set CPU threads
            torch.set_num_threads(4)  # Pi 5 has 4 cores
            self.optimizations.append("CPU threads: 4")
            
            # Disable CUDA even if detected
            torch.cuda.is_available = lambda: False
            
            # Memory optimization
            import gc
            gc.collect()
            self.optimizations.append("Memory optimized")
            
            sprint("   ✅ Optimizations applied:")
            for opt in self.optimizations:
                print(f"      • {opt}")
        else:
            sprint("   ℹ️  Not Raspberry Pi, using default settings")
    
    def monitor_system(self):
        """Monitor system resources"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        if self.is_pi:
            try:
                # Get Pi temperature
                temp = self._get_pi_temperature()
                return {
                    'cpu': cpu_percent,
                    'memory': memory.percent,
                    'temperature': temp
                }
            except:
                pass
        
        return {
            'cpu': cpu_percent,
            'memory': memory.percent,
            'temperature': None
        }
    
    def _get_pi_temperature(self):
        """Get Raspberry Pi CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return temp
        except:
            return None
    
    def check_thermal_throttle(self):
        """Check if Pi is thermal throttling"""
        if not self.is_pi:
            return False
        
        try:
            import subprocess
            result = subprocess.run(
                ['vcgencmd', 'get_throttled'],
                capture_output=True,
                text=True
            )
            throttled = result.stdout.strip()
            return 'throttled=0x0' not in throttled
        except:
            return False


class RaspberryPiDroneDetector:
    """Optimized drone detection for Raspberry Pi 5"""
    
    def __init__(
        self,
        model_path: str = 'yolov8n-obb.pt',
        conf_threshold: float = 0.35,  # Lower default for Pi
        skip_frames: int = 2,  # Skip by default on Pi
        resize_input: bool = False,
        target_width: int = 1280,
        verbose: bool = True
    ):
        """
        Initialize Pi-optimized detector
        
        Args:
            model_path: Path to YOLO model (use yolov8n-obb.pt)
            conf_threshold: Confidence threshold (0.35 recommended for Pi)
            skip_frames: Process every Nth frame (2 = 50% skip)
            resize_input: Resize frames before processing
            target_width: Target width if resizing
            verbose: Print detailed info
        """
        self.verbose = verbose
        self.conf_threshold = conf_threshold
        self.skip_frames = skip_frames
        self.resize_input = resize_input
        self.target_width = target_width
        
        # Initialize Pi optimizer
        self.pi_optimizer = RaspberryPiOptimizer()
        self.pi_optimizer.apply_optimizations()
        
        if self.verbose:
            print("\n" + "="*70)
            sprint("🥧 RASPBERRY PI 5 OPTIMIZED DRONE DETECTION")
            print("="*70)
        
        # Load model
        if self.verbose:
            sprint(f"\n📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to('cpu')  # Force CPU
        
        # Configure model for speed
        self.model.overrides['iou'] = 0.45  # Fast NMS
        self.model.overrides['max_det'] = 50  # Limit detections
        
        if self.verbose:
            sprint(f"   ✅ Model loaded")
            print(f"   • Device: CPU (Raspberry Pi)")
            print(f"   • Confidence: {conf_threshold}")
            print(f"   • Skip factor: {skip_frames}")
            print(f"   • Resize input: {resize_input}")
        
        self.tracker = None
        self.detections = []
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_detections': 0,
            'inference_time': 0,
            'tracking_time': 0,
            'resize_time': 0,
            'temperatures': [],
            'throttle_warnings': 0
        }
    
    def process_video(
        self,
        video_path: str,
        output_csv: str,
        monitor_interval: int = 30
    ):
        """
        Process video with Pi-specific optimizations
        
        Args:
            video_path: Path to input video
            output_csv: Path to output CSV
            monitor_interval: Frames between system monitoring
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate resize dimensions if enabled
        if self.resize_input:
            scale = self.target_width / width
            resize_height = int(height * scale)
            if self.verbose:
                sprint(f"\n📐 Will resize: {width}x{height} → {self.target_width}x{resize_height}")
        
        if self.verbose:
            sprint(f"\n📹 Video: {Path(video_path).name}")
            print(f"   • Original: {width}x{height} @ {fps} FPS")
            print(f"   • Total frames: {total_frames}")
            print(f"   • Estimated time: {total_frames / fps:.1f}s (video duration)")
            
            # Estimate processing time
            if self.skip_frames > 1:
                effective_frames = total_frames // self.skip_frames
                est_time = effective_frames * 0.6  # ~0.6s per frame on Pi 5
                print(f"   • Processing ~{effective_frames} frames (skip={self.skip_frames})")
                print(f"   • Estimated processing: {est_time:.0f}s (~{est_time/60:.1f} min)")
        
        # Initialize tracker
        self.tracker = ByteTrackWrapper(frame_rate=fps)
        
        print("\n" + "="*60)
        sprint("⏳ Processing... (This may take a while on Pi)")
        print("="*60)
        
        # Process frames
        frame_id = 0
        start_time = time.time()
        last_tracked_objects = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Monitor system every N frames
            if frame_id % monitor_interval == 0 and frame_id > 0:
                self._monitor_system(frame_id, total_frames, start_time)
            
            # Skip frame if configured
            if frame_id % self.skip_frames != 0:
                # Use last tracked objects for interpolation
                if last_tracked_objects:
                    for obj in last_tracked_objects:
                        self.detections.append({
                            'frame_id': frame_id,
                            'object_id': obj['object_id'],
                            'center_x': obj['center_x'],
                            'center_y': obj['center_y'],
                            'w': obj['w'],
                            'h': obj['h'],
                            'theta': obj['theta']
                        })
                
                self.stats['frames_skipped'] += 1
                frame_id += 1
                continue
            
            # Resize if enabled
            if self.resize_input:
                resize_start = time.time()
                frame_resized = cv2.resize(
                    frame, 
                    (self.target_width, resize_height),
                    interpolation=cv2.INTER_LINEAR
                )
                self.stats['resize_time'] += time.time() - resize_start
                process_frame = frame_resized
                scale_x = width / self.target_width
                scale_y = height / resize_height
            else:
                process_frame = frame
                scale_x = scale_y = 1.0
            
            # Detection
            inf_start = time.time()
            results = self.model.predict(
                process_frame,
                conf=self.conf_threshold,
                verbose=False
            )[0]
            self.stats['inference_time'] += time.time() - inf_start
            
            # Extract detections
            frame_detections = []
            detection_data = []
            
            if results.obb is not None and len(results.obb) > 0:
                obb = results.obb
                boxes = obb.xywhr.cpu().numpy()
                confs = obb.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    cx, cy, w, h, angle = box
                    
                    # Scale back to original size if resized
                    if self.resize_input:
                        cx *= scale_x
                        cy *= scale_y
                        w *= scale_x
                        h *= scale_y
                    
                    frame_detections.append((cx, cy))
                    detection_data.append({
                        'center': (cx, cy),
                        'size': (w, h),
                        'angle': angle,
                        'conf': float(conf)
                    })
            
            # Tracking
            track_start = time.time()
            tracked = self.tracker.update(frame_detections, frame)
            self.stats['tracking_time'] += time.time() - track_start
            
            # Match and save
            tracked_dict = {tuple(v): k for k, v in tracked.items()}
            last_tracked_objects = []
            
            for det_data in detection_data:
                cx, cy = det_data['center']
                centroid_key = (cx, cy)
                
                if centroid_key in tracked_dict:
                    track_id = tracked_dict[centroid_key]
                    w, h = det_data['size']
                    angle = det_data['angle']
                    
                    # Normalize
                    cx_norm = cx / width
                    cy_norm = cy / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    detection = {
                        'frame_id': frame_id,
                        'object_id': track_id,
                        'center_x': cx_norm,
                        'center_y': cy_norm,
                        'w': w_norm,
                        'h': h_norm,
                        'theta': angle
                    }
                    
                    self.detections.append(detection)
                    last_tracked_objects.append(detection)
                    self.stats['total_detections'] += 1
            
            self.stats['frames_processed'] += 1
            frame_id += 1
        
        cap.release()
        total_time = time.time() - start_time
        
        # Save results
        self._save_results(output_csv, total_time, total_frames)
    
    def _monitor_system(self, frame_id, total_frames, start_time):
        """Monitor system resources"""
        metrics = self.pi_optimizer.monitor_system()
        elapsed = time.time() - start_time
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        progress = frame_id / total_frames * 100
        eta = (total_frames - frame_id) / fps if fps > 0 else 0
        
        sprint(f"📊 Frame {frame_id}/{total_frames} ({progress:.1f}%)")
        print(f"   • FPS: {fps:.2f}")
        print(f"   • CPU: {metrics['cpu']:.1f}%")
        print(f"   • Memory: {metrics['memory']:.1f}%")
        
        if metrics['temperature']:
            temp = metrics['temperature']
            self.stats['temperatures'].append(temp)
            print(f"   • Temp: {temp:.1f}°C", end="")
            
            if temp > 70:
                sprint(" ⚠️  HIGH!")
                if temp > 80:
                    sprint("   ⚠️  WARNING: Thermal throttling likely!")
                    self.stats['throttle_warnings'] += 1
            else:
                sprint(" ✅")
        
        if eta > 0:
            print(f"   • ETA: {eta:.0f}s (~{eta/60:.1f} min)")
        print()
    
    def _save_results(self, output_csv, total_time, total_frames):
        """Save detection results"""
        if self.detections:
            df = pd.DataFrame(self.detections)
            df = df.sort_values(['frame_id', 'object_id'])
            Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_csv, index=False)
            
            unique_objects = df['object_id'].nunique()
            
            print("\n" + "="*60)
            sprint("✅ PROCESSING COMPLETE")
            print("="*60)
            sprint(f"📊 Results:")
            print(f"   • Total frames: {total_frames}")
            print(f"   • Frames processed: {self.stats['frames_processed']}")
            print(f"   • Frames skipped: {self.stats['frames_skipped']}")
            print(f"   • Processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"   • Average FPS: {self.stats['frames_processed']/total_time:.2f}")
            print(f"   • Detections: {len(df)}")
            print(f"   • Unique objects: {unique_objects}")
            
            sprint(f"\n⏱️  Time breakdown:")
            total_proc_time = self.stats['inference_time'] + self.stats['tracking_time']
            if total_proc_time > 0:
                print(f"   • Inference: {self.stats['inference_time']:.1f}s ({self.stats['inference_time']/total_proc_time*100:.1f}%)")
                print(f"   • Tracking: {self.stats['tracking_time']:.1f}s ({self.stats['tracking_time']/total_proc_time*100:.1f}%)")
                if self.resize_input:
                    print(f"   • Resize: {self.stats['resize_time']:.1f}s ({self.stats['resize_time']/total_proc_time*100:.1f}%)")
            
            if self.stats['temperatures']:
                avg_temp = sum(self.stats['temperatures']) / len(self.stats['temperatures'])
                max_temp = max(self.stats['temperatures'])
                sprint(f"\n🌡️  Temperature:")
                print(f"   • Average: {avg_temp:.1f}°C")
                print(f"   • Maximum: {max_temp:.1f}°C")
                
                if max_temp > 70:
                    sprint(f"   ⚠️  High temperature detected!")
                    sprint(f"   💡 Consider adding cooling solution")
                
                if self.stats['throttle_warnings'] > 0:
                    sprint(f"   ⚠️  Throttle warnings: {self.stats['throttle_warnings']}")
            
            if self.skip_frames > 1:
                effective_speedup = total_frames / self.stats['frames_processed']
                sprint(f"\n🚀 Optimization:")
                print(f"   • Skip factor: {self.skip_frames}x")
                print(f"   • Effective speedup: ~{effective_speedup:.1f}x")
            
            sprint(f"\n💾 Output: {output_csv}")
            print("="*60)
        else:
            sprint("\n⚠️  No detections saved")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Raspberry Pi 5 Optimized Drone Detection'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, 
                       default='submissions/p1_raspberry_pi.csv',
                       help='Output CSV file')
    parser.add_argument('--conf', type=float, default=0.35,
                       help='Confidence threshold (default: 0.35 for Pi)')
    parser.add_argument('--skip', type=int, default=2,
                       help='Process every Nth frame (default: 2)')
    parser.add_argument('--resize', action='store_true',
                       help='Resize frames before processing')
    parser.add_argument('--width', type=int, default=1280,
                       help='Target width if resizing (default: 1280)')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='Model path (use yolov8n-obb.pt for Pi)')
    
    args = parser.parse_args()
    
    # Verify running on ARM/Pi
    machine = platform.machine()
    if machine not in ['aarch64', 'armv7l', 'armv8', 'AMD64']:
        sprint(f"⚠️  Warning: Not ARM architecture (detected: {machine})")
        print(f"   This script is optimized for Raspberry Pi 5")
    
    # Create detector
    detector = RaspberryPiDroneDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        skip_frames=args.skip,
        resize_input=args.resize,
        target_width=args.width,
        verbose=True
    )
    
    # Process video
    detector.process_video(
        video_path=args.video,
        output_csv=args.output
    )


if __name__ == '__main__':
    main()
