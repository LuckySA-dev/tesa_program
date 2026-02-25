"""
Raspberry Pi 5 Deployment Guide
Setup and optimization for running the drone detection system on Raspberry Pi 5
"""

import platform
import subprocess
import sys
from pathlib import Path
from hw import sprint


class RaspberryPiChecker:
    """Check system compatibility and requirements for Raspberry Pi 5"""
    
    def __init__(self):
        self.is_raspberry_pi = self._check_raspberry_pi()
        self.requirements = {
            'python_version': (3, 9),  # Minimum Python 3.9
            'ram_gb': 4,  # Minimum 4GB RAM (Pi 5 has 4/8GB options)
            'storage_gb': 10,  # Minimum storage for models + data
        }
    
    def _check_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi"""
        try:
            # Check /proc/cpuinfo for Raspberry Pi signature
            if Path('/proc/cpuinfo').exists():
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                        return True
        except:
            pass
        
        # Check platform
        machine = platform.machine()
        if machine in ['aarch64', 'armv7l', 'armv8']:
            sprint("⚠️  ARM architecture detected (likely Raspberry Pi)")
            return True
        
        return False
    
    def check_system(self):
        """Perform comprehensive system check"""
        print("="*70)
        sprint("🔍 RASPBERRY PI 5 COMPATIBILITY CHECK")
        print("="*70)
        
        # System info
        sprint(f"\n📊 System Information:")
        print(f"   • Platform: {platform.system()}")
        print(f"   • Machine: {platform.machine()}")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • Architecture: {platform.architecture()[0]}")
        
        # Check if Raspberry Pi
        if self.is_raspberry_pi:
            sprint(f"   • Device: ✅ Raspberry Pi detected")
            self._check_pi_model()
        else:
            sprint(f"   • Device: ⚠️  Not Raspberry Pi (testing mode)")
        
        # Check Python version
        sprint(f"\n🐍 Python Version:")
        current = sys.version_info[:2]
        required = self.requirements['python_version']
        if current >= required:
            sprint(f"   ✅ Python {current[0]}.{current[1]} (>= {required[0]}.{required[1]})")
        else:
            sprint(f"   ❌ Python {current[0]}.{current[1]} (need >= {required[0]}.{required[1]})")
        
        # Check RAM (approximate)
        self._check_memory()
        
        # Check storage
        self._check_storage()
        
        # Check dependencies
        self._check_dependencies()
        
        print("\n" + "="*70)
    
    def _check_pi_model(self):
        """Check Raspberry Pi model"""
        try:
            result = subprocess.run(
                ['cat', '/proc/device-tree/model'],
                capture_output=True,
                text=True
            )
            model = result.stdout.strip('\x00')
            print(f"   • Model: {model}")
            
            if 'Raspberry Pi 5' in model:
                sprint(f"   ✅ Raspberry Pi 5 confirmed")
            elif 'Raspberry Pi 4' in model:
                sprint(f"   ⚠️  Raspberry Pi 4 (slower than Pi 5)")
            else:
                sprint(f"   ⚠️  Older model (may be too slow)")
        except:
            print(f"   • Model: Unable to detect")
    
    def _check_memory(self):
        """Check available RAM"""
        sprint(f"\n💾 Memory:")
        try:
            if Path('/proc/meminfo').exists():
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / (1024 ** 2)
                            print(f"   • Total RAM: {mem_gb:.1f} GB")
                            
                            if mem_gb >= self.requirements['ram_gb']:
                                sprint(f"   ✅ Sufficient RAM (>= {self.requirements['ram_gb']} GB)")
                            else:
                                sprint(f"   ⚠️  Low RAM (recommend {self.requirements['ram_gb']} GB)")
                            break
        except:
            print(f"   • RAM: Unable to detect")
    
    def _check_storage(self):
        """Check available storage"""
        sprint(f"\n💿 Storage:")
        try:
            import shutil
            stat = shutil.disk_usage('.')
            free_gb = stat.free / (1024 ** 3)
            total_gb = stat.total / (1024 ** 3)
            
            print(f"   • Total: {total_gb:.1f} GB")
            print(f"   • Free: {free_gb:.1f} GB")
            
            if free_gb >= self.requirements['storage_gb']:
                sprint(f"   ✅ Sufficient storage (>= {self.requirements['storage_gb']} GB)")
            else:
                sprint(f"   ⚠️  Low storage (need {self.requirements['storage_gb']} GB)")
        except:
            print(f"   • Storage: Unable to detect")
    
    def _check_dependencies(self):
        """Check installed packages"""
        sprint(f"\n📦 Dependencies:")
        
        required_packages = [
            ('cv2', 'opencv-python'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('torch', 'torch'),
            ('ultralytics', 'ultralytics'),
            ('supervision', 'supervision'),
        ]
        
        for module_name, package_name in required_packages:
            try:
                __import__(module_name)
                sprint(f"   ✅ {package_name}")
            except ImportError:
                sprint(f"   ❌ {package_name} (not installed)")
    
    def generate_installation_script(self, output_file='setup_raspberry_pi.sh'):
        """Generate installation script for Raspberry Pi"""
        script_content = """#!/bin/bash
# Raspberry Pi 5 Setup Script for TESA Defence Drone Detection System
# Run with: bash setup_raspberry_pi.sh

echo "========================================================================"
echo "RASPBERRY PI 5 SETUP - TESA DEFENCE SYSTEM"
echo "========================================================================"

# Update system
echo ""
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "🔧 Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libhdf5-dev libhdf5-serial-dev
sudo apt-get install -y libharfbuzz-dev libwebp-dev libjasper-dev
sudo apt-get install -y libilmbase-dev libopenexr-dev libgstreamer1.0-dev
sudo apt-get install -y ffmpeg

# Install Python packages (ARM-optimized versions)
echo ""
echo "🐍 Installing Python packages..."

# NumPy (use system package for better ARM support)
sudo apt-get install -y python3-numpy

# Pandas
pip3 install pandas --no-cache-dir

# OpenCV (already installed via apt)
# pip3 install opencv-python  # Skip, use system version

# PyTorch (ARM/CPU version)
echo ""
echo "🔥 Installing PyTorch (ARM CPU version)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Ultralytics YOLO
echo ""
echo "🎯 Installing Ultralytics..."
pip3 install ultralytics --no-cache-dir

# Supervision
echo ""
echo "📹 Installing Supervision..."
pip3 install supervision --no-cache-dir

# XGBoost (for Problem 2)
echo ""
echo "🌲 Installing XGBoost..."
pip3 install xgboost --no-cache-dir

# Additional dependencies
echo ""
echo "📚 Installing additional packages..."
pip3 install scikit-learn matplotlib seaborn tqdm

# Optimize for Raspberry Pi
echo ""
echo "⚙️  Applying Raspberry Pi optimizations..."

# Increase swap (helpful for compilation/large models)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/g' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Enable hardware acceleration
echo ""
echo "🎮 Enabling hardware acceleration..."
# Add user to video group for camera/GPU access
sudo usermod -a -G video $USER

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p ~/tesa_system/videos
mkdir -p ~/tesa_system/models
mkdir -p ~/tesa_system/submissions
mkdir -p ~/tesa_system/output
mkdir -p ~/tesa_system/logs

# Download models (if not already present)
echo ""
echo "📥 Downloading YOLO models..."
cd ~/tesa_system/models
if [ ! -f yolov8n-obb.pt ]; then
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-obb.pt
fi

echo ""
echo "========================================================================"
echo "✅ SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "📋 Next steps:"
echo "   1. Reboot system: sudo reboot"
echo "   2. Copy your project files to ~/tesa_system/"
echo "   3. Test with: python3 problem1_competition.py --video videos/test.mp4"
echo ""
echo "⚡ Performance tips:"
echo "   • Use --skip 2 for 2x faster processing"
echo "   • Reduce confidence threshold if needed"
echo "   • Monitor CPU temperature: vcgencmd measure_temp"
echo "   • Add cooling if temperature > 70°C"
echo ""
echo "========================================================================"
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        sprint(f"\n✅ Generated setup script: {output_file}")
        print(f"   To use: bash {output_file}")


class RaspberryPiOptimizer:
    """Optimize system for Raspberry Pi 5"""
    
    @staticmethod
    def estimate_performance():
        """Estimate performance on Raspberry Pi 5"""
        print("\n" + "="*70)
        sprint("📊 ESTIMATED PERFORMANCE ON RASPBERRY PI 5")
        print("="*70)
        
        benchmarks = {
            'Device': ['Desktop CPU', 'Raspberry Pi 5', 'Raspberry Pi 4'],
            'FPS (baseline)': [5.4, '~1.5-2.5', '~0.5-1.0'],
            'FPS (skip 50%)': [8.4, '~2.5-4.0', '~1.0-2.0'],
            'Processing (120 frames)': ['22s', '~60-90s', '~120-180s'],
            'Optimization': ['None', 'Required', 'Critical'],
        }
        
        print("\nComparison:")
        for key, values in benchmarks.items():
            print(f"{key:<25} {values[0]:<15} {values[1]:<20} {values[2]:<20}")
        
        sprint("\n💡 Recommendations:")
        sprint("   ✅ Use skip_frames=2 (process every 2nd frame)")
        sprint("   ✅ Lower confidence threshold (0.3-0.4)")
        sprint("   ✅ Reduce video resolution if possible")
        sprint("   ✅ Add cooling solution (heatsink + fan)")
        sprint("   ✅ Use lite model (yolov8n-obb.pt)")
        
        sprint("\n⚠️  Limitations on Raspberry Pi 5:")
        print("   • 3-5x slower than desktop CPU")
        print("   • No GPU acceleration for YOLO")
        print("   • May thermal throttle without cooling")
        print("   • Limited to ~2-4 FPS processing")
        
        sprint("\n✅ What works well:")
        print("   • Inference still possible")
        print("   • Skip frames helps significantly")
        print("   • Good for prototyping/demos")
        print("   • Can process shorter videos")
        
        print("="*70)
    
    @staticmethod
    def create_optimized_config():
        """Create optimized configuration for Raspberry Pi"""
        config = {
            'model': 'yolov8n-obb.pt',  # Smallest model
            'confidence': 0.35,  # Lower for better detection
            'skip_frames': 2,  # Process every 2nd frame
            'imgsz': 480,  # Smaller input size
            'max_det': 50,  # Limit max detections
            'device': 'cpu',  # No GPU
            'half': False,  # No FP16 on CPU
            'batch': 1,  # Single image at a time
        }
        
        return config


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Raspberry Pi 5 Deployment Tools')
    parser.add_argument('--check', action='store_true',
                       help='Check system compatibility')
    parser.add_argument('--generate-script', action='store_true',
                       help='Generate installation script')
    parser.add_argument('--estimate', action='store_true',
                       help='Estimate performance on Pi 5')
    parser.add_argument('--all', action='store_true',
                       help='Run all checks and generate script')
    
    args = parser.parse_args()
    
    checker = RaspberryPiChecker()
    optimizer = RaspberryPiOptimizer()
    
    if args.check or args.all:
        checker.check_system()
    
    if args.generate_script or args.all:
        checker.generate_installation_script('setup_raspberry_pi.sh')
        sprint("\n📄 Installation script created: setup_raspberry_pi.sh")
    
    if args.estimate or args.all:
        optimizer.estimate_performance()
    
    if not any([args.check, args.generate_script, args.estimate, args.all]):
        parser.print_help()
        print("\n" + "="*70)
        sprint("💡 QUICK START")
        print("="*70)
        print("\n1. Check compatibility:")
        print("   python raspberry_pi_deployment.py --check")
        print("\n2. Generate setup script:")
        print("   python raspberry_pi_deployment.py --generate-script")
        print("\n3. Estimate performance:")
        print("   python raspberry_pi_deployment.py --estimate")
        print("\n4. Run all:")
        print("   python raspberry_pi_deployment.py --all")
        print("="*70)


if __name__ == '__main__':
    main()
