#!/bin/bash
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
