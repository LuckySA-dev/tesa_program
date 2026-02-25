# 🚀 HOW TO RUN - TESA Drone Detection Project

**Quick Start Guide for Running the Drone Detection System**

Date: November 9, 2025

---

## 📋 Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Running Each Problem](#running-each-problem)
5. [Raspberry Pi 5 Deployment](#raspberry-pi-5)
6. [Dashboards & Monitoring](#dashboards)
7. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start

### **Option 1: Run Everything (Recommended)**
```bash
# Problem 1: Detection + Tracking (MUST RUN FIRST!)
python problem1_competition.py --video videos/video_01.mp4

# Problem 2: Localization (uses Problem 1 output)
python problem2_inference.py --detections submissions/p1_detection_obb.csv --video videos/video_01.mp4
python fix_problem2_competition_format.py  # Convert to competition format

# Problem 3: Complete Pipeline (all-in-one)
python problem3_integration.py --video videos/video_01.mp4

# Check Results (should be all PASS!)
python check_compliance.py
```

### **Option 2: Test Single Problem**
```bash
# Just detection
python problem1_competition.py --video videos/video_01.mp4 --conf 0.55

# Output: submissions/p1_detection_obb.csv
```

---

## 💻 System Requirements

### **Desktop/Laptop:**
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7+ / Ryzen 7+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB free | 20 GB+ |
| Python | 3.9+ | 3.11+ |
| OS | Windows 10/11, Linux, macOS | Any |

### **Raspberry Pi 5:**
| Component | Specification |
|-----------|---------------|
| Model | Raspberry Pi 5 (4GB or 8GB RAM) |
| OS | Raspberry Pi OS 64-bit |
| Storage | 32 GB+ microSD (Class 10) |
| Cooling | Active cooling recommended |
| Python | 3.9+ |

---

## 📦 Installation

### **Step 1: Clone/Download Project**
```bash
# If using Git
git clone https://github.com/LuckySA-dev/tesa.git
cd tesa

# Or download and extract ZIP
cd tesa
```

### **Step 2: Install Python Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually
pip install opencv-python numpy pandas torch torchvision ultralytics xgboost supervision pillow
```

### **Step 3: Verify Installation**
```bash
# Check Python version
python --version
# Should be 3.9 or higher

# Check installed packages
pip list | findstr "opencv torch ultralytics"
```

### **Step 4: Download Models (if needed)**
```bash
# YOLO models should be in project root
# yolov8n-obb.pt (lightweight)
# yolov8m-obb.pt (more accurate)

# XGBoost models should be in models/
# azimuth_deg_xgboost.pkl
# elevation_deg_xgboost.pkl
# range_m_xgboost.pkl
# metadata_xgboost.pkl
```

---

## 🎯 Running Each Problem

### **Problem 1: Drone Detection + Tracking**

**Goal:** Detect drones in video and track them across frames

#### **Basic Usage:**
```bash
python problem1_competition.py --video videos/video_01.mp4
```

#### **With Custom Parameters:**
```bash
python problem1_competition.py \
  --video videos/video_01.mp4 \
  --output submissions/my_detection.csv \
  --conf 0.55 \
  --model yolov8n-obb.pt
```

#### **Parameters:**
- `--video` : Path to input video (required)
- `--output` : Output CSV path (default: submissions/p1_detection_obb.csv)
- `--conf` : Confidence threshold (default: 0.55)
- `--model` : YOLO model path (default: yolov8n-obb.pt)
- `--save-video` : Save annotated video

#### **Expected Output:**
```
Processing: 22.4s
FPS: 5.4
Detections: 248
Unique objects: 3
Output: submissions/p1_detection_obb.csv
```

#### **Output Format (CSV):**
```csv
frame_id,object_id,center_x,center_y,w,h,theta
0,1,0.5234,0.4123,0.0234,0.0156,45.23
0,2,0.7123,0.6234,0.0198,0.0134,12.45
...
```

---

### **Problem 2: GPS Localization (Regression)**

**Goal:** Predict GPS coordinates and altitude from detection bounding boxes

**⚠️ IMPORTANT:** Must run Problem 1 first to generate detections!

#### **Basic Usage:**
```bash
# Step 1: Generate detections (if not done yet)
python problem1_competition.py --video videos/video_01.mp4

# Step 2: Run localization regression
python problem2_inference.py \
  --detections submissions/p1_detection_obb.csv \
  --video videos/video_01.mp4
```

#### **With Custom Parameters:**
```bash
python problem2_inference.py \
  --detections submissions/p1_detection_obb.csv \
  --video videos/video_01.mp4 \
  --output submissions/my_localization.csv \
  --model xgboost
```

#### **Parameters:**
- `--detections` : Path to detections CSV (required) - from Problem 1
- `--video` : Path to video file (for auto-detecting dimensions)
- `--output` : Output CSV path (default: predictions.csv)
- `--model` : Model type - 'xgboost' or 'random_forest' (default: xgboost)
- `--width` : Video width (auto-detected if --video provided)
- `--height` : Video height (auto-detected if --video provided)

#### **Expected Output:**
```
📦 Loading models from: models/
   ✅ Loaded: range_m_xgboost.pkl
   ✅ Loaded: azimuth_deg_xgboost.pkl
   ✅ Loaded: elevation_deg_xgboost.pkl
   • Features: 4
   • Targets: 3

📊 Predictions:
   Frame 0: Range=85.2m, Az=12.3°, El=15.7°
   Frame 0: Range=92.8m, Az=8.5°, El=18.2°
   ...
   
💾 Saved 248 predictions to predictions.csv
```

#### **Output Format (CSV):**
```csv
frame_id,object_id,range_m,azimuth_deg,elevation_deg
0,1,85.234,12.345,15.678
0,2,92.789,8.543,18.234
...
```

**⚠️ IMPORTANT: Convert to Competition Format**

The output needs to be converted to competition format:
```bash
python fix_problem2_competition_format.py
```

This converts:
- `azimuth_deg_pred` → `direction` (0-360°)
- `range_m_pred` → `distance` (meters)
- `elevation_deg_pred` → `height` (calculated: distance × sin(elevation))

**Final Competition Format:**
```csv
frame_id,object_id,direction,distance,height
0,1,12.35,85.23,18.52
0,2,8.54,92.79,13.74
...
```

---

### **Problem 3: Complete Integration**

**Goal:** Full pipeline - detection, tracking, and localization

#### **Basic Usage:**
```bash
python problem3_integration.py --video videos/video_01.mp4
```

#### **With Custom Parameters:**
```bash
python problem3_integration.py \
  --video videos/video_01.mp4 \
  --output submissions/complete_results.csv \
  --conf 0.55 \
  --save-video
```

#### **Parameters:**
- `--video` : Path to input video (required)
- `--output` : Output CSV path (default: submissions/submission.csv)
- `--conf` : Detection confidence (default: 0.55)
- `--save-video` : Save annotated video with tracking

#### **Expected Output:**
```
Stage 1: Detection + Tracking
  - Processed 120 frames
  - Found 3 unique objects
  - Detections: 248

Stage 2: Localization
  - Calculated GPS for 3 drones
  - Average altitude: 87.3m

Output: submissions/submission.csv
Video: output/complete_system.mp4
```

---

## 🥧 Raspberry Pi 5

### **Optimized Version:**
```bash
# Use Pi-optimized script
python problem1_raspberry_pi.py --video videos/video_01.mp4
```

### **With Optimization:**
```bash
python problem1_raspberry_pi.py \
  --video videos/video_01.mp4 \
  --conf 0.35 \
  --skip 2 \
  --resize \
  --width 1280
```

### **Parameters:**
- `--conf 0.35` : Lower confidence (better for Pi)
- `--skip 2` : Process every 2nd frame (2x faster)
- `--resize` : Resize frames before processing
- `--width 1280` : Target width (lower = faster)

### **Expected Performance:**
| Configuration | Desktop | Pi 5 | Speedup |
|---------------|---------|------|---------|
| Baseline | 22s (5.4 FPS) | ~75s (1.6 FPS) | 1x |
| Skip 2 frames | 14s (8.6 FPS) | ~45s (2.7 FPS) | 1.7x |
| Skip + Resize | 14s (8.6 FPS) | ~40s (3.0 FPS) | 1.9x |

### **Full Pi 5 Setup:**
```bash
# On Raspberry Pi 5
chmod +x setup_raspberry_pi.sh
./setup_raspberry_pi.sh

# Test
python raspberry_pi_deployment.py --all
```

---

## 📊 Dashboards & Monitoring

### **1. Performance Dashboard (Static)**
```bash
python dashboard_performance.py
```
**Output:** 
- Opens window with 6 performance graphs
- Saves to `performance_dashboard.png`

### **2. Real-time Monitor (Live)**
```bash
python dashboard_realtime.py
```
**Output:**
- Live FPS tracking
- Temperature monitoring
- Memory usage
- Detection count

### **3. Web Dashboard (Streamlit)**
```bash
# Install Streamlit first
pip install streamlit plotly

# Run web server
streamlit run dashboard_streamlit.py
```
**Output:**
- Opens browser at http://localhost:8501
- Upload videos
- See live processing
- Download results

---

## 🔧 Utilities

### **Validate Submission:**
```bash
# Check CSV format
python check_compliance.py

# Validate specific file
python validate_submission.py submissions/p1_detection_obb.csv
```

### **Fix Problem 2 Format:**
```bash
# Convert format if needed
python fix_problem2_format.py \
  --input submissions/p2_localization.csv \
  --output submissions/p2_localization_final.csv
```

### **Download External Datasets:**
```bash
# Kaggle integration
python kaggle_integration.py --download dasmehdixtr/drone-dataset-uav

# Test with external data
python kaggle_integration.py --test external_data/drone_dataset_yolo/
```

### **Visualize Results:**
```bash
# Create visualization
python visualize.py \
  --video videos/video_01.mp4 \
  --csv submissions/p1_detection_obb.csv \
  --output output/visualization.mp4
```

---

## 🎯 Complete Workflow

### **For Competition Submission:**

```bash
# Step 1: Run all problems
python problem1_competition.py --video videos/video_01.mp4
python problem2_inference.py --detections submissions/p1_detection_obb.csv --video videos/video_01.mp4
python fix_problem2_competition_format.py  # Convert to competition format!
python problem3_integration.py --video videos/video_01.mp4

# Step 2: Validate
python check_compliance.py
# Should show: ALL CHECKS PASSED!

python validate_submission.py submissions/p1_detection_obb.csv
python validate_submission.py submissions/p2_localization_final.csv

# Step 3: Check outputs
dir submissions
# Should see:
#   p1_detection_obb.csv
#   p2_localization_final.csv
#   submission.csv

# Step 4: Submit
# Upload files to competition platform
```

---

## 📁 Expected Outputs

### **After Running Everything:**
```
submissions/
├── p1_detection_obb.csv          ✅ Problem 1 output
├── p2_localization_final.csv     ✅ Problem 2 output
└── submission.csv                ✅ Problem 3 output

output/
├── complete_system.mp4           📹 Annotated video
├── test_tracking.mp4             📹 Tracking visualization
└── visualization.mp4             📹 Custom viz

logs/
└── test_tracking.csv             📊 Debug logs
```

---

## ⚠️ Troubleshooting

### **Issue 1: Module Not Found**
```bash
# Error: No module named 'cv2'
pip install opencv-python

# Error: No module named 'ultralytics'
pip install ultralytics

# Install all dependencies
pip install -r requirements.txt
```

### **Issue 2: CUDA/GPU Errors**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
set CUDA_VISIBLE_DEVICES=       # Windows

# Or edit config.py
# device = 'cpu'
```

### **Issue 3: Model Not Found**
```bash
# Download YOLO model
# Models will auto-download on first run
# Or manually download from Ultralytics

# Check models exist
ls -la *.pt
ls -la models/*.pkl
```

### **Issue 4: Video Not Opening**
```bash
# Check video path
python -c "import cv2; cap = cv2.VideoCapture('videos/video_01.mp4'); print('OK' if cap.isOpened() else 'ERROR')"

# Verify video codec
ffmpeg -i videos/video_01.mp4
```

### **Issue 5: Low FPS/Performance**
```bash
# Use lighter model
python problem1_competition.py --video videos/video_01.mp4 --model yolov8n-obb.pt

# Lower confidence
python problem1_competition.py --video videos/video_01.mp4 --conf 0.4

# Use Pi-optimized version
python problem1_raspberry_pi.py --video videos/video_01.mp4 --skip 2
```

### **Issue 6: Out of Memory**
```bash
# Close other applications
# Use smaller video resolution
# Use skip frames: --skip 2

# For Raspberry Pi
# Add swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### **Issue 7: CSV Format Errors**
```bash
# Auto-fix format
python fix_problem2_format.py --input your_file.csv --output fixed_file.csv

# Validate
python validate_submission.py fixed_file.csv

# Check compliance
python check_compliance.py
```

---

## 💡 Tips & Best Practices

### **Performance Optimization:**
1. **Use yolov8n-obb.pt** for faster processing
2. **Lower confidence** (0.35-0.45) for more detections
3. **Skip frames** (--skip 2) for 2x speed
4. **Resize video** for memory efficiency
5. **Close other apps** when processing

### **Accuracy Optimization:**
1. **Use yolov8m-obb.pt** for better accuracy
2. **Higher confidence** (0.55-0.65) for quality
3. **Process all frames** (no skip)
4. **Full resolution** processing
5. **Tune thresholds** per video

### **For Raspberry Pi:**
1. **Always use cooling** (heatsink + fan)
2. **Use skip frames** (--skip 2 or 3)
3. **Lower confidence** (0.35)
4. **Monitor temperature** with dashboard
5. **Expect 3-5x slower** than desktop

### **For Production:**
1. **Validate all outputs** before submission
2. **Keep backups** of working submissions
3. **Test with multiple videos**
4. **Check CSV format** compliance
5. **Document parameters** used

---

## 📚 Additional Resources

### **Documentation:**
- `PROJECT_ORGANIZATION.md` - Project structure
- `ORGANIZATION.md` - Development guide
- `reports/` - Detailed reports
- `old/README.md` - Archived files reference

### **Optimization Guides:**
- `reports/OPTIMIZATION_REPORT.md` - Performance tuning
- `reports/RASPBERRY_PI_DEPLOYMENT.md` - Pi 5 guide
- `reports/EXTERNAL_DATASET_TESTING.md` - Dataset guide

### **Tools:**
- `organize_old_files.py` - Project cleanup
- `cleanup_project.py` - Remove unnecessary files
- `raspberry_pi_deployment.py` - Pi setup tools

---

## 🎉 Quick Reference

### **Most Common Commands:**

```bash
# 1. Basic run
python problem1_competition.py --video videos/video_01.mp4

# 2. Raspberry Pi
python problem1_raspberry_pi.py --video videos/video_01.mp4 --skip 2

# 3. Complete pipeline
python problem3_integration.py --video videos/video_01.mp4

# 4. Validate
python check_compliance.py

# 5. Web dashboard
streamlit run dashboard_streamlit.py

# 6. Performance analysis
python dashboard_performance.py
```

---

## ✅ Checklist

**Before Running:**
- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models downloaded (*.pt, *.pkl)
- [ ] Video files available
- [ ] Sufficient disk space (10GB+)

**After Running:**
- [ ] CSV files generated in submissions/
- [ ] Output format validated
- [ ] No errors in console
- [ ] Results make sense
- [ ] Ready to submit

---

## 🆘 Getting Help

**If you encounter issues:**

1. **Check error message** - Most errors are self-explanatory
2. **Read troubleshooting** section above
3. **Validate installation** - Check all dependencies
4. **Try simpler command** - Start with basic usage
5. **Check documentation** - Read relevant reports
6. **Review logs** - Check error details

**Common Solutions:**
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Update packages: `pip install --upgrade ultralytics torch opencv-python`
- Clear cache: `rm -rf __pycache__/`
- Restart terminal/IDE
- Check Python version: `python --version`

---

## 🚀 Ready to Go!

**You're all set! Start with:**

```bash
python problem1_competition.py --video videos/video_01.mp4
```

**Good luck! 🎯**

---

*Last updated: November 9, 2025*  
*Project: TESA Defence AI-Based Drone Detection*  
*Deadline: November 11-13, 2025*
