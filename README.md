<p align="center">
  <h1 align="center">TESA Defence AI</h1>
  <p align="center">
    Real-time drone detection, multi-object tracking &amp; GPS localization<br/>
    <em>TESA Top Gun Rally 2024 — Defence AI Challenge</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/YOLOv8-OBB-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"/>
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
    <img src="https://img.shields.io/badge/version-1.0.0-green" alt="v1.0.0"/>
  </p>
</p>

---

## Overview

End-to-end AI pipeline that detects drones from video feeds, tracks them across frames, and estimates their real-world GPS coordinates — built for the TESA Top Gun Rally 2024 Defence AI competition.

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Detection** | YOLOv8-OBB | Oriented bounding box detection (mAP 81%) |
| **Tracking** | ByteTrack | Multi-object tracking with persistent IDs |
| **Localization** | XGBoost + NN | Bbox → distance/bearing → lat, lon, altitude |
| **Dashboard** | Streamlit | 10-page interactive web UI |
| **CLI** | argparse | 12+ commands for batch processing |
| **API** | Multi-protocol | REST, MQTT, WebSocket, gRPC satellite link |
| **Deploy** | Docker | One-command build & run |

---

## Quick Start

### Prerequisites

- Python 3.10+ (tested on 3.11, 3.13)
- pip or Docker

### Option A — Local Install

```bash
# Clone the repository
git clone https://github.com/<your-org>/tesa-defence-ai.git
cd tesa-defence-ai

# Create virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### Option B — Docker (recommended)

```bash
# Build & start dashboard
docker compose up --build

# Open http://localhost:8501
```

---

## Usage

### Streamlit Dashboard

```bash
python -m streamlit run dashboard.py
```

10-page interactive dashboard with pipeline controls, CSV inspector, model inventory, video browser, API tester, and more.

### CLI

```bash
python cli.py --help                  # All commands
python cli.py detect    --video videos/P3_VIDEO.mp4
python cli.py localize  --detections submissions/p1_detection_obb.csv
python cli.py pipeline  --video videos/P3_VIDEO.mp4
python cli.py pipeline-adv            # Production pipeline (P3)
python cli.py api-test  --protocol mock
python cli.py models                  # List models
python cli.py info                    # System info
```

### Run Pipelines Directly

```bash
# Problem 1 — Detection & Tracking
python src/problem1_competition.py --video videos/video_01.mp4

# Problem 2 — GPS Localization
python src/problem2_inference.py --detections submissions/p1_detection_obb.csv

# Problem 3 — Full Pipeline (Detection → Tracking → GPS → Video)
python src/problem_3_pipeline.py

# Quick smoke test
python test_pipeline_short.py
```

---

## Project Structure

```
tesa-defence-ai/
├── src/                            # Core source code
│   ├── problem_3_pipeline.py       #   Main pipeline (P1 + P2 + visualization)
│   ├── detector.py                 #   YOLOv8-OBB wrapper
│   ├── tracker.py                  #   ByteTrack / BoT-SORT tracker
│   ├── localizer.py                #   XGBoost GPS prediction
│   ├── visualizer.py               #   Video annotation & overlay
│   ├── api_client.py               #   Multi-protocol satellite API client
│   ├── hw.py                       #   Hardware detection (CPU/GPU/safe print)
│   ├── config.py                   #   Central configuration
│   ├── problem1_competition.py     #   P1 detection competition entry
│   ├── problem2_inference.py       #   P2 localization inference
│   ├── problem3_integration.py     #   P3 integration (alternative)
│   └── ...                         #   Utilities, validators, formatters
│
├── configs/                        # Tracker & dataset YAML configs
├── models/                         # Trained model weights
│   ├── tomorbest.pt                #   Custom YOLOv8n best model
│   ├── models_approximation/       #   XGBoost sub-models (bbox→GPS)
│   └── yolov8n-obb.pt              #   Pretrained YOLOv8 nano OBB
│
├── runs/detect/                    # YOLO training runs
│   └── drone_detect_v21_max_data/  #   Production model (mAP 81%)
│
├── scripts/                        # Training & analysis scripts
│   ├── 01_data_exploration/        #   EDA notebooks
│   ├── 02_yolo_preparation/        #   Dataset preparation
│   ├── 03_yolo_training/           #   YOLO training configs
│   ├── 04_xgboost_training/        #   XGBoost regression training
│   └── ...                         #   Evaluation, prediction, ensemble
│
├── dashboard.py                    # Streamlit web dashboard
├── cli.py                          # Command-line interface
├── test_pipeline_short.py          # Smoke test
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Docker Compose (dashboard + CLI + GPU)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── setup_raspberry_pi.sh           # Raspberry Pi 5 setup script
```

---

## Pipeline Architecture

```
Video → YOLOv8-OBB Detection (conf=0.10)
      → ByteTrack Tracking (track_buffer=180)
      → Weighted NMS (IOU=0.3)
      → Track Merging (→ 2 drone IDs)
      → GPS Localization (XGBoost bbox→distance/bearing/altitude)
      → Coordinate Smoothing (5-frame moving average)
      → Annotated Video + CSV Submission
```

### Models

| Pipeline | Detection | Localization |
|----------|-----------|-------------|
| **Problem 3** (main) | `drone_detect_v21` — custom YOLOv8n, mAP 81% | XGBoost approximation + NN correction |
| **Problem 1** | `yolov8n-obb.pt` — pretrained | — |
| **Problem 2** | — | XGBoost regression (range, azimuth, elevation) |

### Performance

| Metric | Value |
|--------|-------|
| Detection rate | 99.1% |
| Tracked IDs | 2 (consistent) |
| Processing speed | 12–14 FPS (CPU) |
| XGBoost MAE | range 0.95 m, azimuth 1.43° |

---

## API — Multi-Protocol Satellite Communication

The system can send drone alerts via multiple protocols:

| Protocol | Library | Status |
|----------|---------|--------|
| REST (HTTP/HTTPS) | `requests` | Built-in |
| MQTT | `paho-mqtt` | Built-in |
| WebSocket | `websocket-client` | Built-in |
| gRPC | `grpcio` | Optional |
| Mock | — | Built-in (testing) |

```python
from api_client import create_client

client = create_client("rest", url="https://api.example.com/drone")
client.test_connection()
client.send_first_alarm(drone_count=3, frame=img)
client.send_tracking_data(objects)
client.close()
```

Configure via environment variables or `src/config.py`:

```bash
API_PROTOCOL=rest
API_URL=https://api.tesa.or.th/drone
API_KEY=your-key
API_TIMEOUT=10
API_RETRIES=2
```

---

## Docker

### Services

| Service | Description | Port |
|---------|-------------|------|
| `dashboard` | Streamlit web UI | 8501 |
| `cli` | Interactive CLI | — |
| `dashboard-gpu` | GPU-accelerated dashboard | 8501 |

### Commands

```bash
# Dashboard (default)
docker compose up

# Interactive CLI
docker compose run --rm cli

# GPU mode (requires nvidia-docker)
docker compose --profile gpu up dashboard-gpu

# One-off pipeline run
docker compose run --rm cli python src/problem_3_pipeline.py

# Build manually
docker build -t tesa-defence .
docker run -p 8501:8501 tesa-defence
```

### Volumes

| Host | Container | Mode |
|------|-----------|------|
| `./videos` | `/app/videos` | ro |
| `./outputs` | `/app/outputs` | rw |
| `./submissions` | `/app/submissions` | rw |
| `./.env` | `/app/.env` | ro |

---

## Hardware Compatibility

The system auto-detects hardware and selects the best available device:

| Platform | Support |
|----------|---------|
| x86_64 + NVIDIA GPU (CUDA) | Full acceleration |
| x86_64 CPU-only | Full support (auto-fallback) |
| Raspberry Pi 5 | Optimized pipeline (`problem1_raspberry_pi.py`) |
| Docker (CPU / GPU) | Full support |

> GPU compatibility issues (e.g., unsupported `sm_*` arch) are handled
> automatically — the system falls back to CPU with no manual config needed.

---

## Environment Variables

All optional — sensible defaults are used when unset. See [.env.example](.env.example).

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_CPU` | `0` | Force CPU even if CUDA available |
| `P1_CONF_THRESHOLD` | `0.10` | Detection confidence |
| `P3_VIDEO_PATH` | `videos/P3_VIDEO.mp4` | Input video |
| `API_PROTOCOL` | `rest` | API protocol (rest/mqtt/ws/grpc/mock) |
| `API_URL` | `https://api.tesa.or.th/drone` | API endpoint |
| `API_KEY` | — | API authentication key |
| `STREAMLIT_SERVER_PORT` | `8501` | Dashboard port |

---

## Development

### Project Setup

```bash
git clone https://github.com/<your-org>/tesa-defence-ai.git
cd tesa-defence-ai
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Running Tests

```bash
# Quick smoke test
python test_pipeline_short.py

# API real-world test
python src/test_api_real.py

# Submission format validation
python cli.py compliance
```

### Code Structure Conventions

- All source modules are in `src/` and import hardware utils from `hw.py`
- `sprint()` is used instead of `print()` for safe Unicode output on all platforms
- `get_device()` from `hw.py` handles CPU/GPU selection centrally
- Config is centralized in `src/config.py` with env var overrides

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Detailed run instructions |
| [docs/RASPBERRY_PI_DEPLOYMENT.md](docs/RASPBERRY_PI_DEPLOYMENT.md) | Pi 5 setup guide |
| [docs/OPTIMIZATION_REPORT.md](docs/OPTIMIZATION_REPORT.md) | Performance tuning notes |
| [docs/FINAL_STATUS.md](docs/FINAL_STATUS.md) | Competition submission status |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for <strong>TESA Top Gun Rally 2024</strong> — Defence AI Challenge
</p>
