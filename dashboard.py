"""
TESA Defence AI - Streamlit Dashboard
======================================
Comprehensive visualization and control dashboard for the drone detection system.
All pipeline operations are fully functional with real-time streaming output.

Run with: python -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import glob
import subprocess
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# --------------------------------------------------------------------------- #
#  Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TESA Defence AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
#  Custom CSS – compact log containers
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    /* Cap the height of st.code / st.status output blocks */
    div[data-testid="stStatusWidget"] pre,
    div[data-testid="stCode"] pre {
        max-height: 280px;
        overflow-y: auto;
        font-size: 0.78rem;
        line-height: 1.35;
    }
    /* Compact expander contents */
    div[data-testid="stExpander"] div[data-testid="stCode"] pre {
        max-height: 220px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  Session state defaults
# --------------------------------------------------------------------------- #
if "last_run_output" not in st.session_state:
    st.session_state.last_run_output = ""
if "last_run_returncode" not in st.session_state:
    st.session_state.last_run_returncode = None
if "last_run_page" not in st.session_state:
    st.session_state.last_run_page = ""
if "running_process" not in st.session_state:
    st.session_state.running_process = None
if "current_page" not in st.session_state:
    st.session_state.current_page = ""


def _kill_running_process():
    """Kill any running subprocess (stop button / page change)."""
    proc = st.session_state.get("running_process")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    st.session_state.running_process = None


def _is_pipeline_active() -> bool:
    """Check if a subprocess is currently running."""
    proc = st.session_state.get("running_process")
    return proc is not None and proc.poll() is None


def _safe_upload(uploaded_file, target_dir: Path) -> Path:
    """Save uploaded file, falling back to /tmp when target is read-only."""
    try:
        ensure_dir(target_dir)
        dest = target_dir / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())
        return dest
    except OSError:
        import tempfile as _tf
        tmp = Path(_tf.gettempdir()) / "tesa_uploads"
        ensure_dir(tmp)
        dest = tmp / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())
        st.info(f"Directory read-only; saved to temp: {tmp}")
        return dest


def _fix_video_for_browser(video_path: Path) -> Path:
    """Run ffmpeg -movflags +faststart so the browser can play mp4v videos."""
    import shutil
    if shutil.which("ffmpeg") is None:
        return video_path
    fixed = video_path.parent / f".web_{video_path.name}"
    # Skip if already fixed and up-to-date
    if fixed.exists() and fixed.stat().st_mtime >= video_path.stat().st_mtime:
        return fixed
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-c", "copy", "-movflags", "+faststart", str(fixed)],
            capture_output=True, timeout=120,
        )
        if fixed.exists() and fixed.stat().st_size > 0:
            return fixed
    except Exception:
        pass
    return video_path

# --------------------------------------------------------------------------- #
#  Streaming subprocess helper
# --------------------------------------------------------------------------- #

def run_pipeline_streaming(cmd: list, label: str = "Running pipeline...") -> tuple:
    """
    Execute a subprocess with real-time streaming output in Streamlit.
    Stores the process in session_state so it can be killed on Stop / page change.
    Returns (returncode, full_output_text).
    """
    _kill_running_process()  # kill any leftover

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    output_lines = []

    with st.status(label, expanded=True) as status:
        st.caption(f"Command: `{' '.join(cmd)}`")
        log_area = st.empty()
        progress_text = st.empty()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PROJECT_ROOT),
                env=env,
                bufsize=1,
            )
            st.session_state.running_process = process

            start_time = time.time()

            for line in iter(process.stdout.readline, ""):
                if process.poll() is not None:
                    break
                stripped = line.rstrip()
                if stripped:
                    output_lines.append(stripped)
                    display_text = "\n".join(output_lines[-30:])
                    log_area.code(display_text, language="text")
                    elapsed = time.time() - start_time
                    progress_text.caption(
                        f"Elapsed: {elapsed:.1f}s | Lines: {len(output_lines)}"
                    )

            process.wait(timeout=10)
            elapsed = time.time() - start_time
            st.session_state.running_process = None

            if process.returncode == 0:
                status.update(
                    label=f"{label} -- Done ({elapsed:.1f}s)",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label=f"{label} -- Failed (code {process.returncode})",
                    state="error",
                    expanded=True,
                )

        except FileNotFoundError:
            output_lines.append("ERROR: Python executable not found!")
            status.update(label=f"{label} -- Failed", state="error")
            st.session_state.running_process = None
            return -1, "\n".join(output_lines)
        except Exception as e:
            output_lines.append(f"ERROR: {e}")
            status.update(label=f"{label} -- Failed", state="error")
            st.session_state.running_process = None
            return -1, "\n".join(output_lines)

    full_output = "\n".join(output_lines)
    st.session_state.last_run_output = full_output
    st.session_state.last_run_returncode = process.returncode
    return process.returncode, full_output


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(path)


def load_csv_fresh(path: str) -> pd.DataFrame:
    """Load a CSV file WITHOUT caching (for freshly generated results)."""
    return pd.read_csv(path)


@st.cache_data
def list_models() -> dict:
    """Scan model directories and return structured info."""
    models_dir = PROJECT_ROOT / "models"
    runs_dir = PROJECT_ROOT / "runs"
    info = {
        "yolo": [],
        "xgboost": [],
        "approximation": [],
        "stacking": [],
        "runs": [],
    }

    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                entry = {
                    "name": f.name,
                    "size_mb": round(size_mb, 2),
                    "path": str(f),
                }
                if f.suffix == ".pt":
                    info["yolo"].append(entry)
                elif f.suffix == ".pkl":
                    info["xgboost"].append(entry)

        approx_dir = models_dir / "models_approximation"
        if approx_dir.exists():
            for f in sorted(approx_dir.iterdir()):
                if f.is_file():
                    size_mb = f.stat().st_size / (1024 * 1024)
                    info["approximation"].append(
                        {
                            "name": f.name,
                            "size_mb": round(size_mb, 2),
                            "path": str(f),
                        }
                    )

        stack_dir = models_dir / "models_stacking"
        if stack_dir.exists():
            for f in sorted(stack_dir.iterdir()):
                if f.is_file():
                    size_mb = f.stat().st_size / (1024 * 1024)
                    info["stacking"].append(
                        {
                            "name": f.name,
                            "size_mb": round(size_mb, 2),
                            "path": str(f),
                        }
                    )

    if runs_dir.exists():
        for run_dir in sorted(runs_dir.rglob("weights/best.pt")):
            run_name = run_dir.parent.parent.name
            size_mb = run_dir.stat().st_size / (1024 * 1024)
            info["runs"].append(
                {
                    "name": run_name,
                    "size_mb": round(size_mb, 2),
                    "path": str(run_dir),
                }
            )

    return info


def get_yolo_model_options() -> dict:
    """Return dict of {display_name: full_path} for all YOLO models."""
    models = list_models()
    options = {}
    # Training runs first (best models)
    for m in models["runs"]:
        label = f"🏆 {m['name']} (trained, {m['size_mb']:.1f} MB)"
        options[label] = m["path"]
    # Base models
    for m in models["yolo"]:
        label = f"📦 {m['name']} ({m['size_mb']:.1f} MB)"
        options[label] = m["path"]
    return options


def list_videos() -> list:
    """Return list of video files (including temp uploads in Docker)."""
    vids = []
    vid_dirs = [PROJECT_ROOT / "videos"]
    import tempfile
    tmp_vid = Path(tempfile.gettempdir()) / "tesa_videos"
    if tmp_vid.exists():
        vid_dirs.append(tmp_vid)
    tmp_up = Path(tempfile.gettempdir()) / "tesa_uploads"
    if tmp_up.exists():
        vid_dirs.append(tmp_up)
    for vid_dir in vid_dirs:
        if vid_dir.exists():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                vids += list(vid_dir.glob(ext))
    # Deduplicate by name (prefer primary dir)
    seen = {}
    for v in vids:
        if v.name not in seen:
            seen[v.name] = v
    return sorted(seen.values(), key=lambda p: p.name)


def list_submissions() -> list:
    """Return submission CSV paths."""
    sub_dir = PROJECT_ROOT / "submissions"
    if sub_dir.exists():
        return sorted(sub_dir.glob("*.csv"))
    return []


@st.cache_data
def get_config_dict() -> dict:
    """Import and combine all configs into a single dict."""
    try:
        from config import (
            MODEL_CONFIG,
            TRACKING_CONFIG,
            VISUALIZATION_CONFIG,
            VIDEO_CONFIG,
            LOGGING_CONFIG,
            API_CONFIG,
            HARDWARE_CONFIG,
            ALERT_RULES,
            PATHS,
        )

        return {
            "MODEL_CONFIG": MODEL_CONFIG,
            "TRACKING_CONFIG": TRACKING_CONFIG,
            "VISUALIZATION_CONFIG": {
                k: v
                for k, v in VISUALIZATION_CONFIG.items()
                if k != "track_colors"
            },
            "VIDEO_CONFIG": VIDEO_CONFIG,
            "LOGGING_CONFIG": {
                k: v
                for k, v in LOGGING_CONFIG.items()
                if k != "csv_fields"
            },
            "API_CONFIG": API_CONFIG,
            "HARDWARE_CONFIG": HARDWARE_CONFIG,
            "ALERT_RULES": ALERT_RULES,
            "PATHS": {k: str(v) for k, v in PATHS.items()},
        }
    except Exception as e:
        return {"error": str(e)}


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def show_csv_results(csv_path: Path, title: str = "Results"):
    """Display CSV results with data, stats, and charts."""
    if not csv_path.exists():
        st.info(f"No results file found at: {csv_path.name}")
        return

    df = load_csv_fresh(str(csv_path))
    file_size = format_file_size(csv_path.stat().st_size)
    modified = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.subheader(f"📊 {title}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("File Size", file_size)
    col4.metric("Last Modified", modified)

    tab_data, tab_stats, tab_charts = st.tabs(
        ["📋 Data Table", "📈 Statistics", "📊 Charts"]
    )

    with tab_data:
        st.dataframe(df.head(500), width="stretch", hide_index=True)
        if len(df) > 500:
            st.caption(f"Showing first 500 of {len(df)} rows")

    with tab_stats:
        st.write(df.describe())
        # Specific metrics based on columns
        numeric_cols = [
            c for c in df.columns if df[c].dtype in ("float64", "int64", "float32", "int32")
        ]
        if numeric_cols:
            metric_cols = st.columns(min(len(numeric_cols), 4))
            for idx, col_name in enumerate(numeric_cols[: len(metric_cols)]):
                metric_cols[idx].metric(
                    col_name,
                    f"{df[col_name].mean():.2f} (avg)",
                    f"Range: {df[col_name].min():.2f} — {df[col_name].max():.2f}",
                )

    with tab_charts:
        frame_col = (
            "frame_id"
            if "frame_id" in df.columns
            else ("frame" if "frame" in df.columns else None)
        )

        if frame_col and "object_id" in df.columns:
            detections_per_frame = (
                df.groupby(frame_col).size().reset_index(name="count")
            )
            st.line_chart(detections_per_frame.set_index(frame_col)["count"])
            st.caption("Detections per frame")

        # Scatter for position data
        if "center_x" in df.columns and "center_y" in df.columns:
            st.scatter_chart(df[["center_x", "center_y"]].head(1000))
            st.caption("Detection center positions")
        elif "cx" in df.columns and "cy" in df.columns:
            st.scatter_chart(df[["cx", "cy"]].head(1000))
            st.caption("Detection positions (cx, cy)")

        # Distance vs Height
        if "distance" in df.columns and "height" in df.columns:
            st.scatter_chart(df[["distance", "height"]].head(1000))
            st.caption("Distance vs Height")

        # Prediction columns over time
        if frame_col:
            pred_cols = [c for c in df.columns if "_pred" in c]
            for pc in pred_cols:
                st.line_chart(df.set_index(frame_col)[pc].head(1000))
                st.caption(f"{pc} over frames")

        if "direction" in df.columns and frame_col:
            st.line_chart(df.set_index(frame_col)["direction"].head(500))
            st.caption("Direction over frames")

    # Download button
    csv_data = df.to_csv(index=False)
    st.download_button(
        f"⬇️ Download {csv_path.name}",
        csv_data,
        file_name=csv_path.name,
        mime="text/csv",
        width="stretch",
    )


# --------------------------------------------------------------------------- #
#  Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ TESA Defence AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Detection (P1)",
        "📍 Localization (P2)",
        "🔗 Full Pipeline (P3)",
        "📊 Submissions",
        "✅ Compliance Check",
        "🧠 Models",
        "⚙️ Configuration",
        "🎬 Videos",
        "📡 API Testing",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Project: {PROJECT_ROOT.name}")
st.sidebar.caption(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Kill any running subprocess when the user switches pages
if st.session_state.current_page and st.session_state.current_page != page:
    _kill_running_process()
st.session_state.current_page = page


# =========================================================================== #
#  PAGE: Home
# =========================================================================== #
if page == "🏠 Home":
    st.title("🛡️ TESA Defence AI — Drone Detection System")
    st.markdown("**Competition**: TESA Top Gun Rally 2024 — Defence AI")
    st.markdown("---")

    # Key metrics row
    models = list_models()
    videos = list_videos()
    submissions = list_submissions()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("YOLO Models", len(models["yolo"]) + len(models["runs"]))
    col2.metric("XGBoost Models", len(models["xgboost"]) + len(models["approximation"]))
    col3.metric("Videos", len(videos))
    col4.metric("Submissions", len(submissions))

    st.markdown("---")

    # Pipeline overview
    st.subheader("Pipeline Overview")
    st.markdown(
        """
    ```
    Video Input ──► YOLO-OBB Detection ──► ByteTrack Tracking ──► Track Merging
                        (conf=0.10)          (buffer=180)         (5 tracks→2 IDs)
                                                                        │
                                                                        ▼
    CSV Output  ◄── Coord Smoothing  ◄── XGBoost Localization ◄── Weighted NMS
                     (5-frame avg)        (bbox→range/az/el)       (IOU=0.3)
    ```
    """
    )

    # Performance table
    st.subheader("Model Performance")
    perf_data = {
        "Model": ["drone_detect_v21_max_data", "yolov8n-obb.pt", "tomorbest.pt"],
        "Type": ["YOLOv8-OBB (Custom)", "YOLOv8n-OBB (Base)", "YOLOv8 (Custom)"],
        "mAP@50": ["81.0%", "~65%", "~70%"],
        "Recall": ["90.0%", "~75%", "~78%"],
        "Status": ["✅ Production", "📦 Available", "📦 Available"],
    }
    st.dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

    st.subheader("Quick Start")
    st.code(
        """
# Run full pipeline (Problem 3)
python src/problem_3_pipeline.py

# Run detection only (Problem 1)
python src/problem1_competition.py --video videos/P3_VIDEO.mp4

# Run localization only (Problem 2)
python src/problem2_inference.py --detections submissions/p1_detection_obb.csv

# Use CLI tool
python cli.py pipeline --video videos/P3_VIDEO.mp4
python cli.py detect --video videos/P3_VIDEO.mp4
    """,
        language="bash",
    )

    # System info
    st.markdown("---")
    st.subheader("System Information")
    try:
        from hw import system_summary
        info = system_summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**Python**: {info.get('python', '?')}")
        col2.write(f"**OS**: {info.get('os', '?')}")
        col3.write(f"**PyTorch**: {info.get('torch', 'N/A')} (CUDA: {info.get('cuda_available', False)})")
        col4.write(f"**Device**: {info.get('device', 'cpu')}")
        if info.get("gpu_name"):
            st.info(f"GPU: {info['gpu_name']}")
    except Exception:
        import platform
        col1, col2 = st.columns(2)
        col1.write(f"**Python**: {platform.python_version()}")
        col2.write(f"**OS**: {platform.system()} {platform.release()}")


# =========================================================================== #
#  PAGE: Detection (P1)
# =========================================================================== #
elif page == "🎯 Detection (P1)":
    st.title("🎯 Problem 1 — Drone Detection with OBB")
    st.markdown("YOLOv8-OBB detection + ByteTrack multi-object tracking")
    st.markdown("---")

    with st.expander("ℹ️ About Problem 1", expanded=False):
        st.markdown(
            """
        - **Input**: Video file (MP4)
        - **Output**: CSV with `frame_id, object_id, center_x, center_y, w, h, theta`
        - **Coordinates**: Normalized (0-1)
        - **Model**: YOLOv8-OBB trained on drone dataset
        - **Tracker**: ByteTrack (built into ultralytics)
        """
        )

    # --- Configuration ---
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)

    with col1:
        # Video selection
        videos = list_videos()
        video_names = [v.name for v in videos]
        video_map = {v.name: v for v in videos}
        if video_names:
            selected_video = st.selectbox("Video File", video_names)
        else:
            selected_video = None
            st.warning("No videos found in videos/ directory")

        # Upload video
        uploaded_video = st.file_uploader(
            "Or upload a video",
            type=["mp4", "avi", "mov", "mkv"],
            key="p1_upload",
        )
        if uploaded_video is not None:
            _safe_upload(uploaded_video, PROJECT_ROOT / "videos")
            st.success(f"Uploaded: {uploaded_video.name}")
            selected_video = uploaded_video.name

        # Model selection
        model_options = get_yolo_model_options()
        if model_options:
            model_labels = list(model_options.keys())
            selected_model_label = st.selectbox("🧠 Detection Model", model_labels)
            selected_model_path = model_options[selected_model_label]
        else:
            selected_model_path = "yolov8n-obb.pt"
            st.info("Using default model: yolov8n-obb.pt")

    with col2:
        conf_threshold = st.slider(
            "Confidence Threshold", 0.01, 1.0, 0.10, 0.01, key="p1_conf"
        )
        output_name = st.text_input(
            "Output CSV Name", "p1_detection_output.csv", key="p1_output"
        )
        use_bytetrack = st.checkbox("Use ByteTrack Tracker", value=True, key="p1_bt")
        save_video = st.checkbox(
            "Save Annotated Video", value=False, key="p1_savevid"
        )

        if save_video:
            video_output_name = st.text_input(
                "Output Video Name",
                "p1_detection_output.mp4",
                key="p1_vid_output",
            )

    st.markdown("---")

    # --- Run / Stop Button ---
    if _is_pipeline_active():
        if st.button("Stop Detection", type="secondary", width="stretch", key="p1_run"):
            _kill_running_process()
            st.rerun()
    elif st.button("Run Detection", type="primary", width="stretch", key="p1_run"):
        if selected_video:
            video_path = str(video_map.get(selected_video, PROJECT_ROOT / "videos" / selected_video))
            output_path = str(PROJECT_ROOT / "submissions" / output_name)
            ensure_dir(PROJECT_ROOT / "submissions")

            cmd = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "src" / "problem1_competition.py"),
                "--video",
                video_path,
                "--output",
                output_path,
                "--conf",
                str(conf_threshold),
                "--model",
                selected_model_path,
            ]
            if not use_bytetrack:
                cmd.append("--no-bytetrack")
            if save_video:
                save_vid_path = str(PROJECT_ROOT / "output" / video_output_name)
                ensure_dir(PROJECT_ROOT / "output")
                cmd.extend(["--save-video", save_vid_path])

            returncode, output = run_pipeline_streaming(
                cmd, label="Running P1 Detection..."
            )

            if returncode == 0:
                st.success("Detection completed successfully!")
                st.session_state.last_run_page = "p1"
                st.balloons()
            else:
                st.error("Detection failed. Check the output log above.")
        else:
            st.warning("Please select or upload a video file.")

    # --- View Results ---
    st.markdown("---")
    st.subheader("📊 View Results")

    result_files = []
    output_csv = PROJECT_ROOT / "submissions" / output_name
    default_csv = PROJECT_ROOT / "submissions" / "p1_detection_obb.csv"

    if output_csv.exists():
        result_files.append(output_csv.name)
    if default_csv.exists() and default_csv.name not in result_files:
        result_files.append(default_csv.name)

    # Also show any other p1-related CSVs
    for f in sorted((PROJECT_ROOT / "submissions").glob("p1_*.csv")):
        if f.name not in result_files:
            result_files.append(f.name)

    if result_files:
        selected_result = st.selectbox(
            "Select result file", result_files, key="p1_result_select"
        )
        show_csv_results(
            PROJECT_ROOT / "submissions" / selected_result,
            title=f"P1 Detection Results — {selected_result}",
        )
    else:
        st.info("No P1 results found. Run detection to generate results.")


# =========================================================================== #
#  PAGE: Localization (P2)
# =========================================================================== #
elif page == "📍 Localization (P2)":
    st.title("📍 Problem 2 — Drone Localization")
    st.markdown("XGBoost regression: Bounding box → Range / Azimuth / Elevation")
    st.markdown("---")

    with st.expander("ℹ️ About Problem 2", expanded=False):
        st.markdown(
            """
        - **Input**: Detection CSV from Problem 1 (with center_x, center_y, w, h)
        - **Output**: CSV with predictions (range, azimuth, elevation)
        - **Model**: XGBoost regressors (distance, bearing_sin, bearing_cos, altitude)
        - **Dimensions**: Can auto-detect from video file, or specify manually
        """
        )

    # --- Configuration ---
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)

    with col1:
        # Detection CSV input - from file or upload
        det_files = sorted((PROJECT_ROOT / "submissions").glob("*.csv"))
        det_names = [f.name for f in det_files]

        input_source = st.radio(
            "Input Source",
            ["Select existing CSV", "Upload CSV"],
            horizontal=True,
            key="p2_input_source",
        )

        if input_source == "Select existing CSV":
            if det_names:
                # Default to p1_detection_obb.csv if it exists
                default_idx = (
                    det_names.index("p1_detection_obb.csv")
                    if "p1_detection_obb.csv" in det_names
                    else 0
                )
                selected_det = st.selectbox(
                    "Detection CSV (Input)",
                    det_names,
                    index=default_idx,
                    key="p2_det",
                )
                det_csv_path = str(PROJECT_ROOT / "submissions" / selected_det)
            else:
                selected_det = None
                det_csv_path = None
                st.warning("No CSV files found in submissions/")
        else:
            uploaded_csv = st.file_uploader(
                "Upload Detection CSV", type=["csv"], key="p2_upload"
            )
            if uploaded_csv is not None:
                dest = _safe_upload(uploaded_csv, PROJECT_ROOT / "submissions")
                det_csv_path = str(dest)
                selected_det = uploaded_csv.name
                st.success(f"Uploaded: {uploaded_csv.name}")
            else:
                det_csv_path = None
                selected_det = None

        # Show preview of input CSV
        if det_csv_path and Path(det_csv_path).exists():
            with st.expander("👀 Preview Input CSV"):
                preview_df = pd.read_csv(det_csv_path, nrows=10)
                st.dataframe(preview_df, width="stretch", hide_index=True)
                st.caption(f"Columns: {list(preview_df.columns)}")

    with col2:
        output_name_p2 = st.text_input(
            "Output CSV Name", "p2_localization_output.csv", key="p2_output"
        )
        model_type = st.selectbox(
            "Regression Model", ["xgboost"], key="p2_model"
        )

        st.markdown("**Frame Dimensions**")
        dim_source = st.radio(
            "Dimension source",
            ["Manual", "Auto-detect from video"],
            horizontal=True,
            key="p2_dim_source",
        )

        video_for_dims = None
        if dim_source == "Manual":
            frame_width = st.number_input(
                "Frame Width (px)", value=1920, min_value=1, key="p2_w"
            )
            frame_height = st.number_input(
                "Frame Height (px)", value=1080, min_value=1, key="p2_h"
            )
        else:
            videos = list_videos()
            video_names_p2 = [v.name for v in videos]
            if video_names_p2:
                selected_vid_for_dims = st.selectbox(
                    "Video for dimensions", video_names_p2, key="p2_vid_dims"
                )
                video_for_dims = str(
                    PROJECT_ROOT / "videos" / selected_vid_for_dims
                )
                # Show detected dimensions
                try:
                    import cv2

                    cap = cv2.VideoCapture(video_for_dims)
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        st.info(f"Detected: {w} x {h}")
                        frame_width = w
                        frame_height = h
                    else:
                        st.warning("Could not open video")
                        frame_width, frame_height = 1920, 1080
                        video_for_dims = None
                except ImportError:
                    st.warning("OpenCV not available for auto-detect")
                    frame_width, frame_height = 1920, 1080
                    video_for_dims = None
            else:
                st.warning("No videos found")
                frame_width, frame_height = 1920, 1080

    st.markdown("---")

    # --- Run / Stop Button ---
    if _is_pipeline_active():
        if st.button("Stop Localization", type="secondary", width="stretch", key="p2_run"):
            _kill_running_process()
            st.rerun()
    elif st.button("Run Localization", type="primary", width="stretch", key="p2_run"):
        if det_csv_path and Path(det_csv_path).exists():
            output_path = str(PROJECT_ROOT / "submissions" / output_name_p2)
            ensure_dir(PROJECT_ROOT / "submissions")

            cmd = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "src" / "problem2_inference.py"),
                "--detections",
                det_csv_path,
                "--output",
                output_path,
                "--model",
                model_type,
                "--width",
                str(frame_width),
                "--height",
                str(frame_height),
            ]
            if video_for_dims:
                cmd.extend(["--video", video_for_dims])

            returncode, output = run_pipeline_streaming(
                cmd, label="Running P2 Localization..."
            )

            if returncode == 0:
                st.success("Localization completed successfully!")
                st.session_state.last_run_page = "p2"
                st.balloons()
            else:
                st.error("Localization failed. Check the output log above.")
        else:
            st.warning("Please select or upload a valid detection CSV file.")

    # --- View Results ---
    st.markdown("---")
    st.subheader("📊 View Results")

    result_files_p2 = []
    output_csv_p2 = PROJECT_ROOT / "submissions" / output_name_p2
    default_csv_p2 = PROJECT_ROOT / "submissions" / "p2_localization_final.csv"

    if output_csv_p2.exists():
        result_files_p2.append(output_csv_p2.name)
    if default_csv_p2.exists() and default_csv_p2.name not in result_files_p2:
        result_files_p2.append(default_csv_p2.name)

    for f in sorted((PROJECT_ROOT / "submissions").glob("p2_*.csv")):
        if f.name not in result_files_p2:
            result_files_p2.append(f.name)

    # Also add predictions.csv if it exists
    pred_csv = PROJECT_ROOT / "submissions" / "predictions.csv"
    if pred_csv.exists() and pred_csv.name not in result_files_p2:
        result_files_p2.append(pred_csv.name)

    if result_files_p2:
        selected_result_p2 = st.selectbox(
            "Select result file", result_files_p2, key="p2_result_select"
        )
        show_csv_results(
            PROJECT_ROOT / "submissions" / selected_result_p2,
            title=f"P2 Localization Results — {selected_result_p2}",
        )
    else:
        st.info("No P2 results found. Run localization to generate results.")


# =========================================================================== #
#  PAGE: Full Pipeline (P3)
# =========================================================================== #
elif page == "🔗 Full Pipeline (P3)":
    st.title("🔗 Problem 3 — Full Pipeline Integration")
    st.markdown(
        "End-to-end: Video → Detection → Tracking → Localization → CSV + Satellite API"
    )
    st.markdown("---")

    with st.expander("ℹ️ About Problem 3", expanded=False):
        st.markdown(
            """
        - **Full pipeline**: Detection (YOLO-OBB) → ByteTrack → Weighted NMS → Track Merging → XGBoost Localization → Smoothing → CSV
        - **Production script**: `src/problem_3_pipeline.py` (most optimized)
        - **Alternative**: `src/problem3_integration.py` (simpler, standalone)
        - **Output**: `submission.csv` with video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred
        """
        )

    tab_prod, tab_integ = st.tabs(
        ["🏭 Production Pipeline", "🔧 Integration Pipeline"]
    )

    # ========== Tab: Production Pipeline ==========
    with tab_prod:
        st.subheader("Production Pipeline (Recommended)")
        st.markdown(
            """
        Uses `problem_3_pipeline.py` with advanced **Weighted NMS** + **Track Merging** + **Approximation Localizer**.

        **Default configuration** (hardcoded in script):
        - Model: `drone_detect_v21_max_data/best.pt`
        - Confidence: 0.10
        - IOU: 0.3
        - Track buffer: 180
        - Track merging: Enabled
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            # Check if required files exist
            prod_model = (
                PROJECT_ROOT
                / "runs"
                / "detect"
                / "drone_detect_v21_max_data"
                / "weights"
                / "best.pt"
            )
            prod_video = PROJECT_ROOT / "videos" / "P3_VIDEO.mp4"

            if prod_model.exists():
                st.success(
                    f"✅ Production model found ({format_file_size(prod_model.stat().st_size)})"
                )
            else:
                st.error(f"❌ Production model not found")

        with col2:
            if prod_video.exists():
                st.success(
                    f"✅ Production video found ({format_file_size(prod_video.stat().st_size)})"
                )
            else:
                st.error(f"❌ Production video not found: videos/P3_VIDEO.mp4")

        if _is_pipeline_active():
            if st.button("Stop Production Pipeline", type="secondary", width="stretch", key="p3_prod_run"):
                _kill_running_process()
                st.rerun()
        elif st.button("Run Production Pipeline", type="primary", width="stretch", key="p3_prod_run"):
            if prod_model.exists() and prod_video.exists():
                cmd = [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "src" / "problem_3_pipeline.py"),
                ]

                returncode, output = run_pipeline_streaming(
                    cmd, label="Running Production Pipeline (P3)..."
                )

                if returncode == 0:
                    st.success("Production pipeline completed!")
                    st.session_state.last_run_page = "p3"
                    st.balloons()
                else:
                    st.error("Pipeline failed. Check the output log above.")
            else:
                st.error(
                    "Missing required files. Ensure both the model and video exist."
                )

    # ========== Tab: Integration Pipeline ==========
    with tab_integ:
        st.subheader("Integration Pipeline (Configurable)")
        st.markdown(
            "Simpler standalone pipeline with configurable parameters."
        )

        col1, col2 = st.columns(2)
        with col1:
            videos = list_videos()
            video_names_p3 = [v.name for v in videos]
            if video_names_p3:
                selected_video_p3 = st.selectbox(
                    "📹 Video", video_names_p3, key="p3_video"
                )
            else:
                selected_video_p3 = None
                st.warning("No videos found")

            video_id_p3 = st.text_input(
                "Video ID", "video_01", key="p3_vid_id"
            )

            # Model selection
            model_options_p3 = get_yolo_model_options()
            if model_options_p3:
                model_labels_p3 = list(model_options_p3.keys())
                selected_model_label_p3 = st.selectbox(
                    "🧠 Detection Model", model_labels_p3, key="p3_model"
                )
                selected_model_path_p3 = model_options_p3[
                    selected_model_label_p3
                ]
            else:
                selected_model_path_p3 = "yolov8n-obb.pt"

        with col2:
            conf_p3 = st.slider(
                "Confidence", 0.01, 1.0, 0.10, 0.01, key="p3_conf"
            )
            regression_p3 = st.selectbox(
                "Regression Model", ["xgboost"], key="p3_reg"
            )
            output_name_p3 = st.text_input(
                "Output CSV Name", "submission.csv", key="p3_output"
            )
            save_video_p3 = st.checkbox(
                "Save Annotated Video", value=False, key="p3_save"
            )
            if save_video_p3:
                video_output_name_p3 = st.text_input(
                    "Output Video Name",
                    "p3_integration_output.mp4",
                    key="p3_vid_output",
                )

        if _is_pipeline_active():
            if st.button("Stop Integration Pipeline", type="secondary", width="stretch", key="p3_integ_run"):
                _kill_running_process()
                st.rerun()
        elif st.button("Run Integration Pipeline", type="primary", width="stretch", key="p3_integ_run"):
            if selected_video_p3:
                video_map_p3 = {v.name: v for v in videos}
                video_path = str(video_map_p3.get(selected_video_p3, PROJECT_ROOT / "videos" / selected_video_p3))
                output_path = str(
                    PROJECT_ROOT / "submissions" / output_name_p3
                )
                ensure_dir(PROJECT_ROOT / "submissions")

                cmd = [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "src" / "problem3_integration.py"),
                    "--video",
                    video_path,
                    "--video-id",
                    video_id_p3,
                    "--output",
                    output_path,
                    "--conf",
                    str(conf_p3),
                    "--model",
                    selected_model_path_p3,
                    "--regression",
                    regression_p3,
                ]
                if save_video_p3:
                    save_vid_path = str(
                        PROJECT_ROOT / "output" / video_output_name_p3
                    )
                    ensure_dir(PROJECT_ROOT / "output")
                    cmd.extend(["--save-video", save_vid_path])

                returncode, output = run_pipeline_streaming(
                    cmd, label="Running Integration Pipeline..."
                )

                if returncode == 0:
                    st.success("Integration pipeline completed!")
                    st.session_state.last_run_page = "p3"
                    st.balloons()
                else:
                    st.error(
                        "Pipeline failed. Check the output log above."
                    )
            else:
                st.warning("Please select a video file.")

    # --- View Overlay Video ---
    st.markdown("---")
    st.subheader("Overlay Video")
    st.caption("View annotated output videos generated by the pipeline.")

    overlay_dirs = [
        PROJECT_ROOT / "outputs" / "problem_3" / "final",
        PROJECT_ROOT / "output",
    ]
    overlay_videos = []
    for d in overlay_dirs:
        if d.exists():
            for ext in ("*.mp4", "*.avi", "*.mov"):
                overlay_videos.extend(d.glob(ext))
    overlay_videos = sorted(overlay_videos, key=lambda p: p.stat().st_mtime, reverse=True)

    if overlay_videos:
        ov_names = [f"{v.parent.name}/{v.name}" for v in overlay_videos]
        sel_ov = st.selectbox("Select overlay video", ov_names, key="p3_overlay")
        sel_idx = ov_names.index(sel_ov)
        ov_path = overlay_videos[sel_idx]

        ov_size = format_file_size(ov_path.stat().st_size)
        ov_modified = datetime.fromtimestamp(ov_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        c1, c2, c3 = st.columns(3)
        c1.metric("File", ov_path.name)
        c2.metric("Size", ov_size)
        c3.metric("Modified", ov_modified)

        # Video player — fix moov-atom for browser playback
        try:
            playable = _fix_video_for_browser(ov_path)
            st.video(str(playable), format="video/mp4")
        except Exception:
            st.warning(
                "Cannot play video in browser. "
                "Use the frame-by-frame viewer below instead."
            )

        # Frame-by-frame viewer (always works via OpenCV)
        try:
            import cv2
            cap_ov = cv2.VideoCapture(str(ov_path))
            if cap_ov.isOpened():
                total_ov = int(cap_ov.get(cv2.CAP_PROP_FRAME_COUNT))
                with st.expander("Frame-by-frame viewer", expanded=False):
                    fr = st.slider("Frame", 0, max(total_ov - 1, 0), 0, key="p3_ov_frame")
                    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ret_ov, frame_ov = cap_ov.read()
                    if ret_ov:
                        st.image(
                            cv2.cvtColor(frame_ov, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {fr}",
                            width="stretch",
                        )
            cap_ov.release()
        except ImportError:
            pass
    else:
        st.info(
            "No overlay videos found. Run the pipeline with video output enabled "
            "to generate annotated videos in outputs/problem_3/final/."
        )

    # --- View Results ---
    st.markdown("---")
    st.subheader("Pipeline Results")

    result_files_p3 = []
    for name in ["submission.csv", "submission_normalized.csv"]:
        p = PROJECT_ROOT / "submissions" / name
        if p.exists():
            result_files_p3.append(name)

    # Also check output directory for production pipeline results
    output_dir = PROJECT_ROOT / "outputs" / "problem_3" / "final"
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.csv")):
            result_files_p3.append(f"(outputs) {f.name}")

    for f in sorted((PROJECT_ROOT / "submissions").glob("submission*.csv")):
        if f.name not in result_files_p3:
            result_files_p3.append(f.name)

    if result_files_p3:
        selected_result_p3 = st.selectbox(
            "Select result file", result_files_p3, key="p3_result_select"
        )
        if selected_result_p3.startswith("(outputs) "):
            actual_name = selected_result_p3.replace("(outputs) ", "")
            result_path = output_dir / actual_name
        else:
            result_path = PROJECT_ROOT / "submissions" / selected_result_p3
        show_csv_results(
            result_path, title=f"Pipeline Results — {selected_result_p3}"
        )
    else:
        st.info(
            "No submission results found. Run the pipeline to generate results."
        )


# =========================================================================== #
#  PAGE: Submissions
# =========================================================================== #
elif page == "📊 Submissions":
    st.title("📊 Submission Files")
    st.markdown("Browse, inspect, compare, and validate submission CSV files.")
    st.markdown("---")

    # Upload section
    with st.expander("📤 Upload Submission CSV"):
        uploaded_sub = st.file_uploader(
            "Upload CSV file", type=["csv"], key="sub_upload"
        )
        if uploaded_sub is not None:
            saved = _safe_upload(uploaded_sub, PROJECT_ROOT / "submissions")
            st.success(f"Uploaded: {uploaded_sub.name}")

    submissions = list_submissions()
    if submissions:
        selected_sub = st.selectbox(
            "Select Submission", [s.name for s in submissions]
        )
        sub_path = PROJECT_ROOT / "submissions" / selected_sub

        if sub_path.exists():
            show_csv_results(sub_path, title=selected_sub)
    else:
        st.info("No submission files found in submissions/ directory.")

    # Validation section
    st.markdown("---")
    st.subheader("🔍 Validate Submission")
    col1, col2 = st.columns(2)
    with col1:
        pred_file = st.selectbox(
            "Predictions CSV",
            [s.name for s in submissions] if submissions else ["None"],
            key="val_pred",
        )
    with col2:
        gt_file = st.selectbox(
            "Ground Truth CSV",
            [s.name for s in submissions] if submissions else ["None"],
            key="val_gt",
        )

    threshold = st.slider("Detection Threshold (pixels)", 5, 50, 20)

    if _is_pipeline_active():
        if st.button("Stop Validation", type="secondary", width="stretch", key="val_run"):
            _kill_running_process()
            st.rerun()
    elif st.button("Validate", type="primary", width="stretch", key="val_run"):
        if pred_file != "None" and gt_file != "None":
            cmd = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "src" / "validate_submission.py"),
                "--predictions",
                str(PROJECT_ROOT / "submissions" / pred_file),
                "--ground-truth",
                str(PROJECT_ROOT / "submissions" / gt_file),
                "--threshold",
                str(threshold),
            ]
            returncode, output = run_pipeline_streaming(
                cmd, label="Running Validation..."
            )
            if returncode == 0:
                st.success("Validation completed!")
            else:
                st.error("Validation failed.")
        else:
            st.warning(
                "Please select both prediction and ground truth files."
            )


# =========================================================================== #
#  PAGE: Compliance Check
# =========================================================================== #
elif page == "✅ Compliance Check":
    st.title("✅ Compliance Check")
    st.markdown(
        "Validate all outputs against TESA Defence competition requirements."
    )
    st.markdown("---")

    st.markdown(
        """
    **Expected formats:**
    | Problem | Columns | Notes |
    |---------|---------|-------|
    | P1 | frame_id, object_id, center_x, center_y, w, h, [theta] | Normalized 0-1 |
    | P2 | frame_id, object_id, direction, distance, height | direction: 0-360° |
    | P3 | video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred | cx, cy in pixels |
    """
    )

    if _is_pipeline_active():
        if st.button("Stop Compliance Check", type="secondary", width="stretch", key="compliance_run"):
            _kill_running_process()
            st.rerun()
    elif st.button("Run Full Compliance Check", type="primary", width="stretch", key="compliance_run"):
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "src" / "check_compliance.py"),
        ]
        returncode, output = run_pipeline_streaming(
            cmd, label="Checking Compliance..."
        )
        if returncode == 0:
            st.success("Compliance check completed!")
        else:
            st.warning("Compliance check finished with issues. See log above.")

    # Quick checks per file
    st.markdown("---")
    st.subheader("Quick File Checks")

    checks = [
        (
            "P1 Detection",
            "submissions/p1_detection_obb.csv",
            ["frame_id", "object_id", "center_x", "center_y", "w", "h"],
        ),
        (
            "P2 Localization",
            "submissions/p2_localization_final.csv",
            ["frame_id", "object_id", "direction", "distance", "height"],
        ),
        (
            "P3 Submission",
            "submissions/submission.csv",
            [
                "video_id",
                "frame",
                "cx",
                "cy",
                "range_m_pred",
                "azimuth_deg_pred",
                "elevation_deg_pred",
            ],
        ),
    ]

    for name, path, required_cols in checks:
        full_path = PROJECT_ROOT / path
        with st.expander(f"{name} — `{path}`"):
            if full_path.exists():
                df = load_csv(str(full_path))
                present = [c for c in required_cols if c in df.columns]
                missing = [c for c in required_cols if c not in df.columns]

                if not missing:
                    st.success(
                        f"✅ All required columns present ({len(df)} rows)"
                    )
                else:
                    st.error(f"❌ Missing columns: {missing}")

                st.write(f"Columns: {list(df.columns)}")
                st.dataframe(df.head(5), width="stretch", hide_index=True)

                # Value range checks for P1
                if name == "P1 Detection":
                    for col in ["center_x", "center_y", "w", "h"]:
                        if col in df.columns:
                            vmin, vmax = df[col].min(), df[col].max()
                            if 0 <= vmin and vmax <= 1:
                                st.write(
                                    f"  ✅ `{col}`: {vmin:.4f} — {vmax:.4f} (normalized)"
                                )
                            else:
                                st.write(
                                    f"  ❌ `{col}`: {vmin:.4f} — {vmax:.4f} (NOT normalized!)"
                                )
            else:
                st.warning(f"File not found: {path}")


# =========================================================================== #
#  PAGE: Models
# =========================================================================== #
elif page == "🧠 Models":
    st.title("🧠 Model Management")
    st.markdown("View and manage all models in the system.")
    st.markdown("---")

    models = list_models()

    # YOLO Models
    st.subheader("🎯 YOLO Models (Detection)")
    if models["yolo"]:
        yolo_df = pd.DataFrame(models["yolo"])
        st.dataframe(yolo_df, width="stretch", hide_index=True)
    else:
        st.info("No YOLO models found in models/")

    # Training runs
    st.subheader("🏋️ Training Runs")
    if models["runs"]:
        runs_df = pd.DataFrame(models["runs"])
        st.dataframe(runs_df, width="stretch", hide_index=True)
        st.success(
            "✅ Production model: `drone_detect_v21_max_data` — mAP: 81.0%, Recall: 90.0%"
        )
    else:
        st.info("No training runs found in runs/")

    # XGBoost Models
    st.subheader("📊 XGBoost Models")
    if models["xgboost"]:
        xgb_df = pd.DataFrame(models["xgboost"])
        st.dataframe(xgb_df, width="stretch", hide_index=True)
    else:
        st.info("No standalone XGBoost models found.")

    # Approximation Models
    st.subheader("🎯 Approximation Models (Production)")
    if models["approximation"]:
        approx_df = pd.DataFrame(models["approximation"])
        st.dataframe(approx_df, width="stretch", hide_index=True)
        st.success(
            "✅ These models are used in the production pipeline (problem_3_pipeline.py)"
        )
    else:
        st.info("No approximation models found.")

    # Stacking Models
    st.subheader("📚 Stacking Ensemble Models (Not in Production)")
    if models["stacking"]:
        stack_df = pd.DataFrame(models["stacking"])
        st.dataframe(stack_df, width="stretch", hide_index=True)
        st.warning(
            "⚠️ These models are trained but NOT used in the production pipeline."
        )
    else:
        st.info("No stacking models found.")

    # Total summary
    st.markdown("---")
    total = sum(len(v) for v in models.values())
    st.metric("Total Model Files", total)


# =========================================================================== #
#  PAGE: Configuration
# =========================================================================== #
elif page == "⚙️ Configuration":
    st.title("⚙️ System Configuration")
    st.markdown("View all configuration settings from `src/config.py`.")
    st.markdown("---")

    cfg = get_config_dict()

    if "error" in cfg:
        st.error(f"Failed to load config: {cfg['error']}")
    else:
        for section_name, section_data in cfg.items():
            with st.expander(
                f"📦 {section_name}",
                expanded=section_name == "MODEL_CONFIG",
            ):
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        col1, col2 = st.columns([1, 2])
                        col1.write(f"**{key}**")
                        col2.code(str(value))
                else:
                    st.write(section_data)

    st.markdown("---")
    st.subheader("Configuration File Locations")
    config_files = (
        list((PROJECT_ROOT / "configs").glob("*"))
        if (PROJECT_ROOT / "configs").exists()
        else []
    )
    if config_files:
        for cf in sorted(config_files):
            if cf.is_file():
                size = format_file_size(cf.stat().st_size)
                st.write(f"- `{cf.name}` ({size})")
    else:
        st.info("No extra config files found.")


# =========================================================================== #
#  PAGE: Videos
# =========================================================================== #
elif page == "🎬 Videos":
    st.title("🎬 Video Manager")
    st.markdown("Browse, preview, and upload video files.")
    st.markdown("---")

    # Upload section
    with st.expander("Upload Video"):
        uploaded_vid = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_upload",
        )
        if uploaded_vid is not None:
            _safe_upload(uploaded_vid, PROJECT_ROOT / "videos")
            st.success(f"Uploaded: {uploaded_vid.name}")

    videos = list_videos()
    video_map_vids = {v.name: v for v in videos}
    if videos:
        selected_vid = st.selectbox(
            "Select Video", [v.name for v in videos]
        )
        vid_path = video_map_vids.get(selected_vid, PROJECT_ROOT / "videos" / selected_vid)

        if vid_path.exists():
            size = format_file_size(vid_path.stat().st_size)
            modified = datetime.fromtimestamp(
                vid_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")

            col1, col2, col3 = st.columns(3)
            col1.metric("File Size", size)
            col2.metric("Modified", modified)
            col3.metric("Format", vid_path.suffix.upper())

            # Try to get video info with OpenCV
            try:
                import cv2

                cap = cv2.VideoCapture(str(vid_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Resolution", f"{width}x{height}")
                    col2.metric("FPS", f"{fps:.1f}")
                    col3.metric("Frames", total_frames)
                    col4.metric("Duration", f"{duration:.1f}s")

                    # Frame preview
                    st.markdown("---")
                    st.subheader("Frame Preview")
                    frame_num = st.slider(
                        "Frame Number", 0, max(total_frames - 1, 0), 0
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        )
                        st.image(
                            frame_rgb,
                            caption=f"Frame {frame_num}",
                            width="stretch",
                        )
                    cap.release()
                else:
                    st.warning("Could not open video with OpenCV.")
            except ImportError:
                st.warning("OpenCV not available for video preview.")

            # Streamlit video player
            st.markdown("---")
            st.subheader("Video Player")
            try:
                playable = _fix_video_for_browser(vid_path)
                st.video(str(playable))
            except Exception:
                st.video(str(vid_path))
    else:
        st.info("No video files found in videos/ directory.")


# =========================================================================== #
#  PAGE: API Testing
# =========================================================================== #
elif page == "📡 API Testing":
    st.title("📡 Satellite API Testing")
    st.markdown("Test the drone alert API endpoint for Problem 3.  Supports **REST**, **MQTT**, **WebSocket**, **gRPC**, and **Mock** protocols.")
    st.markdown("---")

    # Import API helpers
    try:
        from api_client import create_client, available_protocols, detect_protocol, format_payload
        api_module_ok = True
    except ImportError as _imp_err:
        api_module_ok = False
        st.error(f"Cannot import api_client module: {_imp_err}")

    if api_module_ok:
        cfg = get_config_dict()
        api_config = cfg.get("API_CONFIG", {})
        avail = available_protocols()

        st.subheader("API Configuration")
        col_proto, col_url = st.columns([1, 3])
        with col_proto:
            protocol = st.selectbox(
                "Protocol",
                options=["rest", "mqtt", "websocket", "grpc", "mock"],
                index=0,
                help=f"Installed: {', '.join(avail)}",
            )
        with col_url:
            default_url = api_config.get("api_url", "https://api.tesa.or.th/drone")
            if protocol == "mock":
                default_url = "mock://localhost"
            api_url = st.text_input("Endpoint URL", value=default_url)

        col_key, col_timeout, col_retries = st.columns(3)
        with col_key:
            api_key = st.text_input("API Key", value=api_config.get("api_key", ""), type="password")
        with col_timeout:
            timeout = st.number_input("Timeout (s)", value=api_config.get("timeout", 10), min_value=1)
        with col_retries:
            retries = st.number_input("Retries", value=api_config.get("retries", 2), min_value=0, max_value=10)

        # MQTT-specific options
        mqtt_topic = "tesa/drone"
        mqtt_qos = 1
        if protocol == "mqtt":
            st.markdown("**MQTT Options**")
            col_t, col_q = st.columns(2)
            with col_t:
                mqtt_topic = st.text_input("Topic", value=api_config.get("mqtt_topic", "tesa/drone"))
            with col_q:
                mqtt_qos = st.selectbox("QoS", options=[0, 1, 2], index=1)

        # Protocol availability warning
        if protocol not in avail and protocol != "mock":
            st.warning(f"Protocol **{protocol}** dependencies are not installed. Install the required package first.")

        st.markdown("---")

        # ---------- Connection Test ----------
        st.subheader("Connection Test")
        if st.button("Test Connection", type="secondary"):
            try:
                client = create_client(
                    protocol=protocol, url=api_url, api_key=api_key,
                    timeout=timeout, retries=0,
                    topic=mqtt_topic, qos=str(mqtt_qos),
                )
                with st.spinner(f"Testing {protocol.upper()} connection..."):
                    result = client.test_connection()
                client.close()

                col1, col2 = st.columns(2)
                col1.metric("Status", "OK" if result["ok"] else "FAIL")
                col2.metric("Latency", f"{result.get('latency_ms', 0):.1f} ms")

                if result["ok"]:
                    st.success(f"Connection OK via {protocol.upper()}")
                else:
                    st.error(f"Connection failed: {result.get('detail', 'unknown')}")

                with st.expander("Full response"):
                    st.json(result)
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")

        # ---------- Send Test Payload ----------
        st.subheader("Send Test Payload")

        test_payload = format_payload(
            [
                {
                    "frame": 0, "object_id": 1, "drone_type": "DJIMavic",
                    "lat": 13.22, "lon": 66.32, "speed_ms": 15.2, "direction_deg": 45.3,
                }
            ]
        )
        payload_str = st.text_area("JSON Payload", value=json.dumps(test_payload, indent=2), height=200)

        if st.button("Send Test Request", type="primary"):
            try:
                client = create_client(
                    protocol=protocol, url=api_url, api_key=api_key,
                    timeout=timeout, retries=retries,
                    topic=mqtt_topic, qos=str(mqtt_qos),
                )
                payload = json.loads(payload_str)
                with st.spinner(f"Sending via {protocol.upper()}..."):
                    ok = client._send_with_retry(payload, topic="/tracking")
                client.close()

                if ok:
                    st.success(f"Sent successfully via {protocol.upper()}!")
                else:
                    st.error(f"Send failed via {protocol.upper()}")

                st.json(client.info())
            except json.JSONDecodeError:
                st.error("Invalid JSON payload.")
            except Exception as e:
                st.error(f"Send failed: {e}")

        # ---------- Mock demo ----------
        if protocol == "mock":
            st.markdown("---")
            st.subheader("Mock Demo")
            if st.button("Run Full Mock Demo"):
                mock = create_client("mock")
                mock.send_first_alarm(3)
                mock.send_tracking_data([
                    {"frame": 1, "object_id": 1, "drone_type": "DJI_Mavic",
                     "lat": 13.7563, "lon": 100.5018, "speed_ms": 15.2, "direction_deg": 45.3},
                    {"frame": 1, "object_id": 2, "drone_type": "DJI_Phantom",
                     "lat": 13.7564, "lon": 100.5019, "speed_ms": 12.8, "direction_deg": 90.0},
                ])
                st.success(f"Mock demo complete: {len(mock.sent)} calls recorded")
                for i, call in enumerate(mock.sent):
                    st.json({"call": i + 1, "topic": call["topic"], "objects": len(call["payload"].get("object", []))})

        st.markdown("---")
        st.subheader("API Payload Format Reference")
        st.json(
            {
                "time": "Unix timestamp (int)",
                "object": [
                    {
                        "frame": "Frame number (int)",
                        "id": "Object ID (int)",
                        "type": "Drone type (str)",
                        "lat": "Latitude (float)",
                        "lon": "Longitude (float)",
                        "velocity": "Velocity m/s (float)",
                        "direction": "Direction degrees (float)",
                    }
                ],
                "image_base64": "Base64 encoded JPEG image (str)",
            }
        )

        st.subheader("Available Protocols")
        proto_data = []
        for p in ["rest", "mqtt", "websocket", "grpc", "mock"]:
            proto_data.append({"Protocol": p.upper(), "Available": "Yes" if p in avail else "No (install dep)"})
        st.table(proto_data)
