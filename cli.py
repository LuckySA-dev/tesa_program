#!/usr/bin/env python3
"""
TESA Defence AI - Command Line Interface
==========================================
Interactive menu + direct command support for all backend operations.

Usage:
    python cli.py              # Interactive menu
    python cli.py <command>    # Direct command (e.g. python cli.py models)
"""

import argparse
import sys
import os
import json
import time
import subprocess
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =========================================================================== #
#  Helpers
# =========================================================================== #
def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def success(msg: str):
    print(f"  [OK] {msg}")


def error(msg: str):
    print(f"  [ERROR] {msg}")


def info(msg: str):
    print(f"  [INFO] {msg}")


def warn(msg: str):
    print(f"  [WARN] {msg}")


def separator():
    print(f"  {'-'*66}")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause():
    print()
    input("  Press Enter to continue...")


def prompt_choice(prompt_text: str, options: list, allow_back: bool = True) -> str:
    """Show numbered options and return selected value. Returns None for 'back'."""
    print()
    for i, (label, _) in enumerate(options, 1):
        print(f"    [{i}] {label}")
    if allow_back:
        print(f"    [0] << Back")
    print()
    while True:
        try:
            raw = input(f"  {prompt_text}: ").strip()
            if raw == "0" and allow_back:
                return None
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
        except (ValueError, KeyboardInterrupt, EOFError):
            pass
        print("  Invalid choice. Try again.")


def prompt_input(prompt_text: str, default: str = "") -> str:
    """Prompt for free text input with optional default."""
    suffix = f" [{default}]" if default else ""
    raw = input(f"  {prompt_text}{suffix}: ").strip()
    return raw if raw else default


def prompt_yesno(prompt_text: str, default: bool = False) -> bool:
    """Prompt for yes/no."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {prompt_text} ({hint}): ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def list_videos_paths() -> list:
    vids = []
    vid_dir = PROJECT_ROOT / "videos"
    if vid_dir.exists():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            vids += list(vid_dir.glob(ext))
    return sorted(vids)


def list_csv_files(subdir: str = "submissions") -> list:
    d = PROJECT_ROOT / subdir
    return sorted(d.glob("*.csv")) if d.exists() else []


# =========================================================================== #
#  Command implementations
# =========================================================================== #
def run_detect(video, output=None, model=None, conf=0.55,
               display=False, save_video=True, no_bytetrack=False):
    header("Problem 1: Drone Detection with OBB")
    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        error(f"Video not found: {video_path}")
        return 1

    output_path = Path(output) if output else PROJECT_ROOT / "submissions" / "p1_detection_output.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    info(f"Video:      {video_path.name}")
    info(f"Output:     {output_path}")
    info(f"Confidence: {conf}")
    separator()

    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "problem1_competition.py"),
           "--video", str(video_path), "--output", str(output_path), "--conf", str(conf)]
    if model:
        cmd.extend(["--model", model])
    if save_video:
        cmd.append("--save-video")
    if display:
        cmd.append("--display")
    if no_bytetrack:
        cmd.append("--no-bytetrack")

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def run_localize(detections, output=None, model="xgboost",
                 width=1920, height=1080, video=None):
    header("Problem 2: Drone Localization")
    det_path = Path(detections)
    if not det_path.is_absolute():
        det_path = PROJECT_ROOT / det_path
    if not det_path.exists():
        error(f"Detection CSV not found: {det_path}")
        return 1

    output_path = Path(output) if output else PROJECT_ROOT / "submissions" / "p2_localization_output.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    info(f"Detections: {det_path.name}")
    info(f"Output:     {output_path}")
    info(f"Model:      {model}")
    separator()

    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "problem2_inference.py"),
           "--detections", str(det_path), "--output", str(output_path),
           "--model", model, "--width", str(width), "--height", str(height)]
    if video:
        v = Path(video)
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        cmd.extend(["--video", str(v)])

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def run_pipeline(video, output=None, video_id="video_01", conf=0.55,
                 model=None, regression="xgboost", display=False, save_video=False):
    header("Problem 3: Integration Pipeline")
    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        error(f"Video not found: {video_path}")
        return 1

    output_path = Path(output) if output else PROJECT_ROOT / "submissions" / "pipeline_output.csv"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    info(f"Video:      {video_path.name}")
    info(f"Output:     {output_path}")
    info(f"Confidence: {conf}")
    separator()

    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "problem3_integration.py"),
           "--video", str(video_path), "--video-id", video_id,
           "--output", str(output_path), "--conf", str(conf), "--regression", regression]
    if model:
        cmd.extend(["--model", model])
    if save_video:
        cmd.append("--save-video")
    if display:
        cmd.append("--display")

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def run_pipeline_adv():
    header("Problem 3: Production Pipeline (Advanced)")
    info("Weighted NMS + Track Merging + Approximation Localizer + Smoothing")
    separator()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "problem_3_pipeline.py")],
        cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        sub_csv = PROJECT_ROOT / "submissions" / "submission.csv"
        if sub_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(sub_csv)
                success(f"Output: {sub_csv}")
                info(f"Records: {len(df)} | Columns: {list(df.columns)}")
            except Exception:
                pass
    return result.returncode


def run_validate(predictions, ground_truth, threshold=20, report=None):
    header("Submission Validation")
    pred_path = Path(predictions)
    gt_path = Path(ground_truth)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / pred_path
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path

    if not pred_path.exists():
        error(f"Predictions not found: {pred_path}"); return 1
    if not gt_path.exists():
        error(f"Ground truth not found: {gt_path}"); return 1

    info(f"Predictions:  {pred_path.name}")
    info(f"Ground Truth: {gt_path.name}")
    info(f"Threshold:    {threshold} px")
    separator()

    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "validate_submission.py"),
           "--predictions", str(pred_path), "--ground-truth", str(gt_path),
           "--threshold", str(threshold)]
    if report:
        cmd.extend(["--report", str(PROJECT_ROOT / report)])
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def run_compliance():
    header("Compliance Check")
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "check_compliance.py")],
        cwd=str(PROJECT_ROOT)).returncode


def show_models(detail=False):
    header("Model Inventory")
    models_dir = PROJECT_ROOT / "models"
    runs_dir = PROJECT_ROOT / "runs"

    print("\n  --- YOLO Detection Models (.pt) ---")
    for f in sorted(models_dir.glob("*.pt")) if models_dir.exists() else []:
        print(f"    {f.name:30s}  {format_size(f.stat().st_size):>10s}")

    print("\n  --- Training Runs (best.pt) ---")
    for w in sorted(runs_dir.rglob("weights/best.pt")) if runs_dir.exists() else []:
        name = w.parent.parent.name
        tag = " << PRODUCTION" if "v21" in name else ""
        print(f"    {name:30s}  {format_size(w.stat().st_size):>10s}{tag}")

    print("\n  --- XGBoost Models ---")
    for f in sorted(models_dir.glob("*.pkl")) if models_dir.exists() else []:
        print(f"    {f.name:40s}  {format_size(f.stat().st_size):>10s}")

    for label, subdir, prod in [
        ("Approximation Models (PRODUCTION)", "models_approximation", True),
        ("Stacking Ensemble (NOT in production)", "models_stacking", False),
    ]:
        d = models_dir / subdir
        print(f"\n  --- {label} ---")
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    print(f"    {f.name:40s}  {format_size(f.stat().st_size):>10s}")
        else:
            print("    (not found)")
    return 0


def show_submissions(view=None, rows=20):
    header("Submission Files")
    sub_dir = PROJECT_ROOT / "submissions"
    csv_files = sorted(sub_dir.glob("*.csv")) if sub_dir.exists() else []
    if not csv_files:
        warn("No CSV files found"); return 0

    if view:
        target = sub_dir / view
        if not target.exists():
            error(f"Not found: {target}"); return 1
        import pandas as pd
        df = pd.read_csv(target)
        print(f"\n  File:    {target.name}")
        print(f"  Rows:    {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        separator()
        print(df.head(rows).to_string(index=False))
        separator()
        print(df.describe().to_string())
        return 0

    import pandas as pd
    print()
    for f in csv_files:
        try:
            df = pd.read_csv(f); r, c = len(df), len(df.columns)
        except Exception:
            r, c = "?", "?"
        size = format_size(f.stat().st_size)
        mod = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"    {f.name:35s}  {str(r):>6s} rows  {str(c):>3s} cols  {size:>8s}  {mod}")
    return 0


def show_videos():
    header("Video Files")
    vid_files = list_videos_paths()
    if not vid_files:
        warn("No videos found"); return 0
    try:
        import cv2; has_cv2 = True
    except ImportError:
        has_cv2 = False
    print()
    for v in vid_files:
        line = f"    {v.name:30s}  {format_size(v.stat().st_size):>10s}"
        if has_cv2:
            cap = cv2.VideoCapture(str(v))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = frames / fps if fps > 0 else 0
                line += f"  {w}x{h}  {fps:.0f}fps  {frames} frames  {dur:.1f}s"
                cap.release()
        print(line)
    return 0


def show_config(section=None):
    header("System Configuration")
    try:
        from config import (MODEL_CONFIG, TRACKING_CONFIG, VISUALIZATION_CONFIG,
                            VIDEO_CONFIG, LOGGING_CONFIG, API_CONFIG,
                            HARDWARE_CONFIG, ALERT_RULES, PATHS)
    except ImportError as e:
        error(f"Cannot import config: {e}"); return 1

    secs = {
        "MODEL_CONFIG": MODEL_CONFIG, "TRACKING_CONFIG": TRACKING_CONFIG,
        "VIDEO_CONFIG": VIDEO_CONFIG,
        "LOGGING_CONFIG": {k: v for k, v in LOGGING_CONFIG.items() if k != "csv_fields"},
        "API_CONFIG": API_CONFIG, "HARDWARE_CONFIG": HARDWARE_CONFIG,
        "ALERT_RULES": ALERT_RULES, "PATHS": {k: str(v) for k, v in PATHS.items()},
    }
    if section:
        key = section.upper()
        if key not in secs:
            error(f"Unknown: {section}. Available: {', '.join(secs.keys())}"); return 1
        secs = {key: secs[key]}
    for name, data in secs.items():
        print(f"\n  [{name}]")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"    {k:30s} = {v}")
    return 0


def show_info():
    header("TESA Defence AI - System Information")
    print(f"\n  Project Root:  {PROJECT_ROOT}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  Platform:      {sys.platform}")
    print(f"  Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    separator()

    print("\n  --- Directories ---")
    for d in ["src", "models", "runs", "configs", "data", "datasets",
              "videos", "submissions", "outputs", "scripts", "docs"]:
        dp = PROJECT_ROOT / d
        if dp.exists():
            count = sum(1 for _ in dp.rglob("*") if _.is_file())
            print(f"    {d + '/':15s}  {count:>5d} files")
        else:
            print(f"    {d + '/':15s}  (not found)")

    separator()
    print("\n  --- Dependencies ---")
    for pkg in ["ultralytics", "torch", "cv2", "numpy", "pandas", "xgboost",
                "scipy", "matplotlib", "seaborn", "tqdm", "streamlit"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"    {pkg:20s}  {ver}")
        except Exception:
            try:
                import importlib.metadata
                ver = importlib.metadata.version(pkg if pkg != "cv2" else "opencv-python")
                print(f"    {pkg:20s}  {ver}")
            except Exception:
                print(f"    {pkg:20s}  NOT INSTALLED")
    return 0


def run_api_test(url="https://api.tesa.or.th/drone", api_key=None,
                 timeout=10, lat=13.22, lon=66.32, protocol=None,
                 retries=2):
    header("Satellite API Test (Multi-Protocol)")
    try:
        from api_client import create_client, available_protocols, detect_protocol
    except ImportError:
        error("api_client module not found"); return 1

    proto = protocol or detect_protocol(url)
    info(f"Protocol: {proto.upper()}")
    info(f"URL:      {url}")
    info(f"Timeout:  {timeout}s")
    info(f"Retries:  {retries}")
    info(f"Available protocols: {', '.join(available_protocols())}")
    separator()

    try:
        client = create_client(
            protocol=proto, url=url, api_key=api_key,
            timeout=timeout, retries=retries,
        )
    except Exception as e:
        error(f"Cannot create {proto} client: {e}"); return 1

    # 1. Connection test
    info("Testing connection...")
    result = client.test_connection()
    if result["ok"]:
        print(f"  Status:  OK")
        print(f"  Latency: {result.get('latency_ms', 0):.1f} ms")
    else:
        print(f"  Status:  FAIL")
        print(f"  Detail:  {result.get('detail', 'unknown')}")
    separator()

    # 2. Send test tracking data
    info("Sending test tracking data...")
    ok = client.send_tracking_data(
        [{"frame": 0, "object_id": 1, "drone_type": "DJIMavic",
          "lat": lat, "lon": lon, "speed_ms": 15.2, "direction_deg": 45.3}]
    )
    print(f"  Sent: {'OK' if ok else 'FAIL'}")
    separator()

    # 3. Send test alarm
    info("Sending test first alarm...")
    ok = client.send_first_alarm(1)
    print(f"  Alarm: {'OK' if ok else 'FAIL'}")
    separator()

    client.close()
    return 0


def launch_dashboard():
    header("Launching Streamlit Dashboard")
    dash = PROJECT_ROOT / "dashboard.py"
    if not dash.exists():
        error("dashboard.py not found!"); return 1
    info("Opening browser at http://localhost:8501")
    info("Press Ctrl+C to stop")
    separator()
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dash)],
                       cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        info("Dashboard stopped.")
    return 0


# =========================================================================== #
#  Interactive Menu System
# =========================================================================== #

BANNER = r"""
  +============================================================+
  |           TESA Defence AI - Drone Detection CLI             |
  |          Detection | Tracking | Localization | Pipeline     |
  +============================================================+
"""

MAIN_MENU = [
    # label                              key            category
    ("Run Detection (Problem 1)",        "detect",      "pipeline"),
    ("Run Localization (Problem 2)",     "localize",    "pipeline"),
    ("Run Pipeline (Problem 3)",         "pipeline",    "pipeline"),
    ("Run Production Pipeline (Adv.)",   "pipeline_adv","pipeline"),
    ("Validate Submission",              "validate",    "tools"),
    ("Compliance Check",                 "compliance",  "tools"),
    ("View Models",                      "models",      "info"),
    ("View Submissions",                 "submissions", "info"),
    ("View Videos",                      "videos",      "info"),
    ("View Configuration",              "config",       "info"),
    ("System Info",                      "info",        "info"),
    ("API Test",                         "api_test",    "tools"),
    ("Launch Dashboard (Streamlit)",     "dashboard",   "tools"),
    ("Exit",                             "exit",        ""),
]


def print_main_menu():
    print(BANNER)
    print("  --- Run Pipeline -----------------------------------------")
    idx = 1
    for label, key, cat in MAIN_MENU:
        if cat == "pipeline":
            print(f"    [{idx:>2}]  {label}")
            idx += 1
    print()
    print("  --- Tools ------------------------------------------------")
    for label, key, cat in MAIN_MENU:
        if cat == "tools":
            print(f"    [{idx:>2}]  {label}")
            idx += 1
    print()
    print("  --- Information ------------------------------------------")
    for label, key, cat in MAIN_MENU:
        if cat == "info":
            print(f"    [{idx:>2}]  {label}")
            idx += 1
    print()
    print(f"    [ 0]  Exit")
    print()


def menu_detect():
    header("Problem 1: Drone Detection")
    vids = list_videos_paths()
    if not vids:
        error("No video files found in videos/"); return
    vid_opts = [(v.name, str(v)) for v in vids]
    video = prompt_choice("Select video", vid_opts)
    if video is None: return

    conf = float(prompt_input("Confidence threshold", "0.55"))
    save_vid = prompt_yesno("Save output video?", True)
    output = prompt_input("Output CSV name", "p1_detection_output.csv")
    separator()
    run_detect(video=video, output=f"submissions/{output}", conf=conf, save_video=save_vid)


def menu_localize():
    header("Problem 2: Localization")
    csvs = list_csv_files("submissions")
    if not csvs:
        error("No CSV files in submissions/"); return
    csv_opts = [(f.name, str(f)) for f in csvs]
    det_file = prompt_choice("Select detection CSV (input)", csv_opts)
    if det_file is None: return

    model = prompt_choice("Regression model", [
        ("XGBoost (default)", "xgboost"),
        ("Stacking Ensemble", "stacking"),
    ], allow_back=False) or "xgboost"

    output = prompt_input("Output CSV name", "p2_localization_output.csv")
    separator()
    run_localize(detections=det_file, output=f"submissions/{output}", model=model)


def menu_pipeline():
    header("Problem 3: Integration Pipeline")
    vids = list_videos_paths()
    if not vids:
        error("No video files found"); return
    vid_opts = [(v.name, str(v)) for v in vids]
    video = prompt_choice("Select video", vid_opts)
    if video is None: return

    conf = float(prompt_input("Confidence threshold", "0.55"))
    vid_id = prompt_input("Video ID", "video_01")
    regression = prompt_choice("Regression model", [
        ("XGBoost (default)", "xgboost"),
        ("Stacking Ensemble", "stacking"),
    ], allow_back=False) or "xgboost"

    save_vid = prompt_yesno("Save output video?", False)
    output = prompt_input("Output CSV name", "pipeline_output.csv")
    separator()
    run_pipeline(video=video, output=f"submissions/{output}",
                 video_id=vid_id, conf=conf, regression=regression, save_video=save_vid)


def menu_validate():
    header("Validate Submission")
    csvs = list_csv_files("submissions")
    if not csvs:
        error("No CSV files found"); return

    csv_opts = [(f.name, str(f)) for f in csvs]
    print("\n  Select PREDICTIONS file:")
    pred = prompt_choice("Predictions", csv_opts)
    if pred is None: return

    print("\n  Select GROUND TRUTH file:")
    gt = prompt_choice("Ground truth", csv_opts)
    if gt is None: return

    threshold = int(prompt_input("Pixel threshold", "20"))
    separator()
    run_validate(predictions=pred, ground_truth=gt, threshold=threshold)


def menu_submissions():
    csvs = list_csv_files("submissions")
    if not csvs:
        show_submissions(); return

    action = prompt_choice("What to do?", [
        ("List all submissions", "list"),
        ("View specific file", "view"),
    ])
    if action is None: return
    if action == "list":
        show_submissions()
    else:
        csv_opts = [(f.name, f.name) for f in csvs]
        fname = prompt_choice("Select file", csv_opts)
        if fname:
            rows = int(prompt_input("Rows to display", "20"))
            show_submissions(view=fname, rows=rows)


def menu_config():
    action = prompt_choice("View which section?", [
        ("All sections",     ""),
        ("MODEL_CONFIG",     "MODEL_CONFIG"),
        ("TRACKING_CONFIG",  "TRACKING_CONFIG"),
        ("VIDEO_CONFIG",     "VIDEO_CONFIG"),
        ("LOGGING_CONFIG",   "LOGGING_CONFIG"),
        ("API_CONFIG",       "API_CONFIG"),
        ("HARDWARE_CONFIG",  "HARDWARE_CONFIG"),
        ("ALERT_RULES",      "ALERT_RULES"),
        ("PATHS",            "PATHS"),
    ])
    if action is None: return
    show_config(section=action if action else None)


def menu_api_test():
    try:
        from api_client import available_protocols
        avail = available_protocols()
    except ImportError:
        avail = ["rest", "mock"]
    proto = prompt_input(f"Protocol ({', '.join(avail)})", "rest")
    if proto == "mock":
        url = "mock://localhost"
    else:
        url = prompt_input("API URL", "https://api.tesa.or.th/drone")
    timeout = int(prompt_input("Timeout (seconds)", "10"))
    retries = int(prompt_input("Retries", "2"))
    run_api_test(url=url, timeout=timeout, protocol=proto, retries=retries)


def interactive_menu():
    clear_screen()

    while True:
        print_main_menu()

        raw = input("  Enter choice: ").strip()
        if not raw:
            continue
        if raw == "0":
            print("\n  Goodbye!\n")
            break

        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(MAIN_MENU) - 1):  # -1 to exclude exit
                print("  Invalid choice.\n")
                continue
        except ValueError:
            print("  Invalid choice.\n")
            continue

        choice = MAIN_MENU[idx][1]

        dispatch = {
            "detect":       menu_detect,
            "localize":     menu_localize,
            "pipeline":     menu_pipeline,
            "pipeline_adv": run_pipeline_adv,
            "validate":     menu_validate,
            "compliance":   run_compliance,
            "models":       show_models,
            "submissions":  menu_submissions,
            "videos":       show_videos,
            "config":       menu_config,
            "info":         show_info,
            "api_test":     menu_api_test,
            "dashboard":    launch_dashboard,
        }

        func = dispatch.get(choice)
        if func:
            func()

        pause()
        clear_screen()


# =========================================================================== #
#  Argparse (direct command mode)
# =========================================================================== #
def build_parser():
    parser = argparse.ArgumentParser(
        prog="tesa-cli",
        description="TESA Defence AI - CLI (run without args for interactive menu)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive mode:  python cli.py
Direct commands:   python cli.py <command> [options]

  python cli.py detect --video videos/P3_VIDEO.mp4
  python cli.py localize --detections submissions/p1_detection_obb.csv
  python cli.py pipeline --video videos/P3_VIDEO.mp4
  python cli.py pipeline-adv
  python cli.py validate --predictions submissions/submission.csv --ground-truth submissions/mock_ground_truth.csv
  python cli.py compliance
  python cli.py models
  python cli.py submissions --view p1_detection_obb.csv
  python cli.py videos
  python cli.py config
  python cli.py info
  python cli.py dashboard
  python cli.py api-test
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Commands (omit for interactive menu)")

    # detect
    p = sub.add_parser("detect", help="Run Problem 1 detection")
    p.add_argument("--video", required=True); p.add_argument("--output")
    p.add_argument("--model"); p.add_argument("--conf", type=float, default=0.55)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--no-bytetrack", action="store_true")
    p.set_defaults(func=lambda a: run_detect(
        a.video, a.output, a.model, a.conf, a.display, a.save_video, a.no_bytetrack))

    # localize
    p = sub.add_parser("localize", help="Run Problem 2 localization")
    p.add_argument("--detections", required=True); p.add_argument("--output")
    p.add_argument("--model", default="xgboost", choices=["xgboost", "stacking"])
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--video")
    p.set_defaults(func=lambda a: run_localize(
        a.detections, a.output, a.model, a.width, a.height, a.video))

    # pipeline
    p = sub.add_parser("pipeline", help="Run Problem 3 integration")
    p.add_argument("--video", required=True); p.add_argument("--output")
    p.add_argument("--video-id", default="video_01")
    p.add_argument("--conf", type=float, default=0.55); p.add_argument("--model")
    p.add_argument("--regression", default="xgboost", choices=["xgboost", "stacking"])
    p.add_argument("--display", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.set_defaults(func=lambda a: run_pipeline(
        a.video, a.output, a.video_id, a.conf, a.model, a.regression, a.display, a.save_video))

    # pipeline-adv
    p = sub.add_parser("pipeline-adv", help="Run production pipeline")
    p.set_defaults(func=lambda a: run_pipeline_adv())

    # validate
    p = sub.add_parser("validate", help="Validate predictions")
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--threshold", type=int, default=20)
    p.add_argument("--report")
    p.set_defaults(func=lambda a: run_validate(a.predictions, a.ground_truth, a.threshold, a.report))

    # compliance
    p = sub.add_parser("compliance", help="Compliance checks")
    p.set_defaults(func=lambda a: run_compliance())

    # models
    p = sub.add_parser("models", help="List models")
    p.add_argument("--detail", action="store_true")
    p.set_defaults(func=lambda a: show_models(a.detail))

    # submissions
    p = sub.add_parser("submissions", help="List/view submissions")
    p.add_argument("--view"); p.add_argument("--rows", type=int, default=20)
    p.set_defaults(func=lambda a: show_submissions(a.view, a.rows))

    # videos
    p = sub.add_parser("videos", help="List videos")
    p.set_defaults(func=lambda a: show_videos())

    # config
    p = sub.add_parser("config", help="Show configuration")
    p.add_argument("--section")
    p.set_defaults(func=lambda a: show_config(a.section))

    # info
    p = sub.add_parser("info", help="System information")
    p.set_defaults(func=lambda a: show_info())

    # dashboard
    p = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p.set_defaults(func=lambda a: launch_dashboard())

    # api-test
    p = sub.add_parser("api-test", help="Test API endpoint (multi-protocol)")
    p.add_argument("--url", default="https://api.tesa.or.th/drone")
    p.add_argument("--protocol", default=None, help="rest|mqtt|websocket|grpc|mock")
    p.add_argument("--api-key"); p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--lat", type=float, default=13.22)
    p.add_argument("--lon", type=float, default=66.32)
    p.set_defaults(func=lambda a: run_api_test(a.url, a.api_key, a.timeout, a.lat, a.lon, a.protocol, a.retries))

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n\n  Goodbye!\n")
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
