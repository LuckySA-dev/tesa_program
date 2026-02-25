"""
TESA Defence AI - Hardware / Device Utilities
==============================================
Centralised device detection that works on any machine:
  - NVIDIA GPU + compatible PyTorch  → cuda
  - NVIDIA GPU but incompatible      → cpu  (automatic fallback)
  - No GPU / FORCE_CPU=1             → cpu

Also provides a safe ``sprint()`` helper that never crashes on
Windows console encodings (emoji characters cause UnicodeEncodeError
when stdout is redirected or the console code-page can't handle them).

Usage:
    from hw import get_device, sprint

    device = get_device()           # 'cuda' or 'cpu'
    sprint("✅ Ready on", device)   # never crashes
"""

import os
import sys
import warnings
import platform

__all__ = ["get_device", "sprint", "system_summary"]


# ---------------------------------------------------------------------------
# Safe print – replaces unencodable chars instead of crashing
# ---------------------------------------------------------------------------

def sprint(*args, **kwargs):
    """Print that never raises UnicodeEncodeError on Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        text = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        print(text, **kwargs)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

_device_cache: str | None = None


def get_device(force: str | None = None) -> str:
    """
    Detect the best available compute device.

    Priority:
        1. ``force`` parameter  ('cpu' or 'cuda')
        2. ``FORCE_CPU=1`` environment variable  → cpu
        3. CUDA available **and** actually works  → cuda
        4. Fallback                               → cpu

    The result is cached after the first call so the (slow) CUDA
    compatibility probe only runs once per process.
    """
    global _device_cache

    # Honour explicit override
    if force and force != "auto":
        return force

    # Environment variable override
    if os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu"

    if _device_cache is not None:
        return _device_cache

    try:
        import torch
    except ImportError:
        _device_cache = "cpu"
        return _device_cache

    if torch.cuda.is_available():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = torch.zeros(1, device="cuda")
                del t
            _device_cache = "cuda"
            sprint(f"[hw] CUDA device detected: {torch.cuda.get_device_name(0)}")
            return _device_cache
        except Exception:
            pass
        # CUDA reports available but the current GPU is not compatible
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sprint("[hw] CUDA available but GPU not compatible with this PyTorch build – using CPU")

    _device_cache = "cpu"
    return _device_cache


# ---------------------------------------------------------------------------
# System summary (for dashboards / CLI info screens)
# ---------------------------------------------------------------------------

def system_summary() -> dict:
    """Return a dict of useful system / hardware info."""
    info: dict = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "device": get_device(),
    }

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except (AssertionError, RuntimeError):
                info["gpu_name"] = "unavailable (hidden or incompatible)"
    except ImportError:
        info["torch"] = "not installed"

    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except (ImportError, AttributeError):
        pass

    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        pass

    return info
