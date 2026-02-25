import torch

print("=" * 50)
print("PyTorch GPU Check")
print("=" * 50)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"\n✅ GPU Support Enabled!")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print(f"\n❌ No GPU Support!")
    print(f"   PyTorch is CPU-only version: torch-{torch.__version__}")
    print(f"\nTo enable GPU:")
    print(f"   1. Uninstall current PyTorch: pip uninstall torch torchvision")
    print(f"   2. Install CUDA version: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
