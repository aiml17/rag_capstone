"""
Test script to verify Apple Silicon MPS (Metal Performance Shaders) support.
Run this to confirm your M4 Mac can use GPU acceleration with PyTorch.
"""

import torch
import platform


def check_system():
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")


def check_mps():
    print("\n" + "=" * 60)
    print("MPS (METAL PERFORMANCE SHADERS) STATUS")
    print("=" * 60)

    # Check if MPS is available
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")

    if mps_available:
        # Check if MPS is built
        mps_built = torch.backends.mps.is_built()
        print(f"MPS built: {mps_built}")

        if mps_built:
            # Test creating a tensor on MPS
            try:
                device = torch.device("mps")
                x = torch.randn(3, 3, device=device)
                print(f"Successfully created tensor on MPS device!")
                print(f"Test tensor:\n{x}")
                print("\n✅ Your M4 Mac is ready for GPU-accelerated PyTorch!")
                return "mps"
            except Exception as e:
                print(f"Error creating tensor on MPS: {e}")
                return "cpu"
    else:
        print("\n⚠️  MPS not available. Will use CPU instead.")
        return "cpu"


if __name__ == "__main__":
    check_system()
    device = check_mps()

    print("\n" + "=" * 60)
    print("RECOMMENDED DEVICE FOR YOUR PROJECT")
    print("=" * 60)
    print(f"Use this in your code: device = torch.device('{device}')")