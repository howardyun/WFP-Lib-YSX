import os
import subprocess

print("=== Basic CUDA Check ===")

print("\n[1] nvidia-smi")
try:
    result = subprocess.run(
        ["nvidia-smi"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
except Exception as e:
    print(f"nvidia-smi failed: {e}")

print("\n[2] PyTorch CUDA")
try:
    import torch

    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"device {i}:", torch.cuda.get_device_name(i))

        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("CUDA tensor test: OK")
    else:
        print("CUDA tensor test: skipped")

except ImportError:
    print("PyTorch not installed")
except Exception as e:
    print(f"PyTorch CUDA test failed: {e}")

print("\n[3] Environment")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PATH:", os.environ.get("PATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
