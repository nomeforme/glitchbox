import os
import torch

# check if MPS is available OSX only M1/M2/M3 chips
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

if torch.cuda.is_available():
    # Use CUDA_DEVICE env var to select GPU index (default: 0)
    cuda_device_idx = int(os.environ.get("CUDA_DEVICE", 0))
    if cuda_device_idx >= torch.cuda.device_count():
        print(f"[device.py] WARNING: CUDA_DEVICE={cuda_device_idx} but only {torch.cuda.device_count()} GPUs visible, using 0")
        cuda_device_idx = 0
    device = torch.device(f"cuda:{cuda_device_idx}")
    print(f"[device.py] Using GPU {cuda_device_idx}: {torch.cuda.get_device_name(cuda_device_idx)}")
elif xpu_available:
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

torch_dtype = torch.float16
if mps_available:
    device = torch.device("mps")
    torch_dtype = torch.float32
