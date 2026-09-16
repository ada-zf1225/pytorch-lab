# 00_env/env_check.py
# Sanity check for a fresh PyTorch install: versions, devices, matmul throughput, autograd.
import time
import platform
import torch

print(f"python {platform.python_version()} | torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  built with CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} | {p.total_memory / 1e9:.1f} GB | sm_{p.major}{p.minor}")
print(f"mps available: {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}\n")


def sync(dev):
    # GPU kernels are launched asynchronously; wait for completion before reading the clock.
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench_matmul(dev, n=4096, reps=10):
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    for _ in range(3):  # warm-up: first calls include one-off initialisation cost
        a @ b
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    sync(dev)
    dt = (time.perf_counter() - t0) / reps
    tflops = 2 * n**3 / dt / 1e12  # an n×n matmul costs ~2n^3 floating-point ops
    print(f"{str(dev):5s} matmul {n}x{n}: {dt*1e3:7.1f} ms  ≈ {tflops:5.1f} TFLOPS")


bench_matmul(torch.device("cpu"), n=2048)
if device.type != "cpu":
    bench_matmul(device)
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")  # let fp32 matmul use TF32 tensor cores
    print("-- TF32 on --")
    bench_matmul(device)

# autograd smoke test: y = sum(x_i^2)  =>  dy/dx_i = 2 x_i
x = torch.tensor([0.0, 1.0, 2.0], device=device, requires_grad=True)
y = (x**2).sum()
y.backward()
print(f"\nautograd: x = {x.tolist()}, dy/dx = {x.grad.tolist()}  (expected [0, 2, 4])")
