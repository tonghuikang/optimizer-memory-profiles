"""

modal run profiling.py

"""

import os
import glob
from typing import Tuple, Callable
from plotting import generate_plot_from_file
import shutil
from concurrent.futures import ThreadPoolExecutor

import modal
import torch

from torch.autograd.profiler import record_function
from torchvision import models
from muon import Muon  # util PyTorch actually releases Muon

def trace_handler(prof: torch.profiler.profile) -> None:
    # Construct the memory timeline file.
    prof.export_memory_timeline("/tmp/profile.html")
    prof.export_memory_timeline("/tmp/profile.raw.json.gz")


def run_resnet50(
    optimizer_type: str,
    device: str,
    use_gradient_accmulation: bool = False,
    num_iters: int = 10,
) -> Tuple[bytes, bytes]:
    assert device in ("cpu", "cuda")
    model = models.resnet50().to(device=device)
    inputs = torch.randn(1, 3, 224, 224, device=device)
    labels = torch.rand_like(model(inputs))

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters())
    elif optimizer_type == "sgd_with_momentum":
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif optimizer_type == "muon":
        optimizer = Muon(model.parameters())
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=10, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for iter_idx in range(num_iters):
            prof.step()
            with record_function("## forward ##"):
                pred = model(inputs)

            with record_function("## backward ##"):
                loss_fn(pred, labels).backward()

            with record_function("## optimizer ##"):
                if use_gradient_accmulation and iter_idx % 4 != 3:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    # Read and return the files
    with open("/tmp/profile.html", "rb") as f:
        html_data: bytes = f.read()

    with open("/tmp/profile.raw.json.gz", "rb") as f:
        json_data: bytes = f.read()

    return html_data, json_data


def profile_memory_local_sgd() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU."""
    html_data, json_data = run_resnet50(optimizer_type="sgd", device="cpu")
    return html_data, json_data


# Modal setup functions

image: modal.Image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
    )
    .pip_install("matplotlib")
)

app: modal.App = modal.App("profiling", image=image)


@app.function(
    cpu=2,
    timeout=3600,
)
def profile_memory_modal_cpu_sgd() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU."""
    html_data, json_data = run_resnet50(optimizer_type="sgd", device="cpu")
    return html_data, json_data


@app.function(
    gpu="A10G",
    cpu=2,
    timeout=3600,
)
def profile_memory_modal_gpu_sgd() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU."""
    assert torch.cuda.is_available()
    device: str = "cuda"
    # Run profiling
    html_data, json_data = run_resnet50(optimizer_type="sgd", device=device)
    return html_data, json_data


@app.function(
    gpu="A10G",
    cpu=2,
    timeout=3600,
)
def profile_memory_modal_gpu_sgd_momentum() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU."""
    assert torch.cuda.is_available()
    device: str = "cuda"
    # Run profiling
    html_data, json_data = run_resnet50(
        optimizer_type="sgd_with_momentum", device=device
    )
    return html_data, json_data


@app.function(
    gpu="A10G",
    cpu=2,
    timeout=3600,
)
def profile_memory_modal_gpu_adam() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU using Adam optimizer."""
    assert torch.cuda.is_available()
    device: str = "cuda"
    # Run profiling with Adam optimizer
    html_data, json_data = run_resnet50(optimizer_type="adam", device=device)
    return html_data, json_data


@app.function(
    gpu="A10G",
    cpu=2,
    timeout=3600,
)
def profile_memory_modal_gpu_adam_gradacc() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU using Adam optimizer."""
    assert torch.cuda.is_available()
    device: str = "cuda"
    # Run profiling with Adam optimizer
    html_data, json_data = run_resnet50(
        optimizer_type="adam", device=device, use_gradient_accmulation=True
    )
    return html_data, json_data


@app.function(
    gpu="A10G",
    cpu=2,
    timeout=3600,
)
def profile_memory_modal_gpu_muon() -> Tuple[bytes, bytes]:
    """Run memory profiling on Modal with GPU using Adam optimizer."""
    assert torch.cuda.is_available()
    device: str = "cuda"
    # Run profiling with Adam optimizer
    html_data, json_data = run_resnet50(optimizer_type="muon", device=device)
    return html_data, json_data


def process(subdirectory: str, func: Callable[[], Tuple[bytes, bytes]]) -> None:
    os.makedirs(f"profiling/{subdirectory}", exist_ok=True)

    html_data, json_data = func()
    # Save HTML file
    with open(f"profiling/{subdirectory}/profile.html", "wb") as f:
        f.write(html_data)
    print(f"Saved HTML file: {len(html_data)} bytes")

    # Save JSON file
    with open(f"profiling/{subdirectory}/profile.raw.json.gz", "wb") as f:
        f.write(json_data)
    print(f"Saved JSON file: {len(json_data)} bytes")


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint to run profiling and save files."""
    os.makedirs("profiling", exist_ok=True)

    for subdir in glob.glob("profiling/*/"):
        shutil.rmtree(subdir)

    # this is calling matplotlib, and we cannot call outside of main thread
    process("local_cpu_sgd", profile_memory_local_sgd)

    directory_and_functions: list[tuple[str, Callable[[], Tuple[bytes, bytes]]]] = [
        # ("modal_cpu_sgd", profile_memory_modal_cpu_sgd.remote),
        # ("modal_gpu_sgd", profile_memory_modal_gpu_sgd.remote),
        # ("modal_gpu_sgd_momentum", profile_memory_modal_gpu_sgd_momentum.remote),
        # ("modal_gpu_adam", profile_memory_modal_gpu_adam.remote),
        # ("modal_gpu_adam_gradacc", profile_memory_modal_gpu_adam_gradacc.remote),
        ("modal_gpu_muon", profile_memory_modal_gpu_muon.remote),
    ]

    # Run all profiling tasks in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        for subdirectory, func in directory_and_functions:
            executor.submit(process, subdirectory, func)

    # Generate plots after all parallel tasks complete
    generate_plot_from_file("profiling/local_cpu_sgd")
    for subdirectory, _ in directory_and_functions:
        # matplotlib cannot be called outside of main thread
        generate_plot_from_file(f"profiling/{subdirectory}")
