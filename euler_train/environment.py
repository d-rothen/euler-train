"""Collect runtime environment metadata for run_environment.json."""
from __future__ import annotations

import os
import platform
import subprocess
import sys


def get_run_environment() -> dict:
    """Return a dict matching the run_environments schema."""
    return {
        "name": _hostname(),
        "python_version": sys.version.split()[0],
        "cuda_version": _cuda_version(),
        "gpu_type": _gpu_type(),
        "gpu_count": _gpu_count(),
        "packages_snapshot": _packages_snapshot(),
        "docker_image": None,
        "docker_digest": None,
        "metadata": None,
    }


def _hostname() -> str | None:
    try:
        return platform.node() or None
    except Exception:
        return None


def _cuda_version() -> str | None:
    # 1. torch.version.cuda
    try:
        import torch
        if torch.version.cuda:
            return torch.version.cuda
    except Exception:
        pass

    # 2. nvcc --version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.1, V12.1.66"
                    parts = line.split("release")[-1].strip().split(",")
                    return parts[0].strip()
    except Exception:
        pass

    # 3. CUDA_VERSION env var
    return os.environ.get("CUDA_VERSION")


def _gpu_type() -> str | None:
    # 1. pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return name
    except Exception:
        pass

    # 2. nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0].strip()
            return first_line or None
    except Exception:
        pass

    return None


def _gpu_count() -> int | None:
    # 1. torch.cuda.device_count()
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass

    # 2. pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        pass

    # 3. CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
        if devices:
            return len(devices)

    return None


def _packages_snapshot() -> dict[str, str] | None:
    # Try uv pip freeze, then pip freeze
    for cmd in (["uv", "pip", "freeze"], ["pip", "freeze"]):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return _parse_freeze(result.stdout)
        except Exception:
            continue
    return None


def _parse_freeze(output: str) -> dict[str, str]:
    packages = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "==" in line:
            name, _, version = line.partition("==")
            packages[name.strip()] = version.strip()
    return packages
