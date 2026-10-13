"""Hardware detection. Reports what PyTorch actually says; never assumes."""
from __future__ import annotations

import os
import platform
from typing import Any, Dict


def detect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the environment
        info.update(torch=None, torch_error=f"{type(exc).__name__}: {exc}",
                    cuda_available=False, mps_available=False, mps_built=False)
        return info
    mps_backend = getattr(torch.backends, "mps", None)
    info.update(
        torch=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        mps_built=bool(mps_backend and mps_backend.is_built()),
        mps_available=bool(mps_backend and mps_backend.is_available()),
    )
    if info["cuda_available"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def choose_device(info: Dict[str, Any], requested: str = "auto") -> str:
    """Pick a device name for `laya.load(device=...)`.

    `requested` is 'auto', 'cpu', 'mps' or 'cuda'. An explicit request for an unavailable
    accelerator raises instead of silently running somewhere else.
    """
    requested = (requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not info.get("mps_available"):
            raise RuntimeError("LAYA_DEVICE=mps was requested but PyTorch reports MPS is not available.")
        return "mps"
    if requested == "cuda":
        if not info.get("cuda_available"):
            raise RuntimeError("LAYA_DEVICE=cuda was requested but PyTorch reports CUDA is not available.")
        return "cuda"
    if requested != "auto":
        raise RuntimeError(f"Unknown LAYA_DEVICE {requested!r}; use auto, cpu, mps or cuda.")
    # Same priority as Laya's own automatic choice: CUDA, then MPS, then CPU.
    if info.get("cuda_available"):
        return "cuda"
    if info.get("mps_available"):
        return "mps"
    return "cpu"
