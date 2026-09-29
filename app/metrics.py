"""Live system metrics: CPU, RAM, network, and (optional) NVIDIA GPU.

GPU metrics use ``pynvml`` (package ``nvidia-ml-py``) when an NVIDIA driver is
present, falling back to ``torch.cuda`` for memory only, and to ``None`` on
CPU-only hosts. All imports are guarded so the endpoint works everywhere.
"""

from __future__ import annotations

import time

_prev_net = {"t": None, "sent": 0, "recv": 0}
_nvml_ready = None  # None=unknown, True/False after first attempt


def _nvml():
    """Return an initialized pynvml module or None."""
    global _nvml_ready
    try:
        import pynvml
    except Exception:
        _nvml_ready = False
        return None
    if _nvml_ready is None:
        try:
            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception:
            _nvml_ready = False
    return pynvml if _nvml_ready else None


def _gpu_metrics():
    pynvml = _nvml()
    if pynvml is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "name": name,
                "util_percent": util.gpu,
                "mem_used_gb": round(mem.used / 1e9, 2),
                "mem_total_gb": round(mem.total / 1e9, 2),
            }
        except Exception:
            pass
    # Fallback: torch memory only (no utilization).
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "name": torch.cuda.get_device_name(0),
                "util_percent": None,
                "mem_used_gb": round((total - free) / 1e9, 2),
                "mem_total_gb": round(total / 1e9, 2),
            }
    except Exception:
        pass
    return None


def get_metrics() -> dict:
    import psutil

    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    net = psutil.net_io_counters()

    now = time.time()
    rate_sent = rate_recv = 0.0
    if _prev_net["t"] is not None:
        dt = max(1e-3, now - _prev_net["t"])
        rate_sent = max(0.0, (net.bytes_sent - _prev_net["sent"]) / dt)
        rate_recv = max(0.0, (net.bytes_recv - _prev_net["recv"]) / dt)
    _prev_net.update(t=now, sent=net.bytes_sent, recv=net.bytes_recv)

    return {
        "cpu_percent": round(cpu, 1),
        "ram_used_gb": round(vm.used / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_percent": vm.percent,
        "net_sent_mbps": round(rate_sent * 8 / 1e6, 2),
        "net_recv_mbps": round(rate_recv * 8 / 1e6, 2),
        "gpu": _gpu_metrics(),
    }
