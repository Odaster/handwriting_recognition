"""Central heavy-model cache with exclusive-GPU eviction.

Large GPU models must not sit in VRAM at the same time (e.g. Chandra ~5B and
SAGE 1.7B still run one at a time on a 12 GB card). Models in the exclusive group are
loaded one at a time: loading one evicts the others (moved off GPU and freed),
so they run sequentially without exhausting VRAM.
"""

from __future__ import annotations

import gc
import logging
import threading
import warnings

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: dict[str, object] = {}

# Keys that must never share GPU memory. Loading one evicts the others.
_EXCLUSIVE = {"chandra", "corrector", "sage"}
_WRAPPER_ATTRS = (
    "model",
    "hf_model",
    "llm",
    "inner",
    "module",
    "language_model",
    "processor",
    "vision_model",
    "text_model",
)


def _move_off_gpu(obj, _seen: set[int] | None = None) -> None:
    """Best-effort: walk wrappers and send torch modules to CPU."""
    if obj is None:
        return
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    to = getattr(obj, "to", None)
    if callable(to) and hasattr(obj, "parameters"):
        try:
            obj.to("cpu")
        except Exception:
            pass
    elif callable(to):
        try:
            obj.to("cpu")
        except Exception:
            pass

    for name in _WRAPPER_ATTRS:
        if hasattr(obj, name):
            try:
                _move_off_gpu(getattr(obj, name), _seen)
            except Exception:
                pass

    if isinstance(obj, (list, tuple)):
        for item in obj:
            _move_off_gpu(item, _seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            _move_off_gpu(item, _seen)


def _empty_cuda() -> None:
    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ipc = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc):
                ipc()
    except Exception:
        pass


def drop_cuda_modules() -> None:
    """Last-resort: CPU-offload any nn.Module still sitting on CUDA."""
    try:
        import torch
        from torch import nn
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    # gc.get_objects() can touch torch.distributed.reduce_op, which emits a
    # FutureWarning even though nothing is wrong.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"torch\.distributed",
        )
        try:
            objects = list(gc.get_objects())
        except Exception:
            objects = []
        for obj in objects:
            try:
                if not isinstance(obj, nn.Module):
                    continue
                param = next(obj.parameters(), None)
                if param is None or not param.is_cuda:
                    continue
                obj.to("cpu")
            except Exception:
                continue
    _empty_cuda()


def vram_allocated_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def _evict(except_key: str) -> None:
    freed = False
    for key in list(_CACHE):
        if key == except_key or key not in _EXCLUSIVE:
            continue
        log.info("GPU: unloading %s to free VRAM", key)
        obj = _CACHE.pop(key)
        _move_off_gpu(obj)
        del obj
        freed = True
    if freed:
        drop_cuda_modules()
        _empty_cuda()


def load(key: str, loader):
    """Return the cached object for ``key``, loading it via ``loader()`` if needed.

    For exclusive-GPU keys, evict the other exclusive models first to free VRAM.
    """
    with _LOCK:
        if key in _EXCLUSIVE:
            _evict(except_key=key)
        obj = _CACHE.get(key)
        if obj is None:
            log.info("Loading model %s", key)
            obj = loader()
            _CACHE[key] = obj
        return obj


def unload(key: str) -> None:
    with _LOCK:
        obj = _CACHE.pop(key, None)
        if obj is not None:
            log.info("GPU: unloading %s", key)
            _move_off_gpu(obj)
            del obj
        drop_cuda_modules()
        _empty_cuda()


def cached_keys() -> set[str]:
    with _LOCK:
        return set(_CACHE)
