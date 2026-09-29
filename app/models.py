"""Central heavy-model cache with exclusive-GPU eviction.

Large GPU models must not sit in VRAM at the same time (e.g. Chandra ~5B and
SAGE-1.7B will not both fit on a 12 GB card). Models in the exclusive group are
loaded one at a time: loading one evicts the others (moved off GPU and freed),
so they run sequentially without exhausting VRAM.
"""

from __future__ import annotations

import gc
import logging
import threading

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: dict[str, object] = {}

# Keys that must never share GPU memory. Loading one evicts the others.
_EXCLUSIVE = {"chandra", "sage"}


def _move_off_gpu(obj) -> None:
    """Best-effort move a model (or (tokenizer/processor, model) tuple) to CPU."""
    parts = obj if isinstance(obj, tuple) else (obj,)
    for part in parts:
        to = getattr(part, "to", None)
        if callable(to):
            try:
                part.to("cpu")
            except Exception:
                pass


def _empty_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
            _empty_cuda()


def cached_keys() -> set[str]:
    with _LOCK:
        return set(_CACHE)
