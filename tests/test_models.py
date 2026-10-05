"""Exclusive-GPU model cache: Chandra and SAGE must not coexist."""

from __future__ import annotations

from app import models


class _Dummy:
    def __init__(self, name: str):
        self.name = name
        self.device = "cuda"

    def to(self, device):  # noqa: A003 - torch-like API
        self.device = device
        return self


def setup_function():
    for key in list(models.cached_keys()):
        models.unload(key)


def teardown_function():
    for key in list(models.cached_keys()):
        models.unload(key)


def test_exclusive_keys_evict_each_other():
    chandra = models.load("chandra", lambda: _Dummy("chandra"))
    assert models.cached_keys() == {"chandra"}
    assert chandra.device == "cuda"

    sage = models.load("corrector", lambda: _Dummy("sage"))
    assert sage.name == "sage"
    assert models.cached_keys() == {"corrector"}
    # The evicted Chandra dummy was moved off GPU before drop.
    assert chandra.device == "cpu"

    models.load("chandra", lambda: _Dummy("chandra-2"))
    assert models.cached_keys() == {"chandra"}
    assert sage.device == "cpu"


def test_nested_wrapper_evicted():
    inner = _Dummy("inner")

    class Wrap:
        def __init__(self, model):
            self.model = model

    models.load("chandra", lambda: Wrap(inner))
    models.unload("chandra")
    assert inner.device == "cpu"


def test_reload_reuses_cached_object():
    first = models.load("corrector", lambda: _Dummy("once"))
    second = models.load("corrector", lambda: _Dummy("twice"))
    assert first is second
    assert first.name == "once"
