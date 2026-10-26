"""Backbone adapters: the attachment points between CALD and in-context
tabular models. ``nanotabiclv2`` is fully functional; ``tabpfn`` and
``tabicl`` are placeholders pending port from the paper repository."""

from .base import CALDBackbone

_REGISTRY = {
    "nanotabiclv2": ("cald.adapters.nanotabiclv2", "NanoTabICLv2Adapter"),
    "tabpfn": ("cald.adapters.tabpfn", "TabPFNAdapter"),
    "tabicl": ("cald.adapters.tabicl", "TabICLAdapter"),
}


def get_adapter(name: str, **kwargs) -> CALDBackbone:
    """Instantiate a backbone adapter by name (lazy import).

    Available names: "nanotabiclv2" (functional), "tabpfn", "tabicl"
    (placeholders).
    """
    try:
        module_name, cls_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown adapter {name!r}; choose from {sorted(_REGISTRY)}")
    import importlib

    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)


__all__ = ["CALDBackbone", "get_adapter"]
