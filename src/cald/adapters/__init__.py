"""Backbone adapters: the attachment points between CALD and in-context
tabular models. ``nanotabiclv2`` is fully functional; ``tabpfn`` and
``tabicl`` are placeholders pending port from the paper repository.

Third-party backbones register themselves with ``register_adapter``::

    from cald.adapters import register_adapter
    register_adapter("mymodel", MyModelAdapter)
    backbone = get_adapter("mymodel")
"""

from .base import CALDBackbone

# name -> CALDBackbone subclass, or a lazy ("module", "ClassName") pair
_REGISTRY: dict[str, type[CALDBackbone] | tuple[str, str]] = {
    "nanotabiclv2": ("cald.adapters.nanotabiclv2", "NanoTabICLv2Adapter"),
    "tabpfn": ("cald.adapters.tabpfn", "TabPFNAdapter"),
    "tabicl": ("cald.adapters.tabicl", "TabICLAdapter"),
}


def register_adapter(name: str, cls: type[CALDBackbone]) -> None:
    """Register a backbone adapter class under ``name`` for ``get_adapter``.

    Re-registering a built-in name is refused; pick a new name instead.
    """
    if name in _REGISTRY and not isinstance(_REGISTRY[name], type):
        raise ValueError(f"{name!r} is reserved by a built-in adapter")
    if not (isinstance(cls, type) and issubclass(cls, CALDBackbone)):
        raise TypeError("adapter must be a CALDBackbone subclass")
    _REGISTRY[name] = cls


def get_adapter(name: str, **kwargs) -> CALDBackbone:
    """Instantiate a backbone adapter by name.

    Built-in names: "nanotabiclv2" (functional), "tabpfn", "tabicl"
    (placeholders). Built-ins import lazily; registered adapters are used
    directly.
    """
    try:
        entry = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown adapter {name!r}; choose from {sorted(_REGISTRY)}")
    if isinstance(entry, type):
        return entry(**kwargs)
    module_name, cls_name = entry
    import importlib

    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)


__all__ = ["CALDBackbone", "get_adapter", "register_adapter"]
