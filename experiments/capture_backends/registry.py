from __future__ import annotations

from importlib import import_module

from .backends.base import CaptureBackend
from .contracts import BackendCapabilities, CaptureConfig


_BACKENDS: dict[str, tuple[str, str]] = {
    "mss": (".backends.mss_backend", "MssBackend"),
    "printwindow": (".backends.printwindow_backend", "PrintWindowBackend"),
    "wgc": (".backends.wgc_backend", "WgcBackend"),
    "dxcam": (".backends.dxcam_backend", "DxcamBackend"),
}


def backend_names() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def _backend_class(name: str) -> type[CaptureBackend]:
    try:
        module_name, class_name = _BACKENDS[name]
    except KeyError as exc:
        choices = ", ".join(_BACKENDS)
        raise ValueError(f"unknown capture backend {name!r}; choose one of: {choices}") from exc
    module = import_module(module_name, package=__package__)
    return getattr(module, class_name)


def create_backend(
    name: str,
    config: CaptureConfig | None = None,
) -> CaptureBackend:
    return _backend_class(name)(config=config)


def probe_backends() -> tuple[BackendCapabilities, ...]:
    return tuple(_backend_class(name).get_capabilities() for name in _BACKENDS)
