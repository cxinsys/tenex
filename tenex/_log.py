"""Lightweight verbosity control for TENEX status messages.

TENEX is silent by default. Status messages are emitted only when verbose
mode is enabled, either globally via :func:`set_verbose` or per call through
the ``verbose`` argument on the public entry points.
"""

_state = {"verbose": False}


def set_verbose(value: bool = True) -> None:
    """Enable or disable TENEX status messages (off by default)."""
    _state["verbose"] = bool(value)


def is_verbose() -> bool:
    """Return whether TENEX verbose mode is currently on."""
    return _state["verbose"]


def vprint(*args, **kwargs) -> None:
    """print() that emits only when TENEX verbose mode is on."""
    if _state["verbose"]:
        print(*args, **kwargs)
