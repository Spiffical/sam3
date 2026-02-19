def main(*args, **kwargs):
    # Lazy import to avoid pulling UI-only dependencies (e.g. bioclip)
    # when callers only need backend utilities.
    from .app import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
