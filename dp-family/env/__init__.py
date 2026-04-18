"""UR5e-only build.

Simulation environments were removed from this repository.
"""

__all__ = []


def __getattr__(name: str):
    if name in {"AdroitEnv", "DexArtEnv"}:
        raise ImportError(
            f"{name} is not available in this UR5e-only repository variant."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

