"""Project path resolution.

Every other module asks this one where things live. Nothing anywhere else
builds a path from ``__file__`` or assumes a current working directory, so
the application behaves identically whether it is launched from the project
root, from an IDE, or from a packaged bundle.
"""

from __future__ import annotations

from pathlib import Path

# core/paths.py -> core/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"


def resolve(path_like: str | Path, base: Path | None = None) -> Path:
    """Turn a configured path into an absolute :class:`Path`.

    Absolute paths are returned unchanged. Relative paths are resolved
    against *base*, defaulting to the project root.
    """
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return ((base or PROJECT_ROOT) / p).resolve()


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
