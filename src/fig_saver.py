from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Optional


_fig_counter = count(1)
_default_fig_ext = "pdf"


def _default_outputs_dir() -> Path:
    """
    Return the repo-consistent outputs directory: code/main_clean/outputs.

    This is resolved from this file's location, so it does not depend on the
    notebook/kernel working directory.
    """
    code_dir = Path(__file__).resolve().parents[1]  # .../code
    return code_dir / "main_clean" / "outputs"


def save_fig(
    name: Optional[str] = None,
    fig=None,
    ext: Optional[str] = None,
    outputs_dir: Optional[Path] = None,
):
    """
    Save the current matplotlib figure (or provided fig) to code/main_clean/outputs.

    Parameters
    ----------
    name:
        Filename (with or without extension). If omitted, uses fig_1, fig_2, ...
    fig:
        Matplotlib figure object. If omitted, uses plt.gcf().
    ext:
        File extension to use if `name` has no suffix (default: pdf).
    outputs_dir:
        Override output directory (default: code/main_clean/outputs).
    """
    import matplotlib.pyplot as plt

    fig = fig or plt.gcf()
    ext = (ext or _default_fig_ext).lstrip(".")
    out_dir = outputs_dir or _default_outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = name or f"fig_{next(_fig_counter)}"
    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(f".{ext}")
    if not path.is_absolute():
        path = out_dir / path

    fig.savefig(path, bbox_inches="tight")
    print(f"Saved figure to {path}")

