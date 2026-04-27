from __future__ import annotations

import importlib
import pickle


def patch_broken_dill() -> None:
    """Patch incomplete dill installs so torch can still import.

    Some local Windows/conda environments end up with a namespace package
    named ``dill`` that is missing the symbols PyTorch expects at import time.
    The training code does not depend on dill directly, so a lightweight
    compatibility patch is enough here.
    """

    try:
        dill = importlib.import_module("dill")
    except Exception:
        return

    if not hasattr(dill, "extend"):
        dill.extend = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    for name in ("dump", "dumps", "load", "loads", "Pickler", "Unpickler"):
        if not hasattr(dill, name):
            setattr(dill, name, getattr(pickle, name))
