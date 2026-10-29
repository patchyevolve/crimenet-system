"""Optional ML dependencies.

ML_ENGINE.md Roadmap 0.4: *"Keep ML dependencies optional until a milestone is
approved by its own acceptance gate. The current pipeline must remain runnable
without PyTorch Geometric or pgmpy."*

Nothing in this module imports a heavy library. Availability is probed with
``importlib.util.find_spec``, which does not execute the package, so a broken
or half-installed dependency cannot break an unrelated import. Callers that
genuinely need a library go through :func:`require`, which raises a named
exception at the point of use rather than at module import time.

The deterministic parts of this package — label inventory, structural
features, the degree/common-neighbor baseline, the metric implementations, and
the exact Bayesian inference engine — deliberately depend on nothing beyond the
standard library. They are the parts that must keep working when the optional
stack is absent.
"""

from __future__ import annotations

import importlib.util
from typing import Final

__all__ = [
    "HAS_PGMPY",
    "HAS_SCIPY",
    "HAS_SKLEARN",
    "HAS_TORCH",
    "HAS_TORCH_GEOMETRIC",
    "MissingMLDependency",
    "availability",
    "require",
]

#: Libraries this package can use but never requires.
HAS_TORCH: Final[bool] = importlib.util.find_spec("torch") is not None
HAS_TORCH_GEOMETRIC: Final[bool] = importlib.util.find_spec(
    "torch_geometric"
) is not None
HAS_PGMPY: Final[bool] = importlib.util.find_spec("pgmpy") is not None
HAS_SKLEARN: Final[bool] = importlib.util.find_spec("sklearn") is not None
HAS_SCIPY: Final[bool] = importlib.util.find_spec("scipy") is not None


class MissingMLDependency(RuntimeError):
    """Raised when an optional ML dependency is required but not installed.

    Never raised at import time. The message names the capability that was
    requested so the failure reads as "this feature is unavailable", not as a
    crash in unrelated pipeline code.
    """


def availability() -> dict[str, bool]:
    """Return the availability map for every optional dependency."""
    return {
        "torch": HAS_TORCH,
        "torch_geometric": HAS_TORCH_GEOMETRIC,
        "pgmpy": HAS_PGMPY,
        "sklearn": HAS_SKLEARN,
        "scipy": HAS_SCIPY,
    }


def require(name: str, capability: str) -> None:
    """Assert that optional dependency ``name`` is installed.

    Args:
        name: Import name, e.g. ``"torch_geometric"``.
        capability: Human description of what needed it, used in the error so
            a missing optional library never looks like a pipeline defect.

    Raises:
        MissingMLDependency: when the dependency is not importable.
    """
    known = availability()
    if name not in known:
        raise MissingMLDependency(
            f"Unknown optional dependency {name!r}; known: {sorted(known)}"
        )
    if not known[name]:
        raise MissingMLDependency(
            f"{capability} needs {name!r}, which is not installed. "
            "The deterministic pipeline does not depend on it and keeps "
            "running without it (ML_ENGINE.md Roadmap 0.4)."
        )
