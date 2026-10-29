"""Shadow-mode artifacts — where ML output is allowed to land.

ML_ENGINE.md sets two rules that together decide this module's existence:

  * R1.5: expose predictions in *"a separate, reviewable prediction artifact.
    Do not inject them into evidence edges or Stage 7 factors automatically."*
  * R2.4: *"run the BN in shadow mode: store posterior output separately...
    It cannot override Stage 7, Stage 8, or observed evidence."*

So the ML layer owns exactly one directory, ``<output_dir>/ml/``, and every
path it writes is asserted to be inside it. :func:`assert_is_ml_artifact`
raises on any path that would land on a stage store, which turns "we must not
write to ``hypotheses.json``" from a code-review item into a runtime guarantee.

Nothing here reads a stage store either, beyond the inputs named by the
evaluation itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ARTIFACTS",
    "assert_is_ml_artifact",
    "ml_directory",
    "read_artifact",
    "write_artifact",
]

#: Files this layer may produce, all inside its own directory.
ARTIFACTS: tuple[str, ...] = (
    "label_inventory.json",
    "baseline_evaluation.json",
    "ml_predictions.json",
    "bn_posteriors.json",
)

#: Stage-owned stores this layer must never write. Listed explicitly so the
#: guard does not depend on someone remembering the pipeline's file list.
_FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    "hypotheses.json",
    "contradictions.json",
    "evidence_gaps.json",
    "graph_edges.json",
    "resolved_entities.json",
    "missing_edges.json",
    "community_assignments.json",
    "extraction_summary.json",
    "audit_trail.json",
    "global_entities.json",
    "global_entity_links.json",
    "cross_case_alerts.json",
    "critic_review.json",
)


def ml_directory(output_dir: str | Path) -> Path:
    """The one directory this layer writes to."""
    return Path(output_dir) / "ml"


def assert_is_ml_artifact(path: str | Path) -> Path:
    """Raise unless ``path`` is an artifact this layer is allowed to own.

    Two checks, deliberately redundant: the resolved path must sit inside a
    directory called ``ml``, and its filename must not be a known stage store.
    A future caller that passes ``.../ml/../hypotheses.json`` fails the first
    check after normalisation rather than escaping through string matching.
    """
    resolved = Path(path).resolve()
    parts = resolved.parts
    if "ml" not in parts:
        raise PermissionError(
            f"{resolved} is outside the ML artifact directory; the ML layer "
            "may only write its own reviewable artifacts (R1.5, R2.4)"
        )
    if resolved.name in _FORBIDDEN_SUFFIXES:
        raise PermissionError(
            f"{resolved.name} is a stage-owned store; ML output must never "
            "overwrite an analytical store (R1.5, R2.4)"
        )
    return resolved


def write_artifact(output_dir: str | Path, name: str, payload: Any) -> Path:
    """Write one ML artifact inside the ML directory."""
    if name not in ARTIFACTS:
        raise ValueError(
            f"unknown ML artifact {name!r}; declared artifacts: {ARTIFACTS}"
        )
    target = ml_directory(output_dir) / name
    assert_is_ml_artifact(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    return target


def read_artifact(output_dir: str | Path, name: str) -> Any:
    """Read back one ML artifact."""
    if name not in ARTIFACTS:
        raise ValueError(f"unknown ML artifact {name!r}")
    target = ml_directory(output_dir) / name
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def artifact_summary(output_dir: str | Path) -> Mapping[str, bool]:
    """Which declared artifacts are present — used by tests and reports."""
    directory = ml_directory(output_dir)
    return {
        name: (directory / name).exists() for name in ARTIFACTS
    }
