"""Versioned, provenance-carrying contracts for ML artifacts.

ML_ENGINE.md Roadmap 0.2 requires *"versioned, provenance-carrying contracts
for features, predictions, hypotheses, and uncertainty. Every prediction must
identify its model/version, run, input snapshot, and supporting observed record
IDs."* Roadmap 0's exit gate adds *"no ML result can be mistaken for an
observation."*

Those two sentences are implemented here as validation rather than as prose,
so an artifact that violates them cannot be constructed at all:

  * :class:`Provenance` refuses an empty model name, model version, run id or
    input snapshot. A prediction with no snapshot hash cannot claim to have
    been computed from a fixed graph.
  * ``as_evidence`` is a read-only field that is always ``False`` and
    ``epistemic_status`` is restricted to ``prediction`` / ``inference``.
    Nothing produced here can be written as ``observation``.
  * :class:`BNPosterior` refuses an unnormalized distribution and refuses an
    empty ``assumptions`` or ``parameter_provenance``. Roadmap 2's exit gate
    requires that the output carry assumptions, uncertainty and provenance;
    a posterior whose priors or CPTs are unattributed is not admissible.
  * Absent uncertainty is :data:`UNCALIBRATED`, which carries
    ``value=None``. It can never be read downstream as a measured ``0.0``.

Every schema carries its own ``*_SCHEMA_VERSION``. Consumers check the version
they were written against instead of guessing at field meaning.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Mapping, Sequence

__all__ = [
    "BNPosterior",
    "EPISTEMIC_INFERENCE",
    "EPISTEMIC_PREDICTION",
    "FEATURE_SCHEMA_VERSION",
    "FeatureVector",
    "LINK_PREDICTION_TASK",
    "POSTERIOR_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "Provenance",
    "UNCALIBRATED",
    "Uncertainty",
    "LinkPrediction",
]

FEATURE_SCHEMA_VERSION: Final[str] = "1.0.0"
PREDICTION_SCHEMA_VERSION: Final[str] = "1.0.0"
POSTERIOR_SCHEMA_VERSION: Final[str] = "1.0.0"

EPISTEMIC_PREDICTION: Final[str] = "prediction"
EPISTEMIC_INFERENCE: Final[str] = "inference"

#: The only epistemic statuses an ML artifact may carry. ``observation`` is
#: deliberately absent: observed records are produced by Stages 1-5, never here.
ADMISSIBLE_EPISTEMIC: Final[frozenset[str]] = frozenset(
    {EPISTEMIC_PREDICTION, EPISTEMIC_INFERENCE}
)

LINK_PREDICTION_TASK: Final[str] = "link_prediction"

_EPS: Final[float] = 1e-9


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: float, message: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{message} (got {value!r})")
    return number


@dataclass(frozen=True)
class Provenance:
    """Who produced an artifact, from what, and against which schema version."""

    schema_version: str
    model_name: str
    model_version: str
    run_id: str
    input_snapshot: str
    supporting_record_ids: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in ("schema_version", "model_name", "model_version", "run_id"):
            _require(
                bool(getattr(self, name)),
                f"Provenance.{name} must be non-empty (Roadmap 0.2 requires "
                "every prediction to identify model/version/run)",
            )
        _require(
            bool(self.input_snapshot),
            "Provenance.input_snapshot must be non-empty: a prediction that "
            "cannot name the graph snapshot it was computed from cannot be "
            "reproduced or checked for leakage (Roadmap 0.2, R1.2)",
        )
        object.__setattr__(
            self, "supporting_record_ids", tuple(self.supporting_record_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Uncertainty:
    """A measured uncertainty, or an explicit statement that none exists.

    ``method == "unavailable"`` forces ``value is None`` so that missing
    calibration is never serialized as a numeric zero.
    """

    method: str
    value: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.method), "Uncertainty.method must be non-empty")
        if self.method == "unavailable":
            _require(
                self.value is None,
                "Uncertainty with method='unavailable' must have value=None; "
                "a number here would be read downstream as a measurement",
            )
            return
        _require(
            self.value is not None,
            f"Uncertainty.method={self.method!r} requires a measured value",
        )
        value = _finite(self.value, "Uncertainty.value must be finite")
        _require(
            0.0 <= value <= 1.0,
            f"Uncertainty.value must lie in [0, 1] (got {value})",
        )

    @property
    def is_measured(self) -> bool:
        return self.method != "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Shared sentinel for "we have not calibrated this yet". Identity comparison
#: is intentional so callers cannot accidentally mint their own.
UNCALIBRATED: Final[Uncertainty] = Uncertainty(method="unavailable")


@dataclass(frozen=True)
class FeatureVector:
    """Features for one subject, computed against a named graph snapshot.

    ``snapshot_hash`` is what makes a feature dataset reproducible and what
    makes leakage auditable: R1.2 requires features be built "from a fixed
    graph snapshot", and this field is the proof that they were.
    """

    schema_version: str
    snapshot_hash: str
    subject: tuple[str, ...]
    values: tuple[tuple[str, float], ...]
    source_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(
            self.schema_version == FEATURE_SCHEMA_VERSION,
            f"FeatureVector.schema_version must be {FEATURE_SCHEMA_VERSION} "
            f"(got {self.schema_version!r}); consumers read this schema",
        )
        _require(
            bool(self.snapshot_hash),
            "FeatureVector.snapshot_hash must be non-empty (R1.2 fixed snapshot)",
        )
        object.__setattr__(self, "subject", tuple(self.subject))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(
            self, "source_record_ids", tuple(self.source_record_ids)
        )
        _require(len(self.subject) >= 1, "FeatureVector.subject must not be empty")
        names = [name for name, _ in self.values]
        _require(
            len(names) == len(set(names)),
            f"FeatureVector values must have unique names (duplicates: "
            f"{sorted({n for n in names if names.count(n) > 1})})",
        )
        for name, value in self.values:
            _require(bool(name), "FeatureVector value names must be non-empty")
            _finite(value, f"FeatureVector value {name!r} must be finite")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.values}


@dataclass(frozen=True)
class LinkPrediction:
    """A scored candidate edge. Never evidence, never an observation."""

    provenance: Provenance
    source_id: str
    target_id: str
    score: float
    task: str = LINK_PREDICTION_TASK
    uncertainty: Uncertainty = field(default_factory=lambda: UNCALIBRATED)
    calibration_status: str = "not_measured"
    shadow_mode: bool = True
    epistemic_status: str = EPISTEMIC_PREDICTION
    as_evidence: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.source_id) and bool(self.target_id),
                 "LinkPrediction endpoints must be non-empty entity ids")
        _require(self.source_id != self.target_id,
                 "LinkPrediction refuses self-loops: a node cannot be "
                 "predicted to link to itself")
        score = _finite(self.score, "LinkPrediction.score must be finite")
        _require(
            0.0 <= score <= 1.0,
            f"LinkPrediction.score must lie in [0, 1] (got {score})",
        )
        _require(
            self.epistemic_status in ADMISSIBLE_EPISTEMIC,
            f"epistemic_status must be one of {sorted(ADMISSIBLE_EPISTEMIC)}; "
            f"{self.epistemic_status!r} would let an ML output masquerade as "
            "an observation (Roadmap 0 exit gate)",
        )
        _require(
            self.as_evidence is False,
            "LinkPrediction.as_evidence is permanently False: an ML result "
            "must never be usable as evidence (Roadmap 0 exit gate)",
        )
        _require(
            self.task == LINK_PREDICTION_TASK,
            f"Unsupported task {self.task!r}; R1 scopes the first model to "
            f"{LINK_PREDICTION_TASK!r} only",
        )
        _require(
            self.calibration_status in {"measured", "not_measured"},
            f"calibration_status must be 'measured' or 'not_measured' "
            f"(got {self.calibration_status!r})",
        )
        if self.calibration_status == "measured":
            _require(
                self.uncertainty.is_measured,
                "calibration_status='measured' requires a measured "
                "Uncertainty; claiming calibration without one is a false "
                "measurement",
            )

    @property
    def key(self) -> tuple[str, str]:
        """Order-independent endpoint pair, used for ranking and joins."""
        return tuple(sorted((self.source_id, self.target_id)))  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = list(self.key)
        return payload


@dataclass(frozen=True)
class BNPosterior:
    """A posterior distribution from Bayesian inference. Shadow-mode only.

    ``assumptions`` and ``parameter_provenance`` are mandatory because
    Roadmap 2's exit gate requires the output to expose its assumptions and
    because R2 forbids inventing priors or CPTs: an unattributed parameter is
    by definition an invented one.
    """

    provenance: Provenance
    query_variable: str
    distribution: tuple[tuple[str, float], ...]
    evidence: tuple[tuple[str, str], ...] = ()
    assumptions: tuple[str, ...] = ()
    parameter_provenance: tuple[str, ...] = ()
    sensitivity: tuple[tuple[str, float], ...] = ()
    uncertainty: Uncertainty = field(default_factory=lambda: UNCALIBRATED)
    shadow_mode: bool = True
    epistemic_status: str = EPISTEMIC_INFERENCE
    as_evidence: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.query_variable), "BNPosterior.query_variable required")
        object.__setattr__(self, "distribution", tuple(self.distribution))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(
            self, "parameter_provenance", tuple(self.parameter_provenance)
        )
        object.__setattr__(self, "sensitivity", tuple(self.sensitivity))

        _require(len(self.distribution) >= 1, "distribution must not be empty")
        seen: set[str] = set()
        total = 0.0
        for state, prob in self.distribution:
            _require(bool(state), "distribution states must be non-empty")
            _require(state not in seen, f"duplicate state {state!r}")
            seen.add(state)
            value = _finite(prob, f"distribution[{state!r}] must be finite")
            _require(
                0.0 <= value <= 1.0,
                f"distribution[{state!r}] must lie in [0, 1] (got {value})",
            )
            total += value
        _require(
            abs(total - 1.0) <= 1e-6,
            f"distribution must sum to 1.0 (got {total}); an unnormalized "
            "posterior is not a posterior",
        )

        _require(
            len(self.assumptions) >= 1,
            "BNPosterior.assumptions must be non-empty: Roadmap 2's exit gate "
            "requires the output to carry its assumptions",
        )
        _require(
            all(a.strip() for a in self.assumptions),
            "assumptions must be non-blank strings",
        )
        _require(
            len(self.parameter_provenance) >= 1,
            "BNPosterior.parameter_provenance must be non-empty: R2 forbids "
            "invented priors/CPTs, so every parameter needs a source",
        )
        _require(
            all(p.strip() for p in self.parameter_provenance),
            "parameter_provenance entries must be non-blank",
        )
        _require(
            self.epistemic_status in ADMISSIBLE_EPISTEMIC,
            f"epistemic_status must be one of {sorted(ADMISSIBLE_EPISTEMIC)}",
        )
        _require(
            self.as_evidence is False,
            "BNPosterior.as_evidence is permanently False: a posterior is an "
            "inference over hypotheses, never observed evidence",
        )

    @property
    def probability_of(self) -> dict[str, float]:
        return {state: float(prob) for state, prob in self.distribution}

    @property
    def most_probable(self) -> str:
        return max(self.distribution, key=lambda item: item[1])[0]

    @property
    def entropy(self) -> float:
        """Shannon entropy in bits over the posterior. Measured, not banded."""
        total = 0.0
        for _, prob in self.distribution:
            if prob > 0.0:
                total -= prob * math.log2(prob)
        return total

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entropy_bits"] = self.entropy
        payload["most_probable"] = self.most_probable
        return payload


def check_versions(payload: Mapping[str, Any], expected: str, what: str) -> None:
    """Assert a deserialized artifact still matches the schema it claims."""
    actual = payload.get("schema_version")
    _require(
        actual == expected,
        f"{what} declares schema_version={actual!r} but this build reads "
        f"{expected!r}; re-run the producing stage instead of guessing",
    )


def sorted_distribution(
    pairs: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    """Canonicalize a distribution: sorted by state, probabilities rounded.

    Rounding is applied only in the serialization path so that identical
    inputs produce byte-identical artifacts across runs.
    """
    return tuple(
        (state, round(float(prob), 12))
        for state, prob in sorted(pairs, key=lambda item: item[0])
    )
