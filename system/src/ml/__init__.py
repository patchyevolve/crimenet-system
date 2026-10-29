"""The ML layer: contracts, baselines and Bayesian inference.

This package is the implementation of ML_ENGINE.md's roadmap, not of its
aspirational layer diagrams. What exists today:

  * :mod:`ml.contracts` — versioned, provenance-carrying artifact contracts
    (Roadmap 0.2). Artifacts that would let an ML result be mistaken for an
    observation cannot be constructed.
  * :mod:`ml.deps` — optional-dependency guard (Roadmap 0.4). Nothing here
    requires ``torch``, ``torch_geometric`` or ``pgmpy``.
  * :mod:`ml.labels` — the label inventory that gates every ML milestone
    (Roadmap 0.3, R1.1).
  * :mod:`ml.snapshot` / :mod:`ml.features` / :mod:`ml.baseline` —
    leakage-resistant features and the non-neural baseline R1.3 requires
    before any learned model.
  * :mod:`ml.metrics` — AUPRC, calibration, coverage and subgroups, written
    here because scikit-learn is not installed.
  * :mod:`ml.bayes` — exact Bayesian inference validated against Monte Carlo
    (Roadmap 2.3), with mandatory parameter provenance (Roadmap 2.2).
  * :mod:`ml.store` — the shadow-mode artifact directory (R1.5, R2.4).

Not implemented, and deliberately not stubbed: the GraphSAGE link predictor,
the domain Bayesian network, information gain, temporal models and adversarial
defence. Roadmap 1.4 permits a neural model only once labels justify it, and
Roadmap 2 forbids a domain BN until priors and CPTs have been reviewed. Their
absence is reported as absence rather than filled with placeholders.
"""

from __future__ import annotations

from .contracts import (
    BNPosterior,
    FEATURE_SCHEMA_VERSION,
    FeatureVector,
    POSTERIOR_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    LinkPrediction,
    Provenance,
    UNCALIBRATED,
    Uncertainty,
)
from .deps import availability

__all__ = [
    "BNPosterior",
    "FEATURE_SCHEMA_VERSION",
    "POSTERIOR_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "FeatureVector",
    "LinkPrediction",
    "Provenance",
    "UNCALIBRATED",
    "Uncertainty",
    "availability",
]
