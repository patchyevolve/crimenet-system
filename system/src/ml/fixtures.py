"""Synthetic networks with known distributions.

Roadmap 2.3 requires *"synthetic unit cases with known distributions to
validate inference and dependency handling"* before any domain model exists.
These fixtures supply that: they are the classic Rain/Sprinkler/Wet structure,
whose posterior can be computed by hand, so :func:`validate_against_sampling`
and the unit tests can check inference against a value that came from
arithmetic rather than from the code under test.

Everything here is explicitly labelled synthetic. No fixture is ever presented
as a domain model, and none of them ships as a posterior — see
:mod:`ml.store`, which refuses to write anything outside its own directory.
"""

from __future__ import annotations

from typing import Final

from .bayes import BayesianNetwork, CPT

__all__ = [
    "SPRINKLER_ASSUMPTIONS",
    "SPRINKLER_PROVENANCE",
    "analytic_sprinkler_posteriors",
    "sprinkler_network",
]

SPRINKLER_PROVENANCE: Final[str] = (
    "synthetic unit-test fixture; probabilities chosen by hand to be "
    "non-degenerate, not estimated from data"
)

SPRINKLER_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "synthetic fixture: not a domain model and not fitted to any case",
    "all variables are binary",
    "Wet is conditionally independent of Cloudy given Rain and Sprinkler",
    "no parameter here is derived from the demo corpus or from Stage 7 factors",
)


def sprinkler_network() -> BayesianNetwork:
    """Rain -> Cloudy, Rain -> Sprinkler, {Rain, Sprinkler} -> Wet."""
    nodes = {
        "Cloudy": ("True", "False"),
        "Rain": ("True", "False"),
        "Sprinkler": ("True", "False"),
        "Wet": ("True", "False"),
    }
    cpts = [
        CPT(
            "Cloudy",
            (),
            ("True", "False"),
            (((), (0.5, 0.5)),),
            SPRINKLER_PROVENANCE,
        ),
        CPT(
            "Rain",
            ("Cloudy",),
            ("True", "False"),
            ((("True",), (0.10, 0.90)), (("False",), (0.80, 0.20))),
            SPRINKLER_PROVENANCE,
        ),
        CPT(
            "Sprinkler",
            ("Rain",),
            ("True", "False"),
            ((("True",), (0.01, 0.99)), (("False",), (0.40, 0.60))),
            SPRINKLER_PROVENANCE,
        ),
        CPT(
            "Wet",
            ("Rain", "Sprinkler"),
            ("True", "False"),
            (
                (("True", "True"), (0.99, 0.01)),
                (("True", "False"), (0.90, 0.10)),
                (("False", "True"), (0.90, 0.10)),
                (("False", "False"), (0.00, 1.00)),
            ),
            SPRINKLER_PROVENANCE,
        ),
    ]
    return BayesianNetwork(
        nodes,
        cpts,
        name="sprinkler_synthetic_fixture",
        model_version="1.0.0",
        assumptions=SPRINKLER_ASSUMPTIONS,
    )


def analytic_sprinkler_posteriors() -> dict[str, float]:
    """Closed-form posteriors computed from the CPTs by arithmetic.

    Kept as an independent reference: the unit tests assert the engine's
    output against these numbers, so a bug in variable elimination cannot hide
    behind agreement with itself.
    """
    # P(Rain=True) = 0.5*0.10 + 0.5*0.80 = 0.45
    prior_rain_true = 0.45
    # P(Wet=True | Rain=True)  = 0.01*0.99 + 0.99*0.90
    # P(Wet=True | Rain=False) = 0.40*0.90 + 0.60*0.00
    wet_given_rain = 0.01 * 0.99 + 0.99 * 0.90
    wet_given_no_rain = 0.40 * 0.90 + 0.60 * 0.00
    numerator = prior_rain_true * wet_given_rain
    denominator = numerator + (1.0 - prior_rain_true) * wet_given_no_rain
    return {
        "prior_rain_true": prior_rain_true,
        "posterior_rain_true_given_wet": numerator / denominator,
        "p_wet_true": denominator,
    }
