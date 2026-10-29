"""Evaluation metrics — implemented here because the dependencies are absent.

scikit-learn and scipy are not installed in this environment (see
:mod:`ml.deps`), and Roadmap 0.4 requires the pipeline to keep running
without optional ML libraries. AUPRC and calibration are therefore computed
directly.

R1.3 is explicit that *"AUC alone is not an acceptance criterion"* and demands
**AUPRC, calibration, coverage, and performance by subgroup**. Each of those is
a separate function below; reporting only :func:`average_precision` would
satisfy the letter of a naive reading and violate its intent, because a model
can score well on ranking while being systematically overconfident or blind on
one subgroup.

Definitions used, stated so the numbers are auditable:

* :func:`average_precision` — area under the precision-recall curve, computed
  as the weighted sum of precision at each positive, which is the standard
  ``AP`` definition. No-skill reference is the positive prevalence, not 0.5.
* :func:`expected_calibration_error` — mean gap between confidence and
  accuracy over equal-width bins, weighted by bin occupancy. Bins with no
  members contribute nothing rather than being invented as ``0.0``.
* :func:`bootstrap_interval` — percentile interval over resampled scores.
  Pure Python so it does not require scipy; seeded so a report is reproducible.
"""

from __future__ import annotations

import random
from typing import Sequence

__all__ = [
    "auc_roc",
    "average_precision",
    "bootstrap_interval",
    "bootstrap_metric",
    "calibration_bins",
    "coverage_at_k",
    "equal_width_bins",
    "expected_calibration_error",
    "no_skill_reference",
    "precision_at_k",
    "score_classification_report",
    "subgroup_reports",
]

DEFAULT_CALIBRATION_BINS: int = 5
DEFAULT_BOOTSTRAP_SAMPLES: int = 1000
DEFAULT_SEED: int = 20260925


def _check_inputs(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError(
            f"labels ({len(labels)}) and scores ({len(scores)}) must align"
        )
    if not labels:
        raise ValueError("cannot evaluate an empty set")


def no_skill_reference(labels: Sequence[int]) -> float:
    """Positive prevalence — what a random ranking achieves on AUPRC."""
    _check_inputs(labels, [0.0] * len(labels))
    return sum(labels) / len(labels)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (average precision).

    Returns ``0.0`` when there are no positives, which is the correct value:
    with nothing to find, nothing found is perfect only in the vacuous sense
    and reporting 1.0 would be misleading.
    """
    _check_inputs(labels, scores)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    order = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )
    retrieved = 0
    hits = 0
    total = 0.0
    previous_recall = 0.0
    for index in order:
        retrieved += 1
        if labels[index]:
            hits += 1
            precision = hits / retrieved
            recall = hits / positives
            total += precision * (recall - previous_recall)
            previous_recall = recall
    return total


def auc_roc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic.

    Ties receive the average of their rank span, which is what makes this
    equal to the trapezoidal ROC area rather than an optimistic variant.
    """
    _check_inputs(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position]] = average_rank
        cursor = end
    positive_rank_sum = sum(
        rank for rank, index in zip(ranks, range(len(labels))) if labels[index]
    )
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def precision_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Precision among the top-``k`` ranked items."""
    _check_inputs(labels, scores)
    if k <= 0:
        raise ValueError("k must be positive")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    top = order[: min(k, len(order))]
    if not top:
        return 0.0
    return sum(labels[index] for index in top) / len(top)


def coverage_at_k(
    labels: Sequence[int], scores: Sequence[float], k: int
) -> float:
    """Fraction of all positives recovered within the top-``k``."""
    _check_inputs(labels, scores)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    return precision_at_k(labels, scores, k) * min(k, len(labels)) / positives


def equal_width_bins(
    scores: Sequence[float], bins: int = DEFAULT_CALIBRATION_BINS
) -> list[list[int]]:
    """Assign indices to equal-width bins over ``[0, 1]``.

    Scores outside ``[0, 1]`` raise rather than being clipped, because a
    out-of-range confidence indicates a broken scorer, not an edge case.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    grouped: list[list[int]] = [[] for _ in range(bins)]
    for index, score in enumerate(scores):
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score {score} outside [0, 1]")
        position = min(int(score * bins), bins - 1)
        grouped[position].append(index)
    return grouped


def calibration_bins(
    labels: Sequence[int],
    scores: Sequence[float],
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> list[dict[str, float | int]]:
    """Per-bin confidence, accuracy and occupancy. Empty bins are omitted."""
    _check_inputs(labels, scores)
    grouped = equal_width_bins(scores, bins)
    width = 1.0 / bins
    report: list[dict[str, float | int]] = []
    for position, indices in enumerate(grouped):
        if not indices:
            continue
        mean_confidence = sum(scores[i] for i in indices) / len(indices)
        accuracy = sum(labels[i] for i in indices) / len(indices)
        report.append(
            {
                "bin": position,
                "lower": round(position * width, 6),
                "upper": round((position + 1) * width, 6),
                "count": len(indices),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": abs(mean_confidence - accuracy),
            }
        )
    return report


def expected_calibration_error(
    labels: Sequence[int],
    scores: Sequence[float],
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """Occupancy-weighted mean |confidence - accuracy| over populated bins."""
    report = calibration_bins(labels, scores, bins)
    total = len(labels)
    if not report or total == 0:
        return 0.0
    return sum(entry["gap"] * entry["count"] for entry in report) / total  # type: ignore[operator]


def bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric computed over ``values``.

    The caller passes already-computed per-resample metric values. The seed is
    fixed so two runs of the same report produce the same interval; an
    interval that moved between runs could not be compared against anything.
    """
    if not values:
        raise ValueError("bootstrap needs at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    lower_index = int(tail * len(ordered))
    upper_index = int((1.0 - tail) * len(ordered))
    upper_index = min(upper_index, len(ordered) - 1)
    return ordered[lower_index], ordered[upper_index]


def bootstrap_metric(
    labels: Sequence[int],
    scores: Sequence[float],
    metric,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Resample ``(labels, scores)`` pairs and return a percentile interval.

    Resampling is over *pairs* so labels stay attached to their scores;
    resampling them independently would break the very association the metric
    measures.
    """
    _check_inputs(labels, scores)
    if len(labels) < 2:
        raise ValueError("bootstrap needs at least two observations")
    generator = random.Random(seed)
    size = len(labels)
    draws: list[float] = []
    for _ in range(samples):
        indices = [generator.randrange(size) for _ in range(size)]
        sampled_labels = [labels[i] for i in indices]
        sampled_scores = [scores[i] for i in indices]
        if sum(sampled_labels) == 0:
            continue
        draws.append(float(metric(sampled_labels, sampled_scores)))
    if not draws:
        return 0.0, 0.0
    return bootstrap_interval(draws)


def score_classification_report(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    """The full R1.3 report: AUPRC, ROC, calibration and fixed-threshold counts."""
    _check_inputs(labels, scores)
    predicted = [1 if score >= threshold else 0 for score in scores]
    true_positives = sum(
        1 for label, guess in zip(labels, predicted) if label and guess
    )
    false_positives = sum(
        1 for label, guess in zip(labels, predicted) if not label and guess
    )
    false_negatives = sum(
        1 for label, guess in zip(labels, predicted) if label and not guess
    )
    positives = sum(labels)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if positives
        else 0.0
    )
    return {
        "n": len(labels),
        "positives": positives,
        "negatives": len(labels) - positives,
        "prevalence": positives / len(labels),
        "average_precision": average_precision(labels, scores),
        "auc_roc": auc_roc(labels, scores),
        "no_skill_ap": positives / len(labels),
        "expected_calibration_error": expected_calibration_error(
            labels, scores, bins
        ),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "calibration": calibration_bins(labels, scores, bins),
    }


def subgroup_reports(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    *,
    minimum_size: int = 1,
) -> dict[str, dict[str, float | int]]:
    """Break the report down by a declared subgroup.

    ``minimum_size`` exists so a two-member group does not get reported with
    the same standing as a fifty-member one; groups below it are still listed,
    but their metrics carry no confidence and are flagged.
    """
    _check_inputs(labels, scores)
    if len(groups) != len(labels):
        raise ValueError("groups must align with labels")
    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        buckets.setdefault(group, []).append(index)
    report: dict[str, dict[str, float | int]] = {}
    for group in sorted(buckets):
        indices = buckets[group]
        group_labels = [labels[i] for i in indices]
        group_scores = [scores[i] for i in indices]
        positives = sum(group_labels)
        report[group] = {
            "n": len(indices),
            "positives": positives,
            "prevalence": positives / len(indices),
            "average_precision": average_precision(group_labels, group_scores),
            "auc_roc": auc_roc(group_labels, group_scores),
            "reliable": len(indices) >= minimum_size and positives > 0,
        }
    return report
