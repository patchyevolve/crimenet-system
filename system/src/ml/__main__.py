"""Run the ML roadmap's current milestone and persist its reviewable artifacts.

    python -m src.ml --output output_geo --run-id <run_id>

Produces, inside ``<output>/ml/``:

  * ``label_inventory.json`` — the Roadmap 0.3 / R1.1 label audit.
  * ``baseline_evaluation.json`` — the R1.3 baseline report with its gate
    verdict, bootstrap intervals and subgroup breakdown.
  * ``ml_predictions.json`` — the baseline's scores as reviewable
    :class:`~ml.contracts.LinkPrediction` records (R1.5).

``bn_posteriors.json`` is intentionally **not** written. There is no domain
Bayesian network: Roadmap 2.2 forbids inventing priors or CPTs, and none has
been reviewed. The engine's correctness is reported instead, under
``bn_engine_validation``, so the absence of a domain model is visible rather
than implied by a missing file.

This entry point is offline and does not run inside ``run.py``. Roadmap 1's
exit gate requires *no production pipeline behaviour changes without a
separate reviewed integration step*.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import POSTERIOR_SCHEMA_VERSION
from .evaluation import evaluate_baseline
from .fixtures import analytic_sprinkler_posteriors, sprinkler_network
from .labels import inventory_labels
from .bayes import parameter_sensitivity, validate_against_sampling
from .store import ARTIFACTS, ml_directory, write_artifact

__all__ = ["build_report", "main"]


def bn_engine_report() -> dict[str, Any]:
    """Validate the inference engine against a known distribution (R2.3)."""
    network = sprinkler_network()
    analytic = analytic_sprinkler_posteriors()
    observed = network.posterior("Rain", {"Wet": "True"})["True"]
    expected = analytic["posterior_rain_true_given_wet"]
    validation = validate_against_sampling(
        network,
        "Rain",
        evidence={"Wet": "True"},
        draws=40000,
        seed=11,
    )
    return {
        "schema_version": POSTERIOR_SCHEMA_VERSION,
        "engine": "variable elimination (pure python, no pgmpy)",
        "network": network.name,
        "model_version": network.model_version,
        "domain_model_built": False,
        "domain_model_blocker": (
            "Roadmap 2.2: priors and CPTs require subject-matter review. "
            "None has been done, so no domain BN exists and no posterior "
            "over this system's hypotheses is produced."
        ),
        "assumptions": list(network.assumptions),
        "parameter_provenance": list(network.parameter_provenance),
        "analytic_reference": {k: round(v, 12) for k, v in analytic.items()},
        "engine_result": {"rain_true_given_wet": round(observed, 12)},
        "analytic_agreement": abs(observed - expected) <= 1e-12,
        "sampling_cross_check": validation,
        "parameter_sensitivity": parameter_sensitivity(
            network, "Rain", evidence={"Wet": "True"}
        ),
    }


def build_report(output_dir: str | Path, *, run_id: str) -> dict[str, Any]:
    """Build every artifact for one run without writing anything."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "label_inventory": inventory_labels(output_dir).to_dict(),
        "baseline_evaluation": evaluate_baseline(output_dir, run_id=run_id),
        "bn_engine_validation": bn_engine_report(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.ml",
        description="Run the ML roadmap milestone and write its artifacts.",
    )
    parser.add_argument("--output", required=True, help="pipeline output directory")
    parser.add_argument("--run-id", default="offline", help="owning run id")
    args = parser.parse_args(argv)

    root = Path(args.output)
    if not (root / "graph_edges.json").exists():
        print(f"error: {root} has no graph_edges.json", file=sys.stderr)
        return 2

    evaluation = evaluate_baseline(root, run_id=args.run_id)
    write_artifact(root, "label_inventory.json", evaluation["label_inventory"])
    write_artifact(
        root,
        "baseline_evaluation.json",
        {k: v for k, v in evaluation.items() if k != "predictions"},
    )
    write_artifact(root, "ml_predictions.json", evaluation.get("predictions", []))

    engine = bn_engine_report()
    print(f"ML artifacts written to {ml_directory(root)}")
    for name in ARTIFACTS:
        state = "present" if (ml_directory(root) / name).exists() else "absent"
        print(f"  {state:7} {name}")
    print("  (bn_posteriors.json stays absent: no reviewed domain CPTs exist)")

    print("\nlabel inventory:")
    counts = evaluation["label_inventory"]["counts"]
    print(
        f"  positives={counts['positives']} negatives={counts['negatives']} "
        f"timestamped={counts['timestamped_positives']} "
        f"train={counts['train']} test={counts['test']}"
    )
    print("\nbaseline gate:")
    print(f"  computable: {evaluation['gate']['computable']}")
    print(f"  acceptance: {evaluation['gate']['acceptance']}")
    for finding in evaluation["gate"]["decisive_findings"]:
        print(f"  - {finding['code']}: {finding['detail']}")
    print("\nBN engine:")
    print(f"  analytic agreement: {engine['analytic_agreement']}")
    print(f"  sampling cross-check passed: {engine['sampling_cross_check']['passed']}")
    print(f"  domain model built: {engine['domain_model_built']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
