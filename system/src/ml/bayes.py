"""Exact Bayesian network inference, plus the validation that makes it trustworthy.

ML_ENGINE.md Roadmap 2 is explicit about *why* this stage is delicate::

    *"Stage 7 currently emits factors, not a calibrated posterior. A Bayesian
    Network cannot be responsibly added by turning those factors into
    probabilities or inventing priors/CPTs."*

    *"Create synthetic unit cases with known distributions to validate
    inference and dependency handling, then evaluate calibration and
    sensitivity on appropriately labeled, held-out cases."*

So this module does the half that is possible without domain review: it builds
and validates an **inference engine**, and it refuses to build a **domain
model**. :class:`CPT` requires a non-empty ``provenance`` string, so a table
whose numbers have no stated source cannot be constructed — the structural
expression of "do not invent priors or CPTs". Nothing here accepts Stage 7's
factors as probabilities, and no API converts a factor score into a
probability.

Why pure Python instead of pgmpy
-------------------------------

``pgmpy`` is not installed and Roadmap 0.4 requires the pipeline to run
without it. Variable elimination over small discrete networks is exact and
straightforward to implement, and owning it lets validation cross-check two
*genuinely independent* algorithms:

  * :func:`exact_posterior` — variable elimination over the joint.
  * :func:`ancestral_sample` — forward Monte-Carlo sampling with rejection.

If those disagree beyond Monte-Carlo tolerance, inference is broken. A library
returning one uncheckable answer would not tell us that.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SENSITIVITY_EPSILON",
    "BayesianNetwork",
    "CPT",
    "ancestral_sample",
    "exact_posterior",
    "parameter_sensitivity",
    "validate_against_sampling",
]

#: Declared perturbation used by :func:`parameter_sensitivity`. Reported
#: verbatim alongside every sensitivity figure; no pass/fail decision depends
#: on its value, so it is an analysis parameter rather than a tuned threshold.
SENSITIVITY_EPSILON: float = 0.05

_TOLERANCE: float = 1e-9


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(values.values())
    _require(total > 0.0, "distribution has zero mass")
    return {state: value / total for state, value in values.items()}


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    states = set(left) | set(right)
    return 0.5 * sum(abs(left.get(s, 0.0) - right.get(s, 0.0)) for s in states)


@dataclass(frozen=True)
class CPT:
    """A conditional probability table with mandatory parameter provenance.

    ``provenance`` is required by Roadmap 2.2 (*"Record parameter provenance
    and uncertainty. Do not learn or tune parameters on evaluation cases."*).
    It must say where the numbers came from — a reviewed domain prior, a
    published base rate, or an explicitly labelled synthetic construction.
    An unattributed number is indistinguishable from an invented one.
    """

    variable: str
    parents: tuple[str, ...]
    states: tuple[str, ...]
    rows: tuple[tuple[tuple[str, ...], tuple[float, ...]], ...]
    provenance: str
    note: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.variable), "CPT.variable must be non-empty")
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(
            self,
            "rows",
            tuple((tuple(key), tuple(float(v) for v in values)) for key, values in self.rows),
        )
        _require(len(self.states) >= 1, f"CPT {self.variable}: no states declared")
        _require(
            len(set(self.states)) == len(self.states),
            f"CPT {self.variable}: duplicate states",
        )
        _require(
            self.variable not in self.parents,
            f"CPT {self.variable}: a variable cannot be its own parent",
        )
        if not self.parents:
            _require(
                len(self.rows) == 1 and self.rows[0][0] == (),
                f"CPT {self.variable}: a root node needs exactly one row keyed "
                "by the empty tuple",
            )
        _require(
            bool(self.provenance.strip()),
            f"CPT {self.variable}: provenance is required — Roadmap 2 forbids "
            "invented priors/CPTs, so every parameter needs a stated source",
        )
        seen: set[tuple[str, ...]] = set()
        for key, values in self.rows:
            _require(
                len(key) == len(self.parents),
                f"CPT {self.variable}: row key {key} does not match parents "
                f"{self.parents}",
            )
            _require(key not in seen, f"CPT {self.variable}: duplicate row {key}")
            seen.add(key)
            _require(
                len(values) == len(self.states),
                f"CPT {self.variable}: row {key} has {len(values)} values for "
                f"{len(self.states)} states",
            )
            total = 0.0
            for value in values:
                _require(
                    math.isfinite(value) and 0.0 <= value <= 1.0,
                    f"CPT {self.variable}: probabilities must lie in [0, 1]",
                )
                total += value
            _require(
                abs(total - 1.0) <= _TOLERANCE,
                f"CPT {self.variable}: row {key} sums to {total}, not 1.0",
            )

    def bind(self, parent_states: Mapping[str, tuple[str, ...]]) -> "CPT":
        """Check that rows cover every parent assignment exactly once."""
        if not self.parents:
            return self
        missing = [p for p in self.parents if p not in parent_states]
        _require(
            not missing,
            f"CPT {self.variable}: parent states unknown for {missing}",
        )
        wanted = {
            tuple(combo)
            for combo in itertools.product(
                *(parent_states[parent] for parent in self.parents)
            )
        }
        declared = {key for key, _ in self.rows}
        _require(
            declared == wanted,
            f"CPT {self.variable}: rows must cover every parent assignment; "
            f"missing {sorted(wanted - declared)}, "
            f"extra {sorted(declared - wanted)}",
        )
        return self

    def row_for(self, parent_values: Sequence[str]) -> tuple[float, ...]:
        """Probability row for one parent assignment, in declared state order."""
        lookup = {key: values for key, values in self.rows}
        key = tuple(parent_values)
        if key not in lookup:
            raise KeyError(f"CPT {self.variable}: no row for parent assignment {key}")
        return lookup[key]

    def as_distribution(
        self, parent_values: Sequence[str]
    ) -> dict[str, float]:
        return dict(zip(self.states, self.row_for(parent_values)))


@dataclass(frozen=True)
class _Factor:
    """A table over a scope of variables."""

    scope: tuple[str, ...]
    values: Mapping[tuple[str, ...], float]

    def restrict(self, evidence: Mapping[str, str]) -> "_Factor":
        """Condition on evidence: keep only matching rows, drop those columns."""
        present = [name for name in self.scope if name in evidence]
        if not present:
            return self
        pinned = {self.scope.index(name): evidence[name] for name in present}
        keep_index = [
            index for index, name in enumerate(self.scope) if name not in pinned
        ]
        keep = tuple(self.scope[index] for index in keep_index)
        new_values: dict[tuple[str, ...], float] = {}
        for key, value in self.values.items():
            if any(key[position] != state for position, state in pinned.items()):
                continue
            new_key = tuple(key[index] for index in keep_index)
            new_values[new_key] = new_values.get(new_key, 0.0) + value
        return _Factor(scope=keep, values=new_values)

    def multiply(self, other: "_Factor") -> "_Factor":
        """Pointwise product. Pairs that disagree on a shared variable
        contribute nothing: the joint is zero there, and adding them anyway
        would smear probability across inconsistent assignments."""
        combined = list(self.scope)
        for name in other.scope:
            if name not in combined:
                combined.append(name)
        scope = tuple(combined)
        shared = tuple(name for name in scope if name in self.scope and name in other.scope)
        left_index = {name: self.scope.index(name) for name in scope if name in self.scope}
        right_index = {name: other.scope.index(name) for name in scope if name in other.scope}
        new_values: dict[tuple[str, ...], float] = {}
        for left_key, left_value in self.values.items():
            for right_key, right_value in other.values.items():
                if any(
                    left_key[left_index[name]] != right_key[right_index[name]]
                    for name in shared
                ):
                    continue
                key = tuple(
                    left_key[left_index[name]]
                    if name in left_index
                    else right_key[right_index[name]]
                    for name in scope
                )
                new_values[key] = (
                    new_values.get(key, 0.0) + left_value * right_value
                )
        return _Factor(scope=scope, values=new_values)

    def marginalise(self, name: str) -> "_Factor":
        if name not in self.scope:
            return self
        position = self.scope.index(name)
        keep = self.scope[:position] + self.scope[position + 1 :]
        new_values: dict[tuple[str, ...], float] = {}
        for key, value in self.values.items():
            new_key = key[:position] + key[position + 1 :]
            new_values[new_key] = new_values.get(new_key, 0.0) + value
        return _Factor(scope=keep, values=new_values)

    def as_single_variable(self) -> dict[str, float]:
        if len(self.scope) != 1:
            raise ValueError(f"expected a single-variable factor, got {self.scope}")
        return _normalise({key[0]: value for key, value in self.values.items()})


class BayesianNetwork:
    """A validated discrete Bayesian network with attributed parameters."""

    def __init__(
        self,
        nodes: Mapping[str, Sequence[str]],
        cpts: Iterable[CPT],
        *,
        name: str,
        model_version: str,
        assumptions: Sequence[str],
    ) -> None:
        _require(bool(name), "BayesianNetwork.name must be non-empty")
        _require(
            bool(model_version),
            "BayesianNetwork.model_version must be non-empty (Roadmap 0.2)",
        )
        _require(
            len(assumptions) >= 1,
            "BayesianNetwork.assumptions must be non-empty: Roadmap 2's exit "
            "gate requires the output to carry its assumptions",
        )
        _require(
            all(str(a).strip() for a in assumptions),
            "assumptions must be non-blank",
        )
        self.name = name
        self.model_version = model_version
        self.assumptions = tuple(str(a) for a in assumptions)

        self._states: dict[str, tuple[str, ...]] = {}
        for variable, states in nodes.items():
            declared = tuple(states)
            _require(bool(declared), f"node {variable} declares no states")
            _require(
                len(set(declared)) == len(declared),
                f"node {variable} declares duplicate states",
            )
            self._states[variable] = declared

        bound: dict[str, CPT] = {}
        for cpt in cpts:
            _require(
                cpt.variable not in bound,
                f"node {cpt.variable} has more than one CPT",
            )
            _require(
                cpt.variable in self._states,
                f"CPT for {cpt.variable} has no declared node states",
            )
            _require(
                cpt.states == self._states[cpt.variable],
                f"CPT {cpt.variable} states {cpt.states} do not match node "
                f"states {self._states[cpt.variable]}",
            )
            for parent in cpt.parents:
                _require(
                    parent in self._states,
                    f"CPT {cpt.variable} references unknown parent {parent!r}",
                )
            bound[cpt.variable] = cpt.bind(self._states)

        self._cpts = bound
        self._order = self._topological_order()
        self._parameter_provenance = tuple(
            sorted({cpt.provenance for cpt in bound.values()})
        )

    def _topological_order(self) -> list[str]:
        pending = dict(self._cpts)
        ordered: list[str] = []
        resolved: set[str] = set()
        while pending:
            ready = sorted(
                variable
                for variable in pending
                if all(p in resolved for p in pending[variable].parents)
            )
            if not ready:
                raise ValueError(
                    "network is cyclic or references an unmodelled parent; "
                    f"unresolved: {sorted(pending)}"
                )
            for variable in ready:
                ordered.append(variable)
                resolved.add(variable)
                del pending[variable]
        orphan = [name for name in self._states if name not in resolved]
        _require(
            not orphan,
            f"nodes declared without a CPT: {sorted(orphan)}",
        )
        return ordered

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._order)

    @property
    def states(self) -> Mapping[str, tuple[str, ...]]:
        return dict(self._states)

    @property
    def parameter_provenance(self) -> tuple[str, ...]:
        return self._parameter_provenance

    def _factors(self) -> list[_Factor]:
        factors: list[_Factor] = []
        for variable in self._order:
            cpt = self._cpts[variable]
            if not cpt.parents:
                row = cpt.row_for(())
                # Keys are ALWAYS 1-tuples, never bare strings: a bare string
                # would be indexed element-wise by the multiplication below
                # ('True'[0] == 'T') and silently corrupt every product.
                factors.append(
                    _Factor(
                        scope=(variable,),
                        values={(state,): probability
                                for state, probability in zip(cpt.states, row)},
                    )
                )
                continue
            scope = tuple(cpt.parents) + (variable,)
            values: dict[tuple[str, ...], float] = {}
            for parent_values in itertools.product(
                *(self._states[parent] for parent in cpt.parents)
            ):
                row = cpt.row_for(parent_values)
                for state, probability in zip(cpt.states, row):
                    values[tuple(parent_values) + (state,)] = probability
            factors.append(_Factor(scope=scope, values=values))
        return factors

    def posterior(
        self, query: str, evidence: Mapping[str, str] | None = None
    ) -> dict[str, float]:
        """Exact ``P(query | evidence)`` by variable elimination."""
        evidence = dict(evidence or {})
        self._validate_query(query, evidence)

        factors = [
            factor.restrict(evidence)
            for factor in self._factors()
        ]
        factors = [factor for factor in factors if factor.scope]

        for name in (
            variable
            for variable in self._order
            if variable != query and variable not in evidence
        ):
            involved = [f for f in factors if name in f.scope]
            if not involved:
                continue
            product = involved[0]
            for factor in involved[1:]:
                product = product.multiply(factor)
            factors = [f for f in factors if name not in f.scope]
            factors.append(product.marginalise(name))

        remaining = [f for f in factors if query in f.scope]
        if not remaining:
            cpt = self._cpts[query]
            if cpt.parents:
                raise ValueError(
                    f"no factor survived elimination for query {query!r}"
                )
            return _normalise(dict(zip(cpt.states, cpt.row_for(()))))

        product = remaining[0]
        for factor in remaining[1:]:
            product = product.multiply(factor)
        for name in [n for n in product.scope if n != query]:
            product = product.marginalise(name)
        return product.as_single_variable()

    def prior(self, query: str) -> dict[str, float]:
        return self.posterior(query, {})

    def _validate_query(self, query: str, evidence: Mapping[str, str]) -> None:
        _require(
            query in self._states,
            f"query {query!r} is not a node of network {self.name!r}",
        )
        for name, value in evidence.items():
            _require(
                name in self._states,
                f"evidence variable {name!r} is not in network {self.name!r}",
            )
            _require(
                value in self._states[name],
                f"evidence {name}={value!r} is not a legal state; "
                f"expected one of {self._states[name]}",
            )
            _require(
                name != query,
                f"query {query!r} also appears in evidence; ask for a "
                "different variable or drop it from the evidence",
            )

    def sample_posterior(
        self,
        query: str,
        *,
        evidence: Mapping[str, str] | None = None,
        draws: int = 20000,
        seed: int = 1,
    ) -> tuple[dict[str, float], int, float]:
        """Monte-Carlo posterior via ancestral sampling with rejection.

        Returns ``(distribution, accepted, acceptance_rate)``. Rejection is
        used rather than Gibbs sampling because it introduces no burn-in or
        autocorrelation assumptions — the samples are independent by
        construction, which is what makes them a fair cross-check of the exact
        result.
        """
        _require(draws > 0, "draws must be positive")
        evidence = dict(evidence or {})
        self._validate_query(query, evidence)

        generator = random.Random(seed)
        counts = {state: 0 for state in self._states[query]}
        accepted = 0
        for _ in range(draws):
            # Sample EVERY variable from its CPT, evidence included, then keep
            # the draw only if it happens to match the evidence. Forcing the
            # evidence values instead would not be rejection sampling: an
            # upstream variable would be drawn from its prior and never learn
            # that the observation downstream contradicted it, so the result
            # would silently equal the prior.
            assignment: dict[str, str] = {}
            for variable in self._order:
                cpt = self._cpts[variable]
                parent_values = [assignment[p] for p in cpt.parents]
                row = dict(zip(cpt.states, cpt.row_for(parent_values)))
                assignment[variable] = _draw(generator, row)
            if all(
                assignment[name] == state for name, state in evidence.items()
            ):
                counts[assignment[query]] += 1
                accepted += 1
        if accepted == 0:
            return (
                {state: 0.0 for state in self._states[query]},
                0,
                0.0,
            )
        total = float(accepted)
        return (
            {state: count / total for state, count in counts.items()},
            accepted,
            accepted / draws,
        )


def _draw(generator: random.Random, weights: Mapping[str, float]) -> str:
    threshold = generator.random()
    running = 0.0
    last = ""
    for state, weight in weights.items():
        last = state
        running += weight
        if threshold <= running:
            return state
    return last


def exact_posterior(
    network: BayesianNetwork,
    query: str,
    evidence: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Module-level alias for :meth:`BayesianNetwork.posterior`."""
    return network.posterior(query, evidence)


def ancestral_sample(
    network: BayesianNetwork,
    query: str,
    *,
    evidence: Mapping[str, str] | None = None,
    draws: int = 20000,
    seed: int = 1,
) -> dict[str, float]:
    """Module-level alias for :meth:`BayesianNetwork.sample_posterior`."""
    distribution, _, _ = network.sample_posterior(
        query, evidence=evidence, draws=draws, seed=seed
    )
    return distribution


def validate_against_sampling(
    network: BayesianNetwork,
    query: str,
    *,
    evidence: Mapping[str, str] | None = None,
    draws: int = 40000,
    seed: int = 1,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Cross-check exact inference against Monte Carlo on the same network.

    This is the Roadmap 2.3 check — *"synthetic unit cases with known
    distributions to validate inference and dependency handling"* — applied to
    whatever network is handed in. A non-zero total variation distance above
    ``tolerance`` means at least one of the two algorithms is wrong, and the
    report says so rather than returning a number for the caller to interpret.
    """
    exact = network.posterior(query, evidence)
    sampled, accepted, rate = network.sample_posterior(
        query, evidence=evidence, draws=draws, seed=seed
    )
    distance = _total_variation(exact, sampled)
    passed = accepted > 0 and distance <= tolerance
    return {
        "network": network.name,
        "model_version": network.model_version,
        "query": query,
        "evidence": dict(evidence or {}),
        "exact": {k: round(v, 12) for k, v in sorted(exact.items())},
        "sampled": {k: round(v, 12) for k, v in sorted(sampled.items())},
        "total_variation_distance": round(distance, 8),
        "tolerance": tolerance,
        "draws": draws,
        "accepted": accepted,
        "acceptance_rate": round(rate, 6),
        "passed": passed,
        "parameter_provenance": list(network.parameter_provenance),
        "assumptions": list(network.assumptions),
    }


def _perturbed(rows: CPT, epsilon: float) -> CPT:
    """Move ``epsilon`` mass from the modal state to the least-probable one.

    Every row stays a valid distribution, so the perturbed network remains a
    well-formed model and the resulting shift isolates parameter sensitivity
    rather than measuring a broken input.
    """
    new_rows: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for key, values in rows.rows:
        if len(values) < 2:
            new_rows.append((key, values))
            continue
        ordered = sorted(range(len(values)), key=lambda i: values[i])
        lowest, highest = ordered[0], ordered[-1]
        if values[highest] - values[lowest] < epsilon:
            new_rows.append((key, values))
            continue
        adjusted = list(values)
        adjusted[highest] -= epsilon
        adjusted[lowest] += epsilon
        new_rows.append((key, tuple(adjusted)))
    return replace(rows, rows=tuple(new_rows))


def parameter_sensitivity(
    network: BayesianNetwork,
    query: str,
    *,
    evidence: Mapping[str, str] | None = None,
    epsilon: float = SENSITIVITY_EPSILON,
) -> list[dict[str, Any]]:
    """How much the posterior moves when each parameter table is perturbed.

    Reported as total-variation distance, a measured quantity with a natural
    ``[0, 1]`` range, so no band label has to be invented to describe it.
    """
    base = network.posterior(query, evidence)
    results: list[dict[str, Any]] = []
    for variable in network.variables:
        original = network._cpts[variable]
        perturbed = _perturbed(original, epsilon)
        if perturbed == original:
            results.append(
                {
                    "variable": variable,
                    "max_total_variation": 0.0,
                    "perturbed": False,
                    "epsilon": epsilon,
                    "provenance": original.provenance,
                }
            )
            continue
        rebuilt = BayesianNetwork(
            network.states,
            [
                perturbed if cpt.variable == variable else cpt
                for cpt in network._cpts.values()
            ],
            name=network.name,
            model_version=network.model_version,
            assumptions=network.assumptions,
        )
        moved = rebuilt.posterior(query, evidence)
        results.append(
            {
                "variable": variable,
                "max_total_variation": round(_total_variation(base, moved), 8),
                "perturbed": True,
                "epsilon": epsilon,
                "provenance": original.provenance,
            }
        )
    results.sort(key=lambda item: item["max_total_variation"], reverse=True)
    return results
