"""Stage 9: Gap Detection."""

from .engine import (
    CANONICAL_FIELDS,
    GAP_KINDS,
    KIND_MISSING_RELATION,
    KIND_UNRESOLVED_ATTRIBUTE,
    attribute_data_source,
    build_gaps,
    generate,
    gap_id,
    load_inputs,
    relation_data_source,
    relation_impact,
)

__all__ = [
    "CANONICAL_FIELDS",
    "GAP_KINDS",
    "KIND_MISSING_RELATION",
    "KIND_UNRESOLVED_ATTRIBUTE",
    "attribute_data_source",
    "build_gaps",
    "generate",
    "gap_id",
    "load_inputs",
    "relation_data_source",
    "relation_impact",
]
