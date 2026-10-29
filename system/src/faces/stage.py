"""Stage 2.5 — Face Processing (RESEARCH_FACE_RECOGNITION/08_PIPELINE_INTEGRATION.md §7).

Pure data stage: the detection+embedding model pass already happened in
Stage 1/2 (one RetinaFace+ArcFace call per image — insightface does both in a
single forward pass, so the doc's split detection→embedding is collapsed to
keep one model pass per file). This stage:

  1. merges faces from previous runs (incremental) with this run's records
  2. finalizes statuses against the doc 08 §3.2 cutoffs
  3. links faces to person entities via doc 11 context (same-file extraction,
     filename tokens) — identity documents confirm outright (§6.1)
  4. pairwise-matches embeddings (doc 08 §4 thresholds; doc 07 §2 confidence)
  5. emits identity candidates (face → person) and person↔person candidates
     (fed to Stage 3 resolution + Stage 5 FACE_MATCH edges)
"""

from __future__ import annotations

from .context import (
    CONTEXT_CONFIDENCE_FLOOR,
    classify_image_input,
    filename_tokens,
    identity_type_from_filename,
    match_face_with_context,
    name_tokens,
)
from .engine import (
    CANDIDATE_SIMILARITY,
    DET_CUTOFF,
    MATCH_SIMILARITY,
    QUALITY_CUTOFF,
    cosine_similarity,
    similarity_to_confidence,
)

_FACEABLE = ("EXTRACTED", "MATCHED", "CONFIRMED")
_DEAD = ("REJECTED", "LOW_QUALITY")


def _context_of(face: dict) -> dict:
    """Doc 11 §5.3 boost inputs for a face's identity attribution."""
    context = face.get("context") or {}
    return {
        "id_type": face.get("identity_document") or identity_type_from_filename(face.get("file_name", "")),
        "source_type": context.get("source_type", ""),
        "investigator_named": face.get("link_basis") == "same_file_context",
        # Only an overlay/EXIF capture time corroborates identity (doc 11
        # §5.3 'timestamp match') — a file modified-time does not.
        "timestamp": face.get("capture_time_source") == "overlay",
    }


def _floor_of(face: dict) -> float:
    """Doc 11 §5.1 minimum confidence for a face whose context supports a
    claim (NO_CONTEXT still floors at the documented 0.50 base)."""
    context_type = (face.get("context") or {}).get("context_type", "NO_CONTEXT")
    return CONTEXT_CONFIDENCE_FLOOR.get(context_type, 0.0)


class FaceProcessingStage:
    """Stage 2.5 processor (doc 08 §7). Stateless; all records pass in/out."""

    def process(
        self,
        face_embeddings,
        existing_persons,
        previous_faces=None,
        file_contexts=None,
        all_files=None,
        file_texts=None,
    ) -> dict:
        file_contexts = file_contexts or {}
        persons = [p for p in (existing_persons or []) if isinstance(p, dict) and p.get("id")]

        # -- merge previous-run faces with this run's (incremental idempotence)
        combined: dict[str, dict] = {}
        for face in previous_faces or []:
            if isinstance(face, dict) and face.get("id"):
                combined[str(face["id"])] = dict(face)
        for face in face_embeddings or []:
            if isinstance(face, dict) and face.get("id"):
                combined[str(face["id"])] = dict(face)
        faces = list(combined.values())

        # -- context fill (doc 11) --------------------------------------------
        for face in faces:
            ctx = file_contexts.get(face.get("file_name", ""), {})
            if not face.get("camera"):
                face["camera"] = ctx.get("camera", "")
            if not face.get("captured_at"):
                overlay = ctx.get("overlay_timestamp", "")
                face["captured_at"] = overlay or ctx.get("captured_at", "")
                if overlay:
                    face["capture_time_source"] = "overlay"
                elif ctx.get("captured_at"):
                    face["capture_time_source"] = "file_mtime"
            if not face.get("context"):
                face["context"] = classify_image_input(
                    face.get("file_path", "") or face.get("file_name", ""),
                    all_files_in_folder=all_files,
                    ocr_text=ctx.get("ocr_text", ""),
                    file_texts=file_texts,
                )

        # -- finalize statuses (doc 08 §3.2) ----------------------------------
        for face in faces:
            if face.get("status") in _DEAD:
                continue
            det = float(face.get("detector_confidence") or 0.0)
            quality = float(face.get("quality_score") or 0.0)
            if det < DET_CUTOFF:
                face["status"], face["embedding_vector"] = "REJECTED", []
            elif quality < QUALITY_CUTOFF:
                face["status"], face["embedding_vector"] = "LOW_QUALITY", []
            elif face.get("status") in (None, "PENDING"):
                face["status"] = "EXTRACTED"

        # -- identity documents confirm outright (doc 11 §6.1) ----------------
        for face in faces:
            if face.get("status") in _DEAD:
                continue
            id_type = identity_type_from_filename(face.get("file_name", ""))
            if id_type:
                face["status"] = "CONFIRMED"
                face["identity_document"] = id_type

        # -- identity linkage (doc 11: same-file extraction, then filename) ----
        # Index persons by BOTH source_id and source_image: extraction merges
        # entities by deterministic id, so a filename-minted 'Amit' can survive
        # with source_id pointing at a text file while attributes.source_image
        # still names the photo the face came from.
        persons_by_file: dict[str, list[dict]] = {}
        for person in persons:
            keys = {
                str(person.get("source_id") or ""),
                str(person.get("source_image") or ""),
            } - {""}
            for key in keys:
                persons_by_file.setdefault(key, []).append(person)
        for face in faces:
            if face.get("person_id"):
                continue
            same_file = persons_by_file.get(face.get("file_name", ""), [])
            if len(same_file) == 1:
                person = same_file[0]
                face["person_id"] = person.get("id", "")
                face["person_name"] = person.get("name", "")
                # A person minted from THIS filename is filename context, not
                # investigator text in the same file (doc 11 §5.1 floors).
                face["link_basis"] = (
                    "filename_context"
                    if person.get("minted_from") == "filename"
                    else "same_file_context"
                )
                continue
            tokens = filename_tokens(face.get("file_name", ""))
            best, best_score, tied = None, 0, False
            for person in persons:
                score = len(tokens & name_tokens(person.get("name", "")))
                if score > best_score:
                    best, best_score, tied = person, score, False
                elif score == best_score and score > 0:
                    tied = True
            if best is not None and not tied:
                face["person_id"] = best.get("id", "")
                face["person_name"] = best.get("name", "")
                face["link_basis"] = "filename_context"

        # -- pairwise matching (doc 08 §4) ------------------------------------
        embeddable = [f for f in faces if f.get("status") in _FACEABLE and f.get("embedding_vector")]
        matches: list[dict] = []
        for i, face_a in enumerate(embeddable):
            for face_b in embeddable[i + 1:]:
                similarity = cosine_similarity(face_a["embedding_vector"], face_b["embedding_vector"])
                if similarity < CANDIDATE_SIMILARITY:
                    continue
                matches.append({
                    "face_a": face_a["id"],
                    "face_b": face_b["id"],
                    "file_a": face_a.get("file_name", ""),
                    "file_b": face_b.get("file_name", ""),
                    "similarity": round(similarity, 4),
                    "confidence": round(similarity_to_confidence(similarity), 4),
                })

        by_id = {f["id"]: f for f in faces}
        best_sim: dict[str, float] = {}
        for match in matches:
            for fid in (match["face_a"], match["face_b"]):
                if match["similarity"] > best_sim.get(fid, -1.0):
                    best_sim[fid] = match["similarity"]
        for face in faces:
            sim = best_sim.get(face.get("id"))
            face["best_similarity"] = round(sim, 4) if sim is not None else None

        # -- identity candidates (doc 08 §3.2 shape; candidate gate = engine's
        # CANDIDATE_SIMILARITY, domain-calibrated) —
        # An unlinked face that matches a linked face yields an identity
        # candidate for the investigator; MATCH/HIGH are merge-action tiers
        # (§4.3), not candidate gates.
        candidates: list[dict] = []
        identity_candidates = 0
        for face in faces:
            if face.get("person_id"):
                continue
            best = None
            for match in matches:
                if face["id"] not in (match["face_a"], match["face_b"]):
                    continue
                other_id = match["face_b"] if match["face_a"] == face["id"] else match["face_a"]
                other = by_id.get(other_id)
                if other and other.get("person_id") and (best is None or match["similarity"] > best[0]):
                    best = (match["similarity"], other)
            if best is None:
                continue
            similarity, partner = best
            # Corroboration comes from either image's context: the linked
            # partner's (ID, filename) and the query scene's (CCTV overlay
            # timestamp, camera). §5.1 floors are minimums — the strongest
            # context floor of the two images still applies.
            sim_conf = similarity_to_confidence(similarity)
            confidence = min(max(
                match_face_with_context(sim_conf, _context_of(partner)),
                match_face_with_context(sim_conf, _context_of(face)),
                _floor_of(partner),
                _floor_of(face),
            ), 1.0)
            identity_candidates += 1
            candidates.append({
                "source_entity_id": face["id"],
                "candidate_entity_id": partner.get("person_id", ""),
                "candidate_person_name": partner.get("person_name", ""),
                "match_type": "FACE",
                "kind": "identity",
                "confidence": round(confidence, 4),
                "signals": {
                    "face_similarity": round(similarity, 4),
                    "face_quality": face.get("quality_score", 0.0),
                    "link_basis": partner.get("link_basis", ""),
                    "context_type": (partner.get("context") or {}).get("context_type", ""),
                },
                "files": sorted({face.get("file_name", ""), partner.get("file_name", "")}),
            })

        # Person↔person aliases: faces linked to two DIFFERENT persons that
        # match at the review floor → Stage 3 candidates + Stage 5 FACE_MATCH.
        person_pairs: dict[tuple, tuple] = {}
        for match in matches:
            if match["similarity"] < MATCH_SIMILARITY:
                continue
            face_a, face_b = by_id.get(match["face_a"]), by_id.get(match["face_b"])
            if not face_a or not face_b:
                continue
            pa, pb = face_a.get("person_id"), face_b.get("person_id")
            if not pa or not pb or pa == pb:
                continue
            key = tuple(sorted((pa, pb)))
            if key not in person_pairs or match["similarity"] > person_pairs[key][0]:
                person_pairs[key] = (match["similarity"], match, face_a, face_b)
        cross_person = 0
        for (pa, pb), (similarity, match, face_a, face_b) in person_pairs.items():
            cross_person += 1
            sim_conf = similarity_to_confidence(similarity)
            confidence = min(max(
                match_face_with_context(sim_conf, _context_of(face_a)),
                match_face_with_context(sim_conf, _context_of(face_b)),
                _floor_of(face_a),
                _floor_of(face_b),
            ), 1.0)
            candidates.append({
                "source_entity_id": pa,
                "candidate_entity_id": pb,
                "match_type": "FACE",
                "kind": "person_alias",
                "confidence": round(confidence, 4),
                "signals": {
                    "face_similarity": round(similarity, 4),
                    "face_quality": face_a.get("quality_score", 0.0),
                    "context_type": (face_a.get("context") or {}).get("context_type", ""),
                },
                "files": sorted({face_a.get("file_name", ""), face_b.get("file_name", "")}),
            })

        # -- MATCHED when a person claim exists (doc 08 §3.2: "MATCHED if
        # candidates") — any match at the engine's candidate floor counts. -----
        identity_faces = {
            c["source_entity_id"] for c in candidates if c.get("kind") == "identity"
        }
        for face in faces:
            sim = face.get("best_similarity")
            if (
                face.get("status") == "EXTRACTED"
                and isinstance(sim, (int, float))
                and sim >= CANDIDATE_SIMILARITY
                and (face.get("person_id") or face.get("id") in identity_faces)
            ):
                face["status"] = "MATCHED"

        # -- per-face record confidence (doc 11 §5.1 floors) -------------------
        identity_by_face = {
            c["source_entity_id"]: c for c in candidates if c.get("kind") == "identity"
        }
        for face in faces:
            sim = face.get("best_similarity")
            match_conf = similarity_to_confidence(sim) if isinstance(sim, (int, float)) else 0.0
            context_conf = 0.0
            if face.get("person_id"):
                context_type = (face.get("context") or {}).get("context_type", "NO_CONTEXT")
                context_conf = CONTEXT_CONFIDENCE_FLOOR.get(context_type, 0.0)
            candidate_conf = float(identity_by_face.get(face.get("id"), {}).get("confidence") or 0.0)
            face["match_confidence"] = round(max(match_conf, context_conf, candidate_conf), 4)

        extracted = sum(1 for f in faces if f.get("status") in _FACEABLE)
        stats = {
            "faces": len(faces),
            "extracted": extracted,
            "low_quality": sum(1 for f in faces if f.get("status") == "LOW_QUALITY"),
            "rejected": sum(1 for f in faces if f.get("status") == "REJECTED"),
            "linked_faces": sum(1 for f in faces if f.get("person_id")),
            "matches": len(matches),
            "high_confidence_matches": sum(1 for m in matches if m["similarity"] >= 0.85),
            "candidates": len(candidates),
            "identity_candidates": identity_candidates,
            "cross_person_matches": cross_person,
        }
        return {"faces": faces, "matches": matches, "candidates": candidates, "stats": stats}
