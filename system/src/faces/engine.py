"""Face detection + embedding engine (insightface RetinaFace + ArcFace 512-d).

Failure-tolerant by contract: every public entry point returns an empty result
when the model stack is unavailable, so ingestion and extraction never fail on
a missing optional dependency. Weights live in ``<system>/.insightface`` which
the API container bind-mounts at ``/app/.insightface`` — host runs and container
runs share the same models.

Scoring follows RESEARCH_FACE_RECOGNITION/07_SCORING.md shape:
  * similarity_to_confidence — §2 breakpoint shape, CCTV-domain calibrated values
  * QUALITY_WEIGHTS — §3.1 factor weights (blur/resolution/angle/lighting/occlusion)
Cutoffs follow doc 08 §3.2 (quality < 0.3 rejected, no embedding stored).
"""

from __future__ import annotations

import math
import re
import threading
from datetime import datetime
from pathlib import Path

# Models are shared host↔container via the `./:/app` bind mount.
_MODEL_ROOT = Path(__file__).resolve().parents[2] / ".insightface"

_APP = None
_APP_LOCK = threading.Lock()
_APP_FAILED = False

# Doc 07 §3.1 — quality factor weights.
QUALITY_WEIGHTS = {
    "blur": 0.20,
    "resolution": 0.20,
    "angle": 0.25,
    "lighting": 0.15,
    "occlusion": 0.20,
}

# Match cutoffs — calibrated on the real evidence domain (buffalo_l embeddings
# over the case's mugshots + CCTV stills): genuine same-person pairs score
# 0.46–0.84 (mugshot↔CCTV across lighting/compression lands 0.46–0.53; clean
# photo↔photo 0.83), while impostor pairs all score ≤ 0.11. The doc-08 §4.2
# breakers (0.40/0.60/0.85) were tuned for studio-quality photos and pushed
# every genuine CCTV↔mugshot pair into the 'candidate' bucket (UI showed 0
# matches, ~0.10 confidence) — a false-negative disaster for surveillance
# evidence. Domain floors: candidate 0.30 (mid-gap of impostor≤0.11 vs
# genuine≥0.46), match 0.45 (just under the weakest genuine pair), high 0.75
# (strong photo-grade identity). Doc 07 §2 breakpoint SHAPE preserved; values
# domain-calibrated.
CANDIDATE_SIMILARITY = 0.30
MATCH_SIMILARITY = 0.45
HIGH_SIMILARITY = 0.75
# Doc 08 §3.2 — quality below 0.3 → no embedding record; detector below 0.5 → rejected.
QUALITY_CUTOFF = 0.30
DET_CUTOFF = 0.50


def similarity_to_confidence(similarity: float) -> float:
    """Cosine similarity → base confidence (doc 07 §2 shape, CCTV-domain values).

    Ramp: <0.30 → 0; 0.30–0.45 → 0–0.30; 0.45–0.75 → 0.30–0.85; ≥0.75 → 0.85+.
    A genuine CCTV↔mugshot pair (0.46–0.53) lands 0.32–0.43 base, reaching
    'medium' after context boosts; impostors (≤0.11) stay at 0.
    """
    if similarity < 0.30:
        return 0.0
    if similarity < 0.45:
        return (similarity - 0.30) * 2.0
    if similarity < 0.75:
        return 0.30 + (similarity - 0.45) * (0.55 / 0.30)
    return min(1.0, 0.85 + (similarity - 0.75) * 0.6)


def weighted_quality(scores: dict) -> float:
    """Weighted overall quality score (doc 07 §3.1)."""
    return sum(float(scores.get(key, 0.0)) * weight for key, weight in QUALITY_WEIGHTS.items())


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity of two embedding vectors; 0.0 on empty/zero input."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def model_available() -> bool:
    """True when insightface imports and weights exist — does NOT load the model."""
    import importlib.util

    if importlib.util.find_spec("insightface") is None:
        return False
    return (_MODEL_ROOT / "models" / "buffalo_l").is_dir()


def _patch_face_align_estimate() -> None:
    """Modernize insightface's ``estimate_norm`` for scikit-image >= 0.26.

    Upstream ``insightface.utils.face_align.estimate_norm`` resolves its
    transform via ``SimilarityTransform.estimate()``, deprecated in
    scikit-image 0.26 (``FutureWarning``) in favor of the
    ``SimilarityTransform.from_estimate()`` constructor. Swap in an equivalent
    implementation so pipeline runs and tests stay FutureWarning-free while
    preserving the exact landmark scaling, ratio math, and ``arcface_dst``
    destination coordinates. No-op when either optional dependency is missing.
    """
    try:
        import numpy as np
        from insightface.utils import face_align
        from skimage import transform as trans
        from skimage.transform import SimilarityTransform
    except Exception:  # noqa: BLE001 — optional stack absent; keep upstream code
        return

    from_estimate = getattr(SimilarityTransform, "from_estimate", None)

    def _estimate_norm(lmk, image_size=112, mode="arcface"):
        assert lmk.shape == (5, 2)
        assert image_size % 112 == 0 or image_size % 128 == 0
        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio
        dst = face_align.arcface_dst * ratio
        dst[:, 0] += diff_x
        if from_estimate is not None:
            tform = SimilarityTransform.from_estimate(lmk, dst)
            if bool(tform):
                return tform.params[0:2, :]
            return np.full((2, 3), np.nan)
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        return tform.params[0:2, :]

    face_align.estimate_norm = _estimate_norm


def _get_face_app():
    """Lazy FaceAnalysis singleton; None (once) when the stack is unavailable."""
    global _APP, _APP_FAILED
    if _APP is not None or _APP_FAILED:
        return _APP
    with _APP_LOCK:
        if _APP is not None or _APP_FAILED:
            return _APP
        try:
            from insightface.app import FaceAnalysis

            _patch_face_align_estimate()

            app = FaceAnalysis(
                name="buffalo_l",
                root=str(_MODEL_ROOT),
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _APP = app
        except Exception as exc:  # noqa: BLE001 — degrade, never fail the run
            _APP_FAILED = True
            print(f"[FACES] Face model unavailable — recognition disabled: {exc}")
        return _APP


def _quality_scores(face, crop, image_shape) -> dict:
    """Per-factor quality scores (doc 07 §3.1); each 0.0–1.0, higher = better."""
    import cv2
    import numpy as np

    top, left = float(face.bbox[1]), float(face.bbox[0])
    bottom, right = float(face.bbox[3]), float(face.bbox[2])
    face_w, face_h = max(0.0, right - left), max(0.0, bottom - top)

    # Blur — Laplacian variance of the face crop (doc 07: "low blur = high score").
    if crop.size == 0:
        blur = 0.0
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur = min(1.0, lap_var / 200.0)

    # Resolution — face pixel size; a 96px+ face scores 1.0.
    resolution = min(1.0, math.sqrt(face_w * face_h) / 96.0) if face_w and face_h else 0.0

    # Angle — roll from the eye line + yaw proxy from nose offset (frontal = best).
    angle = 0.5
    if face.kps is not None and len(face.kps) >= 5:
        kps = np.asarray(face.kps, dtype=float)
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        roll = abs(math.degrees(math.atan2(
            float(right_eye[1]) - float(left_eye[1]),
            float(right_eye[0]) - float(left_eye[0]),
        )))
        roll_score = max(0.0, 1.0 - roll / 30.0)
        eye_mid = (float(left_eye[0]) + float(right_eye[0])) / 2.0
        yaw_score = max(0.0, 1.0 - abs(float(nose[0]) - eye_mid) / max(1.0, face_w * 0.25))
        angle = (roll_score + yaw_score) / 2.0

    # Lighting — exposure closeness to mid-gray (doc 07: "even = best").
    if crop.size == 0:
        lighting = 0.0
    else:
        mean = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        lighting = max(0.0, 1.0 - abs(mean - 127.5) / 127.5)

    # Occlusion — fraction of landmarks with a healthy in-frame margin.
    occlusion = 0.5
    if face.kps is not None and len(face.kps) >= 5:
        margin_x, margin_y = face_w * 0.10, face_h * 0.10
        visible = sum(
            1
            for px, py in face.kps
            if left + margin_x <= px <= right - margin_x
            and top + margin_y <= py <= bottom - margin_y
        )
        occlusion = visible / len(face.kps)

    return {
        "blur": round(blur, 3),
        "resolution": round(resolution, 3),
        "angle": round(angle, 3),
        "lighting": round(lighting, 3),
        "occlusion": round(occlusion, 3),
    }


def analyze_image(file_path: str, file_name: str = "") -> list[dict]:
    """One detection+embedding pass over an image.

    Returns face records with status ``PENDING`` (doc 08 §2: Stage 1 detects,
    Stage 2 finalizes). Empty list when the model stack is unavailable, the
    file is unreadable, or no face is present.
    """
    app = _get_face_app()
    if app is None:
        return []
    try:
        import cv2

        img = cv2.imread(file_path)
        if img is None:
            return []
        detected = app.get(img)
    except Exception as exc:  # noqa: BLE001 — a bad image must not kill a run
        print(f"[FACES] detection failed for {file_name or file_path}: {exc}")
        return []

    records: list[dict] = []
    stem = Path(file_path).stem
    for index, face in enumerate(detected):
        try:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            img_h, img_w = img.shape[:2]
            crop = img[
                max(0, int(y1)):min(img_h, int(y2)),
                max(0, int(x1)):min(img_w, int(x2)),
            ]
            scores = _quality_scores(face, crop, img.shape)
            embedding = [float(v) for v in face.embedding] if face.embedding is not None else []
            landmarks = []
            if face.kps is not None:
                landmarks = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in face.kps]
            records.append({
                "id": f"face_{stem}_{index}",
                "file_path": str(file_path),
                "file_name": Path(file_path).name,
                "face_bbox": {
                    "x": round(x1, 1),
                    "y": round(y1, 1),
                    "width": round(max(0.0, x2 - x1), 1),
                    "height": round(max(0.0, y2 - y1), 1),
                },
                "landmarks": landmarks,
                "detector_confidence": round(float(face.det_score), 3),
                "quality_scores": scores,
                "quality_score": round(weighted_quality(scores), 3),
                "embedding_vector": embedding,
                "status": "PENDING",  # Stage 2 promotes (doc 08 §2.2/§3.2)
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception as exc:  # noqa: BLE001 — skip the bad face, keep the rest
            print(f"[FACES] face record failed for {file_name or file_path}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Overlay context (doc 11 §6.4 — CCTV screenshots carry camera + timestamp)
# ---------------------------------------------------------------------------

# Camera ids: 'CAM_KB7', 'CAM 03' — the (?![A-Za-z]) guard keeps the word
# "Camera" from matching as CAM_ERA.
_CAMERA_RE = re.compile(r"\bCAM(?![A-Za-z])[^A-Za-z0-9]{0,3}([A-Z0-9]{1,8})\b", re.IGNORECASE)
# Overlay stamps in the wild: '14/03/2024 09:30:00', 'Date: 14/03/2024 Time:
# 09:30:00', '2024-06-25 14:32:01'. Date may be separated from the time by
# same-line label text (≤40 chars), never by a newline; separators may be
# '-', '/' or '.' (OCR alternates between them).
_STAMP_RE = re.compile(r"(\d{2}[/.-]\d{2}[/.-]\d{4})[^\n]{0,40}?(\d{2}:\d{2}:\d{2})")
_ISO_STAMP_RE = re.compile(r"(\d{4}[-/.]\d{2}[-/.]\d{2})[^\n]{0,40}?(\d{2}:\d{2}:\d{2})")
_DATE_ONLY_RE = re.compile(r"(\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[/.-]\d{2}[/.-]\d{4})")


def camera_from_text(text: str) -> str:
    """Camera id from overlaid text ('Camera ID: CAM_KB7' → 'CAM_KB7',
    'CAM 4' / 'CAM! 4' OCR variants → 'CAM_4')."""
    match = _CAMERA_RE.search(text or "")
    return f"CAM_{match.group(1).upper()}" if match else ""


def _parse_overlay_date(raw: str) -> datetime | None:
    """'2024-06-25'/'14.03.2024'/'14/03/2024' → datetime (None when invalid)."""
    parts = raw.replace("/", "-").replace(".", "-").split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    try:
        if len(parts[0]) == 4:
            year, month, day = (int(part) for part in parts)
        else:
            day, month, year = (int(part) for part in parts)
        return datetime(year, month, day)
    except ValueError:
        return None


def timestamp_from_text(text: str) -> str:
    """Overlay stamp → ISO ('2024-06-25 14:32:01' → '2024-06-25T14:32:01').

    Accepts DD/MM/YYYY and YYYY-MM-DD (any of '-', '/', '.' separators),
    with time when present; a date-only overlay still pins the day
    ('' when absent).
    """
    text = text or ""
    for pattern, order in ((_STAMP_RE, "dmy"), (_ISO_STAMP_RE, "iso")):
        match = pattern.search(text)
        if not match:
            continue
        date_part = _parse_overlay_date(match.group(1))
        if date_part is None:
            continue
        try:
            dt = date_part.replace(
                hour=int(match.group(2)[0:2]),
                minute=int(match.group(2)[3:5]),
                second=int(match.group(2)[6:8]),
            )
            return dt.isoformat(timespec="seconds")
        except ValueError:
            continue
    match = _DATE_ONLY_RE.search(text)
    if match:
        date_part = _parse_overlay_date(match.group(1))
        if date_part is not None:
            return date_part.date().isoformat()
    return ""
