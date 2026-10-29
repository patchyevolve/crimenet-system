"""
Ingestion Engine — Stage 1
Detects file types and routes to appropriate parser.
Creates RawEvidence with EvidenceIntegrity, SourceReliability, DependencyGroups.
Every pipeline execution creates a PipelineRun.
"""

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..models.schema import (
    PipelineRun,
    EvidenceIntegrity,
    DependencyGroup,
    SourceMetadata,
    DataQualityScore,
    AdversarialCheck,
    SOURCE_RELIABILITY_MATRIX,
    get_reliability,
)


# Source type detection from filename patterns
SOURCE_TYPE_PATTERNS = {
    "fir": [r"FIR", r"fir", r"first_information", r"complaint"],
    "cdr": [r"CDR", r"cdr", r"call_detail", r"call_record"],
    "bank": [r"bank", r"transaction", r"account", r"financial"],
    "cctv": [r"CCTV", r"cctv", r"camera", r"surveillance"],
    "social": [r"social", r"instagram", r"facebook", r"twitter", r"whatsapp", r"telegram"],
    "device": [r"device", r"phone_extract", r"mobile_extract", r"forensic"],
    "text": [r"report", r"statement", r"witness", r"note", r"memo"],
}


def detect_source_type(file_name: str) -> str:
    """Detect source type from filename."""
    name_lower = file_name.lower()
    for source_type, patterns in SOURCE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return source_type
    return "text"  # default


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file — full 64 hex chars for evidence integrity."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


class IngestionEngine:
    """Handles raw file intake, normalization, and metadata creation."""

    SUPPORTED_EXTENSIONS = {
        ".csv": "tabular",
        ".json": "json",
        ".txt": "text",
        ".md": "text",
        ".pdf": "pdf",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".xlsx": "tabular",
        ".xls": "tabular",
        ".docx": "docx",
        ".doc": "docx",
    }

    # Config files to skip
    SKIP_FILES = {
        "ai_providers.json",
        "column_mappings.json",
        "ingestion_config.json",
    }

    def __init__(self):
        self.ingestion_log = []
        self.current_run: Optional[PipelineRun] = None
        self.integrity_records = {}  # file_name → EvidenceIntegrity
        self.dependency_groups = {}  # group_id → DependencyGroup
        self.quality_scores = {}     # file_name → DataQualityScore
        self.adversarial_checks = {} # file_name → AdversarialCheck

    def start_run(self, run_id: str, case_id: str = "default", trigger: str = "new_evidence") -> PipelineRun:
        """Start a new pipeline run — per DATA_FLOW.md Pipeline Run Versioning."""
        self.current_run = PipelineRun(
            run_id=run_id,
            case_id=case_id,
            input_snapshot=[],
            start_time=datetime.now().isoformat(),
            trigger=trigger,
            status="running",
        )
        return self.current_run

    def end_run(self, status: str = "completed"):
        """End the current pipeline run."""
        if self.current_run:
            self.current_run.end_time = datetime.now().isoformat()
            self.current_run.status = status

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file. Returns normalized intermediate format with full metadata."""
        path = Path(file_path)

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        # Skip known config files
        if path.name in self.SKIP_FILES:
            return {"skipped": True, "reason": "config_file", "file_name": path.name}

        ext = path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

        # Compute file hash for integrity
        file_hash = compute_file_hash(file_path)

        # Detect source type from filename
        source_type = detect_source_type(path.name)

        # Create EvidenceIntegrity record
        integrity = EvidenceIntegrity(
            file_hash=file_hash,
            original_filename=path.name,
            ingestion_time=datetime.now().isoformat(),
        )
        integrity.add_custody_event("ingested", by="ingestion_engine")
        self.integrity_records[path.name] = integrity

        # Create SourceMetadata with reliability from matrix
        reliability_occurrence = get_reliability(source_type, "occurrence")
        reliability_identity = get_reliability(source_type, "identity")
        reliability_intent = get_reliability(source_type, "intent")
        reliability_location = get_reliability(source_type, "location")
        reliability_timing = get_reliability(source_type, "timing")

        # Single stat call to avoid TOCTOU race
        file_stats = path.stat()

        file_info = {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "file_ext": ext,
            "detected_type": file_type,
            "source_type": source_type,
            "file_hash": file_hash,
            "file_size_bytes": file_stats.st_size,
            "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "ingestion_time": datetime.now().isoformat(),
            "reliability": {
                "occurrence": reliability_occurrence,
                "identity": reliability_identity,
                "intent": reliability_intent,
                "location": reliability_location,
                "timing": reliability_timing,
            },
            "evidence_integrity": integrity.to_dict(),
        }

        if file_type == "tabular":
            content = self._parse_tabular(file_path)
        elif file_type == "json":
            content = self._parse_json(file_path)
        elif file_type == "text":
            content = self._parse_text(file_path)
        elif file_type == "image":
            content = self._parse_image(file_path)
        elif file_type == "pdf":
            content = self._parse_pdf(file_path)
        elif file_type == "docx":
            content = self._parse_docx(file_path)
        else:
            content = {"error": f"Unsupported file type: {ext}"}

        file_info["content"] = content
        file_info["record_count"] = self._count_records(content)

        # Data quality + completeness assessment
        run_id = self.current_run.run_id if self.current_run else ""
        quality = self._assess_data_quality(file_info, run_id=run_id)
        self.quality_scores[path.name] = quality
        file_info["data_quality"] = quality.to_dict()

        # Adversarial data check
        adversarial = self._check_adversarial(file_info, run_id=run_id)
        self.adversarial_checks[path.name] = adversarial
        file_info["adversarial_check"] = adversarial.to_dict()

        # Stage 1 (face): detect faces in images — RESEARCH_FACE_RECOGNITION
        # doc 08 §2. One RetinaFace+ArcFace pass; records stay PENDING until
        # Stage 2 finalizes them. Degrades to zero faces without the model.
        # Only PHOTOGRAPHIC evidence runs detection: slides/diagrams and text
        # reports rendered as PNGs contain no faces to extract.
        if file_type == "image":
            from ..faces.context import classify_image_evidence
            from ..faces.engine import analyze_image

            ocr_text = str((content or {}).get("ocr_text") or (content or {}).get("content") or "")
            image_class, face_eligible, class_reason = classify_image_evidence(
                path.name, ocr_text, file_path=str(path)
            )
            content["image_class"] = image_class
            file_info["image_class"] = image_class
            if face_eligible:
                face_crops = analyze_image(file_path, file_name=path.name)
                file_info["face_crops"] = face_crops
                file_info["face_count"] = len(face_crops)
                if face_crops:
                    print(f"[FACES]   -> {path.name}: {len(face_crops)} face(s) detected")
            else:
                file_info["face_crops"] = []
                file_info["face_count"] = 0
                file_info["face_skip_reason"] = class_reason
                print(f"[FACES]   -> {path.name}: skipped ({image_class}: {class_reason})")

        # Track in current run
        if self.current_run:
            self.current_run.input_snapshot.append(path.name)

        self.ingestion_log.append(file_info)
        return file_info

    def ingest_directory(self, dir_path: str) -> List[dict]:
        """Ingest all supported files from a directory."""
        results = []
        path = Path(dir_path)

        if not path.exists():
            return [{"error": f"Directory not found: {dir_path}"}]

        # Start a run if not already started
        if not self.current_run:
            self.start_run(f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and not file_path.is_symlink() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                result = self.ingest_file(str(file_path))
                if isinstance(result, dict) and "error" in result:
                    print(f"[INGESTION] SKIP (error): {file_path.name} — {result['error']}")
                    continue
                if not result.get("skipped"):
                    results.append(result)

        # Assign dependency groups
        self._assign_dependency_groups()

        return results

    def _assign_dependency_groups(self):
        """Group files that share sub-sources (e.g., same FIR, same person)."""
        # Simple heuristic: files with similar names or same source type from same directory
        groups_by_type = {}
        for entry in self.ingestion_log:
            st = entry.get("source_type", "text")
            if st not in groups_by_type:
                groups_by_type[st] = []
            groups_by_type[st].append(entry["file_name"])

        for st, files in groups_by_type.items():
            if len(files) > 1:
                group_id = f"grp_{st}_{len(self.dependency_groups)}"
                group = DependencyGroup(
                    group_id=group_id,
                    sources=files,
                    dependency_type="same_source_type",
                    description=f"Multiple {st} files may share sub-sources",
                )
                self.dependency_groups[group_id] = group

    def _parse_tabular(self, file_path: str) -> dict:
        """Parse CSV/Excel files."""
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            return self._parse_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return self._parse_excel(file_path)
        else:
            return {"error": f"Tabular format not yet supported: {ext}"}

    def _parse_csv(self, file_path: str) -> dict:
        """Parse CSV file into structured records."""
        for encoding in ["utf-8", "latin-1", "cp1252", "utf-16"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    lines = f.readlines()

                # Filter out comment lines and blank lines
                    cleaned = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                    if not cleaned:
                        return {"format": "csv", "columns": [], "row_count": 0, "rows": []}

                    import io
                    sample = "".join(cleaned[:10])

                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
                    except csv.Error:
                        delimiter = ","

                    reader = csv.DictReader(io.StringIO("".join(cleaned)), delimiter=delimiter)
                    rows = list(reader)

                    # Filter out rows where all values are None/empty
                    rows = [r for r in rows if any(v for v in r.values() if v)]

                    return {
                        "format": "csv",
                        "columns": list(rows[0].keys()) if rows else [],
                        "row_count": len(rows),
                        "rows": rows,
                    }
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                return {"format": "csv", "error": str(e)}
        return {"format": "csv", "error": "Could not decode file with any supported encoding"}

    def _parse_excel(self, file_path: str) -> dict:
        """Parse Excel file (.xlsx/.xls) using openpyxl."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            wb.close()

            if not rows:
                return {"format": "excel", "columns": [], "row_count": 0, "rows": []}

            # First row = headers
            headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
            data_rows = []
            for row in rows[1:]:
                if all(cell is None for cell in row):
                    continue  # skip empty rows
                record = {}
                for i, val in enumerate(row):
                    if i < len(headers):
                        record[headers[i]] = str(val).strip() if val is not None else ""
                data_rows.append(record)

            return {
                "format": "excel",
                "columns": headers,
                "row_count": len(data_rows),
                "rows": data_rows,
            }
        except ImportError:
            return {"format": "excel", "error": "openpyxl not installed: pip install openpyxl"}
        except Exception as e:
            return {"format": "excel", "error": str(e)}

    def _parse_json(self, file_path: str) -> dict:
        """Parse JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "format": "json",
                "data_type": type(data).__name__,
                "data": data,
                "key_count": len(data) if isinstance(data, dict) else len(data) if isinstance(data, list) else 0,
            }
        except Exception as e:
            return {"format": "json", "error": str(e)}

    def _parse_text(self, file_path: str) -> dict:
        """Parse plain text files."""
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                lines = content.split("\n")

                return {
                    "format": "text",
                    "content": content,
                    "line_count": len(lines),
                    "char_count": len(content),
                }
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                return {"format": "text", "error": str(e)}
        return {"format": "text", "error": "Could not decode file with any supported encoding"}

    def _parse_image(self, file_path: str) -> dict:
        """Parse image file using OCR (pytesseract + Pillow). Supports English + Hindi."""
        try:
            from PIL import Image, ImageEnhance, ImageOps
            import pytesseract
            import shutil

            # Set tesseract path and language data directory
            tessdata_dir = os.path.expanduser("~/tessdata")
            if not shutil.which("tesseract"):
                home_bin = os.path.expanduser("~/bin/tesseract")
                if os.path.exists(home_bin):
                    pytesseract.pytesseract.tesseract_cmd = home_bin

            # Set TESSDATA_PREFIX so tesseract finds eng+hin traineddata
            tessdata_prefix = tessdata_dir if os.path.isdir(tessdata_dir) else None

            img = Image.open(file_path)

            # Resize large images for faster OCR (max 2000px wide)
            max_width = 2000
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

            # OCR with English + Hindi; fall back to English when the hin
            # traineddata is not installed on this host/container.
            #
            # Multi-pass concat: CCTV overlays (green captions on scanlines)
            # are read differently by each pass — the default dense pass gets
            # 'CAM! 4', the sparse --psm 11 pass gets 'CAM 4:', and a
            # preprocessed grayscale+autocontrast+sharpen pass with --psm 6
            # reads the timestamp block '2024-06-25 14:32:01' exactly. No
            # single pass wins, so feed every reading to the extractors and
            # let the timestamp/camera parsers pick the coherent one; the
            # extra noise lines match no downstream pattern.
            #
            # The overlay pass runs eng-only: the hin traineddata degrades
            # digit shapes on stylized fonts ('14:32:01' -> '4:32:0!').
            lang = "hin+eng"
            tess_config = f"--tessdata-dir {tessdata_prefix}" if tessdata_prefix else ""

            pre = ImageOps.autocontrast(img.convert("L").resize(
                (img.width * 2, img.height * 2), Image.LANCZOS))
            pre = ImageEnhance.Sharpness(pre).enhance(2.0)
            variants = ((img, "", lang), (img, "--psm 11", lang),
                        (pre, "--psm 6", "eng"))

            def _ocr_passes(image_variants, base_config: str) -> list[str]:
                texts: list[str] = []
                for image, extra, pass_lang in image_variants:
                    try:
                        texts.append(pytesseract.image_to_string(
                            image, lang=pass_lang,
                            config=f"{base_config} {extra}".strip(),
                        ))
                    except pytesseract.TesseractError:
                        return texts
                return texts

            texts = _ocr_passes(variants, tess_config)
            if not any(t.strip() for t in texts):
                lang = "eng"
                variants = tuple((im, ex, "eng") for im, ex, _ in variants)
                texts = _ocr_passes(variants, tess_config)
            text = "\n".join(t for t in texts if t)

            # Get per-word confidence scores for quality gate
            ocr_data = pytesseract.image_to_data(img, lang=lang,
                                                  output_type=pytesseract.Output.DICT,
                                                  config=tess_config)

            # Calculate quality metrics
            words = []
            confidences = []
            for i, word in enumerate(ocr_data["text"]):
                conf = int(ocr_data["conf"][i]) if ocr_data["conf"][i] != "-1" else 0
                if conf > 0 and word.strip():
                    words.append(word.strip())
                    confidences.append(conf)

            # Quality gate
            meaningful_words = [w for w in words if len(w) > 1 and any(c.isalnum() for c in w)]
            alphanum_chars = sum(1 for c in text if c.isalnum() or c in "अआइईउऊएऐओऔकखगघचछजतदधनपबभमयरलवसशषह")
            total_chars = max(len(text), 1)
            alphanum_ratio = alphanum_chars / total_chars
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            quality_score = (
                min(1.0, len(meaningful_words) / 10) * 0.4 +   # word count component
                (avg_confidence / 100) * 0.35 +                 # confidence component
                alphanum_ratio * 0.25                            # character quality component
            )

            has_useful_text = quality_score > 0.3 and len(meaningful_words) >= 2

            return {
                "format": "image",
                "content": text.strip() if has_useful_text else "",
                "ocr_text": text.strip(),
                "ocr_words_extracted": len(meaningful_words),
                "ocr_quality_score": round(quality_score, 3),
                "ocr_avg_confidence": round(avg_confidence, 1),
                "requires_vision_model": not has_useful_text,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
            }
        except ImportError:
            return {"format": "image", "error": "pytesseract or Pillow not installed", "file_path": file_path}
        except Exception as e:
            return {"format": "image", "error": str(e), "file_path": file_path}

    def _parse_pdf(self, file_path: str) -> dict:
        """Parse PDF: extract text AND tables using pdfplumber. Falls back to OCR for scanned PDFs."""
        try:
            import pdfplumber

            all_text = []
            all_tables = []

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)

                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:
                            # Convert to list of dicts (first row = headers)
                            headers = [str(h).strip() if h else f"col_{j}" for j, h in enumerate(table[0])]
                            rows = []
                            for row in table[1:]:
                                row_dict = {}
                                for j, cell in enumerate(row):
                                    if j < len(headers):
                                        row_dict[headers[j]] = str(cell).strip() if cell else ""
                                rows.append(row_dict)

                            all_tables.append({
                                "page": i + 1,
                                "columns": headers,
                                "row_count": len(rows),
                                "rows": rows,
                            })

            combined_text = "\n\n".join(all_text)

            # If pdfplumber extracted no/little text, this might be a scanned PDF — try OCR
            if not combined_text.strip() or len(combined_text.strip()) < 20:
                ocr_text = self._ocr_pdf(file_path)
                if ocr_text:
                    return {
                        "format": "pdf",
                        "content": ocr_text,
                        "text_length": len(ocr_text),
                        "tables": all_tables,
                        "table_count": len(all_tables),
                        "page_count": total_pages,
                        "ocr_applied": True,
                    }

            return {
                "format": "pdf",
                "content": combined_text,
                "text_length": len(combined_text),
                "tables": all_tables,
                "table_count": len(all_tables),
                "page_count": total_pages,
            }

        except ImportError:
            # Fallback: try pdftotext
            try:
                import subprocess
                result = subprocess.run(
                    ["pdftotext", "--", file_path, "-"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return {
                        "format": "pdf",
                        "content": result.stdout,
                        "text_length": len(result.stdout),
                        "tables": [],
                        "table_count": 0,
                        "note": "Text only — pdfplumber not available for table extraction",
                    }
            except Exception:
                pass

            return {
                "format": "pdf",
                "error": "pdfplumber not installed. Run: pip install pdfplumber",
                "file_path": file_path,
            }

    def _ocr_pdf(self, file_path: str) -> str:
        """OCR a scanned PDF by converting pages to images and running tesseract (Hindi + English)."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import pytesseract
            import io
            import shutil

            # Set tesseract path if needed
            if not shutil.which("tesseract"):
                home_bin = os.path.expanduser("~/bin/tesseract")
                if os.path.exists(home_bin):
                    pytesseract.pytesseract.tesseract_cmd = home_bin

            tessdata_dir = os.path.expanduser("~/tessdata")
            tessdata_prefix = tessdata_dir if os.path.isdir(tessdata_dir) else None
            config = f"--tessdata-dir {tessdata_prefix}" if tessdata_prefix else ""

            doc = fitz.open(file_path)
            all_text = []
            lang = "hin+eng"

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to image (2x for better OCR quality)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                # OCR with Hindi + English
                text = pytesseract.image_to_string(img, lang=lang, config=config)
                if text.strip():
                    all_text.append(text.strip())

            doc.close()
            return "\n\n".join(all_text)

        except ImportError:
            print("[INGESTION] WARNING: PyMuPDF not installed for scanned PDF OCR. Run: pip install PyMuPDF")
            return ""
        except Exception as e:
            print(f"[INGESTION] WARNING: OCR failed for {file_path}: {e}")
            return ""

    def _parse_docx(self, file_path: str) -> dict:
        """Parse DOCX: extract text AND tables using python-docx."""
        try:
            from docx import Document

            doc = Document(file_path)

            # Extract text
            all_text = [para.text for para in doc.paragraphs if para.text.strip()]
            combined_text = "\n".join(all_text)

            # Extract tables
            all_tables = []
            for i, table in enumerate(doc.tables):
                if len(table.rows) < 2:
                    continue

                headers = [cell.text.strip() for cell in table.rows[0].cells]
                rows = []
                for row in table.rows[1:]:
                    row_dict = {}
                    for j, cell in enumerate(row.cells):
                        if j < len(headers):
                            row_dict[headers[j]] = cell.text.strip()
                    rows.append(row_dict)

                all_tables.append({
                    "table_index": i + 1,
                    "columns": headers,
                    "row_count": len(rows),
                    "rows": rows,
                })

            return {
                "format": "docx",
                "content": combined_text,
                "text_length": len(combined_text),
                "tables": all_tables,
                "table_count": len(all_tables),
                "paragraph_count": len(doc.paragraphs),
            }

        except ImportError:
            return {
                "format": "docx",
                "error": "python-docx not installed. Run: pip install python-docx",
                "file_path": file_path,
            }
        except Exception as e:
            return {
                "format": "docx",
                "error": f"Failed to parse docx: {str(e)}",
                "file_path": file_path,
            }

    def _count_records(self, content: dict) -> int:
        """Count records in parsed content (rows + table rows)."""
        count = 0

        # Direct row count
        if "row_count" in content:
            count += content["row_count"]
        elif "key_count" in content:
            count += content["key_count"]
        elif "data" in content:
            data = content["data"]
            if isinstance(data, list):
                count += len(data)
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        count += len(v)
                    elif isinstance(v, dict):
                        count += len(v)
                    else:
                        count += 1

        # Count rows from embedded tables (PDF/DOCX)
        if "tables" in content:
            for table in content["tables"]:
                count += table.get("row_count", 0)

        return count

    def _assess_data_quality(self, file_info: dict, run_id: str = "") -> DataQualityScore:
        """Assess data quality and completeness for an ingested file — per docs."""
        file_name = file_info.get("file_name", "unknown")
        content = file_info.get("content", {})
        file_type = file_info.get("detected_type", "unknown")
        record_count = file_info.get("record_count", 0)

        issues = []
        quality_score = 1.0

        # Structural consistency check
        structural_consistency = 1.0
        if file_type == "tabular":
            rows = content.get("rows", [])
            columns = content.get("columns", [])
            if not columns:
                structural_consistency -= 0.3
                issues.append("no_column_headers")
            if not rows:
                structural_consistency -= 0.5
                issues.append("empty_dataset")
            elif len(rows) < 3:
                structural_consistency -= 0.1
                issues.append("very_few_rows")

            # Check for null/empty field ratio
            if rows and columns:
                total_cells = len(rows) * len(columns)
                empty_cells = sum(1 for row in rows for col in columns if not str(row.get(col, "")).strip())
                empty_ratio = empty_cells / total_cells if total_cells > 0 else 0
                if empty_ratio > 0.5:
                    structural_consistency -= 0.3
                    issues.append(f"high_empty_ratio_{empty_ratio:.0%}")
                elif empty_ratio > 0.2:
                    structural_consistency -= 0.1
                    issues.append(f"moderate_empty_ratio_{empty_ratio:.0%}")

        elif file_type == "text":
            text = content.get("content", "")
            if not text.strip():
                structural_consistency = 0.3
                issues.append("empty_text")
            elif len(text.strip()) < 20:
                structural_consistency = 0.6
                issues.append("very_short_text")

        elif file_type == "json":
            data = content.get("data")
            if data is None:
                structural_consistency = 0.3
                issues.append("null_json_data")

        # Content richness — ratio of non-empty meaningful fields
        content_richness = 1.0
        if file_type == "tabular":
            rows = content.get("rows", [])
            columns = content.get("columns", [])
            if rows and columns:
                meaningful_cells = 0
                total_cells = len(rows) * len(columns)
                for row in rows:
                    for col in columns:
                        val = str(row.get(col, "")).strip()
                        if val and val.lower() not in ("null", "none", "n/a", "na", "-", ""):
                            meaningful_cells += 1
                content_richness = meaningful_cells / total_cells if total_cells > 0 else 0
        elif file_type in ("text", "pdf", "docx"):
            text = content.get("content", "")
            if text:
                words = [w for w in text.split() if len(w) > 2]
                content_richness = min(1.0, len(words) / 50)

        # Composite quality score
        quality_score = (structural_consistency * 0.5 + content_richness * 0.5)
        quality_score = max(0.0, min(1.0, quality_score))

        # Completeness: how much of the expected fields are present
        completeness_score = content_richness  # For tabular, this IS the completeness

        return DataQualityScore(
            file_name=file_name,
            quality_score=round(quality_score, 3),
            completeness_score=round(completeness_score, 3),
            structural_consistency=round(structural_consistency, 3),
            content_richness=round(content_richness, 3),
            issues=issues,
            run_id=run_id,
        )

    def _check_adversarial(self, file_info: dict, run_id: str = "") -> AdversarialCheck:
        """Check for adversarial data patterns — per docs.
        
        Enhanced detection catches:
        1. Basic anomalies (file size, content mismatch, identical rows)
        2. Statistical anomalies (unusual call patterns, transaction amounts)
        3. Temporal anomalies (calls at impossible times, timestamp manipulation)
        4. Cross-file patterns (same numbers in multiple files)
        5. Network anomalies (hub patterns, suspicious communication)
        """
        file_name = file_info.get("file_name", "unknown")
        content = file_info.get("content", {})
        file_type = file_info.get("detected_type", "unknown")
        source_type = file_info.get("source_type", "text")
        file_size = file_info.get("file_size_bytes", 0)

        reasons = []
        suspicious_score = 0.0

        # 1. File size anomaly: extremely small or large for source type
        size_thresholds = {
            "cdr": (500, 50_000_000),     # 500B - 50MB
            "bank": (500, 50_000_000),
            "fir": (1000, 10_000_000),
            "cctv": (100, 50_000_000),
            "social": (200, 10_000_000),
            "device": (500, 50_000_000),
            "text": (100, 5_000_000),
        }
        min_size, max_size = size_thresholds.get(source_type, (100, 10_000_000))
        if file_size < min_size:
            suspicious_score += 0.2
            reasons.append(f"file_size_anomaly: {file_size}B < {min_size}B for {source_type}")
        elif file_size > max_size:
            suspicious_score += 0.15
            reasons.append(f"file_size_large: {file_size}B > {max_size}B for {source_type}")

        # 2. Content/source type mismatch
        if file_type == "tabular" and source_type == "fir":
            suspicious_score += 0.1
            reasons.append("tabular_content_for_fir_source")
        elif file_type == "text" and source_type in ("cdr", "bank"):
            suspicious_score += 0.1
            reasons.append(f"text_content_for_{source_type}_source")

        # 3. Repeated/identical content (check if all rows are same)
        if file_type == "tabular":
            rows = content.get("rows", [])
            if len(rows) > 5:
                # Sanitize rows for comparison (replace None with "", ensure all keys/values are strings)
                def _safe_row(row):
                    return {str(k): (str(v) if v is not None else "") for k, v in row.items()}
                sanitized_rows = [_safe_row(r) for r in rows]
                first_row_str = json.dumps(sanitized_rows[0], sort_keys=True)
                identical_count = sum(1 for r in sanitized_rows if json.dumps(r, sort_keys=True) == first_row_str)
                if identical_count == len(rows):
                    suspicious_score += 0.4
                    reasons.append("all_rows_identical")
                elif identical_count > len(rows) * 0.8:
                    suspicious_score += 0.2
                    reasons.append(f"high_row_duplication_{identical_count}/{len(rows)}")

        # 4. Temporal anomaly: check if timestamps are reasonable
        if file_type == "tabular":
            rows = content.get("rows", [])
            for row in rows[:5]:
                for col, val in row.items():
                    val_str = str(val).strip()
                    # Check for future dates (beyond 2030)
                    if "203" in val_str or "204" in val_str or "205" in val_str:
                        suspicious_score += 0.15
                        reasons.append(f"future_date_detected: {val_str}")
                        break
                    # Check for very old dates (before 1990)
                    if "198" in val_str or "197" in val_str:
                        suspicious_score += 0.1
                        reasons.append(f"very_old_date: {val_str}")
                        break

        # 5. Encoding anomalies: unusual characters
        if file_type in ("text", "pdf"):
            text = content.get("content", "")
            if text:
                unusual_chars = sum(1 for c in text if ord(c) > 65535)
                if unusual_chars > 10:
                    suspicious_score += 0.1
                    reasons.append(f"unusual_unicode_chars: {unusual_chars}")

        # 6. CDR-specific: Call pattern anomalies
        if source_type == "cdr" and file_type == "tabular":
            rows = content.get("rows", [])
            if rows:
                # Check for calls at impossible times (2-5 AM = suspicious)
                late_night_calls = 0
                for row in rows:
                    ts = str(row.get("timestamp", ""))
                    if ts:
                        try:
                            hour = int(ts.split("T")[1].split(":")[0]) if "T" in ts else -1
                            if 2 <= hour <= 5:
                                late_night_calls += 1
                        except (IndexError, ValueError):
                            pass
                if late_night_calls > len(rows) * 0.3:
                    suspicious_score += 0.25
                    reasons.append(f"late_night_calls: {late_night_calls}/{len(rows)} calls between 2-5 AM")

                # Check for unusual call duration patterns (very short calls = possible ping attacks)
                durations = []
                for row in rows:
                    dur = row.get("call_duration", row.get("duration", ""))
                    try:
                        durations.append(int(dur))
                    except (ValueError, TypeError):
                        pass
                if durations:
                    avg_dur = sum(durations) / len(durations)
                    very_short = sum(1 for d in durations if d < 5)
                    if very_short > len(durations) * 0.5:
                        suspicious_score += 0.2
                        reasons.append(f"very_short_calls: {very_short}/{len(durations)} calls under 5s")

                # Check for hub pattern (one number calling many others = possible C2 server)
                caller_counts = {}
                for row in rows:
                    caller = str(row.get("caller_number", row.get("from", "")))
                    if caller:
                        caller_counts[caller] = caller_counts.get(caller, 0) + 1
                if caller_counts:
                    max_calls = max(caller_counts.values())
                    if max_calls > len(rows) * 0.7:
                        hub_number = [k for k, v in caller_counts.items() if v == max_calls][0]
                        suspicious_score += 0.3
                        reasons.append(f"hub_pattern: {hub_number} made {max_calls}/{len(rows)} calls")

        # 7. Bank-specific: Transaction anomalies
        if source_type == "bank" and file_type == "tabular":
            rows = content.get("rows", [])
            if rows:
                # Check for round number transactions (possible layering)
                amounts = []
                for row in rows:
                    amt = row.get("amount", row.get("Amount", ""))
                    try:
                        amounts.append(float(str(amt).replace(",", "").replace("₹", "").strip()))
                    except (ValueError, TypeError):
                        pass
                if amounts:
                    round_amounts = sum(1 for a in amounts if a > 0 and a % 100000 == 0)
                    if round_amounts > len(amounts) * 0.5:
                        # Fraud-typical, not fabrication — evidence weight only,
                        # must not by itself cross the 0.3 suspicion threshold
                        # (round-amount + rapid was flagging the LEGITIMATE
                        # victim bank file 15_Bank_Meena.csv).
                        suspicious_score += 0.1
                        reasons.append(f"round_amount_transactions: {round_amounts}/{len(amounts)} are round numbers")

                    # Check for rapid succession transactions
                    timestamps = []
                    for row in rows:
                        ts = str(row.get("transaction_date", row.get("date", row.get("timestamp", ""))))
                        if ts:
                            timestamps.append(ts)
                    if len(timestamps) >= 2:
                        # Check if multiple transactions within 1 hour
                        try:
                            parsed = []
                            for ts in timestamps:
                                if "T" in ts:
                                    parsed.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                                elif "-" in ts:
                                    parsed.append(datetime.strptime(ts[:10], "%Y-%m-%d"))
                            if len(parsed) >= 2:
                                parsed.sort()
                                rapid_count = 0
                                for i in range(1, len(parsed)):
                                    diff = (parsed[i] - parsed[i-1]).total_seconds()
                                    if diff < 3600:  # Within 1 hour
                                        rapid_count += 1
                                if rapid_count > len(parsed) * 0.5:
                                    # Same demotion as round amounts — rapid
                                    # bursts are fraud-typical, not proof of
                                    # fabricated data (cross-file rules in
                                    # finalize_adversarial_checks catch real
                                    # fabrication).
                                    suspicious_score += 0.1
                                    reasons.append(f"rapid_transactions: {rapid_count} transactions within 1 hour")
                        except (ValueError, TypeError):
                            pass

        # 8. Social media: Profile anomalies
        if source_type == "social" and file_type == "json":
            text = json.dumps(content)
            # Check for very new account with many posts
            if "account_created" in text:
                try:
                    # _parse_json wraps the raw document under "data"
                    payload = content.get("data", content) if isinstance(content, dict) else {}
                    prof = (payload.get("profile", {}) or {}) if isinstance(payload, dict) else {}
                    created_str = prof.get("account_created", "")
                    if created_str:
                        created = datetime.strptime(created_str, "%Y-%m-%d")
                        # Age against the FILE's own activity timeline, not
                        # wall-clock now() — case data is historical (Mar 2024),
                        # so datetime.now() made every account look years old
                        # and the rule never fired.
                        activity_dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
                        ref = max(
                            (datetime.strptime(d, "%Y-%m-%d") for d in activity_dates),
                            default=created,
                        )
                        days_old = (ref - created).days
                        posts = prof.get("posts_count", 0)
                        if days_old < 30 and posts > 50:
                            suspicious_score += 0.2
                            reasons.append(f"new_account_many_posts: {days_old} days old, {posts} posts")
                except (ValueError, TypeError, AttributeError):
                    pass

        # 9. CCTV: Timestamp consistency
        if source_type == "cctv" and file_type == "tabular":
            rows = content.get("rows", [])
            if len(rows) >= 2:
                timestamps = []
                for row in rows:
                    ts = str(row.get("timestamp", ""))
                    if ts:
                        timestamps.append(ts)
                # Check for out-of-order timestamps
                if len(timestamps) >= 2:
                    try:
                        parsed = []
                        for ts in timestamps:
                            if "T" in ts:
                                parsed.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                        if len(parsed) >= 2:
                            out_of_order = sum(1 for i in range(1, len(parsed)) if parsed[i] < parsed[i-1])
                            if out_of_order > 0:
                                suspicious_score += 0.15
                                reasons.append(f"out_of_order_timestamps: {out_of_order} timestamps not in sequence")
                    except (ValueError, TypeError):
                        pass

        # 10. Cross-file: Phone number reuse (same number in multiple unrelated files)
        # This is checked at pipeline level, not per-file

        is_suspicious = suspicious_score > 0.3

        return AdversarialCheck(
            file_name=file_name,
            is_suspicious=is_suspicious,
            behavioral_anomaly=any("identical" in r or "duplication" in r or "hub_pattern" in r or "late_night" in r for r in reasons),
            temporal_anomaly=any("date" in r or "timestamp" in r or "rapid" in r for r in reasons),
            content_anomaly=any("mismatch" in r or "round_amount" in r for r in reasons),
            duplicate_suspect=any("identical" in r or "duplication" in r for r in reasons),
            score=round(min(1.0, suspicious_score), 3),
            reasons=reasons,
            run_id=run_id,
        )

    def finalize_adversarial_checks(self, run_id: str = ""):
        """Cross-file adversarial post-pass — run after all files ingested,
        before extraction reads adversarial flags.

        Per-file checks cannot see fabrication that only exists across files:
          A. bank credits whose claimed funding account had insufficient
             balance in other bank sources BEFORE the credit timestamp
             (funds could not have existed — fabricated ledger entry),
          B. perfect reciprocal echo sweeps fanning out to >=4 partners
             (machine-generated CDR), not human conversation,
          C. CCTV rows copied from another same-schema file at shifted
             timestamps under a uniform synthetic cadence,
          D. DMs to a contact whose profile (living in another social file)
             was created <30 days before the first message and is empty
             (0 posts, 0 followers) — fake interaction bait.

        Each confirmed pattern adds +0.4 (threshold 0.3). Idempotent:
        per-file baselines are captured once and restored on re-run so
        incremental ingest can safely re-invoke this method.
        """
        if not getattr(self, "_adversarial_baselines", None):
            self._adversarial_baselines = {}

        def _restore_baseline(name, chk):
            base = self._adversarial_baselines.get(name)
            if base is None:
                self._adversarial_baselines[name] = (
                    chk.score, list(chk.reasons), chk.is_suspicious,
                    chk.behavioral_anomaly, chk.temporal_anomaly,
                    chk.content_anomaly, chk.duplicate_suspect,
                )
            else:
                (chk.score, chk.reasons, chk.is_suspicious,
                 chk.behavioral_anomaly, chk.temporal_anomaly,
                 chk.content_anomaly, chk.duplicate_suspect) = (
                    base[0], list(base[1]), base[2], base[3],
                    base[4], base[5], base[6],
                )
            return chk

        for name, chk in self.adversarial_checks.items():
            _restore_baseline(name, chk)

        files = [
            f for f in self.ingestion_log
            if isinstance(f, dict) and f.get("content") and not f.get("error")
        ]

        def _add(file_name, points, reason, *, temporal=False, behavioral=False):
            chk = self.adversarial_checks.get(file_name)
            if chk is None:
                return
            chk.score = round(min(1.0, chk.score + points), 3)
            chk.reasons.append(reason)
            if temporal:
                chk.temporal_anomaly = True
            if behavioral:
                chk.behavioral_anomaly = True
            chk.is_suspicious = chk.score > 0.3
            # Rewrite the dict copy carried on file_info
            for f in files:
                if f.get("file_name") == file_name:
                    f["adversarial_check"] = chk.to_dict()

        def _rows(f):
            rows = f.get("content", {}).get("rows", [])
            return [r for r in rows if isinstance(r, dict)]

        def _parse_ts(ts):
            ts = str(ts or "").strip()
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    return datetime.strptime(ts[:26], fmt)
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None

        # ── A. Bank credit funding capability (cross-file ledger) ──
        # A CREDIT is fabricated when the claimed funding account provably
        # could not have paid it: that account's ledger rows in OTHER bank
        # sources strictly BEFORE the credit's timestamp end below the
        # credited amount. Unverifiable cases (unknown counterparty, no
        # prior coverage) are exempt, as are self-referential credits
        # (counterparty == own account) — description quirks, not evidence.
        bank_files = [
            f for f in files
            if f.get("source_type") == "bank" and f.get("detected_type") == "tabular"
        ]

        def _bank_ts(row):
            d = str(row.get("transaction_date", "") or "").strip()
            t = str(row.get("transaction_time", "") or "").strip()
            return _parse_ts(f"{d} {t}".strip()) if d else None

        if bank_files:
            known_accounts = set()
            # (timestamp, balance, source_file) per account, from all bank rows
            ledgers: dict = {}
            for f in bank_files:
                for row in _rows(f):
                    acc = str(row.get("account_number", "") or "").strip()
                    if acc:
                        known_accounts.add(acc)
                    ts = _bank_ts(row)
                    if not acc or ts is None:
                        continue
                    try:
                        bal = float(row.get("balance_after"))
                    except (TypeError, ValueError):
                        continue
                    ledgers.setdefault(acc, []).append((ts, bal, f["file_name"]))
            for acc in ledgers:
                ledgers[acc].sort(key=lambda x: x[0])
            for f in bank_files:
                impossible = []
                for row in _rows(f):
                    if str(row.get("transaction_type", "") or "").strip().upper() != "CREDIT":
                        continue
                    cp = str(row.get("counterparty_account", "") or "").strip()
                    own = str(row.get("account_number", "") or "").strip()
                    try:
                        amt = float(row.get("amount"))
                    except (TypeError, ValueError):
                        continue
                    if amt <= 0:
                        continue
                    if not cp or cp not in known_accounts or cp == own:
                        continue
                    ct = _bank_ts(row)
                    if ct is None:
                        continue
                    prior = [
                        (t, b) for (t, b, src) in ledgers.get(cp, [])
                        if t < ct and src != f["file_name"]
                    ]
                    if not prior:
                        continue
                    if prior[-1][1] < amt:
                        impossible.append(str(row.get("transaction_id", "")))
                if impossible:
                    _add(
                        f["file_name"], 0.4,
                        f"cross_file_timestamp_reconciliation: {len(impossible)} credit(s) "
                        f"{impossible[:3]} at timestamps funding accounts whose ledger "
                        f"balance (other sources, before credit time) is below the credited "
                        f"amount — funds could not have existed, fabricated ledger entry",
                        temporal=True,
                    )

        # ── B. CDR echo sweep ──
        cdr_files = [
            f for f in files
            if f.get("source_type") == "cdr" and f.get("detected_type") == "tabular"
        ]
        for f in cdr_files:
            rows = _rows(f)
            if len(rows) < 4:
                continue
            if not all("caller_number" in r and "callee_number" in r for r in rows):
                continue
            parsed = [(_parse_ts(r.get("timestamp")), r) for r in rows]
            tight = 0
            for i, (ti, r) in enumerate(parsed):
                a, b = str(r.get("caller_number", "")), str(r.get("callee_number", ""))
                if not a or not b or a == b or ti is None:
                    continue
                for j, (tj, r2) in enumerate(parsed):
                    if i == j or tj is None:
                        continue
                    if str(r2.get("caller_number", "")) == b and str(r2.get("callee_number", "")) == a:
                        if abs((ti - tj).total_seconds()) <= 90:
                            tight += 1
                            break
            if tight / len(rows) < 0.9:
                continue
            # Fan-out of the dominant caller distinguishes a fabricated sweep
            # from small paired test fixtures: >=4 distinct partners.
            caller_counts = {}
            for r in rows:
                c = str(r.get("caller_number", ""))
                if c:
                    caller_counts[c] = caller_counts.get(c, 0) + 1
            if not caller_counts:
                continue
            dom = max(caller_counts, key=caller_counts.get)
            partners = {
                str(r.get("callee_number", ""))
                for r in rows
                if str(r.get("caller_number", "")) == dom
                and str(r.get("callee_number", "")) != dom
            }
            if len(partners) < 4:
                continue
            _add(
                f["file_name"], 0.4,
                f"echo_sweep_pattern: {tight}/{len(rows)} calls are perfectly reciprocal "
                f"within 90s and dominant number {dom} fans out to {len(partners)} partners — "
                f"machine-generated call fabrications",
                behavioral=True,
            )

        # ── C. CCTV shifted-duplicate under uniform cadence ──
        cctv_files = [
            f for f in files
            if f.get("source_type") == "cctv" and f.get("detected_type") == "tabular"
        ]
        cctv_headers = {}
        for f in cctv_files:
            hdr = f.get("content", {}).get("columns") or (list(_rows(f)[0].keys()) if _rows(f) else [])
            cctv_headers[f["file_name"]] = tuple(hdr)
        for f in cctv_files:
            fname = f["file_name"]
            rows = _rows(f)
            hdr = cctv_headers.get(fname, ())
            partners = [g for g in cctv_files if g["file_name"] != fname and cctv_headers.get(g["file_name"]) == hdr]
            if not partners or len(rows) < 6:
                continue
            ts_all = [_parse_ts(r.get("timestamp")) for r in rows]
            ts_ok = [t for t in ts_all if t is not None]
            if len(ts_ok) < 6:
                continue
            gaps = [(ts_ok[i + 1] - ts_ok[i]).total_seconds() for i in range(len(ts_ok) - 1)]
            gap_counts = {}
            for g in gaps:
                gap_counts[g] = gap_counts.get(g, 0) + 1
            cadence = max(gap_counts.values()) / len(gaps)
            # 0.75 = at least 3/4 of consecutive gaps identical. The old 0.85
            # threshold left a 0.007 margin over the adversarial file's 6/7
            # cadence (0.857); 0.75 keeps the same pass/fail split with a
            # 0.107 margin over flagged files and 0.083 under clean-but-
            # duplicated ones (0.667).
            if cadence < 0.75:
                continue
            nonts_cols = [c for c in hdr if c and str(c).lower() != "timestamp"]
            shifted_from = None
            for r in rows:
                key = tuple(str(r.get(c, "") or "").strip() for c in nonts_cols)
                if not any(key):
                    continue
                tr = _parse_ts(r.get("timestamp"))
                if tr is None:
                    continue
                for g in partners:
                    for r2 in _rows(g):
                        key2 = tuple(str(r2.get(c, "") or "").strip() for c in nonts_cols)
                        if key != key2:
                            continue
                        t2 = _parse_ts(r2.get("timestamp"))
                        if t2 is not None and abs((tr - t2).total_seconds()) > 60:
                            shifted_from = g["file_name"]
                            break
                    if shifted_from:
                        break
                if shifted_from:
                    break
            if shifted_from:
                _add(
                    fname, 0.4,
                    f"timestamp_manipulation: uniform cadence ({cadence:.2f} equal intervals) with "
                    f"rows shifted-duplicated from {shifted_from} at different timestamps — "
                    f"fabricated CCTV timeline",
                    temporal=True,
                )

        # ── D. Social fake interaction (contact profile from another file) ──
        social_files = [
            f for f in files
            if f.get("source_type") == "social" and isinstance(f.get("content"), dict)
        ]
        profiles = {}  # username -> (created, posts, followers, owner_file)
        for f in social_files:
            content = f.get("content", {})
            # _parse_json wraps the raw document under "data"
            payload = content.get("data", content) if isinstance(content, dict) else {}
            if not isinstance(payload, dict):
                continue
            uname = str(payload.get("username", "") or "")
            prof = payload.get("profile", {}) or {}
            if uname:
                profiles[uname] = (
                    str(prof.get("account_created", "") or ""),
                    prof.get("posts_count", 0),
                    prof.get("followers", 0),
                    f["file_name"],
                )
        for f in social_files:
            content = f.get("content", {})
            payload = content.get("data", content) if isinstance(content, dict) else {}
            if not isinstance(payload, dict):
                continue
            threads = payload.get("direct_messages", []) or []
            for thread in threads:
                if not isinstance(thread, dict):
                    continue
                contact = str(thread.get("contact", "") or "")
                entry = profiles.get(contact)
                if not entry:
                    continue
                created_str, posts, followers, owner_file = entry
                if owner_file == f["file_name"] or not created_str:
                    continue
                if int(posts or 0) != 0 or int(followers or 0) != 0:
                    continue
                dates = [
                    d for d in (
                        str(m.get("date", "") or "") for m in thread.get("messages", []) or []
                        if isinstance(m, dict)
                    ) if d
                ]
                if not dates:
                    continue
                try:
                    first_msg = min(datetime.strptime(d, "%Y-%m-%d") for d in dates)
                    created = datetime.strptime(created_str, "%Y-%m-%d")
                except ValueError:
                    continue
                age_days = (first_msg - created).days
                if 0 <= age_days < 30:
                    _add(
                        f["file_name"], 0.4,
                        f"fake_interaction: DM contact '{contact}' profile (from {owner_file}) "
                        f"created {age_days} days before first message with 0 posts and 0 "
                        f"followers — fabricated interlocutor",
                        behavioral=True,
                    )

        # Mirror updated checks into the ingestion log's summary view
        for name, chk in self.adversarial_checks.items():
            for f in self.ingestion_log:
                if f.get("file_name") == name:
                    f["adversarial_check"] = chk.to_dict()

    def get_ingestion_summary(self) -> dict:
        """Summary of all ingested files with full metadata."""
        source_types = {}
        for f in self.ingestion_log:
            st = f.get("source_type", "unknown")
            source_types[st] = source_types.get(st, 0) + 1

        # Aggregate quality metrics
        quality_scores = [q.quality_score for q in self.quality_scores.values()]
        adversarial_suspicious = sum(1 for a in self.adversarial_checks.values() if a.is_suspicious)

        return {
            "total_files": len(self.ingestion_log),
            "total_records": sum(f.get("record_count", 0) for f in self.ingestion_log),
            "file_types": {
                f["detected_type"]: sum(1 for x in self.ingestion_log if x["detected_type"] == f["detected_type"])
                for f in self.ingestion_log
            },
            "source_types": source_types,
            "dependency_groups": len(self.dependency_groups),
            "integrity_records": len(self.integrity_records),
            "data_quality": {
                "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0,
                "min_quality_score": round(min(quality_scores), 3) if quality_scores else 0,
                "files_assessed": len(self.quality_scores),
            },
            "adversarial": {
                "total_checked": len(self.adversarial_checks),
                "suspicious_count": adversarial_suspicious,
                "suspicious_files": [
                    a.file_name for a in self.adversarial_checks.values() if a.is_suspicious
                ],
            },
            "run_id": self.current_run.run_id if self.current_run else None,
            "files": [
                {
                    "name": f["file_name"],
                    "type": f["detected_type"],
                    "source_type": f.get("source_type", "unknown"),
                    "records": f.get("record_count", 0),
                    "file_hash": f.get("file_hash", ""),
                    "reliability": f.get("reliability", {}),
                    "quality_score": f.get("data_quality", {}).get("quality_score", 0),
                    "adversarial_suspicious": f.get("adversarial_check", {}).get("is_suspicious", False),
                    "adversarial_check": f.get("adversarial_check", {}),
                }
                for f in self.ingestion_log
            ],
        }
