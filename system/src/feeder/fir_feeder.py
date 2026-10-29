"""
FIR Feeder — Multi-FIR input organizer for the pipeline.

Reads a FIR manifest and organizes evidence files by FIR.
Supports two input modes:
  1. Manifest-based: JSON manifest maps files to FIRs
  2. Directory-based: Subdirectories per FIR (future)

The feeder prepares file groups for the pipeline, maintaining FIR provenance.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FIRContext:
    """Context for a single FIR — its metadata and evidence files."""
    fir_id: str
    fir_number: str
    jurisdiction_node_id: str
    description: str = ""
    filed_by_id: str = ""
    investigation_status: str = "REGISTERED"
    files: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)  # Resolved absolute paths


@dataclass
class FeedResult:
    """Result of organizing files for pipeline input."""
    case_id: str
    jurisdiction_node_id: str
    firs: List[FIRContext]
    all_file_paths: List[str]  # All files across all FIRs
    fir_file_map: Dict[str, List[str]]  # fir_id -> [file_paths]
    skip_files: List[str]


class FIRFeeder:
    """
    Organizes evidence files by FIR for pipeline consumption.

    Usage:
        feeder = FIRFeeder(manifest_path="FIR_MANIFEST.json")
        result = feeder.load()

        # Feed to pipeline
        pipeline = Pipeline(
            case_id=result.case_id,
            jurisdiction_node_id=result.jurisdiction_node_id,
            fir_feeder=result,
        )
    """

    SKIP_EXTENSIONS = {".zip", ".tar", ".gz", ".env", ".py", ".js", ".ts", ".lock", ".log"}

    def __init__(
        self,
        manifest_path: str | None = None,
        data_dir: str | None = None,
        fir_ids: List[str] | None = None,
    ):
        """
        Args:
            manifest_path: Path to FIR_MANIFEST.json
            data_dir: Base directory containing evidence files
            fir_ids: If set, only process these FIRs (subset selection)
        """
        self.manifest_path = manifest_path
        self.data_dir = Path(data_dir) if data_dir else None
        self.fir_ids_filter = fir_ids
        self._manifest: dict | None = None

    def load(self) -> FeedResult:
        """Load manifest and resolve file paths."""
        if self.manifest_path:
            return self._load_from_manifest()
        elif self.data_dir:
            return self._load_from_directory()
        else:
            raise ValueError("Either manifest_path or data_dir must be provided")

    def _load_from_manifest(self) -> FeedResult:
        """Load FIR structure from a manifest JSON file."""
        manifest_path = Path(self.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        case_id = self._manifest.get("case_id", "")
        jurisdiction_node_id = self._manifest.get("jurisdiction_node_id", "")
        skip_files = set(self._manifest.get("skip_files", []))

        # Determine data directory (where the actual files are)
        if self.data_dir:
            data_dir = self.data_dir
        else:
            # Default: manifest's parent directory
            data_dir = manifest_path.parent

        firs = []
        all_file_paths = []
        fir_file_map = {}

        for fir_data in self._manifest.get("firs", []):
            fir_id = fir_data.get("fir_id", "")

            # Apply FIR filter if specified
            if self.fir_ids_filter and fir_id not in self.fir_ids_filter:
                continue

            fir_ctx = FIRContext(
                fir_id=fir_id,
                fir_number=fir_data.get("fir_number", ""),
                jurisdiction_node_id=fir_data.get("jurisdiction_node_id", jurisdiction_node_id),
                description=fir_data.get("description", ""),
                filed_by_id=fir_data.get("filed_by_id", ""),
                investigation_status=fir_data.get("investigation_status", "REGISTERED"),
            )

            # Resolve file paths
            for fname in fir_data.get("files", []):
                if fname in skip_files:
                    continue
                fpath = data_dir / fname
                if fpath.exists():
                    fir_ctx.files.append(fname)
                    fir_ctx.file_paths.append(str(fpath.absolute()))
                    all_file_paths.append(str(fpath.absolute()))
                else:
                    print(f"[FEEDER] WARNING: File not found: {fpath}")

            fir_file_map[fir_id] = fir_ctx.file_paths
            firs.append(fir_ctx)

        print(f"[FEEDER] Loaded manifest: {len(firs)} FIRs, {len(all_file_paths)} files")
        for fir in firs:
            print(f"[FEEDER]   {fir.fir_id} ({fir.fir_number}): {len(fir.files)} files")

        return FeedResult(
            case_id=case_id,
            jurisdiction_node_id=jurisdiction_node_id,
            firs=firs,
            all_file_paths=all_file_paths,
            fir_file_map=fir_file_map,
            skip_files=list(skip_files),
        )

    def _load_from_directory(self) -> FeedResult:
        """
        Load from directory structure.

        Expected layout:
          data_dir/
            FIR_001/
              evidence_file1.csv
              evidence_file2.txt
            FIR_002/
              evidence_file3.csv

        Or flat directory (treated as single FIR):
          data_dir/
            evidence_file1.csv
            evidence_file2.txt
        """
        data_dir = self.data_dir
        if not data_dir or not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Check for subdirectories (multi-FIR layout)
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Multi-FIR layout: each subdirectory is a FIR
            firs = []
            all_file_paths = []
            fir_file_map = {}

            for subdir in sorted(subdirs):
                fir_id = subdir.name
                if self.fir_ids_filter and fir_id not in self.fir_ids_filter:
                    continue

                fir_ctx = FIRContext(
                    fir_id=fir_id,
                    fir_number=fir_id,
                    jurisdiction_node_id="",
                )

                for fpath in sorted(subdir.rglob("*")):
                    if fpath.is_file() and fpath.suffix.lower() not in self.SKIP_EXTENSIONS:
                        fir_ctx.files.append(fpath.name)
                        fir_ctx.file_paths.append(str(fpath.absolute()))
                        all_file_paths.append(str(fpath.absolute()))

                fir_file_map[fir_id] = fir_ctx.file_paths
                firs.append(fir_ctx)

            return FeedResult(
                case_id="",
                jurisdiction_node_id="",
                firs=firs,
                all_file_paths=all_file_paths,
                fir_file_map=fir_file_map,
                skip_files=[],
            )
        else:
            # Flat directory: treat as single FIR
            fir_id = data_dir.name
            fir_ctx = FIRContext(
                fir_id=fir_id,
                fir_number=fir_id,
                jurisdiction_node_id="",
            )

            for fpath in sorted(data_dir.iterdir()):
                if fpath.is_file() and fpath.suffix.lower() not in self.SKIP_EXTENSIONS:
                    fir_ctx.files.append(fpath.name)
                    fir_ctx.file_paths.append(str(fpath.absolute()))

            return FeedResult(
                case_id="",
                jurisdiction_node_id="",
                firs=[fir_ctx],
                all_file_paths=fir_ctx.file_paths,
                fir_file_map={fir_id: fir_ctx.file_paths},
                skip_files=[],
            )

    def get_fir_files(self, fir_id: str) -> List[str]:
        """Get file paths for a specific FIR."""
        if not self._manifest:
            raise ValueError("Must call load() first")

        for fir_data in self._manifest.get("firs", []):
            if fir_data.get("fir_id") == fir_id:
                return fir_data.get("files", [])
        return []

    def get_selected_firs(self, fir_ids: List[str]) -> FeedResult:
        """Get a subset of FIRs by ID. Reuses loaded manifest."""
        if not self._manifest:
            raise ValueError("Must call load() first")

        original_filter = self.fir_ids_filter
        self.fir_ids_filter = fir_ids
        result = self._load_from_manifest()
        self.fir_ids_filter = original_filter
        return result
