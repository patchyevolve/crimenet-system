"""
Versioned Response Storage
Stores every LLM call and response for:
  1. Audit trail
  2. Critic review later
  3. Re-analysis with updated prompts
  4. Debugging and quality control
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict


@dataclass
class ResponseRecord:
    """One record of an LLM call."""
    response_id: str
    task: str
    prompt_version: str
    provider: str
    model: str
    system_prompt_hash: str
    input_hash: str
    response_text: str
    usage: dict
    latency_ms: int
    status: str
    timestamp: str
    source_file: Optional[str] = None
    extraction_id: Optional[str] = None
    metadata: dict = None

    def to_dict(self):
        return asdict(self)


class ResponseStore:
    """
    Stores LLM responses with versioning.
    Structure:
      output/llm_responses/
        {task}/
          {response_id}.json    # Full response record
        _index.json             # Quick lookup index
    """

    def __init__(self, base_dir: str = "output/llm_responses"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "_index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"responses": {}, "by_task": {}, "by_file": {}, "total": 0}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def save_response(
        self,
        task: str,
        prompt_version: str,
        provider: str,
        model: str,
        system_prompt: str,
        input_text: str,
        response_text: str,
        usage: dict,
        latency_ms: int,
        status: str,
        source_file: Optional[str] = None,
        extraction_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Save an LLM response. Returns response_id."""
        # Generate unique ID
        content = f"{task}:{input_text[:200]}:{datetime.now().isoformat()}"
        response_id = f"resp_{hashlib.md5(content.encode()).hexdigest()[:12]}"

        record = ResponseRecord(
            response_id=response_id,
            task=task,
            prompt_version=prompt_version,
            provider=provider,
            model=model,
            system_prompt_hash=hashlib.sha256(system_prompt.encode()).hexdigest()[:16],
            input_hash=hashlib.sha256(input_text.encode()).hexdigest()[:16],
            response_text=response_text,
            usage=usage,
            latency_ms=latency_ms,
            status=status,
            timestamp=datetime.now().isoformat(),
            source_file=source_file,
            extraction_id=extraction_id,
            metadata=metadata or {},
        )

        # Save to task directory
        task_dir = self.base_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        record_path = task_dir / f"{response_id}.json"
        with open(record_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        # Update index
        self.index["responses"][response_id] = {
            "task": task,
            "timestamp": record.timestamp,
            "provider": provider,
            "model": model,
            "status": status,
            "source_file": source_file,
        }

        if task not in self.index["by_task"]:
            self.index["by_task"][task] = []
        self.index["by_task"][task].append(response_id)

        if source_file:
            if source_file not in self.index["by_file"]:
                self.index["by_file"][source_file] = []
            self.index["by_file"][source_file].append(response_id)

        self.index["total"] += 1
        self._save_index()

        return response_id

    def get_response(self, response_id: str) -> Optional[dict]:
        """Get a specific response by ID."""
        info = self.index["responses"].get(response_id)
        if not info:
            return None

        record_path = self.base_dir / info["task"] / f"{response_id}.json"
        if record_path.exists():
            with open(record_path) as f:
                return json.load(f)
        return None

    def get_by_task(self, task: str) -> List[dict]:
        """Get all responses for a task."""
        response_ids = self.index["by_task"].get(task, [])
        results = []
        for rid in response_ids:
            resp = self.get_response(rid)
            if resp:
                results.append(resp)
        return results

    def get_by_file(self, source_file: str) -> List[dict]:
        """Get all responses for a source file."""
        response_ids = self.index["by_file"].get(source_file, [])
        results = []
        for rid in response_ids:
            resp = self.get_response(rid)
            if resp:
                results.append(resp)
        return results

    def get_stats(self) -> dict:
        """Get storage statistics."""
        task_counts = {}
        for task, ids in self.index["by_task"].items():
            task_counts[task] = len(ids)

        return {
            "total_responses": self.index["total"],
            "by_task": task_counts,
            "by_provider": self._count_by_field("provider"),
            "by_status": self._count_by_field("status"),
        }

    def _count_by_field(self, field: str) -> dict:
        counts = {}
        for resp_info in self.index["responses"].values():
            val = resp_info.get(field, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts

    def search(self, query: str, task: Optional[str] = None) -> List[dict]:
        """Search responses by content."""
        results = []
        search_dir = self.base_dir / task if task else self.base_dir

        for record_path in search_dir.rglob("*.json"):
            if record_path.name == "_index.json":
                continue
            with open(record_path) as f:
                record = json.load(f)
                if query.lower() in record.get("response_text", "").lower():
                    results.append(record)

        return results

    def export_for_critic(self, task: Optional[str] = None) -> List[dict]:
        """Export responses formatted for critic review."""
        if task:
            responses = self.get_by_task(task)
        else:
            responses = []
            for resp_id in self.index["responses"]:
                resp = self.get_response(resp_id)
                if resp:
                    responses.append(resp)

        return [
            {
                "response_id": r["response_id"],
                "task": r["task"],
                "prompt_version": r["prompt_version"],
                "provider": r["provider"],
                "model": r["model"],
                "response_preview": r["response_text"][:500],
                "status": r["status"],
                "latency_ms": r["latency_ms"],
                "timestamp": r["timestamp"],
                "source_file": r.get("source_file"),
            }
            for r in responses
        ]
