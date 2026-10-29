"""
AI Calling Interface
High-level interface that combines provider + prompts + storage.
All extraction modules call this — never the provider directly.
"""

import json
import time
from typing import Optional, Dict
from .provider import AIProvider, ModelTier
from .prompts.registry import PromptRegistry, PromptTemplate
from .storage import ResponseStore


class AICaller:
    """
    Main AI calling interface.
    Usage:
        ai = AICaller()
        result = ai.extract(task="extract_fir", text="FIR content...", source_file="01_FIR.txt")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        storage_dir: str = "output/llm_responses",
        default_tier: ModelTier = ModelTier.FAST,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        fallbacks: list[dict] | None = None,
    ):
        self.provider = AIProvider(config_path=config_path)
        self.prompts = PromptRegistry()
        self.storage = ResponseStore(base_dir=storage_dir)
        self.default_tier = default_tier
        self.model = model
        self.fallbacks = fallbacks or []
        self.preferred_provider = None
        if provider:
            from .provider import ProviderType
            try:
                self.preferred_provider = ProviderType(provider.lower())
            except ValueError as exc:
                raise ValueError(f"Unknown LLM provider '{provider}'") from exc
        self._last_call_time = 0.0
        self._min_delay = 2.0  # seconds between calls (Groq: 30 RPM = 2s minimum)
        self._recent_tokens = {}  # provider -> [(timestamp, tokens)] for per-provider tracking

    def extract(
        self,
        task: str,
        text: str,
        source_file: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        custom_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 0,  # 0 = auto-detect from model
    ) -> dict:
        """
        Run an extraction task on text.
        Returns: {"response": dict, "response_id": str, "provider": str, "status": str}
        """
        # Enforce minimum delay between calls to avoid 429s
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)

        # Get prompt template
        template = self.prompts.get(task)
        if not template and not custom_prompt:
            return {"error": f"Unknown task: {task}. Available: {self.prompts.list_tasks()}"}

        # Build prompts
        system_prompt = custom_prompt or template.system_prompt
        user_prompt = f"Analyze the following {task} data:\n\n{text}"

        # Call provider
        response = self.provider.call(
            prompt=user_prompt,
            system_prompt=system_prompt,
            tier=tier or self.default_tier,
            preferred_provider=self.preferred_provider,
            model=self.model,
            fallback_models=self.fallbacks,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse response
        parsed_response = self._parse_json_response(response.get("response", ""))

        # Save to storage
        response_id = self.storage.save_response(
            task=task,
            prompt_version=template.version if template else "custom",
            provider=response.get("provider", "unknown"),
            model=response.get("model", "unknown"),
            system_prompt=system_prompt,
            input_text=text,
            response_text=response.get("response", ""),
            usage=response.get("usage", {}),
            latency_ms=response.get("latency_ms", 0),
            status=response.get("status", "unknown"),
            source_file=source_file,
            metadata={"temperature": temperature, "max_tokens": max_tokens,
                      "error": response.get("error"),
                      "fallback_from": response.get("fallback_from")},
        )

        self._last_call_time = time.time()

        return {
            "parsed": parsed_response,
            "raw_response": response.get("response", ""),
            "response_id": response_id,
            "provider": response.get("provider", "unknown"),
            "model": response.get("model", "unknown"),
            "latency_ms": response.get("latency_ms", 0),
            "status": response.get("status", "unknown"),
            "error": response.get("error"),
            "fallback_from": response.get("fallback_from"),
            "task": task,
            "prompt_version": template.version if template else "custom",
        }

    def extract_fir(self, text: str, **kwargs) -> dict:
        """Extract from FIR document."""
        return self.extract(task="extract_fir", text=text, **kwargs)

    def extract_cdr(self, text: str, **kwargs) -> dict:
        """Extract from CDR data."""
        return self.extract(task="extract_cdr", text=text, **kwargs)

    def extract_bank(self, text: str, **kwargs) -> dict:
        """Extract from bank transaction data."""
        return self.extract(task="extract_bank", text=text, **kwargs)

    def extract_social_media(self, text: str, **kwargs) -> dict:
        """Extract from social media data."""
        return self.extract(task="extract_social_media", text=text, **kwargs)

    def extract_cctv(self, text: str, **kwargs) -> dict:
        """Extract from CCTV log data."""
        return self.extract(task="extract_cctv", text=text, **kwargs)

    def extract_device(self, text: str, **kwargs) -> dict:
        """Extract from device extraction data."""
        return self.extract(task="extract_device", text=text, **kwargs)

    def resolve_entities(self, text: str, **kwargs) -> dict:
        """Resolve entity identities."""
        return self.extract(task="resolve_entities", text=text, **kwargs)

    def classify_text(self, text: str, **kwargs) -> dict:
        """Classify what type of investigation document this is."""
        return self.extract(task="classify_document", text=text, **kwargs)

    def extract_relations(self, text: str, **kwargs) -> dict:
        """Extract relations between entities."""
        return self.extract(task="extract_relations", text=text, **kwargs)

    def critic_review(self, text: str, **kwargs) -> dict:
        """Run critic review on extraction results."""
        return self.extract(
            task="critic_review",
            text=text,
            tier=ModelTier.STRONG,
            **kwargs,
        )

    def batch_extract(
        self,
        task: str,
        texts: list,
        source_files: Optional[list] = None,
        tier: Optional[ModelTier] = None,
    ) -> list:
        """Run extraction on multiple texts."""
        results = []
        for i, text in enumerate(texts):
            source = source_files[i] if source_files and i < len(source_files) else None
            result = self.extract(task=task, text=text, source_file=source, tier=tier)
            results.append(result)
        return results

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        return self.storage.get_stats()

    def get_provider_status(self) -> dict:
        """Get provider status."""
        return self.provider.get_status()

    def review_responses(self, task: Optional[str] = None) -> list:
        """Get responses formatted for critic review."""
        return self.storage.export_for_critic(task=task)

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response with repair for common issues."""
        import re
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                return None

            raw_json = json_match.group()

            # Try direct parse first
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                pass

            # Repair common LLM JSON issues
            repaired = raw_json

            # Fix: bare keys like "identity_theft (Section...)" without quotes
            repaired = re.sub(r'(\n\s+)([a-zA-Z_]\w*(?:\s*\([^)]*\))?)\s*:', r'\1"\2":', repaired)

            # Fix: trailing commas before }
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

            # Fix: missing commas between items
            repaired = re.sub(r'"\s*\n\s*"', '",\n  "', repaired)

            # Fix: unquoted keys
            repaired = re.sub(r'(?<=[{,\n])\s*(\w+)\s*:', r' "\1":', repaired)

            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

            # Fix: truncated JSON — count open/close braces and brackets
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            if open_braces > 0 or open_brackets > 0:
                # Remove trailing incomplete value if any
                repaired = re.sub(r',\s*"[^"]*$', '', repaired.rstrip())
                repaired = re.sub(r',\s*\{[^}]*$', '', repaired.rstrip())
                # Close missing brackets/braces in order
                repaired += ']' * open_brackets + '}' * open_braces
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

            # Last resort: try to extract key-value pairs
            result = {}
            for match in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', repaired):
                result[match.group(1)] = match.group(2)
            for match in re.finditer(r'"(\w+)"\s*:\s*\{', repaired):
                key = match.group(1)
                start = match.end() - 1
                depth = 1
                for i, ch in enumerate(repaired[start+1:], start=start+1):
                    if ch == '{': depth += 1
                    elif ch == '}': depth -= 1
                    if depth == 0:
                        try:
                            result[key] = json.loads(repaired[start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break

            return result if result else None

        except Exception:
            return None
