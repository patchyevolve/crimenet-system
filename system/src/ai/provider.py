"""
AI Provider Subsystem
Handles provider selection, model routing, rate limiting, and API key management.
All AI calls go through this — no direct provider calls anywhere else.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import requests


class ProviderType(Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    SILICONFLOW = "siliconflow"
    GOOGLE = "google"
    MISTRAL = "mistral"


class ModelTier(Enum):
    FAST = "fast"         # Quick extraction, high volume
    BALANCED = "balanced" # Good quality, moderate speed
    STRONG = "strong"     # Best quality, slower


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    name: ProviderType
    api_url: str
    api_key_env: str          # env var name for the key
    models: Dict[str, str]    # tier -> model_id
    rate_limit_rpm: int       # requests per minute
    rate_limit_rpd: int       # requests per day
    cost_per_1k_tokens: float
    is_free: bool
    priority: int             # lower = higher priority
    rate_limit_tpm: int = 999999  # tokens per minute
    model_limits: Dict[str, int] = field(default_factory=dict)  # model_id -> max_tokens

    def get_max_tokens(self, model: str) -> int:
        """Get max_tokens for a specific model. Falls back to 4096."""
        return self.model_limits.get(model, 4096)


@dataclass
class RateLimiter:
    """Token bucket rate limiter per provider."""
    rpm_limit: int
    rpd_limit: int
    tpm_limit: int = 999999  # tokens per minute
    requests_this_minute: List[float] = field(default_factory=list)
    tokens_this_minute: List[tuple] = field(default_factory=list)  # (timestamp, tokens)
    requests_today: int = 0
    last_day_reset: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def can_request(self) -> bool:
        with self._lock:
            now = time.time()

            # Reset daily counter
            if now - self.last_day_reset > 86400:
                self.requests_today = 0
                self.last_day_reset = now

            # Clean old minute requests
            self.requests_this_minute = [t for t in self.requests_this_minute if now - t < 60]
            self.tokens_this_minute = [(t, tok) for t, tok in self.tokens_this_minute if now - t < 60]

            if len(self.requests_this_minute) >= self.rpm_limit:
                return False
            if self.requests_today >= self.rpd_limit:
                return False
            if sum(tok for _, tok in self.tokens_this_minute) >= self.tpm_limit:
                return False

            return True

    def record_request(self, tokens: int = 0):
        with self._lock:
            now = time.time()
            self.requests_this_minute.append(now)
            self.tokens_this_minute.append((now, tokens))
            self.requests_today += 1

    def wait_time(self) -> float:
        """Seconds until next request allowed."""
        with self._lock:
            now = time.time()
            if self.requests_today >= self.rpd_limit:
                return 86400 - (now - self.last_day_reset)
            self.requests_this_minute = [t for t in self.requests_this_minute if now - t < 60]
            self.tokens_this_minute = [(t, tok) for t, tok in self.tokens_this_minute if now - t < 60]
            if len(self.requests_this_minute) >= self.rpm_limit:
                oldest = min(self.requests_this_minute)
                return 60 - (now - oldest) + 0.1
            total_tokens = sum(tok for _, tok in self.tokens_this_minute)
            if total_tokens >= self.tpm_limit:
                oldest_token_time = min(t for t, _ in self.tokens_this_minute)
                return 60 - (now - oldest_token_time) + 0.1
            return 0

    def stats(self):
        with self._lock:
            now = time.time()
            self.requests_this_minute = [t for t in self.requests_this_minute if now - t < 60]
            self.tokens_this_minute = [(t, tok) for t, tok in self.tokens_this_minute if now - t < 60]
            return {
                "requests_this_minute": len(self.requests_this_minute),
                "rpm_limit": self.rpm_limit,
                "tokens_this_minute": sum(tok for _, tok in self.tokens_this_minute),
                "tpm_limit": self.tpm_limit,
                "requests_today": self.requests_today,
                "rpd_limit": self.rpd_limit,
            }


# ──────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────

PROVIDERS: List[ProviderConfig] = [
    ProviderConfig(
        name=ProviderType.OLLAMA,
        api_url="http://localhost:11434",
        api_key_env="",
        models={
            ModelTier.FAST.value: "llama3.1:8b",
            ModelTier.BALANCED.value: "llama3.1:8b",
            ModelTier.STRONG.value: "llama3.1:70b",
        },
        rate_limit_rpm=999,
        rate_limit_rpd=9999,
        cost_per_1k_tokens=0.0,
        is_free=True,
        priority=1,
        model_limits={
            "llama3.1:8b": 8192,
            "llama3.1:70b": 8192,
        },
    ),
    ProviderConfig(
        name=ProviderType.GROQ,
        api_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        models={
            ModelTier.FAST.value: "openai/gpt-oss-20b",
            ModelTier.BALANCED.value: "openai/gpt-oss-20b",
            ModelTier.STRONG.value: "openai/gpt-oss-20b",
        },
        rate_limit_rpm=30,
        rate_limit_rpd=7000,
        rate_limit_tpm=12000,
        cost_per_1k_tokens=0.0003,
        is_free=False,
        priority=3,  # Lower priority than Mistral
        model_limits={
            "openai/gpt-oss-20b": 8192,
            "openai/gpt-oss-120b": 8192,
        },
    ),
    ProviderConfig(
        name=ProviderType.SILICONFLOW,
        api_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        models={
            ModelTier.FAST.value: "Qwen/Qwen3-8B",
            ModelTier.BALANCED.value: "Qwen/Qwen2.5-72B-Instruct",
            ModelTier.STRONG.value: "Qwen/Qwen2.5-72B-Instruct",
        },
        rate_limit_rpm=60,
        rate_limit_rpd=10000,
        cost_per_1k_tokens=0.0,
        is_free=True,
        priority=3,
        model_limits={
            "Qwen/Qwen3-8B": 8192,
            "Qwen/Qwen2.5-72B-Instruct": 32768,
        },
    ),
    ProviderConfig(
        name=ProviderType.GOOGLE,
        api_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GOOGLE_API_KEY",
        models={
            ModelTier.FAST.value: "gemini-2.0-flash",
            ModelTier.BALANCED.value: "gemini-2.0-flash",
            ModelTier.STRONG.value: "gemini-2.5-pro",
        },
        rate_limit_rpm=15,
        rate_limit_rpd=1500,
        cost_per_1k_tokens=0.0,
        is_free=True,
        priority=4,
        model_limits={
            "gemini-2.0-flash": 8192,
            "gemini-2.5-pro": 65536,
        },
    ),
    ProviderConfig(
        name=ProviderType.MISTRAL,
        api_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        models={
            ModelTier.FAST.value: "mistral-small-latest",
            ModelTier.BALANCED.value: "mistral-medium-latest",
            ModelTier.STRONG.value: "mistral-large-latest",
        },
        rate_limit_rpm=30,
        rate_limit_rpd=7000,
        rate_limit_tpm=1000000,
        # Mistral API calls are metered. The legacy scalar is the current
        # Small output rate per 1K tokens; actual cost depends on input/output
        # mix and the selected model.
        cost_per_1k_tokens=0.0006,
        is_free=False,
        priority=2,  # Higher priority than Groq
        model_limits={
            "mistral-small-latest": 32768,
            "mistral-medium-latest": 32768,
            "mistral-large-latest": 131072,
        },
    ),
    ProviderConfig(
        name=ProviderType.OPENROUTER,
        api_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        models={
            ModelTier.FAST.value: "meta-llama/llama-3.1-8b-instruct:free",
            ModelTier.BALANCED.value: "qwen/qwen-2.5-72b-instruct:free",
            ModelTier.STRONG.value: "deepseek/deepseek-r1:free",
        },
        rate_limit_rpm=20,
        rate_limit_rpd=200,
        cost_per_1k_tokens=0.0,
        is_free=True,
        priority=6,
        model_limits={
            "meta-llama/llama-3.1-8b-instruct:free": 8192,
            "qwen/qwen-2.5-72b-instruct:free": 32768,
            "deepseek/deepseek-r1:free": 16384,
        },
    ),
]


class AIProvider:
    """
    Main AI calling interface.
    Usage:
        ai = AIProvider(config_path="config/ai_providers.json")
        response = ai.call(
            prompt="Extract entities from this text",
            system_prompt="You are an entity extractor...",
            tier=ModelTier.FAST,
        )
    """

    def __init__(self, config_path: Optional[str] = None):
        self.providers = PROVIDERS.copy()
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.config_path = config_path

        # Load .env file if present
        self._load_env_file()

        # Load config overrides
        if config_path and Path(config_path).exists():
            self._load_config(config_path)

        # Init rate limiters
        for p in self.providers:
            self.rate_limiters[p.name.value] = RateLimiter(
                rpm_limit=p.rate_limit_rpm,
                rpd_limit=p.rate_limit_rpd,
                tpm_limit=p.rate_limit_tpm,
            )

        # Load API keys from env
        self._api_keys: Dict[str, str] = {}
        for p in self.providers:
            if p.api_key_env:
                key = os.environ.get(p.api_key_env, "")
                if key:
                    self._api_keys[p.name.value] = key

    def _load_env_file(self):
        """Load .env file from parent directories."""
        # Look for .env in this file's directory and up to 2 parent directories
        current = Path(__file__).parent
        for _ in range(3):
            env_file = current / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            if val.strip() and key.strip() not in os.environ:
                                os.environ[key.strip()] = val.strip()
                break
            current = current.parent

    def _load_config(self, path: str):
        """Load provider config overrides from JSON with URL validation."""
        from urllib.parse import urlparse
        with open(path) as f:
            config = json.load(f)

        ALLOWED_HOSTS = {"localhost", "127.0.0.1", "api.groq.com", "api.siliconflow.cn",
                         "openrouter.ai", "generativelanguage.googleapis.com", "api.mistral.ai"}

        for p_config in config.get("providers", []):
            # Find existing provider and update
            for p in self.providers:
                if p.name.value == p_config.get("name"):
                    if "api_url" in p_config:
                        parsed = urlparse(p_config["api_url"])
                        if parsed.hostname not in ALLOWED_HOSTS:
                            print(f"[AI] WARNING: Blocked potentially unsafe URL: {p_config['api_url']}")
                            continue
                        p.api_url = p_config["api_url"]
                    if "models" in p_config:
                        p.models.update(p_config["models"])
                    if "rate_limit_rpm" in p_config:
                        p.rate_limit_rpm = p_config["rate_limit_rpm"]
                    if "rate_limit_rpd" in p_config:
                        p.rate_limit_rpd = p_config["rate_limit_rpd"]
                    if "rate_limit_tpm" in p_config:
                        p.rate_limit_tpm = p_config["rate_limit_tpm"]
                    if "model_limits" in p_config:
                        p.model_limits.update(p_config["model_limits"])

        # Honor the configured provider order rather than silently using the
        # hard-coded priorities above.
        priority = config.get("default_provider_priority", [])
        for rank, provider_name in enumerate(priority):
            for provider in self.providers:
                if provider.name.value == provider_name:
                    provider.priority = rank

    def call(
        self,
        prompt: str,
        system_prompt: str = "",
        tier: ModelTier = ModelTier.FAST,
        preferred_provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        fallback_models: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 0,  # 0 = auto-detect from model
        timeout: int = 120,
        retries: int = 3,
    ) -> dict:
        """
        Make an AI call. Auto-selects provider based on availability and tier.
        max_tokens=0 means use the model's default limit (from model_limits).
        Returns: {"response": str, "provider": str, "model": str, "usage": dict, "latency_ms": int}
        """
        # Select the exact primary route, then append only explicitly configured
        # fallbacks. This keeps pipeline routing predictable and auditable.
        provider = self._select_provider(tier, preferred_provider, model=model)
        if not provider:
            reason = f"Configured model unavailable: {model}" if model else "No available providers"
            return {"error": reason, "response": "", "status": "model_unavailable" if model else "no_providers"}
        model = model or provider.models.get(tier.value, next(iter(provider.models.values())))
        auto_max_tokens = max_tokens <= 0
        routes = [(provider, model)]
        for fallback in fallback_models or []:
            try:
                fallback_provider = ProviderType(str(fallback["provider"]).lower())
                fallback_model = str(fallback["model"])
            except (KeyError, ValueError, TypeError):
                continue
            selected = self._select_provider(tier, fallback_provider, model=fallback_model)
            if selected and selected.name == fallback_provider:
                routes.append((selected, fallback_model))

        last_error = None
        previous_route = None
        for route_provider, route_model in routes:
            route_error = None
            limiter = self.rate_limiters[route_provider.name.value]
            for attempt in range(retries + 1):
                if not limiter.can_request():
                    wait = limiter.wait_time()
                    if wait > 30:
                        route_error = "Provider rate limit exceeded"
                        break
                    time.sleep(min(wait, 3))

                token_limit = route_provider.get_max_tokens(route_model) if auto_max_tokens else max_tokens
                start_time = time.time()
                try:
                    result = self._call_provider(
                        route_provider, route_model, prompt, system_prompt,
                        temperature, token_limit, timeout,
                    )
                    limiter.record_request(tokens=len(prompt) // 4 + 500)
                    response = {
                        "response": result.get("response", ""),
                        "provider": route_provider.name.value,
                        "model": route_model,
                        "usage": result.get("usage", {}),
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "status": "success",
                        "attempt": attempt + 1,
                    }
                    if previous_route:
                        response["fallback_from"] = previous_route
                    return response
                except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                    route_error = f"{type(exc).__name__}: {exc}"
                    is_rate_limited = (
                        "429" in str(exc)
                        or "Too Many Requests" in str(exc)
                        or "rate" in str(exc).lower()
                    )
                    if is_rate_limited or attempt >= retries:
                        break
                    time.sleep(1)

            last_error = route_error or "Provider call failed"
            previous_route = {
                "provider": route_provider.name.value,
                "model": route_model,
                "error": last_error,
            }

        status = "rate_limited" if last_error and "rate limit" in last_error.lower() else "failed"
        return {
            "error": last_error,
            "response": "",
            "provider": routes[-1][0].name.value,
            "model": routes[-1][1],
            "status": status,
            "fallback_from": previous_route if len(routes) > 1 else None,
        }

    def _call_provider(
        self,
        provider: ProviderConfig,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> dict:
        """Route call to the right provider implementation."""
        if provider.name == ProviderType.OLLAMA:
            return self._call_ollama(provider, model, prompt, system_prompt, temperature, timeout)
        elif provider.name == ProviderType.GOOGLE:
            return self._call_google(provider, model, prompt, system_prompt, temperature, max_tokens, timeout)
        else:
            return self._call_openai_compatible(provider, model, prompt, system_prompt, temperature, max_tokens, timeout)

    def _call_ollama(self, provider, model, prompt, system_prompt, temperature, timeout) -> dict:
        import requests
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = requests.post(
            f"{provider.api_url}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"response": data.get("response", ""), "usage": {"eval_count": data.get("eval_count", 0)}}

    def _call_openai_compatible(self, provider, model, prompt, system_prompt, temperature, max_tokens, timeout) -> dict:
        import requests
        api_key = self._api_keys.get(provider.name.value, "")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if provider.name == ProviderType.GROQ and model.startswith("openai/gpt-oss-"):
            # GPT-OSS spends part of its completion budget on internal reasoning.
            # Keep that bounded and request JSON for this pipeline's structured tasks.
            request_body.update({
                "reasoning_effort": "low",
                "include_reasoning": False,
                "response_format": {"type": "json_object"},
            })

        resp = requests.post(
            f"{provider.api_url}/chat/completions",
            json=request_body,
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code != 200:
            error_detail = resp.text[:500] if resp.text else "no response body"
            raise requests.HTTPError(f"HTTP {resp.status_code}: {error_detail}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = str(data)
        return {
            "response": content,
            "usage": data.get("usage", {}),
        }

    def _call_google(self, provider, model, prompt, system_prompt, temperature, max_tokens, timeout) -> dict:
        import requests
        api_key = self._api_keys.get(provider.name.value, "")
        url = f"{provider.api_url}/models/{model}:generateContent"

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        resp = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"response": text, "usage": data.get("usageMetadata", {})}

    def _select_provider(
        self,
        tier: ModelTier,
        preferred: Optional[ProviderType] = None,
        exclude: Optional[List[ProviderType]] = None,
        model: Optional[str] = None,
    ) -> Optional[ProviderConfig]:
        """Select best available provider based on rate limits and priority."""
        exclude_names = [e.value for e in (exclude or [])]

        candidates = []
        for p in self.providers:
            if model is not None and model not in p.models.values():
                continue
            if p.name.value in exclude_names:
                continue
            # Check if API key is available
            has_key = bool(self._api_keys.get(p.name.value)) or not p.api_key_env
            if not has_key:
                continue
            # Skip Ollama if not running
            if p.name == ProviderType.OLLAMA and not self._is_ollama_running():
                continue
            limiter = self.rate_limiters[p.name.value]
            if limiter.can_request():
                candidates.append(p)

        if not candidates:
            return None

        # Prefer specific provider if available
        if preferred:
            for p in candidates:
                if p.name == preferred:
                    return p

        # Sort by priority
        candidates.sort(key=lambda p: p.priority)
        return candidates[0]

    def _is_ollama_running(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> dict:
        """Get status of all providers including model limits."""
        status = {}
        for p in self.providers:
            limiter = self.rate_limiters[p.name.value]
            has_key = bool(self._api_keys.get(p.name.value)) or not p.api_key_env
            status[p.name.value] = {
                "available": has_key and limiter.can_request() and (
                    p.name != ProviderType.OLLAMA or self._is_ollama_running()
                ),
                # This is local readiness only. We deliberately do not make a
                # billable provider request just to render the status screen.
                "remote_checked": False,
                "availability_scope": "credentials_and_local_limits_only",
                "has_api_key": has_key,
                "rate_limits": limiter.stats(),
                "is_free": p.is_free,
                "priority": p.priority,
                "models": {
                    tier: {
                        "model_id": model_id,
                        "max_tokens": p.get_max_tokens(model_id),
                    }
                    for tier, model_id in p.models.items()
                },
            }
        return status

    def get_model_limits(self, tier: ModelTier = ModelTier.FAST) -> dict:
        """Get max_tokens for the best available model at a given tier."""
        provider = self._select_provider(tier)
        if not provider:
            return {"max_tokens": 4096, "provider": "none", "model": "none"}
        model = provider.models.get(tier.value, list(provider.models.values())[0])
        return {
            "max_tokens": provider.get_max_tokens(model),
            "provider": provider.name.value,
            "model": model,
        }

    def wait_for_available(self, tier: ModelTier = ModelTier.FAST, max_wait: int = 60):
        """Block until a provider is available."""
        start = time.time()
        while time.time() - start < max_wait:
            provider = self._select_provider(tier)
            if provider:
                return provider
            time.sleep(1)
        return None
