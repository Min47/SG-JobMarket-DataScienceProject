"""Model Gateway for Multi-Provider LLM Support.

This module provides a unified interface for multiple LLM providers with:
- Automatic fallback chains (Vertex AI Gemini → Ollama)
- Cost tracking per request
- Retry logic with exponential backoff
- Request timeout handling
- Usage metrics and logging

Architecture:
    User Request → ModelGateway → [Provider Selection] → LLM API → Response
                        ↓ (on failure)
                   Fallback Provider → Retry

Configuration:
    Set preferred provider via Settings or environment:
    - PREFERRED_PROVIDER="vertexai" (default, cloud-based Gemini)
    - PREFERRED_PROVIDER="ollama" (local Llama 3.1)
    
    Toggle priority with one line change in ModelGateway.__init__:
    self.provider_priority = ["vertexai", "ollama"]  # Vertex AI first
    self.provider_priority = ["ollama", "vertexai"]  # Ollama first

Supported Providers:
    1. Vertex AI Gemini 2.5 Flash (cloud, $0.075/1M input, production-ready)
    2. Ollama Llama 3.1 (local, free, development/testing)

Usage:
    >>> gateway = ModelGateway()
    >>> response = gateway.generate("Rewrite: ML jobs", model="auto")
    >>> print(f"Cost: ${response.cost:.4f}")
"""

from __future__ import annotations

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime, timezone

# Provider-specific imports (lazy loaded to avoid dependency issues)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

try:
    import httpx
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from utils.config import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class GenerationResult:
    """Result from LLM generation with metadata.
    
    Attributes:
        text: Generated text response
        provider: Provider name ("gemini", "ollama")
        model: Specific model used ("gemini-2.5-flash", "qwen3-vl:4b", etc.)
        tokens_input: Number of input tokens consumed
        tokens_output: Number of output tokens generated
        cost: Estimated cost in USD
        latency_ms: Request latency in milliseconds
        timestamp: When request was made
        metadata: Additional provider-specific data
    """
    text: str
    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    cost: float
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class GenerationConfig:
    """Configuration for LLM generation.
    
    Attributes:
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum output tokens
        top_p: Nucleus sampling threshold
        top_k: Top-K sampling (Gemini only)
        stop_sequences: Sequences that stop generation
        timeout_seconds: Request timeout
    """
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    timeout_seconds: int = 60


# =============================================================================
# Abstract Base Provider
# =============================================================================

class BaseProvider(ABC):
    """Abstract base class for all LLM providers.
    
    All providers must implement:
    - generate(): Send prompt and get response
    - is_available(): Check if provider is configured and accessible
    - estimate_cost(): Calculate cost based on token usage
    
    This ensures consistent interface across all providers.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize provider with configuration.
        
        Args:
            settings: Application settings (GCP project, API keys, etc.)
        """
        self.settings = settings or Settings.load()
        self.name: str = "base"  # Override in subclasses
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from prompt.
        
        Args:
            prompt: User prompt/instruction
            config: Generation parameters (temperature, max tokens, etc.)
            
        Returns:
            GenerationResult with text, tokens, cost, latency
            
        Raises:
            RuntimeError: If generation fails
            TimeoutError: If request exceeds timeout
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured.
        
        Returns:
            True if provider can handle requests
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost in USD for given token usage.
        
        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        pass


# =============================================================================
# Model Gateway (Orchestrator)
# =============================================================================

class ModelGateway:
    """Unified gateway for multiple LLM providers with automatic fallback.
    
    This class:
    1. Routes requests to preferred provider
    2. Falls back to alternative providers on failure
    3. Tracks costs and usage per provider
    4. Handles retries with exponential backoff
    5. Logs all requests for observability
    
    Example:
        >>> gateway = ModelGateway()
        >>> result = gateway.generate(
        ...     "Rewrite: ML engineer jobs",
        ...     model="auto",  # Auto-select best available
        ...     fallback=True  # Enable fallback chain
        ... )
        >>> print(f"Provider: {result.provider}, Cost: ${result.cost:.4f}")
    """
    
    # Cost tracking (class-level for all instances)
    _total_costs: Dict[str, float] = {}
    _request_counts: Dict[str, int] = {}
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize gateway with providers.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings.load()
        
        # Initialize providers (lazy loading)
        self.providers: Dict[str, BaseProvider] = {}
        self._init_providers()
        
        # ===================================================================
        # PROVIDER PRIORITY CONFIGURATION (Change this line to toggle)
        # ===================================================================
        # Option 1: Vertex AI first (cloud Gemini 2.5 Flash, production)
        self.provider_priority = ["vertexai", "ollama"]
        
        # Option 2: Ollama first (local Llama 3.1, development) - uncomment:
        # self.provider_priority = ["ollama", "vertexai"]
        
        logger.info(
            f"[ModelGateway] Initialized with {len(self.providers)} providers: "
            f"{list(self.providers.keys())}"
        )
    
    def _init_providers(self):
        """Initialize all available providers."""
        # Vertex AI (Gemini 2.5 Flash)
        if VERTEXAI_AVAILABLE:
            try:
                provider = VertexAIProvider(self.settings)
                if provider.is_available():
                    self.providers["vertexai"] = provider
                    self.providers["gemini"] = provider  # Backward compatibility
                    logger.info("[ModelGateway] Vertex AI (Gemini 2.5 Flash) enabled")
            except Exception as e:
                logger.warning(f"[ModelGateway] Vertex AI init failed: {e}")
        
        # Ollama (Llama 3.1 local)
        if OLLAMA_AVAILABLE:
            try:
                provider = OllamaProvider(self.settings)
                if provider.is_available():
                    self.providers["ollama"] = provider
                    logger.info("[ModelGateway] Ollama (Llama 3.1 local) enabled")
            except Exception as e:
                logger.debug(f"[ModelGateway] Ollama not available: {e}")
        
        if not self.providers:
            logger.warning(
                "[ModelGateway] No providers available! "
                "Install: pip install google-cloud-aiplatform httpx"
            )
    
    def generate(
        self,
        prompt: str,
        model: Literal["auto", "vertexai", "gemini", "ollama"] = "auto",
        config: Optional[GenerationConfig] = None,
        fallback: bool = True,
        max_retries: int = 2,
    ) -> GenerationResult:
        """Generate text using specified or auto-selected provider.
        
        Args:
            prompt: User prompt/instruction
            model: Provider to use ("auto" for priority-based selection)
            config: Generation parameters
            fallback: Enable automatic fallback on failure
            max_retries: Number of retry attempts per provider
            
        Returns:
            GenerationResult with text and metadata
            
        Raises:
            RuntimeError: If all providers fail
        """
        config = config or GenerationConfig()
        
        # Determine provider order
        if model == "auto":
            providers_to_try = self.provider_priority
        else:
            # Specific provider requested
            if model not in self.providers:
                raise ValueError(
                    f"Provider '{model}' not available. "
                    f"Available: {list(self.providers.keys())}"
                )
            providers_to_try = [model] if not fallback else [model] + [
                p for p in self.provider_priority if p != model
            ]
        
        # Try providers in order
        errors = []
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"[ModelGateway] Trying {provider_name} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    
                    result = provider.generate(prompt, config)
                    
                    # Track usage
                    self._track_usage(provider_name, result.cost)
                    
                    logger.info(
                        f"[ModelGateway] Success: {provider_name}, "
                        f"tokens={result.tokens_input}+{result.tokens_output}, "
                        f"cost=${result.cost:.4f}, latency={result.latency_ms:.0f}ms"
                    )
                    
                    return result
                    
                except Exception as e:
                    error_msg = f"{provider_name} attempt {attempt + 1} failed: {e}"
                    errors.append(error_msg)
                    logger.warning(f"[ModelGateway] {error_msg}")
                    
                    # Exponential backoff
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        logger.info(f"[ModelGateway] Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
        
        # All providers failed
        error_summary = "\n".join(errors)
        raise RuntimeError(
            f"All providers failed after retries:\n{error_summary}"
        )
    
    def _track_usage(self, provider: str, cost: float):
        """Track usage statistics per provider."""
        if provider not in self._total_costs:
            self._total_costs[provider] = 0.0
            self._request_counts[provider] = 0
        
        self._total_costs[provider] += cost
        self._request_counts[provider] += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics across all providers.
        
        Returns:
            Dict with costs and request counts per provider
        """
        return {
            "total_cost_usd": sum(self._total_costs.values()),
            "total_requests": sum(self._request_counts.values()),
            "by_provider": {
                provider: {
                    "cost_usd": self._total_costs.get(provider, 0.0),
                    "requests": self._request_counts.get(provider, 0),
                    "avg_cost_per_request": (
                        self._total_costs.get(provider, 0.0) / 
                        self._request_counts.get(provider, 1)
                    )
                }
                for provider in self.providers.keys()
            }
        }


# =============================================================================
# Vertex AI Gemini Provider
# =============================================================================

class VertexAIProvider(BaseProvider):
    """Vertex AI Gemini provider (GCP native).
    
    Models supported:
    - gemini-2.5-flash: Fast, cheap ($0.075/1M input, $0.30/1M output)
    
    Advantages:
    - GCP native (no extra auth needed)
    - Low latency (asia-southeast1)
    - Generous free tier (1500 req/day)
    - Multimodal support (text, images, video)
    
    Configuration:
    - Requires: GCP_PROJECT_ID, GCP_REGION in settings
    - IAM: Service account needs roles/aiplatform.user
    """
    
    # Pricing per 1M tokens (as of Dec 2024)
    PRICING = {
        "gemini-2.5-flash": {
            "input": 0.075,   # $0.075 per 1M input tokens
            "output": 0.30,   # $0.30 per 1M output tokens
        }
    }
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Vertex AI provider.
        
        Args:
            settings: Application settings with GCP config
        """
        super().__init__(settings)
        self.name = "gemini"
        self.default_model = "gemini-2.5-flash"
        
        # Initialize Vertex AI SDK
        if VERTEXAI_AVAILABLE:
            try:
                vertexai.init(
                    project=self.settings.gcp_project_id,
                    location=self.settings.gcp_region
                )
                logger.info(
                    f"[VertexAIProvider] Initialized: "
                    f"project={self.settings.gcp_project_id}, "
                    f"region={self.settings.gcp_region}"
                )
            except Exception as e:
                logger.error(f"[VertexAIProvider] Init failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Vertex AI is configured and accessible.
        
        Returns:
            True if SDK installed and GCP configured
        """
        if not VERTEXAI_AVAILABLE:
            return False
        
        # Check required settings
        if not self.settings.gcp_project_id or not self.settings.gcp_region:
            logger.warning(
                "[VertexAIProvider] Missing GCP_PROJECT_ID or GCP_REGION"
            )
            return False
        
        return True
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using Vertex AI Gemini.
        
        Args:
            prompt: User prompt
            config: Generation parameters
            
        Returns:
            GenerationResult with text and metadata
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_available():
            raise RuntimeError("VertexAI provider not available")
        
        config = config or GenerationConfig()
        start_time = time.time()
        
        try:
            # Initialize model
            model = GenerativeModel(self.default_model)
            
            # Build generation config
            generation_config = {
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "top_p": config.top_p,
            }
            
            if config.top_k:
                generation_config["top_k"] = config.top_k
            
            if config.stop_sequences:
                generation_config["stop_sequences"] = config.stop_sequences
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            # Extract text
            if not response.text:
                raise RuntimeError(
                    "Empty response from Gemini. "
                    "Possible causes: safety filters, empty input"
                )
            
            text = response.text.strip()
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens (Gemini doesn't return token counts directly)
            # Rough estimate: 1 token ≈ 4 characters
            tokens_input = len(prompt) // 4
            tokens_output = len(text) // 4
            
            # Calculate cost
            cost = self.estimate_cost(tokens_input, tokens_output)
            
            logger.info(
                f"[VertexAIProvider] Generated {len(text)} chars, "
                f"~{tokens_output} tokens, ${cost:.6f}, {latency_ms:.0f}ms"
            )
            
            return GenerationResult(
                text=text,
                provider=self.name,
                model=self.default_model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "safety_ratings": str(response.candidates[0].safety_ratings)
                    if response.candidates else None,
                }
            )
            
        except Exception as e:
            logger.error(f"[VertexAIProvider] Generation failed: {e}")
            raise RuntimeError(f"Vertex AI generation failed: {e}")
    
    def estimate_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost for Gemini 2.5 Flash.
        
        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            
        Returns:
            Cost in USD
        """
        pricing = self.PRICING[self.default_model]
        
        input_cost = (tokens_input / 1_000_000) * pricing["input"]
        output_cost = (tokens_output / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost


# =============================================================================
# Ollama Provider (Local LLM)
# =============================================================================

class OllamaProvider(BaseProvider):
    """Ollama local LLM provider (free, private).
    
    Models supported (must be installed via `ollama pull`):
    - qwen3-vl:4b
    - deepseek-r1:8b
    
    Advantages:
    - Free (no API costs)
    - Private (data never leaves machine)
    - No rate limits
    - Good for development/testing
    
    Disadvantages:
    - Requires local installation: https://ollama.ai/download
    - Slower than cloud APIs (depends on hardware)
    - Lower quality than GPT-4/Gemini Pro
    - Requires GPU for good performance
    
    Configuration:
    - Install Ollama: https://ollama.ai/download
    - Pull model: `ollama pull llama3.1`
    - Start server: `ollama serve` (runs on http://localhost:11434)
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Ollama provider.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.name = "ollama"
        self.default_model = "qwen3-vl:4b"
        self.base_url = "http://localhost:11434"
    
    def is_available(self) -> bool:
        """Check if Ollama server is running locally.
        
        Returns:
            True if Ollama server responds
        """
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            # Quick health check
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using Ollama local LLM.
        
        Args:
            prompt: User prompt
            config: Generation parameters
            
        Returns:
            GenerationResult with text and metadata
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_available():
            raise RuntimeError(
                "Ollama server not available. "
                "Install: https://ollama.ai/download, then run: ollama serve"
            )
        
        config = config or GenerationConfig()
        start_time = time.time()
        
        try:
            # Build request
            payload = {
                "model": self.default_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "top_p": config.top_p,
                }
            }
            
            if config.stop_sequences:
                payload["options"]["stop"] = config.stop_sequences
            
            # Generate response
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=config.timeout_seconds,
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract text
            text = data.get("response", "").strip()
            
            if not text:
                raise RuntimeError("Empty response from Ollama")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token counts (Ollama provides these)
            tokens_input = data.get("prompt_eval_count", len(prompt) // 4)
            tokens_output = data.get("eval_count", len(text) // 4)
            
            # Cost is $0 (local)
            cost = 0.0
            
            logger.info(
                f"[OllamaProvider] Generated {len(text)} chars, "
                f"{tokens_output} tokens, FREE, {latency_ms:.0f}ms"
            )
            
            return GenerationResult(
                text=text,
                provider=self.name,
                model=data.get("model", self.default_model),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "load_duration_ms": data.get("load_duration", 0) / 1_000_000,
                    "total_duration_ms": data.get("total_duration", 0) / 1_000_000,
                }
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[OllamaProvider] HTTP error: {e}")
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("[OllamaProvider] Request timeout")
            raise RuntimeError("Ollama request timeout")
        except Exception as e:
            logger.error(f"[OllamaProvider] Generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def estimate_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost for Ollama (always $0).
        
        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            
        Returns:
            Always 0.0 (local, no API costs)
        """
        return 0.0
