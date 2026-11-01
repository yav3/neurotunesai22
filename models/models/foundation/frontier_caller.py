"""Real frontier LLM caller — connects GovernedFrontierBridge to Claude.

This module provides the actual Callable[[str], str|dict] that the
HybridFoundationModel.predict_and_explain() and
UnifiedFoundationFrontierModel.governed_inference() expect as `frontier_call`.

Two implementations:
    ClaudeFrontierCaller   — real Claude API via anthropic SDK
    MockFrontierCaller     — deterministic mock for testing / offline

The caller is deliberately stateless (no conversation memory). Each call
is a single structured query → single JSON response. The GovernedFrontierBridge
handles lineage, not the caller.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FrontierCallerConfig:
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 1024
    temperature: float = 0.2
    timeout: float = 30.0
    api_key_env: str = "ANTHROPIC_API_KEY"


class ClaudeFrontierCaller:
    """Real Claude API caller for frontier reasoning.

    Usage:
        caller = ClaudeFrontierCaller()
        model = HybridFoundationModel(...)
        answer = model.predict_and_explain(..., frontier_call=caller)
    """

    def __init__(self, config: FrontierCallerConfig | None = None):
        self.config = config or FrontierCallerConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic SDK required: pip install anthropic"
                )
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Set {self.config.api_key_env} environment variable"
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def __call__(self, prompt: str) -> dict:
        """Send a governed query to Claude and return parsed JSON."""
        client = self._get_client()
        t0 = time.time()

        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text
        latency_ms = (time.time() - t0) * 1000

        parsed = self._extract_json(raw_text)
        parsed["_latency_ms"] = latency_ms
        parsed["_model"] = self.config.model
        parsed["_input_tokens"] = response.usage.input_tokens
        parsed["_output_tokens"] = response.usage.output_tokens

        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from Claude's response, handling markdown fences."""
        fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence:
            text = fence.group(1)

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {
                "reasoning": text.strip()[:500],
                "confidence": 0.1,
                "evidence_citations": [],
                "suggested_experiments": [],
                "caveats": ["Response was not valid JSON; raw text preserved"],
                "_raw": text,
            }


class MockFrontierCaller:
    """Deterministic mock for testing without API access.

    Returns plausible structured responses based on the query content.
    """

    def __init__(self, confidence: float = 0.7):
        self.confidence = confidence
        self.call_count = 0

    def __call__(self, prompt: str) -> dict:
        self.call_count += 1

        n_up = 0
        n_down = 0
        m_up = re.search(r"(\d+)\s+upregulated", prompt)
        if m_up:
            n_up = int(m_up.group(1))
        m_down = re.search(r"(\d+)\s+downregulated", prompt)
        if m_down:
            n_down = int(m_down.group(1))

        genes = re.findall(r"([A-Z][A-Z0-9]{1,10}):", prompt)[:5]

        return {
            "reasoning": (
                f"The foundation model identifies {n_up} upregulated and "
                f"{n_down} downregulated genes. Top candidates: "
                f"{', '.join(genes) if genes else 'analysis pending'}. "
                f"Pattern consistent with perturbation response."
            ),
            "confidence": self.confidence,
            "evidence_citations": ["PMID:00000000"],
            "suggested_experiments": [
                "Validate top-3 genes with qRT-PCR",
                "Western blot for protein-level confirmation",
            ],
            "caveats": [
                "Computational prediction only",
                "Requires experimental validation",
            ],
        }
