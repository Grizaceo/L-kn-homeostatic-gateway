"""
L-kn Homeostatic Logic
Rule-based and entropy-based mode selection strategies.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class HomeostaticMode:
    """Homeostatic operating modes."""

    FLUIDO = "FLUIDO"
    ANALITICO = "ANALITICO"


class DecisionStrategy:
    """Available decision strategies."""

    RULES = "rules"
    ENTROPY = "entropy"


class ProbeSignals(BaseModel):
    """Cheap request-level signals for rule-based routing."""

    prompt_tokens_est: int
    has_code: bool
    has_math: bool
    has_multi_step: bool
    session_turn_count: int


class HomeostaticConfig(BaseModel):
    """Configuration for homeostatic decision system."""

    decision_strategy: str = Field(default=DecisionStrategy.RULES, pattern="^(rules|entropy)$")
    entropy_threshold: float = 0.6
    probe_top_k: int = 10
    max_tokens_fluido: int = 150
    analytic_system_prompt: str = "Think carefully, verify your assumptions, and reason step-by-step before answering."


class HomeostaticDecision(BaseModel):
    """Result of homeostatic decision."""

    mode: str
    entropy_norm: Optional[float]
    intervention_applied: bool
    rationale: str
    decision_strategy: str
    probe_signals: Optional[ProbeSignals] = None


class HomeostaticSystem:
    """Homeostatic decision and intervention system."""

    _CODE_PATTERN = re.compile(
        r"```|`[^`]+`|\b(def|class|import|function|SELECT|FROM|WHERE|JOIN)\b",
        re.IGNORECASE,
    )
    _MATH_PATTERN = re.compile(
        r"[=+\-*/^]|\b(ecuaci|equation|integr|deriv|matrix|probab|teorema|theorem)\w*\b",
        re.IGNORECASE,
    )
    _MULTI_STEP_PATTERN = re.compile(
        r"paso a paso|step by step|primero.+luego|analiza.+compara|multi.?hop|desglosa",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, config: HomeostaticConfig, engine_client):
        self.config = config
        self.engine_client = engine_client

    def _latest_user_message(self, messages: List[Dict[str, str]]) -> str:
        """Return the latest user message content."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def extract_signals(self, messages: List[Dict[str, str]]) -> ProbeSignals:
        """Extract cheap routing signals without calling the model."""
        user_message = self._latest_user_message(messages)
        token_estimate = int(round(len(user_message.split()) * 1.3))
        session_turn_count = sum(1 for msg in messages if msg.get("role") == "user")

        return ProbeSignals(
            prompt_tokens_est=token_estimate,
            has_code=bool(self._CODE_PATTERN.search(user_message)),
            has_math=bool(self._MATH_PATTERN.search(user_message)),
            has_multi_step=bool(self._MULTI_STEP_PATTERN.search(user_message)),
            session_turn_count=session_turn_count,
        )

    def decide_mode_from_signals(self, signals: ProbeSignals) -> Tuple[str, str]:
        """Deterministic baseline routing rules (MVP v0)."""
        if signals.prompt_tokens_est > self.config.max_tokens_fluido:
            return (
                HomeostaticMode.ANALITICO,
                f"Prompt too long ({signals.prompt_tokens_est} > {self.config.max_tokens_fluido})",
            )

        if signals.has_code:
            return (HomeostaticMode.ANALITICO, "Code pattern detected")

        if signals.has_math:
            return (HomeostaticMode.ANALITICO, "Math pattern detected")

        if signals.has_multi_step:
            return (HomeostaticMode.ANALITICO, "Multi-step task detected")

        return (HomeostaticMode.FLUIDO, "No high-risk signal detected")

    def calculate_entropy(self, logprobs: List[Tuple[str, float]]) -> float:
        """
        Calculate normalized Shannon entropy from logprobs.

        Args:
            logprobs: List of (token, logprob) tuples

        Returns:
            Normalized entropy in [0, 1]
        """
        if not logprobs:
            logger.warning("empty_logprobs")
            return 0.5

        logprob_values = np.array([lp for _, lp in logprobs])
        probs = np.exp(logprob_values)
        probs = probs / probs.sum()

        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        k = len(logprobs)
        max_entropy = math.log2(k) if k > 1 else 1.0
        entropy_norm = entropy / max_entropy

        return float(np.clip(entropy_norm, 0.0, 1.0))

    async def probe_entropy(self, messages: List[Dict[str, str]]) -> Optional[float]:
        """
        Probe engine to estimate entropy of next token distribution.

        Uses native /generate endpoint with logprobs.
        """
        if not messages:
            logger.warning("probe_empty_messages")
            return None

        if self.engine_client is None:
            logger.warning("probe_missing_engine_client")
            return None

        try:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            prompt = "\n".join(prompt_parts) + "\nAssistant:"

            sampling_params = {
                "max_new_tokens": 1,
                "temperature": 1.0,
                "return_logprob": True,
                "top_logprobs_num": self.config.probe_top_k,
            }

            response = await self.engine_client.generate(prompt, sampling_params)

            if "meta_info" in response and "output_top_logprobs" in response["meta_info"]:
                top_logprobs = response["meta_info"]["output_top_logprobs"]
                if top_logprobs:
                    first_token_logprobs = top_logprobs[0]
                    logprobs_list = [(token, lp) for token, lp in first_token_logprobs.items()]
                    entropy_norm = self.calculate_entropy(logprobs_list)
                    logger.info("probe_completed", entropy_norm=round(entropy_norm, 3))
                    return entropy_norm

            logger.warning("probe_schema_mismatch", response_keys=list(response.keys()))
            return None

        except Exception as e:
            from utils import classify_engine_error

            category, reason = classify_engine_error(e)
            logger.error("probe_failed", failure_reason=reason, category=category, error=str(e))
            return None

    async def decide_mode(self, messages: List[Dict[str, str]]) -> HomeostaticDecision:
        """
        Decide operating mode based on configured strategy.
        """
        if self.config.decision_strategy == DecisionStrategy.RULES:
            signals = self.extract_signals(messages)
            mode, rationale = self.decide_mode_from_signals(signals)
            logger.info("lkn_mode_decision", strategy=DecisionStrategy.RULES, mode=mode, rationale=rationale)
            return HomeostaticDecision(
                mode=mode,
                entropy_norm=None,
                intervention_applied=(mode == HomeostaticMode.ANALITICO),
                rationale=rationale,
                decision_strategy=DecisionStrategy.RULES,
                probe_signals=signals,
            )

        entropy_norm = await self.probe_entropy(messages)
        if entropy_norm is None:
            # In entropy mode, fallback to deterministic rules instead of blind FLUIDO.
            signals = self.extract_signals(messages)
            mode, rationale = self.decide_mode_from_signals(signals)
            logger.warning(
                "lkn_mode_decision_fallback",
                strategy=DecisionStrategy.ENTROPY,
                fallback_strategy=DecisionStrategy.RULES,
                mode=mode,
                reason="probe_failed_or_schema_mismatch",
            )
            return HomeostaticDecision(
                mode=mode,
                entropy_norm=None,
                intervention_applied=(mode == HomeostaticMode.ANALITICO),
                rationale=f"Entropy probe failed; {rationale}",
                decision_strategy="entropy_fallback_rules",
                probe_signals=signals,
            )

        if entropy_norm < self.config.entropy_threshold:
            logger.info(
                "lkn_mode_decision",
                strategy=DecisionStrategy.ENTROPY,
                mode=HomeostaticMode.FLUIDO,
                entropy_norm=round(entropy_norm, 3),
            )
            return HomeostaticDecision(
                mode=HomeostaticMode.FLUIDO,
                entropy_norm=entropy_norm,
                intervention_applied=False,
                rationale=f"Low entropy ({entropy_norm:.3f} < {self.config.entropy_threshold})",
                decision_strategy=DecisionStrategy.ENTROPY,
            )

        logger.info(
            "lkn_mode_decision",
            strategy=DecisionStrategy.ENTROPY,
            mode=HomeostaticMode.ANALITICO,
            entropy_norm=round(entropy_norm, 3),
        )
        return HomeostaticDecision(
            mode=HomeostaticMode.ANALITICO,
            entropy_norm=entropy_norm,
            intervention_applied=True,
            rationale=f"High entropy ({entropy_norm:.3f} >= {self.config.entropy_threshold})",
            decision_strategy=DecisionStrategy.ENTROPY,
        )

    def apply_intervention(
        self, messages: List[Dict[str, str]], decision: HomeostaticDecision
    ) -> List[Dict[str, str]]:
        """
        Apply intervention based on mode decision.

        For ANALITICO mode, inject verification system prompt.
        """
        if decision.mode != HomeostaticMode.ANALITICO or not messages:
            return messages

        has_system = any(msg.get("role") == "system" for msg in messages)

        if has_system:
            modified_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    modified_msg = msg.copy()
                    modified_msg["content"] = f"{self.config.analytic_system_prompt}\n\n{msg['content']}"
                    modified_messages.append(modified_msg)
                else:
                    modified_messages.append(msg)
            return modified_messages

        system_msg = {"role": "system", "content": self.config.analytic_system_prompt}
        return [system_msg] + messages
