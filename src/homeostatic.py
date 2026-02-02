"""
L-kn Homeostatic Logic
Entropy-based mode selection and intervention strategies.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class HomeostaticMode:
    """Homeostatic operating modes."""
    FLUIDO = "FLUIDO"
    ANALITICO = "ANALITICO"


class HomeostaticConfig(BaseModel):
    """Configuration for homeostatic decision system."""
    entropy_threshold: float = 0.6
    probe_top_k: int = 10
    analytic_system_prompt: str = "Think carefully, verify your assumptions, and reason step-by-step before answering."


class HomeostaticDecision(BaseModel):
    """Result of homeostatic decision."""
    mode: str
    entropy_norm: Optional[float]
    intervention_applied: bool
    rationale: str


class HomeostaticSystem:
    """Homeostatic decision and intervention system."""
    
    def __init__(self, config: HomeostaticConfig, engine_client):
        self.config = config
        self.engine_client = engine_client
    
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
            return 0.5  # Default to medium entropy
        
        # Convert logprobs to probabilities
        logprob_values = np.array([lp for _, lp in logprobs])
        probs = np.exp(logprob_values)
        
        # Renormalize (in case top-k doesn't sum to 1)
        probs = probs / probs.sum()
        
        # Calculate Shannon entropy: H = -Σ p·log(p)
        # Use log2 for bits
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by max possible entropy for K items: log2(K)
        k = len(logprobs)
        max_entropy = math.log2(k) if k > 1 else 1.0
        entropy_norm = entropy / max_entropy
        
        return float(np.clip(entropy_norm, 0.0, 1.0))
    
    async def probe_entropy(self, messages: List[Dict[str, str]]) -> Optional[float]:
        """
        Probe engine to estimate entropy of next token distribution.
        
        Uses native /generate endpoint with logprobs.
        
        Args:
            messages: Conversation history
        
        Returns:
            Normalized entropy or None if probe fails
        """
        if not messages:
            logger.warning("probe_empty_messages")
            return None
        
        try:
            # Convert messages to single prompt (simple concatenation)
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
            
            # Call /generate with logprobs enabled
            sampling_params = {
                "max_new_tokens": 1,
                "temperature": 1.0,  # Don't bias distribution
                "return_logprob": True,
                "top_logprobs_num": self.config.probe_top_k
            }
            
            response = await self.engine_client.generate(prompt, sampling_params)
            
            # Extract logprobs from response
            # Expected SGLang format: response["meta_info"]["output_top_logprobs"]
            # This is a list of dicts, one per generated token
            # Each dict maps token strings to their logprob values
            if "meta_info" in response and "output_top_logprobs" in response["meta_info"]:
                top_logprobs = response["meta_info"]["output_top_logprobs"]
                if top_logprobs and len(top_logprobs) > 0:
                    # First token's top logprobs
                    first_token_logprobs = top_logprobs[0]
                    # Convert dict to list of tuples
                    logprobs_list = [(token, lp) for token, lp in first_token_logprobs.items()]
                    
                    entropy_norm = self.calculate_entropy(logprobs_list)
                    logger.info("probe_completed", entropy_norm=round(entropy_norm, 3))
                    return entropy_norm
            
            logger.warning("probe_no_logprobs", response_keys=list(response.keys()))
            return None
        
        except Exception as e:
            logger.error("probe_failed", error=str(e), error_type=type(e).__name__)
            return None
    
    async def decide_mode(
        self,
        messages: List[Dict[str, str]]
    ) -> HomeostaticDecision:
        """
        Decide operating mode based on entropy probe.
        
        Args:
            messages: Conversation messages
        
        Returns:
            HomeostaticDecision with mode and metadata
        """
        # Probe entropy
        entropy_norm = await self.probe_entropy(messages)
        
        # Fallback to FLUIDO if probe fails
        if entropy_norm is None:
            logger.warning("lkn_mode_decision", mode=HomeostaticMode.FLUIDO, reason="probe_failed")
            return HomeostaticDecision(
                mode=HomeostaticMode.FLUIDO,
                entropy_norm=None,
                intervention_applied=False,
                rationale="Probe failed, defaulting to FLUIDO mode"
            )
        
        # Decision logic
        if entropy_norm < self.config.entropy_threshold:
            # Low entropy: model is confident
            logger.info("lkn_mode_decision", mode=HomeostaticMode.FLUIDO, entropy_norm=round(entropy_norm, 3))
            return HomeostaticDecision(
                mode=HomeostaticMode.FLUIDO,
                entropy_norm=entropy_norm,
                intervention_applied=False,
                rationale=f"Low entropy ({entropy_norm:.3f} < {self.config.entropy_threshold})"
            )
        
        else:
            # High entropy: model is uncertain
            logger.info("lkn_mode_decision", mode=HomeostaticMode.ANALITICO, entropy_norm=round(entropy_norm, 3))
            return HomeostaticDecision(
                mode=HomeostaticMode.ANALITICO,
                entropy_norm=entropy_norm,
                intervention_applied=True,
                rationale=f"High entropy ({entropy_norm:.3f} >= {self.config.entropy_threshold})"
            )
    
    def apply_intervention(
        self,
        messages: List[Dict[str, str]],
        decision: HomeostaticDecision
    ) -> List[Dict[str, str]]:
        """
        Apply intervention based on mode decision.
        
        For ANALÍTICO mode, inject verification system prompt.
        
        Args:
            messages: Original messages
            decision: Mode decision
        
        Returns:
            Modified messages
        """
        if decision.mode != HomeostaticMode.ANALITICO or not messages:
            return messages
        
        # Check if system message already exists
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        if has_system:
            # Prepend to existing system message
            modified_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    modified_msg = msg.copy()
                    modified_msg["content"] = f"{self.config.analytic_system_prompt}\n\n{msg['content']}"
                    modified_messages.append(modified_msg)
                else:
                    modified_messages.append(msg)
            return modified_messages
        
        else:
            # Inject new system message at the beginning
            system_msg = {
                "role": "system",
                "content": self.config.analytic_system_prompt
            }
            return [system_msg] + messages
