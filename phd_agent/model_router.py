"""Configurable model selection with explicit, auditable escalation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


TIERS = ("LUNA", "TERRA", "SOL")
TASK_TIERS: Mapping[str, str] = {
    "PAGE_EXTRACTION": "LUNA",
    "NORMALIZATION": "LUNA",
    "PAGE_CLASSIFICATION": "LUNA",
    "REQUIREMENT_EXTRACTION": "LUNA",
    "CONTEXT_RETRIEVAL": "LUNA",
    "PUBLICATION_CLASSIFICATION": "LUNA",
    "REPLY_CLASSIFICATION": "LUNA",
    "FORM_FIELD_MAPPING": "LUNA",
    "SUMMARY": "LUNA",
    "DISCOVERY_PLANNING": "TERRA",
    "PROGRAMME_TRIAGE": "TERRA",
    "FACULTY_SYNTHESIS": "TERRA",
    "OVERLAP_ANALYSIS": "TERRA",
    "ELIGIBILITY_INTERPRETATION": "TERRA",
    "PROFESSOR_EMAIL": "TERRA",
    "APPLICATION_ANSWER": "TERRA",
    "AMBIGUOUS_EXTRACTION": "TERRA",
    "FINAL_RESEARCH_FIT": "SOL",
    "TAILORED_SOP": "SOL",
    "RESEARCH_PROPOSAL": "SOL",
    "RESEARCH_CV": "SOL",
    "NUANCED_ALIGNMENT": "SOL",
    "CONFLICTING_REQUIREMENTS": "SOL",
    "FINAL_FACTUAL_REVIEW": "SOL",
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "openai"
    luna: str = "gpt-5.6-luna"
    terra: str = "gpt-5.6-terra"
    sol: str = "gpt-5.6-sol"

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> "ModelConfig":
        env = os.environ if environ is None else environ
        return cls(
            provider=(env.get("PHD_AGENT_MODEL_PROVIDER") or "openai").strip(),
            luna=(env.get("PHD_AGENT_MODEL_LUNA") or "gpt-5.6-luna").strip(),
            terra=(env.get("PHD_AGENT_MODEL_TERRA") or "gpt-5.6-terra").strip(),
            sol=(env.get("PHD_AGENT_MODEL_SOL") or "gpt-5.6-sol").strip(),
        )

    def model_for(self, tier: str) -> str:
        tier = tier.upper()
        if tier not in TIERS:
            raise ValueError("Unknown model tier")
        return getattr(self, tier.casefold())


@dataclass(frozen=True)
class RouteDecision:
    task_type: str
    tier: str
    provider: str
    model: str
    reasons: tuple[str, ...]
    policy_version: str = "model-router-v1"


class ModelRouter:
    """Choose a configured model. Safety and permission decisions stay outside this router."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig.from_environ()

    def route(self, task_type: str, *, confidence: float = 1.0,
              conflicting_evidence: bool = False, unsupported_output: bool = False,
              operator_tier: str | None = None) -> RouteDecision:
        task = task_type.strip().upper()
        if task not in TASK_TIERS:
            raise ValueError("Task type has no reviewed model route")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        tier = TASK_TIERS[task]
        reasons = [f"default:{tier}"]
        if confidence < 0.65 and tier == "LUNA":
            tier = "TERRA"
            reasons.append("low_confidence")
        if (conflicting_evidence or unsupported_output) and tier != "SOL":
            tier = "SOL" if tier == "TERRA" or confidence < 0.4 else "TERRA"
            reasons.append("conflict" if conflicting_evidence else "unsupported_output")
        if operator_tier:
            requested = operator_tier.strip().upper()
            if requested not in TIERS:
                raise ValueError("Operator model tier must be Luna, Terra, or Sol")
            if TIERS.index(requested) < TIERS.index(tier):
                raise ValueError("Operator override cannot reduce a reviewed task risk tier")
            tier = requested
            reasons.append("operator_request")
        return RouteDecision(task, tier, self.config.provider, self.config.model_for(tier), tuple(reasons))
