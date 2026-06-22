"""MADARA: Model-Adaptive Document Assessment Routing Architecture.

A training-free pipeline that diagnoses a model-task pair with the
Reasoning-Score Coupling (RSC) probe and the No-Filter baseline, then routes it
to the cheapest treatment that works: PDE, SDA, CoT De-Polarization, or ATF.
"""

from .metrics import exact_match, token_f1
from .model_client import MockModelClient, ModelClient
from .router import MADARARouter, RoutingDecision
from .rsc import RSCProbe, RSCResult, score_entropy
from .treatments import (
    atf_filter,
    pde_answer,
    sda_select,
    three_agent_scores,
)

__all__ = [
    "MADARARouter",
    "RoutingDecision",
    "RSCProbe",
    "RSCResult",
    "score_entropy",
    "ModelClient",
    "MockModelClient",
    "three_agent_scores",
    "sda_select",
    "atf_filter",
    "pde_answer",
    "exact_match",
    "token_f1",
]

__version__ = "0.1.0"
