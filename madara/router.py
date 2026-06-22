"""The MADARA routing protocol (paper Section 4.5 and Appendix E).

Phase 1 runs a one-time RSC probe and a No-Filter (NF) baseline on a small
calibration set to pick a single treatment for the model-task pair:

    if EM_NF < tau_NF:          mode = PDE     (weak baseline -> isolation)
    elif rho_star > -1.0:       mode = SDA     (stochastic scoring)
    elif rho_1 < 0.5:           mode = CoT     (weakly coupled, quality-ordered)
    else:                       mode = ATF     (strongly coupled, quality-ordered)

Phase 2 applies the chosen treatment to every incoming query. The thresholds
(tau_NF = 30%, rho_star = -1.0, rho_1 = 0.5) were fixed on a Mistral-7B pilot
and applied zero-shot to every other model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import exact_match
from .model_client import ModelClient
from .rsc import RSCProbe, RSCResult
from .treatments import (
    aggregate,
    atf_filter,
    generate,
    pde_answer,
    rerank_topk,
    sda_select,
    three_agent_scores,
)

TAU_NF = 0.30  # No-Filter baseline accuracy below which a model is "weak"


@dataclass
class RoutingDecision:
    mode: str               # "PDE" | "SDA" | "CoT" | "ATF"
    em_nf: float
    rsc: RSCResult
    reason: str


class MADARARouter:
    """Model-Adaptive Document Assessment Routing Architecture."""

    def __init__(
        self,
        client: ModelClient,
        k: int = 5,
        tau_nf: float = TAU_NF,
        seed: int = 42,
    ):
        self.client = client
        self.k = k
        self.tau_nf = tau_nf
        self.probe = RSCProbe(seed=seed)
        self.decision: RoutingDecision | None = None

    # ----- Phase 1: one-time calibration -------------------------------------
    def calibrate(self, calibration: list[dict], task: str = "factoid",
                  model_name: str = "model") -> RoutingDecision:
        """Profile the model on a calibration set and select a treatment.

        Each calibration example is ``{"question", "documents": [...], "answer": ...}``.
        The RSC probe is label-free; only the NF baseline reads the gold answer.
        """
        rsc = self.probe.probe(self.client, calibration, task=task, model_name=model_name)
        em_nf = self._eval_nf(calibration, task)

        if em_nf < self.tau_nf:
            mode, reason = "PDE", f"NF {em_nf:.0%} < tau_NF {self.tau_nf:.0%}: weak baseline -> isolation"
        elif rsc.classification == "stochastic":
            mode, reason = "SDA", "stochastic scoring -> distribution alignment"
        elif rsc.shuffled_rho < self.probe.strength_threshold:
            mode, reason = "CoT", f"quality-ordered, weakly coupled (rho_1={rsc.shuffled_rho} < 0.5)"
        else:
            mode, reason = "ATF", f"quality-ordered, strongly coupled (rho_1={rsc.shuffled_rho} >= 0.5)"

        self.decision = RoutingDecision(mode=mode, em_nf=round(em_nf, 4), rsc=rsc, reason=reason)
        return self.decision

    # ----- Phase 2: per-query assessment -------------------------------------
    def answer(self, question: str, documents: list[str], task: str = "factoid") -> str:
        """Answer one query using the calibrated treatment."""
        if self.decision is None:
            raise RuntimeError("Call calibrate() before answer().")
        mode = self.decision.mode

        if mode == "SDA":
            scores = three_agent_scores(self.client, question, documents)
            selected = sda_select(scores, self.k)
            return generate(self.client, question, [documents[i] for i in selected], task)
        if mode == "CoT":
            scores = three_agent_scores(self.client, question, documents, use_cot=True)
            selected = rerank_topk(aggregate(scores), self.k)
            return generate(self.client, question, [documents[i] for i in selected], task)
        if mode == "ATF":
            scores = three_agent_scores(self.client, question, documents)
            selected = atf_filter(aggregate(scores), self.k)
            return generate(self.client, question, [documents[i] for i in selected], task)
        # PDE
        scores = three_agent_scores(self.client, question, documents)
        return pde_answer(self.client, question, documents, aggregate(scores), self.k, task)

    def _eval_nf(self, examples: list[dict], task: str) -> float:
        """No-Filter baseline: feed all documents to the generator (standard RAG)."""
        hits = [
            exact_match(generate(self.client, ex["question"], ex["documents"], task), ex["answer"])
            for ex in examples
            if "answer" in ex
        ]
        return float(np.mean(hits)) if hits else 0.0
