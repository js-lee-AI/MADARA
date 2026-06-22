"""The four MADARA treatment strategies and the three-agent scorer.

Each treatment addresses a distinct failure mode diagnosed by RSC and the
No-Filter (NF) baseline (paper Section 4):

* CoT De-Polarization - reasoning before scoring, for weakly coupled
  quality-ordered models whose baseline scores collapse to the extremes.
* SDA (Score Distribution Alignment) - percentile-rank aggregation, for
  stochastic models where reasoning does not drive scores.
* ATF (Adaptive Threshold Filtering) - keep documents scoring above
  mu - 0.5 sigma, for strongly coupled models whose rankings are already good.
* PDE (Per-Document Extraction) - answer from each document in isolation and
  vote, for weak-baseline models that drown in multi-document context.

For a clean, backend-agnostic interface the three assessors each return a 0..5
score per document; the baseline aggregates them with weights (0.4, 0.3, 0.3),
while SDA uses uniform weights.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .model_client import ModelClient
from .prompts import (
    BASE_ASSESSORS,
    GENERATOR_FACTOID,
    GENERATOR_FEVER,
    build_generator_user_prompt,
    cot_assessor,
)
from .rsc import _parse_score

BASELINE_WEIGHTS = (0.4, 0.3, 0.3)
UNIFORM_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
ATF_KAPPA = 0.5


def three_agent_scores(
    client: ModelClient,
    question: str,
    documents: list[str],
    use_cot: bool = False,
) -> np.ndarray:
    """Score every document with the three assessors. Returns an (m, 3) matrix."""
    assessors = [cot_assessor(a) for a in BASE_ASSESSORS] if use_cot else list(BASE_ASSESSORS)
    matrix = np.zeros((len(documents), len(assessors)))
    for j, system in enumerate(assessors):
        systems = [system] * len(documents)
        users = [_assessor_user(question, documents, i) for i in range(len(documents))]
        for i, resp in enumerate(client.chat_batch(systems, users, max_tokens=512)):
            matrix[i, j] = _parse_score(resp)
    return matrix


def aggregate(scores: np.ndarray, weights=BASELINE_WEIGHTS) -> np.ndarray:
    """Weighted aggregation of an (m, A) score matrix into (m,) document scores."""
    w = np.asarray(weights)[: scores.shape[1]]
    w = w / w.sum()
    return scores @ w


def rerank_topk(agg_scores: np.ndarray, k: int) -> list[int]:
    """Indices of the top-k documents by aggregated score (CoT / baseline path)."""
    return list(np.argsort(-agg_scores)[:k])


def sda_select(scores: np.ndarray, k: int, weights=UNIFORM_WEIGHTS) -> list[int]:
    """Score Distribution Alignment (Algorithm 1).

    Convert each agent's raw scores to percentile ranks, aggregate with uniform
    weights, map to [0, 5], and return the top-k document indices. Percentile
    normalization is scale-invariant, so it adapts to heterogeneous per-agent
    score distributions where a fixed-constant fusion (e.g. RRF) cannot.
    """
    m, a = scores.shape
    if m < 2:
        return list(range(m))
    w = np.asarray(weights)[:a]
    w = w / w.sum()
    ranks = np.zeros_like(scores, dtype=float)
    for j in range(a):
        order = np.argsort(np.argsort(scores[:, j]))  # 0..m-1 ascending ranks
        ranks[:, j] = order / (m - 1)
    calibrated = (ranks @ w) * 5.0
    return list(np.argsort(-calibrated)[:k])


def atf_filter(agg_scores: np.ndarray, k: int, kappa: float = ATF_KAPPA) -> list[int]:
    """Adaptive Threshold Filtering: keep documents with score >= mu - kappa*sigma.

    Retains at least 2 and at most k documents and adds no LLM calls.
    """
    tau = agg_scores.mean() - kappa * agg_scores.std()
    order = list(np.argsort(-agg_scores))
    kept = [i for i in order if agg_scores[i] >= tau]
    if len(kept) < 2:
        kept = order[: min(2, len(order))]
    return kept[:k]


def generate(client: ModelClient, question: str, documents: list[str], task: str) -> str:
    """Generate a final answer from the selected documents."""
    system = GENERATOR_FEVER if task == "fever" else GENERATOR_FACTOID
    return client.chat(system, build_generator_user_prompt(question, documents), max_tokens=64).strip()


def pde_answer(
    client: ModelClient,
    question: str,
    documents: list[str],
    agg_scores: np.ndarray,
    k: int,
    task: str = "factoid",
) -> str:
    """Per-Document Extraction: answer from each top-k document in isolation, then
    group candidates by normalized string match and return the group with the
    highest cumulative document score."""
    topk = list(np.argsort(-agg_scores)[:k])
    groups: dict[str, dict] = defaultdict(lambda: {"weight": 0.0, "answer": ""})
    for idx in topk:
        candidate = generate(client, question, [documents[idx]], task)
        key = _normalize(candidate)
        groups[key]["weight"] += float(agg_scores[idx])
        groups[key]["answer"] = candidate
    if not groups:
        return ""
    best = max(groups.values(), key=lambda g: g["weight"])
    return best["answer"]


def _assessor_user(question: str, documents: list[str], target: int) -> str:
    target_doc = documents[target]
    others = "\n".join(
        f"Doc {i}: {documents[i][:200]}" for i in range(len(documents)) if i != target
    )
    return (
        f"Question: {question}\n\nDocument to evaluate:\n{target_doc}\n\n"
        f"Other retrieved documents (for consistency/conflict checks):\n{others}"
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
