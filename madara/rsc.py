"""Reasoning-Score Coupling (RSC) diagnostic.

RSC is the label-free *diagnostic* arm of MADARA. It detects whether a model's
document scores degrade monotonically as the reasoning behind them is
progressively corrupted. If they do, the model's scoring is reasoning-driven
(*quality-ordered*); if not, it is *stochastic*.

Protocol (see paper Section 3 and Appendix C):

1. Collect the model's normal CoT score and reasoning chain for each document.
2. Re-score each document three times under increasingly severe perturbations
   of that reasoning: Shuffled (1), Contradicted (2), Random (3).
3. For each level k, compute the Spearman correlation rho_k between the normal
   and perturbed scores across all calibration documents.
4. The trend coefficient is rho_star = Spearman([1, 2, 3], [rho_1, rho_2, rho_3]).
   rho_star == -1.0 (perfect monotonic degradation, rho_1 > rho_2 > rho_3)
   classifies the model-task pair as quality-ordered; otherwise stochastic.
5. rho_1 measures baseline coupling strength: strongly coupled if rho_1 >= 0.5.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .model_client import ModelClient
from .prompts import RELEVANCE_ASSESSOR, cot_assessor

# Default routing thresholds, derived from the Mistral-7B pilot and applied
# zero-shot to every other model (paper Section 5.5).
TREND_RHO_THRESHOLD = -0.9   # rho_star below this == monotonic (rho_star = -1.0)
TREND_P_THRESHOLD = 0.05
STRENGTH_THRESHOLD = 0.5     # rho_1 >= this -> strongly coupled

_FORCED_REASONING_SYSTEM = (
    "You are a Relevance Assessor. A previous analysis of this document has been "
    "completed. Your task is to review the provided reasoning and assign a final "
    "relevance score. You MUST base your score on the provided reasoning "
    "analysis. Do NOT re-analyze from scratch. Scoring guide: 5 = directly and "
    "comprehensively answers; 4 = highly relevant; 3 = moderately relevant; "
    "2 = tangentially relevant; 1 = minimally relevant; 0 = completely "
    'irrelevant. Respond in JSON: {"score": <integer 0-5>, "confidence": '
    "<integer 1-5>}."
)

_POSITIVE_NEGATIVE = [
    ("consistent", "inconsistent"), ("agrees", "disagrees"),
    ("supports", "contradicts"), ("relevant", "irrelevant"),
    ("accurate", "inaccurate"), ("reliable", "unreliable"),
    ("coherent", "incoherent"), ("aligned", "misaligned"),
    ("confirms", "refutes"), ("correct", "incorrect"),
    ("valid", "invalid"), ("helpful", "unhelpful"), ("useful", "useless"),
]


@dataclass
class RSCResult:
    model_name: str
    task: str
    shuffled_rho: float
    contradicted_rho: float
    random_rho: float
    trend_rho: float
    trend_p_value: float
    classification: str          # "quality-ordered" or "stochastic"
    strength: str                # "strongly-coupled" / "weakly-coupled" / "stochastic"
    n_queries: int
    n_docs: int
    normal_mean: float
    normal_extreme_pct: float
    score_entropy: float

    @property
    def is_quality_ordered(self) -> bool:
        return self.classification == "quality-ordered"


def score_entropy(scores, n_bins: int = 6) -> float:
    """Shannon entropy (bits) of a 0..5 score distribution.

    Entropy measures distribution spread, NOT reasoning-score coupling: a model
    can have well-spread scores yet be stochastic. RSC captures the distinction
    that entropy cannot.
    """
    scores = np.asarray(scores)
    counts = np.array([np.sum(scores == s) for s in range(n_bins)], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def shuffle_steps(steps: list[str], rng: random.Random) -> list[str]:
    """Level 1 (mild): reorder reasoning steps, preserving their content."""
    if len(steps) <= 1:
        return list(steps)
    out = list(steps)
    rng.shuffle(out)
    return out


def contradict_steps(steps: list[str]) -> list[str]:
    """Level 2 (moderate): negate directional cues via antonym substitution."""
    out = []
    for step in steps:
        s = step
        for pos, neg in _POSITIVE_NEGATIVE:
            s = s.replace(pos, f"__NEG_{neg}__")
        for _, neg in _POSITIVE_NEGATIVE:
            s = s.replace(f"__NEG_{neg}__", neg)
        out.append(s)
    return out


def random_steps(all_steps: list[list[str]], current_index: int, rng: random.Random) -> list[str]:
    """Level 3 (severe): borrow a reasoning chain from a different document."""
    pool = [s for i, s in enumerate(all_steps) if i != current_index and s]
    return list(rng.choice(pool)) if pool else []


def _parse_score(text: str) -> int:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = json.loads(match.group(0))["score"] if match else 0
        return min(5, max(0, int(raw)))
    except (ValueError, TypeError, KeyError, AttributeError):
        digits = re.findall(r"[0-5]", text)
        return int(digits[0]) if digits else 0


def _split_steps(reasoning: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", reasoning.strip())
    return [p for p in parts if p]


class RSCProbe:
    """Runs the three-level RSC trend probe with a :class:`ModelClient`."""

    def __init__(
        self,
        trend_rho_threshold: float = TREND_RHO_THRESHOLD,
        trend_p_threshold: float = TREND_P_THRESHOLD,
        strength_threshold: float = STRENGTH_THRESHOLD,
        seed: int = 42,
    ):
        self.trend_rho_threshold = trend_rho_threshold
        self.trend_p_threshold = trend_p_threshold
        self.strength_threshold = strength_threshold
        self.seed = seed

    def probe(self, client: ModelClient, examples: list[dict], task: str = "calibration",
              model_name: str = "model") -> RSCResult:
        """Run the probe on ``examples`` = ``[{"question", "documents": [...]}]``."""
        rng = random.Random(self.seed)
        cot_system = cot_assessor(RELEVANCE_ASSESSOR)

        normal_scores: list[int] = []
        reasoning_chains: list[list[str]] = []
        doc_texts: list[str] = []
        questions: list[str] = []

        for ex in examples:
            q, docs = ex["question"], ex["documents"]
            systems = [cot_system] * len(docs)
            users = [f"Question: {q}\n\nDocument to evaluate:\n{d}" for d in docs]
            for d, resp in zip(docs, client.chat_batch(systems, users, max_tokens=512)):
                normal_scores.append(_parse_score(resp))
                reasoning_chains.append(_split_steps(self._extract_reasoning(resp)))
                doc_texts.append(d)
                questions.append(q)

        normal = np.array(normal_scores)
        levels = {
            "shuffled": [shuffle_steps(s, rng) for s in reasoning_chains],
            "contradicted": [contradict_steps(s) for s in reasoning_chains],
            "random": [random_steps(reasoning_chains, i, rng) for i in range(len(reasoning_chains))],
        }

        rhos = {}
        for name, perturbed_chains in levels.items():
            systems = [_FORCED_REASONING_SYSTEM] * len(doc_texts)
            users = [
                self._forced_user(questions[i], doc_texts[i], perturbed_chains[i])
                for i in range(len(doc_texts))
            ]
            perturbed = np.array([_parse_score(r) for r in client.chat_batch(systems, users, max_tokens=256)])
            rho = stats.spearmanr(normal, perturbed)[0]
            rhos[name] = 0.0 if np.isnan(rho) else float(rho)

        return self._classify(rhos, normal, task, model_name, len(examples))

    def classify_from_rhos(self, shuffled_rho: float, contradicted_rho: float,
                           random_rho: float, model_name: str = "model",
                           task: str = "calibration") -> RSCResult:
        """Classify directly from pre-computed per-level correlations."""
        rhos = {"shuffled": shuffled_rho, "contradicted": contradicted_rho, "random": random_rho}
        return self._classify(rhos, np.array([]), task, model_name, 0)

    def _classify(self, rhos: dict, normal: np.ndarray, task: str,
                  model_name: str, n_queries: int) -> RSCResult:
        levels = [rhos["shuffled"], rhos["contradicted"], rhos["random"]]
        if len(set(np.round(levels, 9))) == 1:
            # Constant per-level correlations: no monotonic degradation trend.
            trend_rho, trend_p = 0.0, 1.0
        else:
            trend_rho, trend_p = stats.spearmanr([1, 2, 3], levels)
            trend_rho = 0.0 if np.isnan(trend_rho) else float(trend_rho)
            trend_p = 1.0 if np.isnan(trend_p) else float(trend_p)

        quality_ordered = trend_rho < self.trend_rho_threshold and trend_p < self.trend_p_threshold
        classification = "quality-ordered" if quality_ordered else "stochastic"
        if not quality_ordered:
            strength = "stochastic"
        elif rhos["shuffled"] >= self.strength_threshold:
            strength = "strongly-coupled"
        else:
            strength = "weakly-coupled"

        return RSCResult(
            model_name=model_name,
            task=task,
            shuffled_rho=round(rhos["shuffled"], 3),
            contradicted_rho=round(rhos["contradicted"], 3),
            random_rho=round(rhos["random"], 3),
            trend_rho=round(trend_rho, 3),
            trend_p_value=round(trend_p, 4),
            classification=classification,
            strength=strength,
            n_queries=n_queries,
            n_docs=int(normal.size),
            normal_mean=round(float(normal.mean()), 3) if normal.size else 0.0,
            normal_extreme_pct=round(float(np.mean((normal == 0) | (normal == 5)) * 100), 1) if normal.size else 0.0,
            score_entropy=round(score_entropy(normal), 3) if normal.size else 0.0,
        )

    @staticmethod
    def _extract_reasoning(resp: str) -> str:
        try:
            match = re.search(r"\{.*\}", resp, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            steps = data.get("reasoning_steps") or data.get("reasoning") or ""
            return " ".join(steps) if isinstance(steps, list) else str(steps)
        except (ValueError, TypeError, AttributeError):
            return resp

    @staticmethod
    def _forced_user(question: str, document: str, steps: list[str]) -> str:
        reasoning = "\n".join(f"- {s}" for s in steps) if steps else "(No reasoning available)"
        return (
            f"Question: {question}\n\nDocument to evaluate:\n{document[:2000]}\n\n"
            f"=== PREVIOUS ANALYSIS ===\n{reasoning}\n=== END ANALYSIS ===\n\n"
            "Based on the above reasoning analysis, provide your final relevance score and confidence."
        )
