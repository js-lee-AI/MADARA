"""Backend-agnostic model-client interface for MADARA.

MADARA is training-free: every component (the RSC probe, the four treatments,
and the router) only needs text in and text out from an instruction-tuned chat
model. Implement :meth:`ModelClient.chat` for your backend (a local vLLM or
transformers server, or a hosted chat API) and pass the client through.

A small :class:`MockModelClient` is included so ``example.py`` runs with no GPU
and no network access.
"""

from __future__ import annotations

import abc
import hashlib
import json
import re
from typing import Sequence


class ModelClient(abc.ABC):
    """Minimal chat contract MADARA depends on."""

    @abc.abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        """Return the assistant message for a single (system, user) turn."""
        raise NotImplementedError

    def chat_batch(
        self,
        systems: Sequence[str],
        users: Sequence[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> list[str]:
        """Score a batch of turns. Override for true server-side batching."""
        return [
            self.chat(s, u, max_tokens=max_tokens, temperature=temperature)
            for s, u in zip(systems, users)
        ]


def _seeded_unit(*parts: str) -> float:
    """Deterministic pseudo-random value in [0, 1) from string parts."""
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockModelClient(ModelClient):
    """Offline stand-in used by the example and tests.

    Scoring agents receive a lexical-overlap score in 0..5 (so stronger
    overlap yields higher scores), and generators echo the most overlapping
    candidate answer. Outputs are deterministic given the inputs.
    """

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        sys_l = system.lower()
        if "supports or refutes" in sys_l:
            return "SUPPORTS" if _seeded_unit(user) > 0.5 else "REFUTES"
        if "give only the answer" in sys_l or "answer the question based only" in sys_l:
            return self._mock_answer(user)
        if '"score"' in system or "score" in sys_l:
            return self._mock_score(user)
        return self._mock_answer(user)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}

    _STRUCTURAL = {"document", "documents", "question", "doc", "other", "retrieved"}

    def _mock_score(self, user: str) -> str:
        question, document = self._split_question_document(user)
        q, d = self._tokens(question), self._tokens(document)
        overlap = len(q & d) / max(1, len(q))
        noise = (_seeded_unit(question, document) - 0.5) * 0.4
        score = int(round(min(1.0, max(0.0, overlap + noise)) * 5))
        return json.dumps({"score": score, "reasoning": "lexical overlap heuristic"})

    def _mock_answer(self, user: str) -> str:
        question, document = self._split_question_document(user)
        caps = [
            w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", document)
            if w.lower() not in self._STRUCTURAL
        ]
        if not caps:
            return "unknown"
        # Return the most frequently mentioned entity (ties broken by first seen).
        return max(caps, key=lambda w: (caps.count(w), -caps.index(w)))

    @staticmethod
    def _split_question_document(user: str) -> tuple[str, str]:
        if "Document to evaluate:" in user:           # assessor turn
            question, _, document = user.partition("Document to evaluate:")
            return question, document
        if "Question:" in user and "Documents:" in user:  # generator turn
            documents, _, question = user.partition("Question:")
            return question, documents
        return user, user
