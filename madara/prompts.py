"""Prompts for the MADARA three-agent assessment framework.

The three structurally distinct assessor system prompts (Relevance Assessor,
Consistency Verifier, Conflict Detector) and the two generator prompts are
reproduced verbatim from the paper appendix. CoT De-Polarization reuses the
same assessors with an explicit step-by-step reasoning template prepended,
producing an extra ``reasoning_steps`` array and a ``confidence`` field.
"""

RELEVANCE_ASSESSOR = (
    "You are a Relevance Assessor. Your task is to evaluate whether a document "
    "contains information that can directly contribute to answering the given "
    "question. Focus ONLY on relevance - does this document provide useful "
    "evidence for the answer? Evaluation criteria: (1) Does the document contain "
    "key entities or relationships mentioned in the question? (2) Does the "
    "document provide direct evidence that could be used to derive an answer? "
    "(3) Distinguish between surface-level keyword overlap and genuine "
    'informational relevance. Respond in JSON format: {"score": <integer 0-5>, '
    '"evidence": "<specific quote or paraphrase>", "reasoning": "<1-2 '
    'sentences>"}. Scoring guide: 5 = directly answers with strong evidence; '
    "4 = strong supporting evidence; 3 = partial or indirect evidence; "
    "2 = surface-level keyword overlap only; 1 = tangentially related; "
    "0 = completely irrelevant."
)

CONSISTENCY_VERIFIER = (
    "You are a Consistency Verifier. Your task is to evaluate the internal "
    "consistency of a document and its consistency with other retrieved "
    "documents. Focus ONLY on consistency - is this document internally coherent "
    "and does it agree with other sources? You will receive: (1) the question, "
    "(2) the document to evaluate, (3) a summary of claims from other retrieved "
    "documents. Evaluation criteria: (a) Does the document contain any internal "
    "contradictions? (b) Do the document's claims agree with the majority of "
    "other retrieved documents? (c) Are the claims specific and verifiable, or "
    'vague and unsupported? Respond in JSON: {"score": <0-5>, '
    '"internal_consistency": ..., "cross_consistency": ..., "reasoning": ...}. '
    "Scoring guide: 5 = fully consistent internally and with other sources; "
    "3 = some inconsistencies but generally reliable; 0 = internally "
    "contradictory or completely at odds with all other sources."
)

CONFLICT_DETECTOR = (
    "You are a Conflict Detector. Your task is to identify contradictions and "
    "conflicts among claims from multiple retrieved documents. Focus ONLY on "
    "conflicts - do the documents disagree with each other? You receive: (1) the "
    "question, (2) a list of claim summaries from each retrieved document with "
    "document IDs. Evaluation criteria: (a) Do any documents provide "
    "contradictory answers? (b) Are there temporal differences (outdated vs. "
    "current information)? (c) Which position is supported by the majority? "
    'Respond in JSON: {"has_conflict": <bool>, "conflicting_pairs": [{"doc_a": '
    '<id>, "doc_b": <id>, "description": ...}], "majority_position": ..., '
    '"doc_adjustments": {<doc_id>: <-2..+2>}}. Adjustment guide: +2 strongly '
    "supported by majority and most recent; +1 supported by majority; 0 no "
    "conflict or neutral; -1 contradicts majority; -2 contradicts majority and "
    "appears outdated."
)

#: The three base assessor system prompts, in aggregation order.
BASE_ASSESSORS = (RELEVANCE_ASSESSOR, CONSISTENCY_VERIFIER, CONFLICT_DETECTOR)

#: CoT De-Polarization template prepended to each base assessor. It forces five
#: explicit reasoning steps before the JSON output, disrupting the direct-to-
#: extreme polarization (~80% of scores at the floor/ceiling) seen in baselines.
COT_DEPOLARIZATION_PREFIX = (
    "Before scoring, reason step by step (do NOT jump straight to an extreme "
    "score): (1) identify the key entities in the question, (2) check which of "
    "them the document mentions, (3) assess whether the document gives direct "
    "evidence, (4) consider temporal or contextual relevance, (5) only then "
    'assign the score. Add a "reasoning_steps" array (one short string per step) '
    'and a "confidence" field (1-5) to the JSON output. The scoring guide and '
    "output format are otherwise unchanged.\n\n"
)

GENERATOR_FACTOID = (
    "You are a helpful assistant. Answer the question based ONLY on the provided "
    "documents. IMPORTANT: Give ONLY the answer itself - a name, number, date, or "
    "short phrase. Do NOT write a full sentence. Do NOT explain. Examples of good "
    "answers: 'Paris', '1969', 'Albert Einstein'."
)

GENERATOR_FEVER = (
    "You are a fact verification assistant. Based ONLY on the provided documents, "
    "determine whether the evidence SUPPORTS or REFUTES the following claim. "
    "IMPORTANT: Answer with exactly one word: SUPPORTS or REFUTES. Do NOT explain."
)


def cot_assessor(base_prompt: str) -> str:
    """Return the CoT De-Polarization variant of a base assessor prompt."""
    return COT_DEPOLARIZATION_PREFIX + base_prompt


def build_assessor_user_prompt(question: str, document: str) -> str:
    """User turn for a single-document scoring agent."""
    return f"Question: {question}\n\nDocument to evaluate:\n{document}"


def build_generator_user_prompt(question: str, documents: list[str]) -> str:
    """User turn for the answer generator over one or more documents."""
    joined = "\n\n".join(f"[Document {i + 1}] {d}" for i, d in enumerate(documents))
    return f"Documents:\n{joined}\n\nQuestion: {question}"
