"""Runnable MADARA demo using the offline mock client (no GPU, no network).

It shows the two phases of the pipeline:

1. Calibrate: run the RSC probe and the No-Filter baseline on a tiny calibration
   set, which selects a single treatment for the model-task pair.
2. Answer: apply the selected treatment to held-out queries.

Swap MockModelClient for a real ModelClient (a local vLLM/transformers server or
a hosted chat API) to reproduce the paper's pipeline. The mock's scores are a
lexical-overlap heuristic, so the exact route here is illustrative only.
"""

from madara import MADARARouter, MockModelClient, RSCProbe

CALIBRATION = [
    {
        "question": "What is the capital of France?",
        "documents": [
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe with several overseas regions.",
            "The Eiffel Tower is a landmark located in Paris, France.",
            "Berlin is the capital of Germany.",
            "Lyon is a major city in central-eastern France.",
        ],
        "answer": "Paris",
    },
    {
        "question": "Who wrote the play Romeo and Juliet?",
        "documents": [
            "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
            "William Shakespeare was an English playwright born in Stratford-upon-Avon.",
            "The Globe Theatre staged many Shakespeare plays in London.",
            "Christopher Marlowe was a contemporary English playwright.",
            "Romeo and Juliet has been adapted into numerous films and operas.",
        ],
        "answer": "William Shakespeare",
    },
]

HELD_OUT = [
    {
        "question": "What is the largest planet in the Solar System?",
        "documents": [
            "Jupiter is the largest planet in the Solar System by mass and volume.",
            "Saturn is the second largest planet and is known for its rings.",
            "Earth is the third planet from the Sun.",
            "Jupiter is a gas giant composed mainly of hydrogen and helium.",
            "Mars is often called the Red Planet.",
        ],
    },
]


def main() -> None:
    client = MockModelClient()

    # Inspect the RSC diagnostic on the calibration set.
    rsc = RSCProbe().probe(client, CALIBRATION, task="factoid", model_name="mock")
    print("RSC probe")
    print(f"  per-level rho : shuffled={rsc.shuffled_rho}, "
          f"contradicted={rsc.contradicted_rho}, random={rsc.random_rho}")
    print(f"  trend rho*    : {rsc.trend_rho} (p={rsc.trend_p_value})")
    print(f"  class         : {rsc.classification} ({rsc.strength})")
    print(f"  score entropy : {rsc.score_entropy} bits")

    # Calibrate the router (Phase 1) and answer held-out queries (Phase 2).
    router = MADARARouter(client, k=3)
    decision = router.calibrate(CALIBRATION, task="factoid", model_name="mock")
    print("\nRouting decision")
    print(f"  NF baseline   : {decision.em_nf:.0%}")
    print(f"  selected mode : {decision.mode}")
    print(f"  reason        : {decision.reason}")

    print("\nHeld-out answers")
    for ex in HELD_OUT:
        answer = router.answer(ex["question"], ex["documents"], task="factoid")
        print(f"  Q: {ex['question']}")
        print(f"  A: {answer}  (via {decision.mode})")


if __name__ == "__main__":
    main()
