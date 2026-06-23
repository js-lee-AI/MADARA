# MADARA: Model-Adaptive Document Assessment Routing Architecture

Reference implementation of MADARA, a **training-free** pipeline for
cost-efficient multi-agent retrieval-augmented generation (RAG). Multi-agent
document assessment multiplies inference cost, yet whether it helps depends
sharply on the model. MADARA diagnoses each model-task pair and routes it to the
cheapest assessment treatment that actually works.

The pipeline has two diagnostics and four treatments:

* **RSC (Reasoning-Score Coupling)** is a label-free probe. It perturbs the
  reasoning behind a model's document scores in three increasing steps
  (shuffle, contradict, randomize) and checks whether the scores degrade
  monotonically. Monotonic degradation means scoring is reasoning-driven
  (*quality-ordered*); otherwise it is *stochastic*. Strength is read from the
  first-level correlation `rho_1`.
* **NF (No-Filter) baseline** measures the model's raw context-handling capacity.

Routing (thresholds fixed on a single Mistral-7B pilot, applied zero-shot):

```
if   EM_NF < 30%            -> PDE   (weak baseline: isolate documents)
elif scoring is stochastic  -> SDA   (align score distributions)
elif rho_1 < 0.5            -> CoT   (weakly coupled: reason before scoring)
else                        -> ATF   (strongly coupled: threshold-filter)
```

The four treatments:

* **PDE (Per-Document Extraction)** answers from each top-k document in
  isolation and votes, curing multi-document context confusion.
* **SDA (Score Distribution Alignment)** converts each agent's scores to
  percentile ranks before aggregating, for models whose reasoning does not drive
  scores.
* **CoT De-Polarization** forces step-by-step reasoning before scoring, undoing
  the direct-to-extreme polarization of weakly coupled models.
* **ATF (Adaptive Threshold Filtering)** keeps documents scoring above
  `mu - 0.5*sigma`, adding no extra LLM calls.

> Paper: *To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient
> Multi-Agent RAG.* A BibTeX entry will be added once the arXiv version is
> available.

## What's in this repository

```
madara/
  rsc.py            RSC diagnostic: perturbation probe, trend rho, classification
  treatments.py     three-agent scorer + PDE / SDA / CoT / ATF
  prompts.py        the three assessor prompts, CoT variant, and generators
  router.py         MADARA two-phase routing protocol
  metrics.py        Exact Match and Token F1
  model_client.py   backend-agnostic model-client interface (plus a mock)
example.py          runnable demo using the mock client (no GPU needed)
```

## Installation

```bash
git clone https://github.com/js-lee-AI/MADARA.git && cd MADARA
pip install -r requirements.txt   # numpy, scipy
python example.py
```

## Usage

```python
from madara import MADARARouter, MockModelClient

client = MockModelClient()                 # swap for your real model client
router = MADARARouter(client, k=5)

# Phase 1: one-time calibration on a small labelled set (RSC is label-free;
# only the No-Filter baseline reads the gold answer).
decision = router.calibrate(calibration_set, task="factoid")
print(decision.mode, decision.em_nf, decision.reason)

# Phase 2: answer queries with the selected treatment.
answer = router.answer(question, documents, task="factoid")
```

Run the RSC probe on its own to inspect scoring behaviour:

```python
from madara import RSCProbe

rsc = RSCProbe().probe(client, calibration_set, model_name="my-model")
print(rsc.classification, rsc.strength, rsc.trend_rho, rsc.shuffled_rho)
```

### Model-client contract

MADARA is backend-agnostic. Provide any object with a `chat` method:

```python
chat(system, user, *, max_tokens=512, temperature=0.6) -> str   # assistant text
```

See `madara/model_client.py` for the abstract base class and a mock used by the
example. Implement it for your model (a local vLLM or transformers server, or a
hosted chat API). Override `chat_batch` for server-side batching.

## Benchmarks

The paper evaluates on public datasets, which are **not redistributed here**.
Obtain them from their original sources under their respective licenses:
CONFLICTS, FEVER, TriviaQA, and MuSiQue.

## Results

MADARA routes five open-weight 7B-9B instruction-tuned models zero-shot. The
routing thresholds are derived from a single pilot model (Mistral-7B) and
applied unchanged to the other four families. Exact Match (%); the strategy is
chosen automatically from the RSC class and the NF baseline.

| Model | Task | RSC | NF | Best component | MADARA strategy | MADARA EM | vs NF |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | CONFLICTS | QO | 13.9 | 24.1 | **PDE** | **50.2** | +36.3 |
| Llama-3.1-8B | FEVER | QO | 88.7 | 88.4 | **ATF** | **90.9** | +2.2 |
| Mistral-7B-v0.3 | CONFLICTS | QO | 18.1 | 22.8 | **PDE** | **43.5** | +25.4 |
| Mistral-7B-v0.3 | FEVER | QO | 90.7 | 92.6 | **CoT** | **92.6** | +1.9 |
| Qwen3-8B | CONFLICTS | S | 60.8 | 63.0 | **SDA** | **63.0** | +2.2 |
| Qwen3-8B | FEVER | QO | 88.6 | 90.0 | **ATF** | **90.9** | +2.3 |
| Qwen2.5-7B | CONFLICTS | S | 59.9 | 65.4 | **SDA** | **65.4** | +5.5 |
| Qwen2.5-7B | FEVER | QO | 87.1 | 90.5 | ATF | 88.2 | +1.1 |
| Gemma-2-9B | CONFLICTS | S | 60.8 | 64.6 | **SDA** | **63.3** | +2.5 |
| Gemma-2-9B | FEVER | QO | 92.2 | 92.5 | **ATF** | **93.6** | +1.4 |

We find out
* For weak-baseline models, **per-document isolation drives outsized gains**
  (+36.3pp for Llama, +25.4pp for Mistral on CONFLICTS; up to +49.8pp on
  TriviaQA), and assessment-free random isolation matches the full pipeline,
  cutting inference calls roughly 4x.
* For strong baselines, scoring quality matters and RSC selects the right scoring
  treatment, matching the oracle treatment 3x more often than a score-entropy
  heuristic (3/10 vs 1/10).
* The isolation finding persists under dense retrieval and generative reranking,
  and the diagnostic boundary transfers zero-shot to four unseen model families.

`S` = stochastic, `QO` = quality-ordered. See the paper for Token F1, MuSiQue
multi-hop results, significance tests, and the full ablations.

## Citation

A BibTeX entry will be added once the arXiv version is available.

## License

The code in this repository is released under the [MIT License](LICENSE). The
paper itself is distributed under CC BY 4.0 via arXiv.
