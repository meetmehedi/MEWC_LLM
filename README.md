# MEWC-LLM 🧠
### Memory-Efficient Weight Consolidation for Large Language Models

> **Adaptive Stability–Plasticity Optimization for Factual Continual Learning in Large Language Models**

[![Research Domain](https://img.shields.io/badge/Domain-AI%20%2F%20NLP-blueviolet)](#)
[![Focus](https://img.shields.io/badge/Focus-Hallucination%20Reduction-red)](#)
[![Method](https://img.shields.io/badge/Method-Continual%20Learning-green)](#)
[![Status](https://img.shields.io/badge/Status-Pre--publication%20Draft-orange)](#)

---

## 📌 Overview

**MEWC-LLM** is a novel continual learning architecture that addresses LLM hallucination at its **root cause** — the destabilization of factual knowledge representations during parameter updates.

Unlike existing mitigation strategies (RAG, replay buffers, post-hoc fact-checking) that treat hallucination as an inference-time artifact, MEWC-LLM reframes hallucination as a **learning-dynamics failure** — a manifestation of the classical stability–plasticity dilemma in continual learning. When LLMs are fine-tuned on new data, gradient updates can catastrophically overwrite verified factual knowledge, directly producing hallucinations.

---

## 🧩 Core Architecture

MEWC-LLM integrates **four interconnected mechanisms**:

| Component | Description |
|-----------|-------------|
| **Truth-Aware Adaptive Consolidation (TAAC)** | Uses gradient cosine similarity to detect and suppress conflicts between new-knowledge gradients and verified factual gradients |
| **Fisher-Based Truth Importance Scoring** | Adapts the Fisher Information Matrix to protect parameters critical for factual reliability |
| **Conflict-Aware Truth Preservation (CATP)** | A gradient-level immune system with 3-tier adaptive response (low / moderate / high conflict) |
| **Modular Factual Memory Architecture** | Structurally separates stable reasoning (frozen backbone) from adaptable domain knowledge (factual heads) |

> ✅ **No replay buffers required** — making it suitable for edge AI, private LLMs, and real-time continual learning environments.

---

## 📐 Mathematical Foundation

The MEWC-LLM training objective is a modified loss function:

```
L_MEWC(θ) = L_new(θ; D_new) + λ · Σᵢ Fᵢ^truth · (θᵢ − θᵢ*)²
```

Where:
- `L_new` — standard training loss on new data
- `Fᵢ^truth` — Truth Importance Score for parameter θᵢ (Fisher-based)
- `θᵢ*` — parameter values encoding verified factual knowledge
- `λ` — adaptive consolidation coefficient controlled by gradient conflict detection

**Conflict Signal (TAAC):**
```
conflict = −cos(g_t, g_prev) = −(g_t · g_prev) / (||g_t|| · ||g_prev||)
```

**Adaptive Lambda:**
```
λ_adaptive = λ_base · (1 + α · max(0, conflict))
```

---

## 📊 Empirical Results (GPT-2 124M Validation)

| Configuration | Factual PPL (TruthfulQA) ↓ | Adaptation PPL (PubMedQA) ↓ | HR Reduction |
|:---|:---:|:---:|:---:|
| Base Model | 25.38 | 45.20 | — |
| Baseline LoRA | 38.60 | 12.40 | 0% |
| **MEWC-LLM** | **26.80** | **14.20** | **62.4%** |

**Key Findings:**
1. 🔒 **Destabilization Prevention:** Baseline fine-tuning causes a **52% spike** in factual perplexity. MEWC-LLM restricts this drift to just **5.6%**.
2. ⚡ **Minimal Plasticity Penalty:** Adaptation quality (14.2 PPL) remains highly competitive with the unconstrained baseline (12.4 PPL).
3. 🎯 **Conflict Awareness:** Gradient conflict detection identified **15.7%** of parameter updates as potentially harmful to factual integrity.

**Projected Results at Scale (7B–13B Models):**
- **40–70%** standalone hallucination reduction in continual learning scenarios
- **80–95%** reduction combined with RAG + confidence calibration in bounded domains

---

## 🔬 Evaluation Benchmarks

| Benchmark | Purpose |
|-----------|---------|
| **TruthfulQA** | Primary hallucination rate measurement (817 adversarially crafted questions) |
| **HaluEval** | Claim-level hallucination detection |
| **HotpotQA** | Multi-hop factual reasoning preservation |
| **MMLU** | 57-domain knowledge breadth |
| **FEVER** | Fact verification against knowledge databases |
| **Natural Questions** | Open-domain QA factual retrieval |

---

## 🆚 Why Not Just Use RAG / RLHF / Replay?

| Approach | Fundamental Limitation |
|----------|----------------------|
| RAG | External patch; doesn't prevent internal representation degradation |
| RLHF | Shapes output distribution but doesn't protect specific factual parameters |
| Replay Buffers | Computationally expensive; privacy-violating; not feasible for edge/private deployments |
| Chain-of-Thought | Inference-time scaffold; doesn't address underlying parameter degradation |
| Post-hoc Fact-Checking | Reactive, not preventive; requires external ground truth |

**MEWC-LLM** is the first framework to address hallucination **preventively at the parameter level**, without replay, with edge-feasible compute requirements.

---

## 📁 Repository Structure

```
MEWC_LLM/
├── MEWC-LLM.md                    # Full research paper (pre-publication draft)
├── MEWC-LLM.pdf                   # PDF version of the paper
├── MEWC_LLM_Proposal.md           # Original research proposal
├── 625_Camera Ready.pdf           # Camera-ready conference submission
├── Stanford Agentic Reviewer - View Review.pdf  # Peer review feedback
└── Paper_Draft/
    ├── MEWC_LLM_Paper.tex         # LaTeX source
    ├── train_mewc_llm.py          # MEWC-LLM training script
    ├── mewc_lora_optimizer.py     # MEWC LoRA optimizer implementation
    ├── data_pipeline.py           # Data pipeline utilities
    ├── evaluate_mewc.py           # Evaluation scripts
    ├── run_experiments.py         # Experiment runner
    ├── api_server.py              # API server for inference
    ├── final_experiment_summary.json  # Experiment results summary
    ├── results_base.json          # Base model results
    ├── results_baseline_lora_weights.json  # Baseline LoRA results
    ├── results_mewc_lora_weights.json      # MEWC-LLM results
    ├── baseline_lora_weights/     # Baseline LoRA adapter weights
    └── mewc_lora_weights/         # MEWC-LLM adapter weights
```

---

## 🚀 CATP Response Protocol

The Conflict-Aware Truth Preservation layer implements a three-tier gradient immune response:

| Conflict Level | Threshold | Response |
|:---:|:---:|:---|
| 🟢 **Low** | cos(g_t, g_prev) > −0.1 | Normal update proceeds; standard λ applies |
| 🟡 **Moderate** | −0.5 < cos ≤ −0.1 | Learning rate scaled down; λ_adaptive activated; soft gradient projection |
| 🔴 **High** | cos ≤ −0.5 | Update rejected for factual parameters; retrieval augmented; conflict logged |

---

## 🏗️ Modular Factual Memory Architecture

```
┌──────────────────────────────────────────────┐
│          Shared Reasoning Backbone            │  ← Frozen / max TAAC consolidation
│   (syntax, logic, cross-domain semantics)     │
└───────────────┬──────────────────────────────┘
                │
    ┌───────────┼──────────────┐
    ▼           ▼              ▼
┌────────┐ ┌────────┐    ┌──────────┐
│Medicine│ │  Law   │ …  │ Science  │  ← Domain-Specific Factual Heads (LoRA adapters)
│ Head   │ │  Head  │    │  Head    │
└────────┘ └────────┘    └──────────┘
    ↑
┌─────────────────────┐
│  Verification Modules│  ← Confidence scoring → gates CATP thresholds
└─────────────────────┘
```

---

## 📚 Key References

- Kirkpatrick et al. (2017) — Elastic Weight Consolidation (EWC) — *PNAS*
- Yu et al. (2020) — Gradient Surgery / PCGrad — *NeurIPS*
- Hu et al. (2022) — LoRA — *ICLR*
- Ji et al. (2023) — Survey of Hallucination in NLG — *ACM Computing Surveys*
- Xu et al. (2024) — Hallucination is Inevitable — *arXiv:2401.11817*
- Lewis et al. (2020) — Retrieval-Augmented Generation — *NeurIPS*

---

## 🔭 Research Questions

1. Can gradient cosine similarity reliably detect hallucination-inducing parameter updates?
2. Does Fisher-based weight consolidation targeting *factual truth* outperform task-based EWC?
3. To what extent does CATP reduce catastrophic forgetting of verified factual representations?
4. Does modular memory architecture reduce hallucination propagation vs. monolithic LLMs?
5. Is MEWC-LLM computationally feasible for edge and private LLM deployments?

---

## ⚠️ Limitations

- **FAS Curation Overhead** — Maintaining a high-quality Factual Anchor Set requires ongoing expert effort
- **Fisher Approximation Fidelity** — Diagonal FIM may miss cross-parameter correlations relevant to factual reliability
- **False-Positive Conflict Detection** — Cosine similarity may incorrectly flag factual *refinements* as threats
- **Contested Facts** — The framework assumes stable, unambiguous facts; context-dependent facts are not yet handled
- **Scale Validation Gap** — Validation at 70B+ parameter scale is future work

---

## 🌐 Broader Vision

MEWC-LLM is a foundational component toward a **self-correcting lifelong LLM** — a system that:
- Continuously ingests new information
- Protects verified knowledge at the parameter level
- Verifies its own factual claims against internal and external sources
- Refuses to generate content it cannot reliably ground

Combined with **HALO** (Hybrid Anti-Hallucination LLM Orchestration — a parallel inference-time verification framework), MEWC-LLM forms the internal learning-dynamics layer of a comprehensive anti-hallucination system stack.

---

## 📄 Citation

```bibtex
@article{mewcllm2025,
  title     = {MEWC-LLM: Memory-Efficient Weight Consolidation for Large Language Models —
               Adaptive Stability–Plasticity Optimization for Factual Continual Learning},
  author    = {Hasan, Md. Mehedi},
  year      = {2025},
  note      = {Pre-publication research draft},
  url       = {https://github.com/meetmehedi/MEWC_LLM}
}
```

---

## 📬 Contact

**Research Domain:** Artificial Intelligence / NLP  
**Classification:** Research Proposal / Pre-publication Draft  
**GitHub:** [@meetmehedi](https://github.com/meetmehedi)

---

*Keywords: LLM hallucination, continual learning, stability–plasticity dilemma, elastic weight consolidation, gradient conflict detection, factual memory, catastrophic forgetting, truth-preserving fine-tuning*