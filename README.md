# Pain LLM Fine-tuning

Local, HIPAA-compliant evaluation of small LLMs for chronic pain clinical scoring.

## Overview

This project evaluates whether small, locally-run LLMs can replicate the clinical scoring 
of a large model (Llama 405B) on chronic pain patient interviews. All processing is fully 
local on Apple M4 — no data leaves the machine (HIPAA-compliant).

The 405B scores are used as ground truth, validated against clinical questionnaires 
in a submitted manuscript (Norel et al., under review).

## Data
Transcript data from University of Rochester (not included — HIPAA).
CLBP cohort only (n=67 participants).
- `data/transcripts/general/` — general interview section (TXT, all participants)
- `data/transcripts/dx/` — diagnostic section (TXT, CLBP participants only)
- `data/transcripts/full/` — full interview (TXT, general + dx combined)

## Environment
- Apple M4, 24GB unified memory
- MLX — fine-tuning framework, optimized for Apple Silicon
- Ollama — local model serving
- ChromaDB — vector store for RAG

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py  # build RAG knowledge base from literature PDFs
```

## Pipeline

### Scripts
| Script | Description |
|--------|-------------|
| `src/llm_scorer.py` | Baseline scoring via Ollama (2 prompts, 9 metrics) |
| `src/llm_scorer_rag.py` | RAG scoring — 1 prompt per metric with clinical literature context |
| `src/run_scoring.py` | Batch baseline scoring over all subjects |
| `src/run_scoring_rag.py` | Batch RAG scoring |
| `src/run_scoring_finetuned.py` | Batch scoring with fine-tuned model (MLX) |
| `src/ingest.py` | Ingest clinical literature PDFs into ChromaDB |
| `src/prepare_finetune_data.py` | Prepare JSONL dataset for LoRA fine-tuning |
| `src/compare_all.py` | Compare all conditions vs 405B ground truth |

### Metrics (0.0–10.0, higher = worse)
Nine clinical metrics extracted per transcript, matching the published pipeline:
Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety, Catastrophizing, 
Rumination, Narrative_Fragmentation, Agency_Deficit

---

## Results

### Baseline: Llama 3.2 3B vs Llama 405B (ground truth)

Spearman correlations on CLBP cohort (n=67). Higher r = 3B scores agree more with 405B.

| Metric | r | p-value |
|--------|---|---------|
| Physical_Pain | 0.369 | 0.002 ** |
| Emotional_Pain | 0.343 | 0.005 ** |
| Depression | 0.396 | 0.001 *** |
| poor_QoL | 0.309 | 0.011 * |
| Anxiety | 0.302 | 0.013 * |
| Catastrophizing | 0.212 | 0.085 |
| Rumination | 0.311 | 0.010 * |
| Narrative_Fragmentation | 0.340 | 0.005 ** |
| Agency_Deficit | 0.306 | 0.012 * |

*p<0.05, **p<0.01, ***p<0.001

The 3B model shows moderate agreement with the 405B across most metrics. 
Catastrophizing is the only non-significant metric — consistent with it being 
a cognitively complex construct requiring deeper clinical reasoning.

---

### Complete Comparison: All Conditions (Spearman r vs 405B ground truth, n=67)

| Metric | Baseline | RAG | FT-v1 | FT-v2 | FT-v3 | RAG+FT |
|--------|----------|-----|-------|-------|-------|--------|
| Physical_Pain | 0.369** | 0.367** | nan | -0.068 | nan | n/a† |
| Emotional_Pain | 0.343** | 0.290* | 0.261* | 0.152 | 0.261* | n/a† |
| Depression | 0.396*** | 0.375** | 0.386** | 0.354** | 0.464*** | n/a† |
| poor_QoL | 0.309* | 0.559*** | -0.148 | 0.022 | 0.182 | n/a† |
| Anxiety | 0.302* | 0.425*** | 0.293* | 0.108 | 0.067 | n/a† |
| Catastrophizing | 0.212 | 0.263* | 0.137 | 0.153 | -0.041 | n/a† |
| Rumination | 0.311* | 0.349** | -0.035 | 0.117 | -0.035 | n/a† |
| Narrative_Fragmentation | 0.340** | 0.347** | 0.202 | -0.028 | 0.268* | n/a† |
| Agency_Deficit | 0.306* | 0.326** | 0.158 | 0.130 | 0.138 | n/a† |

*p<0.05, **p<0.01, ***p<0.001  
nan = model output was constant (no variance), Spearman undefined  
†RAG+FT: model collapsed to constant output (7.2) — fine-tuning overfit overrides RAG context

---

## Key Findings

**1. RAG improves agreement with 405B on most metrics**
RAG scores agree more with 405B ground truth than baseline, especially:
- poor_QoL: 0.309 → 0.559*** (largest improvement)
- Anxiety: 0.302 → 0.425***
- Catastrophizing: 0.212 (ns) → 0.263* (became significant — clinical literature helps)

**2. Fine-tuning with n=67 leads to severe overfitting**
All fine-tuning versions (v1, v2, v3) memorize a constant value (7.2) for Physical_Pain.
The model learns the most frequent training value instead of the underlying construct.

**3. Exception: FT-v3 wins on Depression**
FT-v3 (general + dx transcripts, no duplication) achieves r=0.464*** for Depression —
the best result across all conditions including RAG (0.375**).

**4. RAG + Fine-tuned: negative result**
Combining RAG with the fine-tuned model collapses to constant output — 
parameter memorization overrides retrieved context entirely.

---

## Fine-tuning Details

| Version | Dataset | Train/Val | Iters | Val loss (start→end) | Notes |
|---------|---------|-----------|-------|----------------------|-------|
| v1 | 67 (general only) | 53/14, ordered | 100 | 2.792→2.298 | baseline FT |
| v2 | 201 (general+full+dx) | 160/41, shuffled | 200 | 2.964→2.521 | full duplicates general+dx |
| v3 | 134 (general+dx) | 107/27, shuffled | 200 | 2.836→2.385 | best val loss |

**Method:** LoRA via MLX — 0.108% trainable parameters (3.47M / 3,212M), last 8 layers  
**Peak memory:** 6.6 GB (v1) / 14.0 GB (v2, v3) of 24 GB available

---

## Conclusions

**RAG > Baseline >> Fine-tuning (n=67)**

For small LLMs in specialized clinical domains, retrieval-augmented generation 
outperforms parameter fine-tuning when labeled data is scarce. RAG with domain-specific 
clinical literature is the most practical approach for HIPAA-compliant deployment.

---

## Next Steps
1. Expand RAG knowledge base with pain-specific papers
2. Re-run RAG with improved knowledge base and compare
3. Fix fine-tuning overfitting before attempting RAG+FT
