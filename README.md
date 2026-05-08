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

**Note:** Transcripts exceeding 2000 words are truncated before scoring. 
Longest transcript observed: 3156 words (subject 1248).

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
| `src/compare_all.py` | Compare all conditions vs 405B ground truth (general section) |
| `src/compare_all_dx.py` | Compare baseline and RAG vs 405B ground truth (dx section) |
| `src/ingest.py` | Ingest clinical literature PDFs into ChromaDB |
| `src/prepare_finetune_data.py` | Prepare JSONL dataset for LoRA fine-tuning |

### Metrics (0.0–10.0, higher = worse)
Nine clinical metrics extracted per transcript, matching the published pipeline:
Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety, Catastrophizing, 
Rumination, Narrative_Fragmentation, Agency_Deficit

---

## Results

### General Section: Agreement between Llama 3.2 3B and Llama 405B
Spearman r — higher values indicate stronger agreement with 405B ground truth  
(n=67 CLBP participants, general interview section — participants not discussing their condition)

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

### Diagnostic Section (dx): Agreement between Llama 3.2 3B and Llama 405B
Spearman r — higher values indicate stronger agreement with 405B ground truth  
(n=67 CLBP participants, condition-specific interview section)

| Metric | Baseline | RAG |
|--------|----------|-----|
| Physical_Pain | 0.682*** | 0.600*** |
| Emotional_Pain | 0.599*** | 0.578*** |
| Depression | 0.572*** | 0.518*** |
| poor_QoL | 0.632*** | 0.534*** |
| Anxiety | 0.445*** | 0.491*** |
| Catastrophizing | 0.425*** | 0.082 |
| Rumination | 0.549*** | 0.527*** |
| Narrative_Fragmentation | 0.261* | 0.451*** |
| Agency_Deficit | 0.328** | 0.345** |

*p<0.05, **p<0.01, ***p<0.001

---

## Key Findings

**1. Small LLMs capture clinical signal from non-condition-specific narratives**
In the general section, participants do not discuss their condition directly — yet the 3B 
model shows significant agreement with 405B ground truth on 8/9 metrics. This demonstrates 
that small LLMs can infer clinical state from naturalistic conversation, without explicit 
symptom reporting.

**2. RAG with clinical literature improves general section scoring**
When transcript content is non-specific, retrieving relevant clinical literature as context 
improves agreement with 405B ground truth:
- poor_QoL: 0.309 → 0.559*** (largest improvement)
- Anxiety: 0.302 → 0.425***
- Catastrophizing: 0.212 (ns) → 0.263* (became significant)

**3. Diagnostic section: RAG does not add value over baseline**
In dx, the transcript content is already condition-specific — RAG context does not improve 
and can hurt (Catastrophizing: 0.425*** → 0.082ns). 
Exception: Narrative_Fragmentation benefits from RAG (0.261* → 0.451***), suggesting 
this construct requires external clinical context regardless of transcript specificity.

**4. Fine-tuning with n=67 leads to severe overfitting**
All fine-tuning versions (v1–v3) memorize a constant value for Physical_Pain.
Exception: FT-v3 achieves best Depression agreement across all conditions (r=0.464***).

**5. RAG + Fine-tuned: negative result**
Combining RAG with the fine-tuned model collapses to constant output — 
parameter memorization overrides retrieved context entirely.

---

## Fine-tuning Details

| Version | Dataset | Ground truth | Train/Val | Iters | Val loss (start→end) | Notes |
|---------|---------|-------------|-----------|-------|----------------------|-------|
| v1 | 67 (general) | 405b_general | 53/14, ordered | 100 | 2.792→2.298 | baseline FT |
| v2 | 201 (general+full+dx) | 405b_general | 160/41, shuffled | 200 | 2.964→2.521 | full duplicates general+dx |
| v3 | 134 (general+dx) | 405b_general | 107/27, shuffled | 200 | 2.836→2.385 | best val loss, but wrong GT for dx |
| v4 | 134 (general+dx) | 405b_general + 405b_dx | 107/27, shuffled | TBD | TBD | correct GT per section |

**Method:** LoRA via MLX — 0.108% trainable parameters (3.47M / 3,212M), last 8 layers  
**Peak memory:** 6.6 GB (v1) / 14.0 GB (v2, v3) of 24 GB available

---

## Conclusions

**For general section: RAG > Baseline >> Fine-tuning (n=67)**
**For dx section: Baseline > RAG**

- Small LLMs can capture clinical signal from non-condition-specific narratives (key finding)
- RAG with domain-specific literature is most effective when transcript content is non-specific
- dx transcripts are sufficiently informative without additional context
- Fine-tuning v4 (with correct ground truth per section) is the next step
- All processing is fully local and HIPAA-compliant (Apple M4, 24GB)

---

## Next Steps
1. Fine-tuning v4 — use 405b_dx as ground truth for dx transcripts
2. Expand RAG knowledge base with pain-specific papers (RAG v2)
3. Re-run RAG v2 on general section and compare

---

## Publication Target

**Proposed angle:**
> "HIPAA-compliant local deployment of small LLMs for clinical NLP — small models capture clinical signal from non-condition-specific narratives, RAG helps when content is non-specific"

**Target journals:**
- JMIR AI — strong fit, publishes LLM methodology in health, moderate N acceptable
- Journal of Biomedical Informatics — methodological focus, small N ok if technical contribution is clear
- Frontiers in Digital Health — flexible, publishes pipelines and methods
- PLOS ONE — if angle is reproducibility and open methodology
