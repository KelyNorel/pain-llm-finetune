# Pain LLM Fine-tuning

Local, HIPAA-compliant fine-tuning of LLMs on chronic pain interview data.

## Data
Clinical and transcript data from University of Rochester (not included).
- `data/clinical/` — clinical variables (CSV)
- `data/transcripts/full/` — full interviews (TXT)
- `data/transcripts/general/` — general section (TXT)
- `data/transcripts/dx/` — diagnostic section (patients only, TXT)

## Environment
- Apple M4, 24GB unified memory
- MLX for fine-tuning (Apple Silicon optimized)
- Ollama for local model serving

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Results

### Baseline: Small Model (Llama 3.2 3B) vs Large Model (Llama 405B)

Spearman correlations between 3B and 405B scores on CLBP cohort (n=67):

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

**Hypothesis:** Fine-tuning improves small LLM replication of large model clinical 
scoring, particularly for cognitively complex constructs (Catastrophizing, Agency Deficit).
