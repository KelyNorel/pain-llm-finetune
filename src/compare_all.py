import pandas as pd
from scipy import stats

# Cargar todos los CSVs
ground_truth = pd.read_csv("data/processed/llm_scores_clbp_405b.csv")
baseline = pd.read_csv("data/processed/llm_scores_clbp_small.csv")
rag = pd.read_csv("data/processed/llm_scores_clbp_rag.csv")
finetuned = pd.read_csv("data/processed/llm_scores_clbp_finetuned.csv")
finetuned_v2 = pd.read_csv("data/processed/llm_scores_clbp_finetuned_v2.csv")
finetuned_v3 = pd.read_csv("data/processed/llm_scores_clbp_finetuned_v3.csv")
rag2 = pd.read_csv("data/processed/llm_scores_clbp_rag_v2.csv")


# Merge todo por study_id
df = ground_truth.merge(baseline, on="study_id", suffixes=("_405b", "_baseline"))
df = df.merge(rag, on="study_id", suffixes=("", "_rag"))
df = df.merge(finetuned, on="study_id", suffixes=("", "_finetuned"))
df = df.merge(finetuned_v2, on="study_id", suffixes=("", "_v2"))
df = df.merge(finetuned_v3, on="study_id", suffixes=("", "_v3"))
df = df.merge(rag2, on="study_id", suffixes=("", "_rag2"))

print(f"Subjects matched: {len(df)}\n")

METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL",
           "Anxiety", "Catastrophizing", "Rumination",
           "Narrative_Fragmentation", "Agency_Deficit"]


print(f"{'Metric':<25} {'Baseline':>10} {'RAG-v1':>10} {'RAG-v2':>10} {'FT-v1':>10} {'FT-v2':>10} {'FT-v3':>10}")
print("-" * 62)

for metric in METRICS:
    col_405b = f"{metric}_405b"
    col_base = f"{metric}_baseline"
    col_rag = f"{metric}_rag"
    col_rag2 = f"{metric}_rag2"
    col_ft = f"{metric}_finetuned"
    col_v2 = f"{metric}_v2"
    col_v3 = f"{metric}_v3"

    # Renombrar columnas del merge
    if col_rag not in df.columns:
        col_rag = metric + "_rag" if metric + "_rag" in df.columns else metric
    if col_ft not in df.columns:
        col_ft = metric + "_finetuned" if metric + "_finetuned" in df.columns else metric

    results = {}
    for name, col in [("baseline", col_base), ("rag", col_rag), ("rag2", col_rag2), ("ft", col_ft),("v2", col_v2),("v3", col_v3)]:
        if col in df.columns:
            r, p = stats.spearmanr(df[col_405b], df[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            results[name] = f"{r:.3f}{sig}"
        else:
            results[name] = "N/A"

    print(f"{metric:<25} {results.get('baseline','N/A'):>10} {results.get('rag','N/A'):>10} {results.get('rag2','N/A'):>10} {results.get('ft','N/A'):>12} {results.get('v2','N/A'):>10} {results.get('v3','N/A'):>10}")