import pandas as pd
from scipy import stats

ground_truth = pd.read_csv("data/processed/llm_scores_clbp_dx_405b.csv")
small = pd.read_csv("data/processed/llm_scores_clbp_dx_small.csv")
rag = pd.read_csv("data/processed/llm_scores_clbp_dx_rag.csv")

df = ground_truth.merge(small, on="study_id", suffixes=("_405b", "_small"))
df = df.merge(rag, on="study_id", suffixes=("", "_rag"))

print(f"Subjects matched: {len(df)}\n")

METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL",
           "Anxiety", "Catastrophizing", "Rumination",
           "Narrative_Fragmentation", "Agency_Deficit"]

print(f"{'Metric':<25} {'Baseline':>10} {'RAG':>10}")
print("-" * 50)

for metric in METRICS:
    col_405b = f"{metric}_405b"
    col_small = f"{metric}_small"
    col_rag = metric if metric in df.columns else f"{metric}_rag"

    results = {}
    for name, col in [("baseline", col_small), ("rag", col_rag)]:
        if col in df.columns:
            r, p = stats.spearmanr(df[col_405b], df[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            results[name] = f"{r:.3f}{sig}"
        else:
            results[name] = "N/A"

    print(f"{metric:<25} {results.get('baseline','N/A'):>10} {results.get('rag','N/A'):>10}")