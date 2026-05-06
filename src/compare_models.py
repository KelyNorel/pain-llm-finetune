import pandas as pd
from scipy import stats

# Cargar ambos CSVs
small = pd.read_csv("data/processed/llm_scores_clbp_small.csv")
big = pd.read_csv("data/processed/llm_scores_clbp_405b.csv")

# Merge por study_id
df = small.merge(big, on="study_id", suffixes=("_small", "_405b"))
print(f"Subjects matched: {len(df)}\n")

METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL", 
           "Anxiety", "Catastrophizing", "Rumination", 
           "Narrative_Fragmentation", "Agency_Deficit"]

print(f"{'Metric':<25} {'Spearman r':>10} {'p-value':>10}")
print("-" * 50)

for metric in METRICS:
    col_small = f"{metric}_small"
    col_big = f"{metric}_405b"
    
    if col_small in df.columns and col_big in df.columns:
        r, p = stats.spearmanr(df[col_small], df[col_big])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{metric:<25} {r:>10.3f} {p:>10.4f} {sig}")