import json
import time
import chromadb
import pandas as pd
from pathlib import Path
from mlx_lm import load, generate

TRANSCRIPTS_DIR = Path("data/transcripts/general")
OUTPUT_DIR = Path("data/processed")
CHROMA_DIR = Path("data/chroma_db")

MODEL_PATH = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER_PATH = "models/lora_adapters_v3"

METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL",
           "Anxiety", "Catastrophizing", "Rumination",
           "Narrative_Fragmentation", "Agency_Deficit"]

# ChromaDB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection("clinical_literature")

METRICS_QUERIES = {
    "Physical_Pain": "physical pain chronic back pain intensity assessment",
    "Emotional_Pain": "emotional pain negative affect chronic pain",
    "Depression": "depression symptoms assessment chronic pain",
    "poor_QoL": "quality of life chronic pain functional impact",
    "Anxiety": "anxiety symptoms chronic pain assessment",
    "Catastrophizing": "catastrophizing chronic pain cognitive assessment",
    "Rumination": "rumination repetitive thinking chronic pain depression",
    "Narrative_Fragmentation": "narrative coherence fragmentation autobiographical memory",
    "Agency_Deficit": "sense of agency personal control locus of control depression"
}

def retrieve_context(query: str, n_results: int = 2) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = ""
    for chunk, source in zip(chunks, sources):
        context += f"[{source}]\n{chunk}\n\n"
    return context

def score_transcript_rag_ft(transcript: str, model, tokenizer) -> dict:
    results = {}
    for metric in METRICS:
        context = retrieve_context(METRICS_QUERIES[metric])
        prompt = f"""You are a knowledgeable psychiatrist evaluating a patient interview. Use this clinical reference:

{context}

Based on the transcript below, score ONLY this ONE metric:
{metric}: 0.0 to 10.0 (higher = worse)

Respond with ONLY this JSON, nothing else:
{{"{metric}": VALUE}}

Replace VALUE with a decimal number. Do not include other metrics.

Transcript:
{transcript}

Response:
{{"{metric}": """

        # DEBUG
        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        print(f"  [{metric}] Raw: '{response[:200]}'")
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
            raw = response.strip()
            raw = raw.rstrip('}')
            raw = '{' + f'"{metric}": ' + raw + '}'
            score = json.loads(raw)
            value = float(list(score.values())[0])
            print(f"  {metric}: {value}")
            results[metric] = value
        except Exception as e:
            print(f"  ERROR {metric}: {e}")

    return results

def extract_study_id(filename: str) -> str:
    return filename.replace("_CLBP.txt", "")

def run_batch():
    print("Loading fine-tuned model with RAG...")
    model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print("Model loaded!\n")

    start_total = time.time()
    txt_files = sorted(TRANSCRIPTS_DIR.glob("*_CLBP.txt"))
    print(f"Found {len(txt_files)} transcripts\n")

    results = []

    for i, txt_path in enumerate(txt_files):
        study_id = extract_study_id(txt_path.name)
        print(f"Processing {study_id}... ({i+1}/{len(txt_files)})")
        start = time.time()
       

        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            scores = score_transcript_rag_ft(transcript, model, tokenizer)
            scores["study_id"] = int(study_id)
            results.append(scores)
            elapsed = time.time() - start
            print(f"  ✓ Done ({elapsed:.1f}s)\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            print(f"  Raw: '{response[:200] if 'response' in locals() else 'no response'}'")

    total = time.time() - start_total
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Average per subject: {total/len(results):.1f}s")

    df = pd.DataFrame(results)
    for col in ["study_id"] + METRICS:
        if col not in df.columns:
            df[col] = None
    df = df[["study_id"] + METRICS]

    output_path = OUTPUT_DIR / "llm_scores_clbp_rag_finetuned.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    df = run_batch()
    print(df.describe())