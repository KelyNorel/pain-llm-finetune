import json
import time
import pandas as pd
from pathlib import Path
from mlx_lm import load, generate

TRANSCRIPTS_DIR = Path("data/transcripts/general")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "mlx-community/Llama-3.2-3B-Instruct-4bit"
#ADAPTER_PATH = "models/lora_adapters"
ADAPTER_PATH =  "models/lora_adapters_v2"

METRICS = ["Physical_Pain", "Emotional_Pain", "Depression", "poor_QoL",
           "Anxiety", "Catastrophizing", "Rumination",
           "Narrative_Fragmentation", "Agency_Deficit"]

PROMPT_TEMPLATE = """You are a knowledgeable psychiatrist evaluating a patient interview. Score these 9 clinical metrics from 0.0 to 10.0 (higher = worse):
Physical_Pain, Emotional_Pain, Depression, poor_QoL, Anxiety, Catastrophizing, Rumination, Narrative_Fragmentation, Agency_Deficit

Transcript:
{transcript}

Response:
"""

def extract_study_id(filename: str) -> str:
    return filename.replace("_CLBP.txt", "")

def parse_scores(raw: str) -> dict:
    """Parsea JSON de la respuesta del modelo."""
    try:
        # Limpiar markdown si viene
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        # Buscar primer { hasta último }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as e:
        print(f"  Parse error: {e}")
    return {}

def run_batch():
    print("Loading fine-tuned model...")
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
            prompt = PROMPT_TEMPLATE.format(transcript=transcript)

            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=200,
                verbose=False
            )

            scores = parse_scores(response)
            scores["study_id"] = int(study_id)
            results.append(scores)
            elapsed = time.time() - start
            print(f"  Scores: {scores}")
            print(f"  ✓ Done ({elapsed:.1f}s)\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")

    total = time.time() - start_total
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Average per subject: {total/len(results):.1f}s")

    df = pd.DataFrame(results)
    
    # Asegurar que todas las columnas existen
    for col in ["study_id"] + METRICS:
        if col not in df.columns:
            df[col] = None
    
    df = df[["study_id"] + METRICS]
    output_path = OUTPUT_DIR / "llm_scores_clbp_finetuned_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    df = run_batch()
    print(df.describe())
