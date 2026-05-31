# make_dataset.py -- load sequences into a HuggingFace Dataset and save as parquet
import json, argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
SEQUENCES = HERE / "sequences"

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def flatten_full(row: dict) -> dict:
    msgs = row["messages"]
    user_msg = msgs[0]["content"]
    asst = msgs[1]["content"]
    code = next(c["text"] for c in asst if c["type"] == "code")
    img  = next(c["path"] for c in asst if c["type"] == "image")
    return {"type": row["type"], "domain": row["domain"],
            "prompt": user_msg, "code": code, "image_path": img}

def flatten_trace(row: dict) -> dict:
    return {"type": row["type"], "domain": row["domain"],
            "prompt": "",
            "code": row["steps"][-1],       # final (complete) program
            "image_path": row["final_image"],
            "steps": json.dumps(row["steps"])}

def flatten_prompt_free(row: dict) -> dict:
    asst = row["messages"][0]["content"]
    code = next(c["text"] for c in asst if c["type"] == "code")
    img  = next(c["path"] for c in asst if c["type"] == "image")
    return {"type": row["type"], "domain": row["domain"],
            "prompt": "", "code": code, "image_path": img}

FLATTENERS = {
    "full": flatten_full,
    "trace": flatten_trace,
    "prompt_free": flatten_prompt_free,
}

def build_dataframe(domain: str) -> pd.DataFrame:
    rows = []
    for split in ("full", "trace", "prompt_free"):
        for raw in load_jsonl(SEQUENCES / domain / f"{split}.jsonl"):
            rows.append(FLATTENERS[split](raw))
    df = pd.DataFrame(rows)
    # resolve relative image paths to absolute
    def _abs(p: str) -> str:
        path = Path(p)
        return str(HERE / p) if not path.is_absolute() else p
    df["image_path"] = df["image_path"].apply(_abs)
    df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="tikz")
    p.add_argument("--out", default=None, help="output parquet path")
    p.add_argument("--hf", action="store_true", help="also save as HuggingFace Dataset")
    args = p.parse_args()

    print(f"loading {args.domain} sequences...")
    df = build_dataframe(args.domain)

    print(df[["type", "domain", "image_exists"]].value_counts().to_string())
    print(f"\ntotal rows: {len(df)}")
    print(df[["prompt", "code"]].head(3).to_string())

    out = args.out or str(SEQUENCES / args.domain / "dataset.parquet")
    df.to_parquet(out, index=False)
    print(f"\nsaved parquet -> {out}")

    if args.hf:
        from datasets import Dataset, Image as HFImage
        hf_ds = Dataset.from_pandas(df.drop(columns=["steps", "image_exists"],
                                             errors="ignore"))
        hf_ds = hf_ds.cast_column("image_path", HFImage())
        hf_out = str(SEQUENCES / args.domain / "hf_dataset")
        hf_ds.save_to_disk(hf_out)
        print(f"saved HF dataset -> {hf_out}")

if __name__ == "__main__":
    main()
