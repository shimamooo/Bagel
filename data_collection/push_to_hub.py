# push_to_hub.py -- upload dataset to HuggingFace Hub
# usage: python push_to_hub.py --repo your-username/tikz-bagel-dataset
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset, Features, Value, Image as HFImage

HERE = Path(__file__).parent
PARQUET = HERE / "sequences/tikz/dataset.parquet"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True,
                   help="HuggingFace repo id, e.g. your-username/tikz-bagel")
    p.add_argument("--private", action="store_true")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    print(f"loading {PARQUET}")
    df = pd.read_parquet(PARQUET)
    print(f"{len(df)} rows, columns: {df.columns.tolist()}")

    # drop helper column, keep image_path as the source for HFImage
    df = df.drop(columns=["image_exists"], errors="ignore")
    # rename so datasets knows to treat it as an image
    df = df.rename(columns={"image_path": "image"})

    print("building HuggingFace Dataset (embeds images -- may take a while)...")
    features = Features({
        "type":   Value("string"),
        "domain": Value("string"),
        "prompt": Value("string"),
        "code":   Value("string"),
        "image":  HFImage(),
        "steps":  Value("string"),   # JSON string for trace rows, null otherwise
    })
    ds = Dataset.from_pandas(df, features=features)

    print(f"pushing to hub: {args.repo} (split={args.split}, private={args.private})")
    ds.push_to_hub(
        repo_id=args.repo,
        split=args.split,
        private=args.private,
    )
    print(f"done: https://huggingface.co/datasets/{args.repo}")

if __name__ == "__main__":
    main()
