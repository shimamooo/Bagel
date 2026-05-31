# dedup.py -- AST-hash and CLIP-embedding deduplication
import re, json, hashlib
from pathlib import Path

RAW = Path("raw")
RENDERED = Path("rendered")
DEDUPED = Path("deduped")

def _normalize(code: str) -> str:
    code = re.sub(r"%[^\n]*", "", code) # LaTeX/TikZ comments
    code = re.sub(r"#[^\n]*", "", code) # Python comments
    code = re.sub(r"<!--.*?-->", "", code, flags=re.DOTALL) # XML comments
    return re.sub(r"\s+", " ", code).strip()

def ast_dedup(domain: str):
    src = RAW / domain
    dst = DEDUPED / domain
    dst.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = dropped = 0
    for jf in sorted(src.glob("*.json")):
        data = json.loads(jf.read_text())
        h = hashlib.md5(_normalize(data["code"]).encode()).hexdigest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        (dst / jf.name).write_text(jf.read_text())
        kept += 1
    print(f"{domain} ast_dedup: kept {kept} / dropped {dropped}")

# CLIP embedding dedup (visual duplicates by CLIP)

def clip_dedup(domain: str, radius: float = 0.05, batch: int = 256):
    import torch
    import numpy as np
    from PIL import Image

    src = DEDUPED / domain
    img_dir = RENDERED / domain
    out_dir = DEDUPED / f"{domain}_clip"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32").to(device)
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    files = [jf for jf in sorted(src.glob("*.json"))
             if (img_dir / f"{jf.stem}.png").exists()]
    uids: list[str] = []
    chunks: list[np.ndarray] = []

    for i in range(0, len(files), batch):
        batch_files = files[i:i + batch]
        imgs, valid = [], []
        for jf in batch_files:
            try:
                imgs.append(Image.open(img_dir / f"{jf.stem}.png").convert("RGB"))
                valid.append(jf.stem)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(device)
            feats = model(**inp).image_embeds
            feats = feats / feats.norm(dim=-1, keepdim=True)
        uids.extend(valid)
        chunks.append(feats.cpu().float().numpy())
        if i % 5000 == 0:
            print(f"{domain} clip embed: {i}/{len(files)}")

    if not chunks:
        return
    E = np.vstack(chunks).astype("float32")

    # greedy dedup with faiss inner-product index
    try:
        import faiss
        index = faiss.IndexFlatIP(E.shape[1])
        kept_mask = np.ones(len(uids), dtype=bool)
        k = min(64, len(uids))
        for i in range(len(uids)):
            if not kept_mask[i]:
                continue
            D, I = index.search(E[i:i + 1], k)
            for j, d in zip(I[0], D[0]):
                if j != -1 and j != i and d > 1.0 - radius:
                    kept_mask[j] = False
            index.add(E[i:i + 1])
    except ImportError:
        # faiss not available: fall back
        print("faiss not found")
        kept_mask = np.ones(len(uids), dtype=bool)
        for i in range(len(uids)):
            if not kept_mask[i]:
                continue
            sims = E[i] @ E[i + 1:].T
            drop = np.where(sims > 1.0 - radius)[0] + (i + 1)
            kept_mask[drop] = False

    kept = dropped = 0
    for idx, uid in enumerate(uids):
        if kept_mask[idx]:
            (out_dir / f"{uid}.json").write_text(
                (src / f"{uid}.json").read_text())
            kept += 1
        else:
            dropped += 1
    print(f"{domain} clip_dedup: kept {kept} / dropped {dropped}")
