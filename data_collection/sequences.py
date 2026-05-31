import re, json, random
from pathlib import Path
from xml.etree import ElementTree as ET

DEDUPED = Path("deduped")
RENDERED = Path("rendered")
SEQUENCES = Path("sequences")

def _split_tikz(code: str) -> list[str]:
    lines = code.splitlines()
    prefixes: list[str] = []
    accumulated: list[str] = []
    for line in lines:
        accumulated.append(line)
        s = line.strip()
        if (s.startswith(r"\draw") or s.startswith(r"\node")) and s.endswith(";"):
            prefixes.append("\n".join(accumulated))
    return prefixes if len(prefixes) > 1 else [code]

def _split_matplotlib(code: str) -> list[str]:
    plot_re = re.compile(
        r"^\s*(plt|ax|axes)\.(plot|scatter|bar|barh|hist|imshow|contour|"
        r"contourf|fill_between|errorbar|pie|boxplot|violinplot|step|stem|"
        r"quiver|streamplot|hexbin|pcolormesh)\b",
        re.MULTILINE)
    lines = code.splitlines()
    split_lines: set[int] = set()
    pos = 0
    for i, line in enumerate(lines):
        if plot_re.match(line):
            split_lines.add(i)
        pos += len(line) + 1
    if not split_lines:
        return [code]
    prefixes: list[str] = []
    for i, line in enumerate(lines):
        if i in split_lines and i > 0:
            prefixes.append("\n".join(lines[:i]))
    prefixes.append(code)
    return prefixes if len(prefixes) > 1 else [code]

def _split_svg(code: str) -> list[str]:
    try:
        tree = ET.fromstring(code)
    except ET.ParseError:
        return [code]
    ns_match = re.match(r"\{([^}]+)\}", tree.tag)
    ns = f"{{{ns_match.group(1)}}}" if ns_match else ""
    svg_tag = re.match(r"<svg[^>]*>", code)
    header = svg_tag.group(0) if svg_tag else "<svg>"
    children = list(tree)
    if not children:
        return [code]
    prefixes: list[str] = []
    body = header
    for child in children:
        child_str = ET.tostring(child, encoding="unicode")
        body += "\n" + child_str
        prefixes.append(body + "\n</svg>")
    return prefixes if len(prefixes) > 1 else [code]

SPLITTERS = {
    "tikz": _split_tikz,
    "svg": _split_svg,
    "matplotlib": _split_matplotlib,
    "manim": lambda c: [c],
}

def _full(code: str, caption: str, img_path: str, domain: str) -> dict:
    prompt = caption.strip() if caption.strip() else f"Generate a {domain} program."
    return {
        "type": "full",
        "domain": domain,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": [
                {"type": "code", "text": code},
                {"type": "image", "path": img_path},
            ]},
        ],
    }

def _prompt_free(code: str, img_path: str, domain: str) -> dict:
    return {
        "type": "prompt_free",
        "domain": domain,
        "messages": [
            {"role": "assistant", "content": [
                {"type": "code", "text": code},
                {"type": "image", "path": img_path},
            ]},
        ],
    }

def _trace(prefixes: list[str], img_path: str, domain: str) -> dict:
    return {
        "type": "trace",
        "domain": domain,
        "steps": prefixes,
        "final_image": img_path,
    }

# Main builder

def build_sequences(domain: str, seed: int = 42):
    random.seed(seed)
    src = DEDUPED / f"{domain}_clip"
    if not src.exists():
        src = DEDUPED / domain
    img_dir = RENDERED / domain
    out_dir = SEQUENCES / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [jf for jf in sorted(src.glob("*.json"))
             if (img_dir / f"{jf.stem}.png").exists()]
    print(f"{domain}: {len(files)} samples")

    splitter = SPLITTERS.get(domain, lambda c: [c])
    full_f = (out_dir / "full.jsonl").open("w")
    trace_f = (out_dir / "trace.jsonl").open("w")
    pfree_f = (out_dir / "prompt_free.jsonl").open("w")

    for jf in files:
        data = json.loads(jf.read_text())
        code = data["code"]
        caption = data.get("caption", "")
        img_path = str(img_dir / f"{jf.stem}.png")
        r = random.random()
        if r < 0.70:
            full_f.write(json.dumps(_full(code, caption, img_path, domain)) + "\n")
        elif r < 0.90:
            prefixes = splitter(code)
            if len(prefixes) > 1:
                trace_f.write(json.dumps(_trace(prefixes, img_path, domain)) + "\n")
            else:
                full_f.write(json.dumps(_full(code, caption, img_path, domain)) + "\n")
        else:
            pfree_f.write(json.dumps(_prompt_free(code, img_path, domain)) + "\n")

    full_f.close()
    trace_f.close()
    pfree_f.close()
    print(f"{domain}: sequences written to {out_dir}")
