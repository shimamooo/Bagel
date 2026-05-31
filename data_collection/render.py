# render.py, execute code and produce PNG images
import os, re, sys, json, subprocess, tempfile
from pathlib import Path

RAW = Path("raw")
RENDERED = Path("rendered")
TIMEOUT = 30

# TikZ

_TIKZ_DOC = r"""\documentclass[border=2pt]{{standalone}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepackage{{amsmath,amssymb}}
\begin{{document}}
{body}
\end{{document}}
"""

def render_tikz(code: str, out_png: Path) -> bool:
    body = code if r"\begin{tikzpicture}" in code else (
        r"\begin{tikzpicture}" + "\n" + code + "\n" + r"\end{tikzpicture}")
    with tempfile.TemporaryDirectory() as tmp:
        tex = os.path.join(tmp, "fig.tex")
        pdf = os.path.join(tmp, "fig.pdf")
        Path(tex).write_text(_TIKZ_DOC.format(body=body))
        try:
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", tmp, tex],
                capture_output=True, timeout=TIMEOUT)
            if r.returncode != 0 or not os.path.exists(pdf):
                return False
            # pdf2image via poppler
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf, dpi=150)
            if not pages:
                return False
            pages[0].save(str(out_png))
            return True
        except (subprocess.TimeoutExpired, Exception):
            return False

# SVG

def render_svg(code: str, out_png: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
        f.write(code)
        svg_path = f.name
    try:
        r = subprocess.run(["resvg", svg_path, str(out_png)],
                           capture_output=True, timeout=TIMEOUT)
        return r.returncode == 0 and out_png.exists()
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(svg_path)
        except OSError:
            pass

# Matplotlib

_MPL_WRAP = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

{code}

plt.savefig({out!r}, bbox_inches="tight", dpi=150)
plt.close("all")
"""

def render_matplotlib(code: str, out_png: Path) -> bool:
    script = _MPL_WRAP.format(code=code, out=str(out_png))
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(script)
        sp = f.name
    try:
        r = subprocess.run(
            [sys.executable, sp], capture_output=True, timeout=TIMEOUT,
            env={**os.environ, "MPLBACKEND": "Agg"})
        return r.returncode == 0 and out_png.exists()
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(sp)
        except OSError:
            pass

# Manim

def render_manim(code: str, out_png: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        sp = os.path.join(tmp, "scene.py")
        Path(sp).write_text(code)
        m = re.search(r"class (\w+)\(", code)
        if not m:
            return False
        try:
            r = subprocess.run(
                ["manim", sp, m.group(1), "-ql", "--format=png",
                 "--media_dir", tmp],
                capture_output=True, timeout=60)
            pngs = sorted(Path(tmp).glob("**/*.png"))
            if not pngs:
                return False
            import shutil
            shutil.copy(str(pngs[0]), str(out_png))
            return True
        except (subprocess.TimeoutExpired, Exception):
            return False

# Dispatch

RENDERERS = {
    "tikz": render_tikz,
    "svg": render_svg,
    "matplotlib": render_matplotlib,
    "manim": render_manim,
}

def render_all(domain: str):
    src = RAW / domain
    prerendered = RAW / f"{domain}_prerendered"  # images bundled with dataset
    if not src.exists():
        print(f"no raw data for {domain}")
        return
    dst = RENDERED / domain
    dst.mkdir(parents=True, exist_ok=True)
    ok = fail = skipped = 0
    for jf in sorted(src.glob("*.json")):
        out_png = dst / f"{jf.stem}.png"
        if out_png.exists():
            continue
        # use pre-rendered image if available (e.g. from DaTikZv2)
        pre = prerendered / f"{jf.stem}.png"
        if pre.exists():
            import shutil
            shutil.copy(str(pre), str(out_png))
            skipped += 1
            continue
        data = json.loads(jf.read_text())
        if RENDERERS[domain](data["code"], out_png):
            ok += 1
        else:
            fail += 1
        if (ok + fail) % 500 == 0:
            print(f"{domain}: {ok} rendered / {fail} failed / {skipped} pre-rendered")
    print(f"{domain} done: {ok} rendered / {fail} failed / {skipped} pre-rendered")
