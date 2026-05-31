# collect.py -- download raw (code, caption) pairs from each source
import re, io, json, tarfile, hashlib, time, requests
from pathlib import Path
from typing import Iterator

OUT = Path("raw")

def _save(domain: str, code: str, caption: str, uid: str):
    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{uid}.json"
    if not p.exists():
        p.write_text(json.dumps({"code": code, "caption": caption}))

# TikZ

def collect_datikzv2(limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset("nllg/datikz-v2", split="train")
    total = min(limit, len(ds)) if limit else len(ds)
    if limit:
        ds = ds.select(range(total))
    print(f"  datikzv2: {total} rows to collect")
    img_dir = OUT / "tikz_prerendered"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(ds):
        uid = hashlib.md5(row["code"].encode()).hexdigest()
        _save("tikz", row["code"], row.get("caption", ""), uid)
        img = row.get("image")
        if img is not None:
            out = img_dir / f"{uid}.png"
            if not out.exists():
                img.save(str(out))
        if (i + 1) % 1000 == 0:
            print(f"  datikzv2: {i + 1}/{total}")

# TikZ: arXiv source tarballs

TIKZ_PAT = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL)

def _arxiv_ids(category: str, limit: int) -> Iterator[str]:
    base = "http://export.arxiv.org/api/query"
    start = 0
    while start < limit:
        r = requests.get(base, params={
            "search_query": f"cat:{category}", "start": start,
            "max_results": 100, "sortBy": "submittedDate", "sortOrder": "descending"})
        ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", r.text)
        if not ids:
            break
        yield from ids
        start += 100
        time.sleep(3)

def _fetch_tarball(arxiv_id: str) -> bytes | None:
    try:
        r = requests.get(f"https://arxiv.org/src/{arxiv_id}", timeout=30)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None

def collect_arxiv_tikz(category: str = "cs.CV", max_papers: int = 1000):
    print(f"  arxiv {category}: up to {max_papers} papers")
    seen = total_tikz = 0
    for arxiv_id in _arxiv_ids(category, max_papers * 2):
        if seen >= max_papers:
            break
        data = _fetch_tarball(arxiv_id)
        if not data:
            continue
        paper_tikz = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                for member in tf.getmembers():
                    if not member.name.endswith(".tex"):
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    text = f.read().decode("utf-8", errors="ignore")
                    for m in TIKZ_PAT.finditer(text):
                        code = m.group(0)
                        uid = hashlib.md5(code.encode()).hexdigest()
                        _save("tikz", code, "", uid)
                        paper_tikz += 1
        except Exception:
            pass
        seen += 1
        total_tikz += paper_tikz
        if seen % 50 == 0:
            print(f"  arxiv: {seen}/{max_papers} papers, {total_tikz} tikz snippets")
        time.sleep(1)
    print(f"  arxiv done: {seen} papers, {total_tikz} tikz snippets")

# SVG: SVG-Stack corpus + Wikimedia Commons

def collect_svg_stack_hf(limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset("darknoon/svg-stack-filtered", split="train")
    total = min(limit, len(ds)) if limit else len(ds)
    if limit:
        ds = ds.select(range(total))
    print(f"  svg-stack-hf: {total} rows to collect")
    saved = skipped = 0
    for i, row in enumerate(ds):
        code = row.get("svg") or row.get("code") or ""
        if not code or "<svg" not in code[:2000]:
            skipped += 1
            continue
        caption = row.get("description") or row.get("caption") or ""
        uid = hashlib.md5(code.encode()).hexdigest()
        _save("svg", code, caption, uid)
        saved += 1
        if (i + 1) % 1000 == 0:
            print(f"  svg-stack-hf: {i + 1}/{total} processed ({saved} saved, {skipped} skipped)")

def collect_svg_stack(jsonl_path: str):
    # SVG-Stack JSONL: fields "svg" and "description"
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            uid = hashlib.md5(row["svg"].encode()).hexdigest()
            _save("svg", row["svg"], row.get("description", ""), uid)

_HEADERS = {"User-Agent": "bagel-data-collector/1.0 (research; contact via github)"}

def collect_wikimedia_svg(max_items: int = 10000):
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "generator": "categorymembers",
        "gcmtitle": "Category:SVG files", "gcmtype": "file",
        "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|mime",
        "format": "json"}
    fetched = 0
    while fetched < max_items:
        resp = requests.get(api, params=params, headers=_HEADERS, timeout=15)
        if not resp.ok or not resp.text.strip():
            print(f"wikimedia API error: {resp.status_code} -- stopping")
            break
        r = resp.json()
        for page in r.get("query", {}).get("pages", {}).values():
            ii = page.get("imageinfo", [{}])[0]
            if ii.get("mime") != "image/svg+xml":
                continue
            try:
                resp = requests.get(ii["url"], timeout=10, headers=_HEADERS)
                svg = resp.text
                # skip HTML error pages
                if "<svg" not in svg[:2000]:
                    continue
                uid = hashlib.md5(svg.encode()).hexdigest()
                _save("svg", svg, page.get("title", ""), uid)
                fetched += 1
            except Exception:
                pass
        if fetched % 200 == 0 and fetched:
            print(f"wikimedia svg: {fetched} collected")
        cont = r.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(1)

# Matplotlib: Stack Overflow accepted answers

_CODE_PAT = re.compile(r"<code>(.*?)</code>", re.DOTALL)

def collect_stackoverflow_matplotlib(api_key: str = "", max_pages: int = 100):
    base = "https://api.stackexchange.com/2.3"
    total_saved = 0
    for page in range(1, max_pages + 1):
        r = requests.get(f"{base}/questions", params={
            "tagged": "matplotlib", "site": "stackoverflow",
            "filter": "withbody", "pagesize": 100, "page": page,
            "sort": "votes", "order": "desc", "key": api_key}).json()
        page_saved = 0
        quota = r.get("quota_remaining", "?")
        for q in r.get("items", []):
            aid = q.get("accepted_answer_id")
            if not aid:
                continue
            ar = requests.get(f"{base}/answers/{aid}", params={
                "site": "stackoverflow", "filter": "withbody",
                "key": api_key}).json()
            body = ar.get("items", [{}])[0].get("body", "")
            for m in _CODE_PAT.finditer(body):
                code = m.group(1)
                if "plt." not in code and "matplotlib" not in code:
                    continue
                uid = hashlib.md5(code.encode()).hexdigest()
                _save("matplotlib", code, q.get("title", ""), uid)
                page_saved += 1
        total_saved += page_saved
        print(f"  stackoverflow p{page}/{max_pages}: +{page_saved} snippets "
              f"(total {total_saved}, quota remaining: {quota})")
        if not r.get("has_more"):
            print(f"  stackoverflow: no more pages")
            break
        time.sleep(1)

# Manim: GitHub search

def collect_manim_github(token: str, max_repos: int = 200):
    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github.v3+json"}
    url = "https://api.github.com/search/code"
    seen_repos: set[str] = set()
    total_scenes = 0
    page = 1
    while len(seen_repos) < max_repos:
        r = requests.get(url, headers=headers, params={
            "q": "manim Scene construct language:Python",
            "per_page": 50, "page": page}).json()
        items = r.get("items", [])
        rate = r.get("message", "")
        if rate and "rate limit" in rate.lower():
            print(f"  github rate limited -- stopping at {len(seen_repos)} repos")
            break
        if not items:
            print(f"  github: no more results at page {page}")
            break
        page_scenes = 0
        for item in items:
            repo = item["repository"]["full_name"]
            if repo in seen_repos:
                continue
            seen_repos.add(repo)
            raw_url = item["html_url"].replace(
                "https://github.com", "https://raw.githubusercontent.com"
            ).replace("/blob/", "/")
            try:
                code = requests.get(raw_url, timeout=10).text
                for scene in re.finditer(
                        r"(class \w+\([^)]*Scene[^)]*\):.*?)(?=\nclass |\Z)",
                        code, re.DOTALL):
                    s = scene.group(1).strip()
                    uid = hashlib.md5(s.encode()).hexdigest()
                    _save("manim", s, "", uid)
                    page_scenes += 1
            except Exception:
                pass
        total_scenes += page_scenes
        print(f"  github p{page}: {len(seen_repos)}/{max_repos} repos, "
              f"+{page_scenes} scenes (total {total_scenes})")
        page += 1
        time.sleep(2)
