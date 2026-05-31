# pipeline.py -- CLI
import argparse, sys

DOMAINS = ["tikz", "svg", "matplotlib", "manim"]

def run_collect(args):
    from collect import (collect_datikzv2, collect_arxiv_tikz,
                         collect_svg_stack, collect_wikimedia_svg,
                         collect_stackoverflow_matplotlib, collect_manim_github)
    if args.domain in ("tikz", "all"):
        print("collecting DaTikZv2.")
        collect_datikzv2(limit=args.limit)
        if args.arxiv_papers > 0:
            print("collecting arXiv TikZ")
            collect_arxiv_tikz(max_papers=args.arxiv_papers)
    if args.domain in ("svg", "all"):
        if args.svg_jsonl:
            print("collecting SVG-Stack")
            collect_svg_stack(args.svg_jsonl)
        print("collecting Wikimedia SVG.")
        collect_wikimedia_svg(max_items=args.wikimedia_max)
    if args.domain in ("matplotlib", "all"):
        if not args.so_key:
            print("--so-key required for matplotlib", file=sys.stderr)
        else:
            print("collecting Stack Overflow matplotlib.")
            collect_stackoverflow_matplotlib(args.so_key)
    if args.domain in ("manim", "all"):
        if not args.gh_token:
            print("gh token")
        else:
            print("collecting manim")
            from collect import collect_manim_github
            collect_manim_github(args.gh_token)

def run_render(args):
    from render import render_all
    domains = DOMAINS if args.domain == "all" else [args.domain]
    for d in domains:
        print(f"rendering {d}")
        render_all(d)

def run_dedup(args):
    from dedup import ast_dedup, clip_dedup
    domains = DOMAINS if args.domain == "all" else [args.domain]
    for d in domains:
        print(f"dedup {d}")
        ast_dedup(d)
        clip_dedup(d, radius=args.clip_radius)

def run_sequences(args):
    from sequences import build_sequences
    domains = DOMAINS if args.domain == "all" else [args.domain]
    for d in domains:
        print(f"building sequences for {d}")
        build_sequences(d)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "render", "dedup", "sequences", "all"])
    p.add_argument("--domain", default="all", choices=DOMAINS + ["all"])
    p.add_argument("--limit", type=int, default=0,
                   help="max samples to collect per source")
    p.add_argument("--arxiv-papers", type=int, default=0,
                   help="arXiv papers to scrape for TikZ (0=skip)")
    p.add_argument("--wikimedia-max", type=int, default=10000)
    p.add_argument("--svg-jsonl", default=None)
    p.add_argument("--so-key", default=None, help="Stack Overflow API key")
    p.add_argument("--gh-token", default=None, help="GitHub token")
    p.add_argument("--clip-radius", type=float, default=0.05)
    args = p.parse_args()

    if args.stage == "all":
        run_collect(args)
        run_render(args)
        run_dedup(args)
        run_sequences(args)
    elif args.stage == "collect":
        run_collect(args)
    elif args.stage == "render":
        run_render(args)
    elif args.stage == "dedup":
        run_dedup(args)
    elif args.stage == "sequences":
        run_sequences(args)

if __name__ == "__main__":
    main()
