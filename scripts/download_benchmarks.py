"""Download Tier 1 benchmark datasets for Loqi evaluation.

Benchmarks:
  - MuSiQue (multi-hop QA, 2-4 hops) — tests focused/diffuse graph traversal
  - HotpotQA (multi-hop QA, 2 hops) — standard baseline, used by all graph-RAG papers
  - LongMemEval (long-term memory) — tests trigger recall of standing instructions
  - MemoryAgentBench (memory agent competencies) — tests Hebbian learning/forgetting

All data is downloaded to data/raw/{benchmark_name}/.
Run: python scripts/download_benchmarks.py [--benchmark NAME]
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_musique():
    """Download MuSiQue via HuggingFace datasets API."""
    from datasets import load_dataset

    out = DATA_DIR / "musique"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "validation.jsonl").exists():
        print("  MuSiQue already downloaded, skipping.")
        return

    print("  Downloading MuSiQue (answerable subset)...")
    ds = load_dataset("bdsaglam/musique", split="validation")

    with open(out / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Saved {len(ds)} validation examples to {out / 'validation.jsonl'}")

    # Also grab a small train sample for Hebbian learning tests
    print("  Downloading MuSiQue train sample (first 1000)...")
    ds_train = load_dataset("bdsaglam/musique", split="train[:1000]")

    with open(out / "train_sample.jsonl", "w", encoding="utf-8") as f:
        for row in ds_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Saved {len(ds_train)} train examples to {out / 'train_sample.jsonl'}")


def download_hotpotqa():
    """Download HotpotQA distractor setting via HuggingFace."""
    from datasets import load_dataset

    out = DATA_DIR / "hotpotqa"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "validation.jsonl").exists():
        print("  HotpotQA already downloaded, skipping.")
        return

    print("  Downloading HotpotQA (distractor, validation)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    with open(out / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Saved {len(ds)} validation examples to {out / 'validation.jsonl'}")


def download_longmemeval():
    """Download LongMemEval oracle and small-scale chat histories."""
    import urllib.request

    out = DATA_DIR / "longmemeval"
    out.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

    files = {
        "longmemeval_oracle.json": "oracle (evidence-only, ~500 questions)",
        "longmemeval_s_cleaned.json": "small-scale (~40 sessions per question)",
    }

    for filename, desc in files.items():
        target = out / filename
        if target.exists():
            print(f"  {filename} already downloaded, skipping.")
            continue

        print(f"  Downloading {desc}...")
        url = f"{base_url}/{filename}"
        urllib.request.urlretrieve(url, target)

        # Validate JSON
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Saved {len(data)} questions to {target}")


def download_memoryagentbench():
    """Download MemoryAgentBench via HuggingFace datasets API."""
    from datasets import load_dataset

    out = DATA_DIR / "memoryagentbench"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "manifest.json").exists():
        print("  MemoryAgentBench already downloaded, skipping.")
        return

    print("  Downloading MemoryAgentBench...")
    ds = load_dataset("ai-hyz/MemoryAgentBench")

    manifest = {}
    for split_name in ds:
        split_path = out / f"{split_name}.jsonl"
        split_ds = ds[split_name]

        with open(split_path, "w", encoding="utf-8") as f:
            for row in split_ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest[split_name] = len(split_ds)
        print(f"  Saved {len(split_ds)} examples to {split_path}")

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


BENCHMARKS = {
    "musique": download_musique,
    "hotpotqa": download_hotpotqa,
    "longmemeval": download_longmemeval,
    "memoryagentbench": download_memoryagentbench,
}


def main():
    parser = argparse.ArgumentParser(description="Download Loqi evaluation benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Which benchmark to download (default: all)",
    )
    args = parser.parse_args()

    targets = BENCHMARKS if args.benchmark == "all" else {args.benchmark: BENCHMARKS[args.benchmark]}

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, fn in targets.items():
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            print("  Continuing with next benchmark...", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
