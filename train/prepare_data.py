"""
Data preparation pipeline:
  1. Read all .txt files from data_dir
  2. Tokenize with tiktoken (no training needed)
  3. Save as binary .bin files (train / val)

Usage:

python -m train.prepare_data \
    --data-dir ~/pet_projects/datasets/llm_pretrain/test_pretrain/raw_txt \
    --output-dir ~/pet_projects/datasets/llm_pretrain/test_pretrain/tokenized_binary_texts \
    --encoding cl100k_base
"""

import argparse
import glob
import json
import os

import numpy as np
import tiktoken
from tqdm import tqdm


def read_corpus(data_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    texts = []
    for path in tqdm(paths, desc="Reading files"):
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())

    print(f"  {len(paths)} files read")
    return "\n\n".join(texts)


def prepare(
    data_dir: str,
    output_dir: str,
    encoding: str,
    val_ratio: float,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Read corpus ────────────────────────────────────────────────────────
    print("\n[1/3] Reading corpus...")
    corpus = read_corpus(data_dir)
    print(f"  corpus size: {len(corpus):,} characters")

    # ── 2. Tokenize with tiktoken ─────────────────────────────────────────────
    print(f"\n[2/3] Tokenizing with tiktoken encoding='{encoding}'...")
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode_ordinary(corpus)
    print(f"  vocab size:   {enc.n_vocab:,}")
    print(f"  total tokens: {len(tokens):,}")
    print(f"  compression:  {len(corpus) / len(tokens):.2f} chars/token")

    # ── 3. Save binary + metadata ─────────────────────────────────────────────
    print(f"\n[3/3] Saving binary files → {output_dir}")
    arr = np.array(tokens, dtype=np.int32)

    split = int(len(arr) * (1 - val_ratio))
    splits = {"train": arr[:split], "val": arr[split:]}

    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.bin")
        data.tofile(path)
        print(f"  {name:5s}: {len(data):>10,} tokens → {path}")

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"encoding": encoding, "vocab_size": enc.n_vocab}, f, indent=2)
    print(f"  meta  → {meta_path}")

    print("\n✓ Done. Ready to train.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="~/pet_projects/datasets/llm_pretrain/test_pretrain")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--encoding",   default="cl100k_base")
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    args = parser.parse_args()

    prepare(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        encoding=args.encoding,
        val_ratio=args.val_ratio,
    )
