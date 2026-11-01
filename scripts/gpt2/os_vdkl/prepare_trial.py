"""Download the GPT-2 trial assets and build a small pretraining stand-in.

The stand-in matches the sequence length of the OpenWebText rows used by the
launcher, so a measured step includes the same student language-model forward.
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils.indexed_dataset import make_builder  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

TEACHER = os.path.join(ROOT, "results", "gpt2", "train", "sft", "gpt2-xlarge")
STUDENT = os.path.join(ROOT, "results", "gpt2", "train", "init", "gpt2-base")
DOLLY = os.path.join(ROOT, "processed_data", "dolly")
LM_DIR = os.path.join(ROOT, "processed_data", "openwebtext_smoke")


def _ready(path, filename):
    return os.path.exists(os.path.join(path, filename))


def download():
    if not _ready(TEACHER, "config.json"):
        print("downloading teacher MiniLLM/teacher-gpt2-1.5B", flush=True)
        snapshot_download("MiniLLM/teacher-gpt2-1.5B", local_dir=TEACHER)
    else:
        print("teacher checkpoint present", flush=True)

    if not _ready(STUDENT, "config.json"):
        print("downloading student MiniLLM/init-gpt2-120M", flush=True)
        snapshot_download("MiniLLM/init-gpt2-120M", local_dir=STUDENT)
    else:
        print("student checkpoint present", flush=True)

    train_bin = os.path.join(DOLLY, "full", "gpt2", "train_0.bin")
    if not os.path.exists(train_bin):
        print("downloading MiniLLM/dolly-processed full/gpt2", flush=True)
        snapshot_download(
            "MiniLLM/dolly-processed",
            repo_type="dataset",
            allow_patterns=["full/gpt2/*"],
            local_dir=DOLLY,
        )
    else:
        print("dolly data present", flush=True)


def build_lm_standin(num_sequences=64, seq_len=512):
    bin_path = os.path.join(LM_DIR, "train_0.bin")
    idx_path = os.path.join(LM_DIR, "train_0.idx")
    if os.path.exists(bin_path) and os.path.exists(idx_path):
        print(f"lm stand-in present at {LM_DIR}", flush=True)
        return
    os.makedirs(LM_DIR, exist_ok=True)
    builder = make_builder(bin_path, impl="mmap", dtype=np.uint16)
    for _ in range(num_sequences):
        tokens = torch.randint(0, 50257, (seq_len,), dtype=torch.int32)
        builder.add_item(tokens)
        builder.end_document()
    builder.finalize(idx_path)
    print(f"wrote {num_sequences} x {seq_len} lm rows to {LM_DIR}", flush=True)


def count_train_rows():
    path = os.path.join(DOLLY, "full", "gpt2", "train.jsonl")
    if not os.path.exists(path):
        return
    with open(path) as handle:
        rows = sum(1 for _ in handle)
    print(f"dolly train rows: {rows}", flush=True)


def clean_hub_cache():
    """Drop the Hub blob cache once the local copies exist. The cache duplicates the weights."""
    home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    hub = os.path.join(home, "hub")
    if os.path.isdir(hub):
        import shutil
        shutil.rmtree(hub, ignore_errors=True)
        print(f"removed hub cache {hub}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    download()
    build_lm_standin()
    count_train_rows()
    clean_hub_cache()
