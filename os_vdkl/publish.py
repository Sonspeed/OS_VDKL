"""Upload a finished OS-VDKL checkpoint to the Hugging Face Hub."""

import json
import os
import sys


def render_card(repo_id, estimate):
    estimate = estimate or {}
    full_eta = estimate.get("full_eta", "n/a")
    step = estimate.get("avg_step_seconds")
    step_text = f"{step:.2f}s" if isinstance(step, (int, float)) else "n/a"
    gpu = estimate.get("gpu", "n/a")
    batch = estimate.get("batch_size", "n/a")
    epochs = estimate.get("full_epochs", "n/a")
    steps = estimate.get("full_steps", "n/a")
    per_epoch = estimate.get("steps_per_epoch", "n/a")
    train = estimate.get("full_train", "n/a")
    eval_time = estimate.get("full_eval", "n/a")
    measured = estimate.get("measured_steps", "n/a")
    return f"""---
license: mit
base_model: openai-community/gpt2
tags:
- os-vdkl
- knowledge-distillation
- gpt2
datasets:
- databricks/databricks-dolly-15k
---

# {repo_id}

Trial checkpoint of Orthogonal-Subspace Variational Dual-KL distillation.
The student is GPT-2 (124M), initialized from `MiniLLM/init-gpt2-120M`.
The teacher is `MiniLLM/teacher-gpt2-1.5B`.

This upload is the short trial run ({measured} measured optimizer steps on {gpu}, batch {batch}).
It checks the training loop and the Hub path. It is a smoke checkpoint, and the 20-epoch schedule has not been finished.

## Time estimate

Measured on **{gpu}**, batch **{batch}**:

| | |
| --- | --- |
| seconds / optimizer step | {step_text} |
| steps / epoch | {per_epoch} |
| full schedule | {epochs} epochs, {steps} steps |
| training | {train} |
| validation (initial + one per epoch) | {eval_time} |
| **total** | **{full_eta}** |

`time_estimate.json` has the raw numbers. The step time covers distillation on Dolly tokens plus the language-model forward at the launcher's initial adaptive threshold. Steps that later trigger on-policy student sampling also generate `max_length - max_prompt_length` tokens.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
```

`AutoModelForCausalLM` loads the distilled GPT-2 weights. The branch adapters are in `os_multi_lora.pt` and are attached by `os_vdkl.multi_lora.attach_os_multi_lora`.
"""


def push_checkpoint(folder, repo_id, private=False):
    from huggingface_hub import HfApi

    folder = os.path.abspath(folder)
    estimate = None
    estimate_path = os.path.join(folder, "time_estimate.json")
    if os.path.exists(estimate_path):
        with open(estimate_path) as handle:
            estimate = json.load(handle)
    card_path = os.path.join(folder, "README.md")
    if not os.path.exists(card_path):
        with open(card_path, "w") as handle:
            handle.write(render_card(repo_id, estimate))

    api = HfApi()
    try:
        api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
    except Exception as exc:
        print(
            f"Hugging Face upload failed for {repo_id}. The checkpoint is saved at {folder}.\n"
            "Log in with an account that can write to that namespace, then retry:\n"
            f"  hf auth login\n"
            f"  python -m os_vdkl.publish {folder} {repo_id}",
            flush=True,
        )
        raise exc
    commit = api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload OS-VDKL trial checkpoint",
    )
    print(f"Pushed {folder} to https://huggingface.co/{repo_id}", flush=True)
    print(commit, flush=True)
    return commit


def main():
    if len(sys.argv) < 3:
        print("usage: python -m os_vdkl.publish <checkpoint_dir> <repo_id> [--private]", file=sys.stderr)
        sys.exit(2)
    push_checkpoint(sys.argv[1], sys.argv[2], private="--private" in sys.argv[3:])


if __name__ == "__main__":
    main()
