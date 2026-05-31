#!/usr/bin/env python3
"""
_check_token_alignment.py

Regression test for the token-position bug:
  - STLoRA's mask must align against the *chat-templated* Qwen prompt
    (the same sequence Flux2KleinPipeline._get_qwen3_prompt_embeds encodes).
  - The CSV (steering vector) must use the same position space.

This script needs only the Qwen tokenizer (CPU, no GPU, no model load).
It exits non-zero on any mismatch so it can be wired into CI / pre-commit.

Usage:
  python training/_check_token_alignment.py
  python training/_check_token_alignment.py --pretrained black-forest-labs/FLUX.2-klein-base-9B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import Qwen2TokenizerFast

from visual_analogy.utils.hf_utils import resolve_hf_snapshot_path
from visual_analogy.utils.selective_lora import (
    _apply_klein_chat_template,
    klein_find_substring_token_indices,
    klein_templated_prompt_token_positions,
)
from training.train_stlora_flux2_klein import build_analogy_prompt, build_token_mask


EDITS = ["change pose to running", "change style to oil painting"]
MAX_LEN = 512


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _check(cond: bool, msg: str) -> bool:
    print((_green("[ok]   ") if cond else _red("[FAIL] ")) + msg)
    return bool(cond)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str,
                        default="black-forest-labs/FLUX.2-klein-base-9B")
    args = parser.parse_args()

    local_path = resolve_hf_snapshot_path(args.pretrained)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        local_path, subfolder="tokenizer", local_files_only=True)

    prompt = build_analogy_prompt(EDITS, EDITS)
    print(f"prompt: {prompt!r}")
    templated = _apply_klein_chat_template(prompt, tokenizer)
    print(f"templated len(chars)={len(templated)}; "
          f"raw len(chars)={len(prompt)}")

    all_ok = True

    # ---- 1. Each edit aligns and decodes back to itself ----
    enc = tokenizer(
        templated,
        max_length=MAX_LEN, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"][0].tolist()

    for i, edit in enumerate(EDITS):
        positions = klein_find_substring_token_indices(
            prompt, edit, tokenizer, max_length=MAX_LEN, use_chat_template=True)
        decoded = tokenizer.decode([ids[p] for p in positions],
                                   skip_special_tokens=False)
        ok = decoded.strip() == edit.strip()
        all_ok &= _check(
            ok,
            f"edit {i+1} aligns: positions={positions[0]}..{positions[-1]} "
            f"decoded={decoded!r}"
        )

    # ---- 2. Per-edit positions are a strict subset of the all-prompt positions ----
    all_positions = set(klein_templated_prompt_token_positions(
        prompt, tokenizer, max_length=MAX_LEN))
    for i, edit in enumerate(EDITS):
        edit_positions = set(klein_find_substring_token_indices(
            prompt, edit, tokenizer, max_length=MAX_LEN, use_chat_template=True))
        all_ok &= _check(
            edit_positions.issubset(all_positions),
            f"edit {i+1} positions are a subset of templated-prompt positions"
        )

    # ---- 3. build_token_mask produces a non-empty mask for any non-empty edit list ----
    import torch
    mask = build_token_mask(
        prompt, EDITS, tokenizer, device=torch.device("cpu"),
        max_seq_len=MAX_LEN, use_chat_template=True,
    )
    n_true = int(mask.sum().item())
    expected_min = sum(len(klein_find_substring_token_indices(
        prompt, e, tokenizer, max_length=MAX_LEN, use_chat_template=True))
        for e in EDITS)
    all_ok &= _check(
        n_true > 0,
        f"build_token_mask covers {n_true} positions (expected >= 1)"
    )
    all_ok &= _check(
        n_true == expected_min,
        f"build_token_mask True-count == sum of per-edit positions "
        f"({n_true} vs {expected_min})"
    )

    # ---- 4. Templated alignment differs from raw alignment (regression-canary) ----
    raw_positions = klein_find_substring_token_indices(
        prompt, EDITS[0], tokenizer, max_length=MAX_LEN, use_chat_template=False)
    tmpl_positions = klein_find_substring_token_indices(
        prompt, EDITS[0], tokenizer, max_length=MAX_LEN, use_chat_template=True)
    all_ok &= _check(
        raw_positions != tmpl_positions,
        "raw vs templated positions differ (chat template adds a prefix) — "
        f"raw={raw_positions[:3]}..  tmpl={tmpl_positions[:3]}.."
    )

    # ---- 5. build_token_mask raises (not silently zeroes) when an edit is missing ----
    bogus_edit = "this exact phrase definitely is not in the prompt 12345"
    raised = False
    try:
        build_token_mask(
            prompt, [bogus_edit], tokenizer, device=torch.device("cpu"),
            max_seq_len=MAX_LEN, use_chat_template=True,
        )
    except RuntimeError:
        raised = True
    all_ok &= _check(
        raised,
        "build_token_mask RAISES on missing edit (no silent empty mask)"
    )

    if all_ok:
        print(_green("\nAll token-alignment checks passed."))
        sys.exit(0)
    print(_red("\nOne or more token-alignment checks FAILED."))
    sys.exit(1)


if __name__ == "__main__":
    main()
