#!/usr/bin/env python3
"""
steering_dom.py

Paper-style difference-of-means (DoM) text-embedding steering for the
visual-analogy pipeline (Diffusion Sliders — Ekin & Gandelsman,
arXiv:2603.17998).

Per edit, we:

  1. Use the already-loaded Flux2-Klein Qwen3 text encoder (which is a
     ``Qwen3ForCausalLM`` — it can do ``.generate()``) to produce N
     contrastive pos/neg sentence pairs, in the JSONL schema the paper's
     ``dataset/generate.py`` uses:
         {"pos_style": "...", "neg_style": "...",
          "pos": "...",       "neg": "..."}
     The paper's OpenAI system prompt is reused verbatim (shortened where
     appropriate for local generation).

  2. Encode each sentence via ``Flux2KleinPipeline._get_qwen3_prompt_embeds``
     (the exact same code path as diffusion inference), pool at the
     ``pos_style`` / ``neg_style`` token positions (aligned against the Qwen
     chat-templated sequence so the indices match the pooled ``prompt_embeds``
     tensor), and compute

         v = normalize(mean(pos_pooled) − mean(neg_pooled))

     mirroring ``steering/vectors.py::compute_difference_of_means`` in the
     paper repo.

  3. Cache both the raw pairs (``pairs.jsonl``) and the unit-norm vector
     (``vector.pt``) under ``cache_dir/<sha1(edit)[:12]>/``.

The returned vector has the same hidden dimension as Flux2-Klein's prompt
embeddings (3 × hidden_size of Qwen3 — layers {9,18,27} concatenated), so it
can be added directly to ``prompt_embeds`` at edit-token positions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from diffusers import Flux2KleinPipeline

from visual_analogy.utils.selective_lora import klein_find_substring_token_indices


# ---------------------------------------------------------------------------
# Contrastive dataset generation (Qwen3 causal-LM)
# ---------------------------------------------------------------------------


REQUIRED_KEYS = {"pos_style", "neg_style", "pos", "neg"}


def _build_generation_prompt(edit: str, n_pairs: int) -> str:
    """The paper's unbiased-dataset system prompt, lightly adapted for local LLMs.

    Source: diffusion-sliders/dataset/generate.py::build_prompt and
    diffusion-sliders/system_prompts/unbiased_dataset_generation.txt.
    """
    return f"""You are an advanced data generation assistant.

Your task is to create a contrastive dataset of {n_pairs} examples for \
computing a steering vector.

The steering concept to focus on is: {edit}

Output exactly {n_pairs} JSON objects, one per line (JSON Lines), with no list \
brackets, no extra commentary, and no markdown. Each line must be:
{{"pos_style": "<positive identifier>", "neg_style": "<negative identifier>", \
"pos": "<positive full sentence>", "neg": "<negative full sentence>"}}

Rules (STRICT):
- PARALLELISM: "pos" and "neg" MUST share the same syntactic skeleton, subject, \
setting and composition. Only the identifier tokens differ.
- MINIMAL DELTA: The only difference between "pos" and "neg" is the tokens that \
express the concept contrast.
- STYLE NEUTRALITY: Do NOT change rendering domain, lighting, camera, or layout.
- IDENTIFIERS: Use the SAME "pos_style" and "neg_style" identifiers for all \
{n_pairs} lines, and the identifiers MUST appear verbatim in the corresponding \
sentences.
- DIVERSITY: Vary the subject, setting, viewpoint and wording across the \
{n_pairs} lines so the steering direction generalises.
- Use neutral terms for people ("person", "figure"); avoid age, gender, race \
unless the concept itself requires them.

Example (concept: "bright vs dark"):
{{"pos_style": "bright", "neg_style": "dark", "pos": "A bright living room \
with large windows.", "neg": "A dark living room with large windows."}}

Now generate EXACTLY {n_pairs} JSONL lines for the concept \"{edit}\" following \
these instructions. Do not number the lines. Do not wrap them in code fences.
"""


def _parse_jsonl_records(text: str) -> List[dict]:
    """Parse JSONL lines, robust to stray code fences and commentary."""
    records: List[dict] = []
    # Drop markdown code-fence wrappers if present
    text = re.sub(r"^```[a-zA-Z0-9]*\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate trailing commas
        line = line.rstrip(",").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not REQUIRED_KEYS.issubset(obj.keys()):
            continue
        # Validate identifiers appear in sentences (case-insensitive). Skip otherwise.
        if obj["pos_style"].lower() not in obj["pos"].lower():
            continue
        if obj["neg_style"].lower() not in obj["neg"].lower():
            continue
        records.append({k: obj[k] for k in ("pos_style", "neg_style", "pos", "neg")})
    return records


_FALLBACK_SCAFFOLDS = [
    "A photo of a {X}.",
    "A detailed close-up of a {X}.",
    "A cinematic portrait showing a {X}.",
    "An outdoor scene featuring a {X}.",
    "A studio photograph of a {X}.",
    "A high-resolution image of a {X}.",
    "An editorial shot of a {X}.",
    "A candid photograph of a {X}.",
]


def _fallback_pairs(edit: str, n_pairs: int) -> List[dict]:
    """Last-resort: build trivial parallel pairs by scaffolding the raw edit
    against a generic "no edit" control. Used only if Qwen's JSONL output is
    unparseable; produces a much weaker steering signal than the full LLM
    pipeline but keeps the run going.
    """
    pos_style = edit
    neg_style = "unchanged"
    out = []
    i = 0
    while len(out) < n_pairs:
        scaffold = _FALLBACK_SCAFFOLDS[i % len(_FALLBACK_SCAFFOLDS)]
        pos = scaffold.format(X=f"subject showing {pos_style}")
        neg = scaffold.format(X=f"subject, {neg_style}")
        out.append({"pos_style": pos_style, "neg_style": neg_style,
                    "pos": pos, "neg": neg})
        i += 1
    return out


@torch.no_grad()
def generate_pairs_with_qwen(
    edit: str,
    text_encoder,
    tokenizer,
    n_pairs: int,
    device,
    max_new_tokens_per_pair: int = 90,
) -> Tuple[List[dict], int]:
    """Ask Qwen3 (``text_encoder``, which is ``Qwen3ForCausalLM``) for
    ``n_pairs`` contrastive pos/neg sentences about ``edit``.

    Returns
    -------
    (records, n_requested)  where ``records`` has length in
    ``[0, n_requested]`` — callers should top up with ``_fallback_pairs``.
    """
    user_prompt = _build_generation_prompt(edit, n_pairs)
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    was_training = text_encoder.training
    text_encoder.eval()
    try:
        gen_out = text_encoder.generate(
            **inputs,
            max_new_tokens=max(256, n_pairs * max_new_tokens_per_pair),
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        if was_training:
            text_encoder.train()

    # Strip the prompt prefix so we only decode the model's reply.
    generated = gen_out[0, inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True)

    records = _parse_jsonl_records(reply)
    return records, n_pairs


# ---------------------------------------------------------------------------
# Pooled prompt-embedding extraction + difference-of-means
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_prompt(text: str, text_encoder, tokenizer, device,
                   max_sequence_length: int = 512) -> torch.Tensor:
    """Run a single prompt through the same encoder path as diffusion inference."""
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=[text],
        device=device,
        max_sequence_length=max_sequence_length,
    )
    return prompt_embeds  # (1, seq_len, hidden_dim)


def _templated_substring_indices(
    prompt: str,
    substr: str,
    tokenizer,
    max_length: int = 512,
) -> List[int]:
    """Locate ``substr`` tokens inside the Qwen chat-templated version of
    ``prompt`` — matches the sequence that ``_get_qwen3_prompt_embeds``
    actually encodes.

    First tries the canonical chat-template-aware locator (case-sensitive).
    Falls back to a case-insensitive scan because the contrastive-dataset
    parser validates ``pos_style`` / ``neg_style`` containment with
    ``.lower()``, so an LLM that produced "Bright" in a sentence and
    ``"bright"`` as the style identifier still passes parsing.

    Returns ``[]`` if the substring is missing — DoM pooling treats a miss
    as "skip this pair", not as a hard failure.
    """
    try:
        return klein_find_substring_token_indices(
            prompt, substr, tokenizer,
            max_length=max_length,
            use_chat_template=True,
        )
    except AssertionError:
        pass

    # Case-insensitive fallback: re-locate the char span, then reuse the
    # offset-mapping logic the canonical helper would have applied.
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    char_start = templated.lower().find(substr.lower())
    if char_start < 0:
        return []
    char_end = char_start + len(substr)
    enc = tokenizer(
        templated,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    out = []
    for i, (cs, ce) in enumerate(offsets):
        if cs == ce == 0 and i > 0:
            continue
        if ce > char_start and cs < char_end:
            out.append(i)
    return out


@torch.no_grad()
def pool_edit_embedding(
    sentence: str,
    style: str,
    text_encoder,
    tokenizer,
    device,
    max_sequence_length: int = 512,
) -> Optional[torch.Tensor]:
    """Encode ``sentence`` and mean-pool ``prompt_embeds`` at the token
    positions covering ``style``. Returns ``None`` if alignment fails."""
    positions = _templated_substring_indices(
        sentence, style, tokenizer, max_length=max_sequence_length)
    if not positions:
        return None
    prompt_embeds = _encode_prompt(
        sentence, text_encoder, tokenizer, device, max_sequence_length)
    # (1, S, D) → (D,) via mean over style positions
    pooled = prompt_embeds[0, positions, :].float().mean(dim=0).cpu()
    return pooled


@torch.no_grad()
def compute_dom_vector(
    pairs: Sequence[dict],
    text_encoder,
    tokenizer,
    device,
    max_sequence_length: int = 512,
) -> Tuple[torch.Tensor, dict]:
    """Compute a unit-norm difference-of-means steering vector from a list of
    ``{pos, neg, pos_style, neg_style}`` records.

    Returns
    -------
    (v, stats) where
        ``v``      : unit-norm tensor of shape ``(hidden_dim,)`` on ``device``
        ``stats``  : dict with ``n_usable``, ``pre_norm``, ``pos_count``, ``neg_count``
    """
    pos_vecs: List[torch.Tensor] = []
    neg_vecs: List[torch.Tensor] = []

    for rec in pairs:
        p = pool_edit_embedding(
            rec["pos"], rec["pos_style"],
            text_encoder, tokenizer, device, max_sequence_length)
        n = pool_edit_embedding(
            rec["neg"], rec["neg_style"],
            text_encoder, tokenizer, device, max_sequence_length)
        if p is None or n is None:
            continue
        pos_vecs.append(p)
        neg_vecs.append(n)

    if not pos_vecs or not neg_vecs:
        raise RuntimeError(
            "compute_dom_vector: no pooled embeddings could be computed — "
            "check that pos_style / neg_style appear verbatim in pos / neg "
            "and that the encoder is loaded.")

    pos_mean = torch.stack(pos_vecs).mean(dim=0)
    neg_mean = torch.stack(neg_vecs).mean(dim=0)
    diff = (pos_mean - neg_mean).float()
    pre_norm = float(diff.norm())
    if pre_norm < 1e-8:
        raise RuntimeError(
            f"compute_dom_vector: DoM direction has near-zero norm "
            f"({pre_norm:.2e}); contrastive pairs may be degenerate.")
    v_unit = diff / pre_norm
    stats = {
        "n_usable": len(pos_vecs),
        "pre_norm": pre_norm,
        "pos_count": len(pos_vecs),
        "neg_count": len(neg_vecs),
    }
    return v_unit.to(device), stats


# ---------------------------------------------------------------------------
# Disk caching
# ---------------------------------------------------------------------------


def _edit_cache_dir(cache_root: Path, edit: str) -> Path:
    sha = hashlib.sha1(edit.encode("utf-8")).hexdigest()[:12]
    return Path(cache_root) / sha


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def get_or_build_steering_vector(
    edit: str,
    n_pairs: int,
    text_encoder,
    tokenizer,
    device,
    cache_root: Path,
    regen: bool = False,
    max_sequence_length: int = 512,
) -> Tuple[torch.Tensor, dict]:
    """Top-level entry point used by ``infer_single.py``.

    - If ``cache_root / <sha1>/ vector.pt`` exists and ``regen`` is False,
      return the cached vector (and whatever ``pairs.jsonl`` metadata is
      present, if any).
    - Otherwise generate pairs with Qwen (falling back to the scaffold list
      if parse yield < 50 %), compute the DoM vector, persist both artifacts
      to disk, and return.

    Returns ``(v_unit, meta)`` where ``meta`` captures enough info for
    ``infer_single.py`` to emit sanity logs.
    """
    edit_dir = _edit_cache_dir(cache_root, edit)
    pairs_path = edit_dir / "pairs.jsonl"
    vec_path = edit_dir / "vector.pt"
    meta_path = edit_dir / "meta.json"

    if not regen and vec_path.exists():
        v = torch.load(vec_path, map_location=device)
        meta: dict = {"source": "cache", "vec_path": str(vec_path)}
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return v.to(device=device, dtype=torch.float32), meta

    # ---- Generate via Qwen ----
    print(f"[Steering DoM] Generating {n_pairs} contrastive pairs via Qwen3 for edit: {edit!r}")
    records, n_requested = generate_pairs_with_qwen(
        edit, text_encoder, tokenizer, n_pairs, device)
    n_parsed = len(records)
    used_fallback = False
    if n_parsed < max(4, n_pairs // 2):
        print(f"[Steering DoM] Qwen parse yield {n_parsed}/{n_requested} too low — "
              f"padding with scaffold fallback.")
        existing_keys = {(r["pos"], r["neg"]) for r in records}
        for fb in _fallback_pairs(edit, n_pairs - n_parsed):
            if (fb["pos"], fb["neg"]) in existing_keys:
                continue
            records.append(fb)
            existing_keys.add((fb["pos"], fb["neg"]))
            if len(records) >= n_pairs:
                break
        used_fallback = True

    # ---- Encode & DoM ----
    print(f"[Steering DoM] Encoding {len(records)} pairs and computing DoM vector …")
    v_unit, stats = compute_dom_vector(
        records, text_encoder, tokenizer, device, max_sequence_length)

    # ---- Persist ----
    edit_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(pairs_path, records)
    torch.save(v_unit.detach().cpu().float(), vec_path)
    meta = {
        "edit": edit,
        "n_requested": n_requested,
        "n_parsed_from_llm": n_parsed,
        "n_records_saved": len(records),
        "n_usable_after_pooling": stats["n_usable"],
        "pre_normalization_norm": stats["pre_norm"],
        "used_fallback_padding": used_fallback,
        "source": "generated",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Steering DoM] Cached → {edit_dir}")
    return v_unit.to(device=device, dtype=torch.float32), meta
