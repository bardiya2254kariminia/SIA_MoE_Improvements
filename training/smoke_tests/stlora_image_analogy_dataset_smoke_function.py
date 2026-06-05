"""
Smoke test for ImageAnalogyDataset from train_stlora_flux2_klein.py.

Runs against the real training dataset and saves outputs to
SIA_MoE_Improvements/temp/stlora_dataset_smoke/.

Output layout
─────────────
temp/stlora_dataset_smoke/
    getitem_samples/
        sample_0/
            A.png  A_prime.png  A_prime_total.png  A_prime_partial.png (if present)
            B.png  B_prime.png  B_total.png
            strip.png
            prompt.json
        sample_1/ … sample_N/
    val_pair/
        A.png  A_prime.png  A_prime_partial.png (if present)
        B.png  B_prime.png  B_total.png
        strip.png
        prompt.json
    helper_unit_tests.txt
    summary.txt

Usage
─────
    python3 training/stlora_image_analogy_dataset_smoke_function.py [--n_samples N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ── Repo root on sys.path ──────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # minimal stub — no torch needed for data-loading smoke test
        pass

# ── Real dataset root ──────────────────────────────────────────────────────
_DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "SIA_Dataset_creation" / "datasets" / "SIA_datasets" / "train"
)

# ============================================================================
# Helpers — verbatim from train_stlora_flux2_klein.py
# ============================================================================

def _parse_combined_filename(filename: str):
    m = re.match(r'combined(\d+)\.png$', filename)
    if not m:
        return None
    return sorted(int(d) for d in m.group(1))


def _get_combined_files(data_dir: Path):
    result = []
    for f in data_dir.iterdir():
        indices = _parse_combined_filename(f.name)
        if indices is not None:
            result.append((f.name, indices))
    return result


def _tier_probs(n_tiers: int) -> list:
    if n_tiers <= 1:
        return [1.0] * max(n_tiers, 1)
    if n_tiers == 2:
        return [0.70, 0.30]
    if n_tiers == 3:
        return [0.70, 0.20, 0.10]
    tail = 0.10 / (n_tiers - 2)
    return [0.70, 0.20] + [tail] * (n_tiers - 2)


def _sample_combined(combined_files: list):
    if not combined_files:
        raise ValueError("combined_files is empty")
    tier_dict: dict = {}
    for fname, indices in combined_files:
        t = len(indices)
        tier_dict.setdefault(t, []).append((fname, indices))
    sorted_tiers = sorted(tier_dict.keys(), reverse=True)
    probs = _tier_probs(len(sorted_tiers))
    chosen_tier = random.choices(sorted_tiers, weights=probs)[0]
    return random.choice(tier_dict[chosen_tier])


def resize_to_match(img, target_w, target_h):
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_keep_ratio(img, max_side=512):
    w, h = img.size
    scale = max_side / max(w, h)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def harmonize_images(a, a_prime, b, b_prime, max_side=512):
    a        = resize_keep_ratio(a, max_side)
    a_w, a_h = a.size
    a_prime  = resize_to_match(a_prime, a_w, a_h)
    b        = resize_keep_ratio(b, max_side)
    b_w, b_h = b.size
    b_prime  = resize_to_match(b_prime, b_w, b_h)
    return a, a_prime, b, b_prime


def build_analogy_prompt(pair_a_edits, pair_b_edits, suppress_indices=None):
    suppress  = set(suppress_indices or [])
    kept      = [e for i, e in enumerate(pair_a_edits) if i not in suppress and e]
    edits_part = " ".join(f"Edit {n + 1}: {e}." for n, e in enumerate(kept))
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )


# ============================================================================
# Dataset — exact copy from train_stlora_flux2_klein.py
# ============================================================================

class ImageAnalogyDataset(Dataset):
    """Image analogy dataset for STLoRA training (SIA_datasets/train structure)."""

    def __init__(self, data_root):
        self.data_root = Path(data_root)

        self.categories: dict = {}
        n_skipped = 0

        for cat_dir in sorted(self.data_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            self.categories[cat] = {}

            for setting_dir in sorted(cat_dir.iterdir()):
                if not setting_dir.is_dir():
                    continue
                setting = setting_dir.name
                samples = []

                for data_dir in sorted(setting_dir.iterdir()):
                    if not data_dir.is_dir():
                        continue
                    if not (data_dir / 'input.png').exists():
                        n_skipped += 1
                        continue
                    if not (data_dir / 'prompt.json').exists():
                        n_skipped += 1
                        continue
                    combined = _get_combined_files(data_dir)
                    if len(combined) < 2:
                        n_skipped += 1
                        continue
                    try:
                        prompt_dict = json.loads((data_dir / 'prompt.json').read_text())
                    except Exception:
                        n_skipped += 1
                        continue
                    samples.append((data_dir, combined, prompt_dict))

                if len(samples) >= 2:
                    self.categories[cat][setting] = samples

        self.cat_names = sorted(self.categories.keys())
        total = sum(len(s) for c in self.categories.values() for s in c.values())
        print(f"SIADataset (STLoRA): {len(self.cat_names)} categories, "
              f"{total} samples ({n_skipped} skipped)")
        for cat in self.cat_names:
            n = sum(len(s) for s in self.categories[cat].values())
            print(f"  {cat}: {len(self.categories[cat])} settings, {n} samples")

    def __len__(self):
        return sum(len(s) for c in self.categories.values() for s in c.values())

    @staticmethod
    def _longest_combined(combined_files: list):
        return max(combined_files, key=lambda x: len(x[1]))

    @staticmethod
    def _partial_candidates(combined_files: list, total_indices: list):
        total_set = set(total_indices)
        total_len = len(total_indices)
        one_shorter = [
            (fn, idx) for fn, idx in combined_files
            if len(idx) == total_len - 1 and set(idx).issubset(total_set)
        ]
        if one_shorter:
            return one_shorter
        shorter = [(fn, idx) for fn, idx in combined_files if len(idx) < total_len]
        return shorter if shorter else combined_files

    def sample_val_pair(self):
        cat     = random.choice(self.cat_names)
        setting = random.choice(list(self.categories[cat].keys()))
        samples = self.categories[cat][setting]
        sample_a, sample_b = random.sample(samples, 2)
        data_dir_a, combined_a, prompt_a = sample_a
        data_dir_b, combined_b, prompt_b = sample_b

        fname_a_total, total_indices_a = self._longest_combined(combined_a)
        fname_b_total, total_indices_b = self._longest_combined(combined_b)

        candidates_b           = self._partial_candidates(combined_b, total_indices_b)
        fname_b_prime, b_prime_indices = random.choice(candidates_b)

        suppressed_digits = sorted(set(total_indices_b) - set(b_prime_indices))
        suppress_index    = (suppressed_digits[0] - 1) if suppressed_digits else 0

        map_a           = {frozenset(idx): fn for fn, idx in combined_a}
        fname_a_partial = map_a.get(frozenset(b_prime_indices))

        pair_a_edits = [prompt_a.get(f'edit{i}', '') for i in sorted(total_indices_a)]
        pair_b_edits = [prompt_b.get(f'edit{i}', '') for i in sorted(total_indices_b)]
        full_prompt  = build_analogy_prompt(pair_a_edits, pair_b_edits)

        return {
            'a_path':               str(data_dir_a / 'input.png'),
            'a_prime_path':         str(data_dir_a / fname_a_total),
            'a_prime_partial_path': str(data_dir_a / fname_a_partial) if fname_a_partial else None,
            'b_path':               str(data_dir_b / 'input.png'),
            'b_prime_path':         str(data_dir_b / fname_b_prime),
            'b_total_path':         str(data_dir_b / fname_b_total),
            'edit_name':            f"{cat}/{setting}",
            'prompt':               full_prompt,
            'pair_a_edits':         pair_a_edits,
            'pair_b_edits':         pair_b_edits,
            'suppress_index':       suppress_index,
        }

    def __getitem__(self, idx):
        cat      = random.choice(self.cat_names)
        settings = self.categories[cat]
        setting  = random.choice(list(settings.keys()))
        samples  = settings[setting]

        sample_a, sample_b = random.sample(samples, 2)
        data_dir_a, combined_a, prompt_a = sample_a
        data_dir_b, combined_b, prompt_b = sample_b

        fname_a_total, total_indices_a = self._longest_combined(combined_a)
        fname_b_total, total_indices_b = self._longest_combined(combined_b)

        candidates_b             = self._partial_candidates(combined_b, total_indices_b)
        fname_b_prime, b_prime_indices = random.choice(candidates_b)

        suppressed_digits = sorted(set(total_indices_b) - set(b_prime_indices))
        if suppressed_digits:
            chosen_digit   = random.choice(suppressed_digits)
            suppress_index = chosen_digit - 1
        else:
            suppress_index = 0

        map_a           = {frozenset(idx): fn for fn, idx in combined_a}
        fname_a_partial = map_a.get(frozenset(b_prime_indices))

        pair_a_edits = [prompt_a.get(f'edit{i}', '') for i in sorted(total_indices_a)]
        pair_b_edits = [prompt_b.get(f'edit{i}', '') for i in sorted(total_indices_b)]
        full_prompt  = build_analogy_prompt(pair_a_edits, pair_b_edits)

        a        = Image.open(data_dir_a / 'input.png').convert('RGB')
        a_total  = Image.open(data_dir_a / fname_a_total).convert('RGB')
        b        = Image.open(data_dir_b / 'input.png').convert('RGB')
        b_prime  = Image.open(data_dir_b / fname_b_prime).convert('RGB')
        b_total  = Image.open(data_dir_b / fname_b_total).convert('RGB')
        a_partial = (
            Image.open(data_dir_a / fname_a_partial).convert('RGB')
            if fname_a_partial else None
        )

        a, a_total, b, b_prime = harmonize_images(a, a_total, b, b_prime, max_side=512)
        b_total = resize_to_match(b_total, *b.size)
        if a_partial is not None:
            a_partial = resize_to_match(a_partial, *a.size)

        return {
            'A':               a,
            'A_prime':         a_total,
            'A_prime_total':   a_total,
            'A_prime_partial': a_partial,
            'B':               b,
            'B_prime':         b_prime,
            'B_total':         b_total,
            'edit_name':       f"{cat}/{setting}",
            'prompt':          full_prompt,
            'pair_a_edits':    pair_a_edits,
            'pair_b_edits':    pair_b_edits,
            'suppress_index':  suppress_index,
        }


# ============================================================================
# Output helpers
# ============================================================================

def _try_load_font(size: int = 14):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return None


def _make_strip(images: list[Image.Image], labels: list[str],
                cell_w: int = 256, cell_h: int = 256) -> Image.Image:
    """Horizontal strip of resized images with text labels."""
    font   = _try_load_font(14)
    n      = len(images)
    strip  = Image.new("RGB", (cell_w * n, cell_h + 24), (255, 255, 255))
    draw   = ImageDraw.Draw(strip)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
        x_off = i * cell_w + (cell_w - thumb.width) // 2
        strip.paste(thumb, (x_off, 24))
        draw.text((i * cell_w + 4, 4), lbl, fill=(0, 0, 0),
                  font=font if font else None)
    return strip


def save_getitem_sample(sample: dict, out_dir: Path, index: int) -> None:
    """Save all images, strip, and prompt JSON for one __getitem__ call."""
    sd = out_dir / f"sample_{index}"
    sd.mkdir(parents=True, exist_ok=True)

    sample['A'].save(sd / 'A.png')
    sample['A_prime'].save(sd / 'A_prime.png')
    sample['A_prime_total'].save(sd / 'A_prime_total.png')
    sample['B'].save(sd / 'B.png')
    sample['B_prime'].save(sd / 'B_prime.png')
    sample['B_total'].save(sd / 'B_total.png')

    if sample['A_prime_partial'] is not None:
        sample['A_prime_partial'].save(sd / 'A_prime_partial.png')
    else:
        (sd / 'A_prime_partial_NONE.txt').write_text(
            "No matching partial exists in data_a for this sample.")

    # Strip: A | A_prime_total | A_prime_partial | B | B_prime | B_total
    placeholder = Image.new("RGB", (256, 256), (220, 220, 220))
    _make_strip(
        [sample['A'],
         sample['A_prime_total'],
         sample['A_prime_partial'] if sample['A_prime_partial'] is not None else placeholder,
         sample['B'],
         sample['B_prime'],
         sample['B_total']],
        ['A', "A' total", "A' partial\n(None=grey)", 'B', "B' target", 'B total'],
    ).save(sd / 'strip.png')

    # prompt.json — all metadata the training loop uses
    (sd / 'prompt.json').write_text(json.dumps({
        'edit_name':      sample['edit_name'],
        'prompt':         sample['prompt'],
        'pair_a_edits':   sample['pair_a_edits'],
        'pair_b_edits':   sample['pair_b_edits'],
        'suppress_index': sample['suppress_index'],
        'suppressed_edit':
            sample['pair_a_edits'][sample['suppress_index']]
            if sample['suppress_index'] < len(sample['pair_a_edits']) else None,
    }, indent=2, ensure_ascii=False))

    print(f"  [getitem #{index}]  {sample['edit_name']}  "
          f"suppress_idx={sample['suppress_index']}  →  {sd.name}/")


def save_val_pair(vp: dict, out_dir: Path) -> None:
    """Save images and prompt JSON from sample_val_pair()."""
    vpd = out_dir / 'val_pair'
    vpd.mkdir(parents=True, exist_ok=True)

    path_map = {
        'A':       'a_path',
        'A_prime': 'a_prime_path',
        'B':       'b_path',
        'B_prime': 'b_prime_path',
        'B_total': 'b_total_path',
    }
    loaded = {}
    for img_name, path_key in path_map.items():
        p = vp[path_key]
        if p and Path(p).exists():
            img = Image.open(p).convert('RGB')
            img.save(vpd / f'{img_name}.png')
            loaded[img_name] = img

    partial_p = vp.get('a_prime_partial_path')
    if partial_p and Path(partial_p).exists():
        img = Image.open(partial_p).convert('RGB')
        img.save(vpd / 'A_prime_partial.png')
        loaded['A_prime_partial'] = img
    else:
        (vpd / 'A_prime_partial_NONE.txt').write_text(
            f"a_prime_partial_path = {partial_p!r}")

    # Strip without partial (partial may be None)
    strip_imgs   = [loaded[k] for k in ('A', 'A_prime', 'B', 'B_prime', 'B_total') if k in loaded]
    strip_labels = ['A', "A' (total)", 'B', "B' target", 'B total'][:len(strip_imgs)]
    _make_strip(strip_imgs, strip_labels).save(vpd / 'strip.png')

    (vpd / 'prompt.json').write_text(json.dumps({
        'edit_name':            vp['edit_name'],
        'prompt':               vp['prompt'],
        'pair_a_edits':         vp['pair_a_edits'],
        'pair_b_edits':         vp['pair_b_edits'],
        'suppress_index':       vp['suppress_index'],
        'suppressed_edit':
            vp['pair_a_edits'][vp['suppress_index']]
            if vp['suppress_index'] < len(vp['pair_a_edits']) else None,
        'a_path':               vp['a_path'],
        'a_prime_path':         vp['a_prime_path'],
        'a_prime_partial_path': vp['a_prime_partial_path'],
        'b_path':               vp['b_path'],
        'b_prime_path':         vp['b_prime_path'],
        'b_total_path':         vp['b_total_path'],
    }, indent=2, ensure_ascii=False))

    print(f"  [val_pair]  {vp['edit_name']}  suppress_idx={vp['suppress_index']}  →  {vpd.name}/")


# ============================================================================
# Helper unit tests (pure-logic, no real images)
# ============================================================================

def run_helper_unit_tests() -> str:
    lines = ["STLoRA helper unit tests", "─" * 44]

    combined = [
        ("combined1.png",   [1]),
        ("combined12.png",  [1, 2]),
        ("combined123.png", [1, 2, 3]),
    ]

    longest = ImageAnalogyDataset._longest_combined(combined)
    assert longest == ("combined123.png", [1, 2, 3])
    lines.append(f"[PASS] _longest_combined: {longest[0]}")

    partials = ImageAnalogyDataset._partial_candidates(combined, [1, 2, 3])
    assert len(partials) == 1 and partials[0][0] == "combined12.png"
    lines.append(f"[PASS] _partial_candidates (3→2): {partials[0][0]}")

    partials2 = ImageAnalogyDataset._partial_candidates(combined, [1, 2])
    assert len(partials2) == 1 and partials2[0][0] == "combined1.png"
    lines.append(f"[PASS] _partial_candidates (2→1): {partials2[0][0]}")

    no_partial = [("combined12.png", [1, 2]), ("combined123.png", [1, 2, 3])]
    fb = ImageAnalogyDataset._partial_candidates(no_partial, [1, 2, 3])
    assert any(p[0] == "combined12.png" for p in fb)
    lines.append(f"[PASS] _partial_candidates (fallback): {[p[0] for p in fb]}")

    edits_a = ["add hat", "add glasses", "change background"]
    edits_b = edits_a[:]

    full = build_analogy_prompt(edits_a, edits_b)
    assert "add hat" in full and "add glasses" in full and "change background" in full
    lines.append(f"[PASS] build_analogy_prompt (full)")

    supp = build_analogy_prompt(edits_a, edits_b, suppress_indices=[1])
    assert "add glasses" not in supp and "add hat" in supp and "change background" in supp
    lines.append(f"[PASS] build_analogy_prompt (suppress idx 1 → 'add glasses' absent)")

    assert abs(sum(_tier_probs(3)) - 1.0) < 1e-6
    lines.append("[PASS] _tier_probs(3) sums to 1.0")

    lines += ["─" * 44, "All helper unit tests passed."]
    return "\n".join(lines)


# ============================================================================
# Main smoke test
# ============================================================================

def run_smoke_test(data_root: Path, out_root: Path,
                   n_samples: int = 5, seed: int = 42) -> None:
    random.seed(seed)

    print("\n" + "=" * 64)
    print("STLoRA ImageAnalogyDataset — smoke test (real dataset)")
    print("=" * 64)
    print(f"data_root : {data_root}")
    print(f"out_root  : {out_root}")

    getitem_dir = out_root / 'getitem_samples'
    getitem_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Helper unit tests ──────────────────────────────────────────────
    print("\n[1/4] Running helper unit tests …")
    helper_report = run_helper_unit_tests()
    print(helper_report)
    (out_root / 'helper_unit_tests.txt').write_text(helper_report)

    # ── 2. Instantiate ────────────────────────────────────────────────────
    print("\n[2/4] Instantiating ImageAnalogyDataset …")
    dataset = ImageAnalogyDataset(data_root)

    length = len(dataset)
    print(f"      __len__ = {length}")
    assert length > 0, "Dataset is empty — check data_root."

    # ── 3. __getitem__ samples ────────────────────────────────────────────
    print(f"\n[3/4] Saving {n_samples} __getitem__ samples …")
    for i in range(n_samples):
        sample = dataset[i]
        save_getitem_sample(sample, getitem_dir, index=i)

    # ── 4. sample_val_pair ────────────────────────────────────────────────
    print("\n[4/4] Saving sample_val_pair output …")
    vp = dataset.sample_val_pair()
    save_val_pair(vp, out_root)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = "\n".join([
        "STLoRA ImageAnalogyDataset smoke test — PASSED",
        "",
        f"data_root      : {data_root}",
        f"out_root       : {out_root}",
        f"dataset length : {length}",
        f"categories     : {dataset.cat_names}",
        f"getitem samples: {n_samples}",
        "",
        "val_pair:",
        f"  edit_name      : {vp['edit_name']}",
        f"  suppress_index : {vp['suppress_index']}",
        f"  suppressed edit: {vp['pair_a_edits'][vp['suppress_index']] if vp['suppress_index'] < len(vp['pair_a_edits']) else 'n/a'}",
        f"  prompt         : {vp['prompt']}",
        f"  a_path         : {vp['a_path']}",
        f"  a_prime_path   : {vp['a_prime_path']}",
        f"  a_prime_partial: {vp['a_prime_partial_path']}",
        f"  b_path         : {vp['b_path']}",
        f"  b_prime_path   : {vp['b_prime_path']}",
        f"  b_total_path   : {vp['b_total_path']}",
    ])
    (out_root / 'summary.txt').write_text(summary)
    print("\n" + summary)
    print("\nAll assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STLoRA dataset smoke test")
    parser.add_argument("--data_root", type=str, default=str(_DEFAULT_DATA_ROOT),
                        help="Path to SIA_datasets/train directory")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of __getitem__ samples to draw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _out = Path(__file__).resolve().parent.parent / 'temp' / 'stlora_dataset_smoke'
    run_smoke_test(Path(args.data_root), _out,
                   n_samples=args.n_samples, seed=args.seed)
