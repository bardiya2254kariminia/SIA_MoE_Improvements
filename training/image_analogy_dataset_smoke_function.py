"""
Smoke test for ImageAnalogyDataset from train_gated_moe_lora_flux2_klein.py.

Runs against the real training dataset and saves outputs to
SIA_MoE_Improvements/temp/moe_dataset_smoke/.

Output layout
─────────────
temp/moe_dataset_smoke/
    getitem_samples/
        sample_0/
            A.png  A_prime.png  B.png  B_prime.png
            strip.png
            prompt.json
        sample_1/ … sample_N/
    val_pair/
        A.png  A_prime.png  B.png  B_prime.png
        strip.png
        prompt.json
    summary.txt

Usage
─────
    python3 training/image_analogy_dataset_smoke_function.py [--n_samples N] [--seed S]
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
# Helpers — verbatim from train_gated_moe_lora_flux2_klein.py
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


def _build_prompt_from_indices(edit_indices: list, prompt_dict: dict) -> str:
    edits_part = " ".join(
        f"Edit {i}: {prompt_dict[f'edit{i}']}."
        for i in sorted(edit_indices)
        if prompt_dict.get(f'edit{i}', '')
    )
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )


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


def harmonize_images(a, a_prime, b, b_prime):
    target_w, target_h = a.size
    a_prime = resize_to_match(a_prime, target_w, target_h)
    b       = resize_to_match(b,       target_w, target_h)
    b_prime = resize_to_match(b_prime, target_w, target_h)
    return a, a_prime, b, b_prime


# ============================================================================
# Dataset — exact copy from train_gated_moe_lora_flux2_klein.py
# ============================================================================

class ImageAnalogyDataset(Dataset):
    """Image analogy dataset reading from the SIA_datasets/train structure."""

    def __init__(self, data_root, mode="concat"):
        self.data_root = Path(data_root)
        self.mode = mode

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
                    if not combined:
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
        print(f"SIADataset: {len(self.cat_names)} categories, {total} samples "
              f"({n_skipped} skipped)")
        for cat in self.cat_names:
            n = sum(len(s) for s in self.categories[cat].values())
            print(f"  {cat}: {len(self.categories[cat])} settings, {n} samples")

    def __len__(self):
        return sum(len(s) for c in self.categories.values() for s in c.values())

    def sample_val_pair(self):
        cat     = random.choice(self.cat_names)
        setting = random.choice(list(self.categories[cat].keys()))
        samples = self.categories[cat][setting]
        sample_a, sample_b = random.sample(samples, 2)
        data_dir_a, combined_a, prompt_a = sample_a
        data_dir_b, combined_b, prompt_b = sample_b

        map_a  = {frozenset(idx): fn for fn, idx in combined_a}
        map_b  = {frozenset(idx): fn for fn, idx in combined_b}
        common = set(map_a) & set(map_b)
        pool   = [(map_a[k], sorted(k)) for k in common] if common else combined_a
        fname_a, indices = _sample_combined(pool)
        fname_b = map_b.get(frozenset(indices), _sample_combined(combined_b)[0])

        prompt = _build_prompt_from_indices(indices, prompt_a)
        return {
            'a_path':       str(data_dir_a / 'input.png'),
            'a_prime_path': str(data_dir_a / fname_a),
            'b_path':       str(data_dir_b / 'input.png'),
            'b_prime_path': str(data_dir_b / fname_b),
            'edit_name':    f"{cat}/{setting}",
            'prompt':       prompt,
        }

    def __getitem__(self, idx):
        cat     = random.choice(self.cat_names)
        settings = self.categories[cat]
        setting  = random.choice(list(settings.keys()))
        samples  = settings[setting]

        sample_a, sample_b = random.sample(samples, 2)
        data_dir_a, combined_a, prompt_a = sample_a
        data_dir_b, combined_b, prompt_b = sample_b

        map_a  = {frozenset(idx): fn for fn, idx in combined_a}
        map_b  = {frozenset(idx): fn for fn, idx in combined_b}
        common = set(map_a) & set(map_b)

        if common:
            pool    = [(map_a[k], sorted(k)) for k in common]
            fname_a, indices = _sample_combined(pool)
            fname_b = map_b[frozenset(indices)]
        else:
            fname_a, indices = _sample_combined(combined_a)
            fname_b, _       = _sample_combined(combined_b)

        a       = Image.open(data_dir_a / 'input.png').convert('RGB')
        a_prime = Image.open(data_dir_a / fname_a).convert('RGB')
        b       = Image.open(data_dir_b / 'input.png').convert('RGB')
        b_prime = Image.open(data_dir_b / fname_b).convert('RGB')
        a, a_prime, b, b_prime = harmonize_images(a, a_prime, b, b_prime)

        combined_prompt = _build_prompt_from_indices(indices, prompt_a)
        if random.random() < 0.1:
            combined_prompt = ""

        return {
            'A':         a,
            'A_prime':   a_prime,
            'B':         b,
            'B_prime':   b_prime,
            'edit_name': f"{cat}/{setting}",
            'prompt':    combined_prompt,
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
    """Save A/A_prime/B/B_prime PNGs, a strip, and prompt JSON for one __getitem__ call."""
    sd = out_dir / f"sample_{index}"
    sd.mkdir(parents=True, exist_ok=True)

    sample['A'].save(sd / 'A.png')
    sample['A_prime'].save(sd / 'A_prime.png')
    sample['B'].save(sd / 'B.png')
    sample['B_prime'].save(sd / 'B_prime.png')

    _make_strip(
        [sample['A'], sample['A_prime'], sample['B'], sample['B_prime']],
        ['A (input)', "A' (edited)", 'B (input)', "B' (target)"],
    ).save(sd / 'strip.png')

    (sd / 'prompt.json').write_text(json.dumps({
        'edit_name': sample['edit_name'],
        'prompt':    sample['prompt'],
    }, indent=2, ensure_ascii=False))

    print(f"  [getitem #{index}]  {sample['edit_name']}  →  {sd.name}/")


def save_val_pair(vp: dict, out_dir: Path) -> None:
    """Save images and prompt JSON from sample_val_pair()."""
    vpd = out_dir / 'val_pair'
    vpd.mkdir(parents=True, exist_ok=True)

    imgs = {
        'A':       Image.open(vp['a_path']).convert('RGB'),
        'A_prime': Image.open(vp['a_prime_path']).convert('RGB'),
        'B':       Image.open(vp['b_path']).convert('RGB'),
        'B_prime': Image.open(vp['b_prime_path']).convert('RGB'),
    }
    for name, img in imgs.items():
        img.save(vpd / f'{name}.png')

    _make_strip(
        list(imgs.values()),
        ['A (input)', "A' (edited)", 'B (input)', "B' (target)"],
    ).save(vpd / 'strip.png')

    (vpd / 'prompt.json').write_text(json.dumps({
        'edit_name':    vp['edit_name'],
        'prompt':       vp['prompt'],
        'a_path':       vp['a_path'],
        'a_prime_path': vp['a_prime_path'],
        'b_path':       vp['b_path'],
        'b_prime_path': vp['b_prime_path'],
    }, indent=2, ensure_ascii=False))

    print(f"  [val_pair]  {vp['edit_name']}  →  {vpd.name}/")


# ============================================================================
# Main smoke test
# ============================================================================

def run_smoke_test(data_root: Path, out_root: Path,
                   n_samples: int = 5, seed: int = 42) -> None:
    random.seed(seed)

    print("\n" + "=" * 64)
    print("MoE ImageAnalogyDataset — smoke test (real dataset)")
    print("=" * 64)
    print(f"data_root : {data_root}")
    print(f"out_root  : {out_root}")

    getitem_dir = out_root / 'getitem_samples'
    getitem_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Instantiate ────────────────────────────────────────────────────
    print("\n[1/3] Instantiating ImageAnalogyDataset …")
    dataset = ImageAnalogyDataset(data_root, mode="concat")

    length = len(dataset)
    print(f"      __len__ = {length}")
    assert length > 0, "Dataset is empty — check data_root."

    # ── 2. __getitem__ samples ────────────────────────────────────────────
    print(f"\n[2/3] Saving {n_samples} __getitem__ samples …")
    for i in range(n_samples):
        sample = dataset[i]
        save_getitem_sample(sample, getitem_dir, index=i)

    # ── 3. sample_val_pair ────────────────────────────────────────────────
    print("\n[3/3] Saving sample_val_pair output …")
    vp = dataset.sample_val_pair()
    save_val_pair(vp, out_root)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = "\n".join([
        "MoE ImageAnalogyDataset smoke test — PASSED",
        "",
        f"data_root      : {data_root}",
        f"out_root       : {out_root}",
        f"dataset length : {length}",
        f"categories     : {dataset.cat_names}",
        f"getitem samples: {n_samples}",
        "",
        "val_pair:",
        f"  edit_name   : {vp['edit_name']}",
        f"  prompt      : {vp['prompt']}",
        f"  a_path      : {vp['a_path']}",
        f"  a_prime_path: {vp['a_prime_path']}",
        f"  b_path      : {vp['b_path']}",
        f"  b_prime_path: {vp['b_prime_path']}",
    ])
    (out_root / 'summary.txt').write_text(summary)
    print("\n" + summary)
    print("\nAll assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE dataset smoke test")
    parser.add_argument("--data_root", type=str, default=str(_DEFAULT_DATA_ROOT),
                        help="Path to SIA_datasets/train directory")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of __getitem__ samples to draw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _out = Path(__file__).resolve().parent.parent / 'temp' / 'moe_dataset_smoke'
    run_smoke_test(Path(args.data_root), _out,
                   n_samples=args.n_samples, seed=args.seed)
