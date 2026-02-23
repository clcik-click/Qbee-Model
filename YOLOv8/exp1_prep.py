from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import re

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Pair:
    img: Path
    lbl: Path


def filter_to_queen(label_path: Path, queen_id_src: int, queen_id_new: int) -> str:
    """
    Keep ONLY queen boxes (class == queen_id_src) and remap to queen_id_new.
    If no queen boxes, return "" (empty) -> negative example.
    """
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return ""

    lines_out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        cls = int(float(parts[0]))  # robust if "3.0"
        if cls == queen_id_src:
            parts[0] = str(queen_id_new)
            lines_out.append(" ".join(parts))

    return "\n".join(lines_out) + ("\n" if lines_out else "")


def list_pairs(src_selected: Path) -> list[Pair]:
    src_img = src_selected / "images"
    src_lbl = src_selected / "labels"

    if not src_img.exists():
        raise FileNotFoundError(f"Missing images folder: {src_img}")
    if not src_lbl.exists():
        raise FileNotFoundError(f"Missing labels folder: {src_lbl}")

    pairs: list[Pair] = []
    missing_lbl = 0

    for img in src_img.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            missing_lbl += 1
            continue
        pairs.append(Pair(img=img, lbl=lbl))

    print(f"Found pairs: {len(pairs)} | Missing labels: {missing_lbl}")
    return pairs


def hard_reset_out(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    train_img = out_dir / "train/images"
    train_lbl = out_dir / "train/labels"
    val_img = out_dir / "val/images"
    val_lbl = out_dir / "val/labels"

    for p in [train_img, train_lbl, val_img, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    return train_img, train_lbl, val_img, val_lbl


def write_split_include_negatives(
    pairs: list[Pair],
    img_out: Path,
    lbl_out: Path,
    queen_id_src: int,
    queen_id_new: int,
) -> tuple[int, int]:
    pos = 0
    neg = 0

    for pair in pairs:
        new_txt = filter_to_queen(pair.lbl, queen_id_src, queen_id_new)

        # copy image always
        shutil.copy2(pair.img, img_out / pair.img.name)

        # write label always (empty file = negative)
        out_lbl = lbl_out / (pair.img.stem + ".txt")
        out_lbl.write_text(new_txt, encoding="utf-8")

        if new_txt.strip():
            pos += 1
        else:
            neg += 1

    return pos, neg


def sanity_report(img_dir: Path, lbl_dir: Path, name: str) -> None:
    imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    lbls = sorted([p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}

    missing_lbl = sorted(img_stems - lbl_stems)
    missing_img = sorted(lbl_stems - img_stems)

    empty_lbl = 0
    nonempty_lbl = 0
    for p in lbls:
        if p.read_text(encoding="utf-8", errors="ignore").strip() == "":
            empty_lbl += 1
        else:
            nonempty_lbl += 1

    print(f"\n--- {name} ---")
    print(f"Images: {len(imgs)}")
    print(f"Labels: {len(lbls)}")
    print(f"Non-empty labels (queen present): {nonempty_lbl}")
    print(f"Empty labels (no queen): {empty_lbl}")

    if missing_lbl:
        print(f"⚠️ Images missing labels: {len(missing_lbl)} (showing up to 10)")
        print(missing_lbl[:10])
    else:
        print("✅ Every image has a label file")

    if missing_img:
        print(f"⚠️ Labels missing images: {len(missing_img)} (showing up to 10)")
        print(missing_img[:10])
    else:
        print("✅ Every label corresponds to an image")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Experiment1 dataset (queen-only, single-class).")

    ap.add_argument("--root", default=".", help="Project root (default: current directory).")

    # UPDATED to your new structure
    ap.add_argument(
        "--src_selected",
        default="data/data-with-labels/Selected67",
        help="Folder containing images/ and labels/ (output of collect_pictures).",
    )

    ap.add_argument(
        "--out_dir",
        default="YOLOv8/Experiment1",
        help="Output dataset folder. Will be DELETED and recreated each run.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)

    ap.add_argument("--queen_id_src", type=int, default=3, help="Queen class index in source labels.")
    ap.add_argument("--queen_id_new", type=int, default=0, help="New class id for queen in output labels.")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    src_selected = (root / args.src_selected).resolve()
    out_dir = (root / args.out_dir).resolve()

    pairs = list_pairs(src_selected)
    if not pairs:
        print("No valid (image,label) pairs found. Aborting.")
        return

    random.seed(args.seed)
    random.shuffle(pairs)

    split_idx = int(args.train_ratio * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")

    train_img, train_lbl, val_img, val_lbl = hard_reset_out(out_dir)

    pos_tr, neg_tr = write_split_include_negatives(
        train_pairs, train_img, train_lbl, args.queen_id_src, args.queen_id_new
    )
    pos_va, neg_va = write_split_include_negatives(
        val_pairs, val_img, val_lbl, args.queen_id_src, args.queen_id_new
    )

    print(f"✅ Train positives (has queen): {pos_tr}, negatives (no queen): {neg_tr}")
    print(f"✅ Val   positives (has queen): {pos_va}, negatives (no queen): {neg_va}")

    sanity_report(train_img, train_lbl, "Experiment1 / train")
    sanity_report(val_img, val_lbl, "Experiment1 / val")

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()