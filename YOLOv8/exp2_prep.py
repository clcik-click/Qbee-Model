from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def read_label_text(label_path: Path | None) -> str:
    """Return label text with trailing newline. If None -> negative -> empty."""
    if label_path is None:
        return ""
    txt = label_path.read_text(encoding="utf-8", errors="ignore")
    return txt if txt.endswith("\n") or txt == "" else (txt + "\n")


def write_split(pairs: list[tuple[Path, Path | None]], img_out: Path, lbl_out: Path) -> tuple[int, int]:
    """Copy images and write label files (empty txt = negative). Returns (pos, neg)."""
    pos = 0
    neg = 0

    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)

        out_lbl_path = lbl_out / (img.stem + ".txt")
        txt = read_label_text(lbl)
        out_lbl_path.write_text(txt, encoding="utf-8")

        if txt.strip():
            pos += 1
        else:
            neg += 1

    return pos, neg


def stems_in(folder: Path, exts: set[str]) -> set[str]:
    return {p.stem for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Experiment2 dataset with test/train/val splits.")

    ap.add_argument("--root", default=".", help="Project root (default: current directory).")

    # UPDATED PATHS (matches your screenshot)
    ap.add_argument(
        "--src",
        default="data/data-with-labels/from-DrE",
        help="Source folder containing images/ and labels/.",
    )

    ap.add_argument(
        "--out",
        default="YOLOv8/Experiment2",
        help="Output folder. WILL BE DELETED and recreated each run.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.30, help="Test split ratio.")
    ap.add_argument("--train_ratio_within_70", type=float, default=0.80, help="Train ratio within remaining 70%.")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    src = (root / args.src).resolve()
    out = (root / args.out).resolve()

    src_img = src / "images"
    src_lbl = src / "labels"

    if not src_img.exists():
        raise FileNotFoundError(f"Missing images folder: {src_img}")
    if not src_lbl.exists():
        raise FileNotFoundError(f"Missing labels folder: {src_lbl}")

    # Hard reset output (replace old one)
    if out.exists():
        shutil.rmtree(out)

    test_img = out / "test_30/images"
    test_lbl = out / "test_30/labels"
    train_img = out / "train_70/train/images"
    train_lbl = out / "train_70/train/labels"
    val_img = out / "train_70/val/images"
    val_lbl = out / "train_70/val/labels"

    for d in [test_img, test_lbl, train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print("✅ Folders ready")

    # Build (img, label|None) pairs
    images = [p for p in src_img.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    images.sort()

    pairs: list[tuple[Path, Path | None]] = []
    missing_label_files = 0

    for img in images:
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            missing_label_files += 1
            pairs.append((img, None))  # negative
        else:
            pairs.append((img, lbl))

    print(f"Images found: {len(images)}")
    print(f"Missing label files (will become negatives): {missing_label_files}")

    # Split
    random.seed(args.seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_test = int(args.test_ratio * n_total)

    test_pairs = pairs[:n_test]
    train70_pairs = pairs[n_test:]

    split_idx = int(args.train_ratio_within_70 * len(train70_pairs))
    train_pairs = train70_pairs[:split_idx]
    val_pairs = train70_pairs[split_idx:]

    print(f"Test {int(args.test_ratio*100)}%:  {len(test_pairs)}")
    print(f"Train {100-int(args.test_ratio*100)}%: {len(train70_pairs)}")
    print(f"  Train:   {len(train_pairs)}")
    print(f"  Val:     {len(val_pairs)}")

    # Write outputs
    pos_te, neg_te = write_split(test_pairs, test_img, test_lbl)
    pos_tr, neg_tr = write_split(train_pairs, train_img, train_lbl)
    pos_va, neg_va = write_split(val_pairs, val_img, val_lbl)

    print(f"✅ test_30  positives: {pos_te}, negatives: {neg_te}")
    print(f"✅ train    positives: {pos_tr}, negatives: {neg_tr}")
    print(f"✅ val      positives: {pos_va}, negatives: {neg_va}")

    # Sanity checks: image/label matching
    for name, img_dir, lbl_dir in [
        ("test_30", test_img, test_lbl),
        ("train", train_img, train_lbl),
        ("val", val_img, val_lbl),
    ]:
        img_stems = stems_in(img_dir, IMG_EXTS)
        lbl_stems = stems_in(lbl_dir, {".txt"})
        print(f"\n--- {name} ---")
        print("Images:", len(img_stems), "Labels:", len(lbl_stems))
        print("✅ missing labels:", len(img_stems - lbl_stems))
        print("✅ missing images:", len(lbl_stems - img_stems))

    # Overlap checks
    test_stems = stems_in(test_img, IMG_EXTS)
    train_stems = stems_in(train_img, IMG_EXTS)
    val_stems = stems_in(val_img, IMG_EXTS)

    print("\n--- overlap checks ---")
    print("test ∩ train:", len(test_stems & train_stems))
    print("test ∩ val:  ", len(test_stems & val_stems))
    print("train ∩ val: ", len(train_stems & val_stems))

    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()