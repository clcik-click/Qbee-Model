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


def write_split(
    pairs: list[tuple[Path, Path | None]],
    img_out: Path,
    lbl_out: Path,
) -> tuple[int, int]:
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


def build_pairs(src_img: Path, src_lbl: Path) -> tuple[list[tuple[Path, Path | None]], int]:
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

    return pairs, missing_label_files


def sanity_check_split(name: str, img_dir: Path, lbl_dir: Path) -> None:
    img_stems = stems_in(img_dir, IMG_EXTS)
    lbl_stems = stems_in(lbl_dir, {".txt"})
    print(f"\n--- {name} ---")
    print("Images:", len(img_stems), "Labels:", len(lbl_stems))
    print("✅ missing labels:", len(img_stems - lbl_stems))
    print("✅ missing images:", len(lbl_stems - img_stems))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare both Experiment2 and Experiment3 datasets from the same source dataset."
    )

    ap.add_argument("--root", default=".", help="Project root (default: current directory).")

    ap.add_argument(
        "--src",
        default="data/data-with-labels/from-DrE",
        help="Source folder containing images/ and labels/.",
    )

    ap.add_argument(
        "--out_exp2",
        default="YOLOv8/Experiment2",
        help="Output folder for Experiment2. WILL BE DELETED and recreated each run.",
    )

    ap.add_argument(
        "--out_exp3",
        default="YOLOv8/Experiment3",
        help="Output folder for Experiment3. WILL BE DELETED and recreated each run.",
    )

    ap.add_argument("--seed", type=int, default=42)

    # Experiment 2
    ap.add_argument("--test_ratio", type=float, default=0.30, help="Test split ratio for Experiment2.")
    ap.add_argument(
        "--train_ratio_within_70",
        type=float,
        default=0.80,
        help="Train ratio within remaining 70 percent for Experiment2.",
    )

    # Experiment 3
    ap.add_argument("--train_ratio_exp3", type=float, default=0.80, help="Train split ratio for Experiment3.")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    src = (root / args.src).resolve()
    out_exp2 = (root / args.out_exp2).resolve()
    out_exp3 = (root / args.out_exp3).resolve()

    src_img = src / "images"
    src_lbl = src / "labels"

    if not src_img.exists():
        raise FileNotFoundError(f"Missing images folder: {src_img}")
    if not src_lbl.exists():
        raise FileNotFoundError(f"Missing labels folder: {src_lbl}")

    # Reset outputs
    for out in [out_exp2, out_exp3]:
        if out.exists():
            shutil.rmtree(out)

    # -------------------------
    # Build source pairs once
    # -------------------------
    pairs, missing_label_files = build_pairs(src_img, src_lbl)

    print("✅ Source loaded")
    print(f"Images found: {len(pairs)}")
    print(f"Missing label files (will become negatives): {missing_label_files}")

    # =========================
    # Experiment 2
    # =========================
    exp2_test_img = out_exp2 / "test_30/images"
    exp2_test_lbl = out_exp2 / "test_30/labels"
    exp2_train_img = out_exp2 / "train_70/train/images"
    exp2_train_lbl = out_exp2 / "train_70/train/labels"
    exp2_val_img = out_exp2 / "train_70/val/images"
    exp2_val_lbl = out_exp2 / "train_70/val/labels"

    for d in [
        exp2_test_img, exp2_test_lbl,
        exp2_train_img, exp2_train_lbl,
        exp2_val_img, exp2_val_lbl,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    pairs_exp2 = pairs.copy()
    random.seed(args.seed)
    random.shuffle(pairs_exp2)

    n_total = len(pairs_exp2)
    n_test = int(args.test_ratio * n_total)

    exp2_test_pairs = pairs_exp2[:n_test]
    exp2_train70_pairs = pairs_exp2[n_test:]

    split_idx_exp2 = int(args.train_ratio_within_70 * len(exp2_train70_pairs))
    exp2_train_pairs = exp2_train70_pairs[:split_idx_exp2]
    exp2_val_pairs = exp2_train70_pairs[split_idx_exp2:]

    print("\n================ Experiment 2 ================")
    print(f"Test {int(args.test_ratio * 100)}%: {len(exp2_test_pairs)}")
    print(f"Train+Val {100 - int(args.test_ratio * 100)}%: {len(exp2_train70_pairs)}")
    print(f"  Train: {len(exp2_train_pairs)}")
    print(f"  Val:   {len(exp2_val_pairs)}")

    pos_te, neg_te = write_split(exp2_test_pairs, exp2_test_img, exp2_test_lbl)
    pos_tr2, neg_tr2 = write_split(exp2_train_pairs, exp2_train_img, exp2_train_lbl)
    pos_va2, neg_va2 = write_split(exp2_val_pairs, exp2_val_img, exp2_val_lbl)

    print(f"✅ test_30  positives: {pos_te}, negatives: {neg_te}")
    print(f"✅ train    positives: {pos_tr2}, negatives: {neg_tr2}")
    print(f"✅ val      positives: {pos_va2}, negatives: {neg_va2}")

    sanity_check_split("exp2 test_30", exp2_test_img, exp2_test_lbl)
    sanity_check_split("exp2 train", exp2_train_img, exp2_train_lbl)
    sanity_check_split("exp2 val", exp2_val_img, exp2_val_lbl)

    exp2_test_stems = stems_in(exp2_test_img, IMG_EXTS)
    exp2_train_stems = stems_in(exp2_train_img, IMG_EXTS)
    exp2_val_stems = stems_in(exp2_val_img, IMG_EXTS)

    print("\n--- exp2 overlap checks ---")
    print("test ∩ train:", len(exp2_test_stems & exp2_train_stems))
    print("test ∩ val:  ", len(exp2_test_stems & exp2_val_stems))
    print("train ∩ val: ", len(exp2_train_stems & exp2_val_stems))

    # =========================
    # Experiment 3
    # =========================
    exp3_train_img = out_exp3 / "train/images"
    exp3_train_lbl = out_exp3 / "train/labels"
    exp3_val_img = out_exp3 / "val/images"
    exp3_val_lbl = out_exp3 / "val/labels"

    for d in [exp3_train_img, exp3_train_lbl, exp3_val_img, exp3_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    pairs_exp3 = pairs.copy()
    random.seed(args.seed)
    random.shuffle(pairs_exp3)

    split_idx_exp3 = int(args.train_ratio_exp3 * len(pairs_exp3))
    exp3_train_pairs = pairs_exp3[:split_idx_exp3]
    exp3_val_pairs = pairs_exp3[split_idx_exp3:]

    print("\n================ Experiment 3 ================")
    print(f"Train {int(args.train_ratio_exp3 * 100)}%: {len(exp3_train_pairs)}")
    print(f"Val   {100 - int(args.train_ratio_exp3 * 100)}%: {len(exp3_val_pairs)}")

    pos_tr3, neg_tr3 = write_split(exp3_train_pairs, exp3_train_img, exp3_train_lbl)
    pos_va3, neg_va3 = write_split(exp3_val_pairs, exp3_val_img, exp3_val_lbl)

    print(f"✅ train  positives: {pos_tr3}, negatives: {neg_tr3}")
    print(f"✅ val    positives: {pos_va3}, negatives: {neg_va3}")

    sanity_check_split("exp3 train", exp3_train_img, exp3_train_lbl)
    sanity_check_split("exp3 val", exp3_val_img, exp3_val_lbl)

    exp3_train_stems = stems_in(exp3_train_img, IMG_EXTS)
    exp3_val_stems = stems_in(exp3_val_img, IMG_EXTS)

    print("\n--- exp3 overlap checks ---")
    print("train ∩ val:", len(exp3_train_stems & exp3_val_stems))

    print(f"\nExperiment2 output: {out_exp2}")
    print(f"Experiment3 output: {out_exp3}")


if __name__ == "__main__":
    main()


# python YOLOv8/exp2_exp3_prep.py --seed 42