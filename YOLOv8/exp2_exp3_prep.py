from __future__ import annotations

import argparse
import random
import shutil
import re
from collections import defaultdict
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
    if not folder.exists():
        return set()
    return {p.stem for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts}


def is_frame_image(stem: str) -> bool:
    """
    Detect extracted video frame names like:
    IMG_5594_0
    IMG_5594_1
    """
    return re.fullmatch(r"IMG_\d+_\d+", stem) is not None


def get_frame_group(stem: str) -> str | None:
    """
    For IMG_5594_23 -> IMG_5594
    """
    m = re.fullmatch(r"(IMG_\d+)_\d+", stem)
    return m.group(1) if m else None


def build_pairs_grouped(
    src_frames: Path,
    src_images: Path,
    src_lbl: Path,
) -> tuple[dict[str, list[tuple[Path, Path | None]]], list[tuple[Path, Path | None]], int]:
    """
    Returns:
      frame_groups: {video_prefix: [(img, lbl_or_none), ...]}
      random_images: [(img, lbl_or_none), ...]
      missing_label_files: int
    """
    frame_groups: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    random_images: list[tuple[Path, Path | None]] = []
    missing_label_files = 0

    # Read extracted video frames from frames/
    frame_files = [p for p in src_frames.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    frame_files.sort()

    for img in frame_files:
        lbl = src_lbl / (img.stem + ".txt")
        lbl_or_none = lbl if lbl.exists() else None

        if lbl_or_none is None:
            missing_label_files += 1

        group = get_frame_group(img.stem)
        if group is None:
            # fallback in case some frame file does not match expected naming
            random_images.append((img, lbl_or_none))
        else:
            frame_groups[group].append((img, lbl_or_none))

    # Read standalone/random images from images/
    image_files = [p for p in src_images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    image_files.sort()

    for img in image_files:
        lbl = src_lbl / (img.stem + ".txt")
        lbl_or_none = lbl if lbl.exists() else None

        if lbl_or_none is None:
            missing_label_files += 1

        random_images.append((img, lbl_or_none))

    return frame_groups, random_images, missing_label_files


def sanity_check_split(name: str, img_dir: Path, lbl_dir: Path) -> None:
    img_stems = stems_in(img_dir, IMG_EXTS)
    lbl_stems = stems_in(lbl_dir, {".txt"})
    print(f"\n--- {name} ---")
    print("Images:", len(img_stems), "Labels:", len(lbl_stems))
    print("missing labels:", len(img_stems - lbl_stems))
    print("missing images:", len(lbl_stems - img_stems))


def summarize_pairs(name: str, pairs: list[tuple[Path, Path | None]]) -> None:
    pos = sum(1 for _, lbl in pairs if lbl is not None and read_label_text(lbl).strip())
    neg = len(pairs) - pos
    print(f"{name}: {len(pairs)} images | Pos: {pos} | Neg: {neg}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare Experiment2 (70/20/10, video-aware) and Experiment3 (train+val vs test)."
    )

    ap.add_argument("--root", default=".", help="Project root (default: current directory).")

    ap.add_argument(
        "--src",
        default="data/data-with-labels/from-DrE",
        help="Source folder containing frames/, images/, and labels/.",
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

    ap.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio for frame groups.")
    ap.add_argument("--val_ratio", type=float, default=0.20, help="Val ratio for frame groups.")
    ap.add_argument("--test_ratio", type=float, default=0.10, help="Test ratio for frame groups.")

    args = ap.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
        )

    root = Path(args.root).resolve()
    src = (root / args.src).resolve()
    out_exp2 = (root / args.out_exp2).resolve()
    out_exp3 = (root / args.out_exp3).resolve()

    src_frames = src / "frames"
    src_images = src / "images"
    src_lbl = src / "labels"

    if not src_frames.exists():
        raise FileNotFoundError(f"Missing frames folder: {src_frames}")
    if not src_images.exists():
        raise FileNotFoundError(f"Missing images folder: {src_images}")
    if not src_lbl.exists():
        raise FileNotFoundError(f"Missing labels folder: {src_lbl}")

    # Reset outputs
    for out in [out_exp2, out_exp3]:
        if out.exists():
            shutil.rmtree(out)

    # -------------------------
    # Build source pairs once
    # -------------------------
    frame_groups, random_images, missing_label_files = build_pairs_grouped(
        src_frames, src_images, src_lbl
    )

    total_frame_images = sum(len(v) for v in frame_groups.values())
    total_random_images = len(random_images)
    total_images = total_frame_images + total_random_images

    print("Source loaded")
    print(f"Frames folder: {src_frames}")
    print(f"Images folder: {src_images}")
    print(f"Labels folder: {src_lbl}")
    print(f"Total images found: {total_images}")
    print(f"Frame groups found: {len(frame_groups)}")
    print(f"Frame images found: {total_frame_images}")
    print(f"Random images found: {total_random_images}")
    print(f"Missing label files (will become negatives): {missing_label_files}")

    # =========================
    # Experiment 2
    # =========================
    exp2_train_img = out_exp2 / "train/images"
    exp2_train_lbl = out_exp2 / "train/labels"
    exp2_val_img = out_exp2 / "val/images"
    exp2_val_lbl = out_exp2 / "val/labels"
    exp2_test_img = out_exp2 / "test/images"
    exp2_test_lbl = out_exp2 / "test/labels"

    for d in [
        exp2_train_img, exp2_train_lbl,
        exp2_val_img, exp2_val_lbl,
        exp2_test_img, exp2_test_lbl,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Split frame groups by video prefix
    group_names = sorted(frame_groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_names)

    n_groups = len(group_names)
    n_train_groups = int(n_groups * args.train_ratio)
    n_val_groups = int(n_groups * args.val_ratio)

    train_group_names = group_names[:n_train_groups]
    val_group_names = group_names[n_train_groups:n_train_groups + n_val_groups]
    test_group_names = group_names[n_train_groups + n_val_groups:]

    exp2_train_pairs: list[tuple[Path, Path | None]] = []
    exp2_val_pairs: list[tuple[Path, Path | None]] = []
    exp2_test_pairs: list[tuple[Path, Path | None]] = []

    for g in train_group_names:
        exp2_train_pairs.extend(frame_groups[g])
    for g in val_group_names:
        exp2_val_pairs.extend(frame_groups[g])
    for g in test_group_names:
        exp2_test_pairs.extend(frame_groups[g])

    # All random images go to train
    exp2_train_pairs.extend(random_images)

    print("\n================ Experiment 2 ================")
    print(f"Frame groups total: {n_groups}")
    print(f"Train groups: {len(train_group_names)}")
    print(f"Val groups:   {len(val_group_names)}")
    print(f"Test groups:  {len(test_group_names)}")
    print(f"Random images added to train: {len(random_images)}")

    summarize_pairs("Exp2 train", exp2_train_pairs)
    summarize_pairs("Exp2 val", exp2_val_pairs)
    summarize_pairs("Exp2 test", exp2_test_pairs)

    pos_tr2, neg_tr2 = write_split(exp2_train_pairs, exp2_train_img, exp2_train_lbl)
    pos_va2, neg_va2 = write_split(exp2_val_pairs, exp2_val_img, exp2_val_lbl)
    pos_te2, neg_te2 = write_split(exp2_test_pairs, exp2_test_img, exp2_test_lbl)

    print(f"Written train positives: {pos_tr2}, negatives: {neg_tr2}")
    print(f"Written val   positives: {pos_va2}, negatives: {neg_va2}")
    print(f"Written test  positives: {pos_te2}, negatives: {neg_te2}")

    sanity_check_split("exp2 train", exp2_train_img, exp2_train_lbl)
    sanity_check_split("exp2 val", exp2_val_img, exp2_val_lbl)
    sanity_check_split("exp2 test", exp2_test_img, exp2_test_lbl)

    exp2_train_stems = stems_in(exp2_train_img, IMG_EXTS)
    exp2_val_stems = stems_in(exp2_val_img, IMG_EXTS)
    exp2_test_stems = stems_in(exp2_test_img, IMG_EXTS)

    print("\n--- exp2 overlap checks ---")
    print("train ∩ val :", len(exp2_train_stems & exp2_val_stems))
    print("train ∩ test:", len(exp2_train_stems & exp2_test_stems))
    print("val ∩ test  :", len(exp2_val_stems & exp2_test_stems))

    # =========================
    # Experiment 3
    # final-output model:
    # train = exp2 train + exp2 val
    # val   = exp2 test
    # =========================
    exp3_train_img = out_exp3 / "train/images"
    exp3_train_lbl = out_exp3 / "train/labels"
    exp3_val_img = out_exp3 / "val/images"
    exp3_val_lbl = out_exp3 / "val/labels"

    for d in [exp3_train_img, exp3_train_lbl, exp3_val_img, exp3_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    exp3_train_pairs = exp2_train_pairs + exp2_val_pairs
    exp3_val_pairs = exp2_test_pairs

    print("\n================ Experiment 3 ================")
    print("Exp3 train = Exp2 train + Exp2 val")
    print("Exp3 val   = Exp2 test")
    summarize_pairs("Exp3 train", exp3_train_pairs)
    summarize_pairs("Exp3 val", exp3_val_pairs)

    pos_tr3, neg_tr3 = write_split(exp3_train_pairs, exp3_train_img, exp3_train_lbl)
    pos_va3, neg_va3 = write_split(exp3_val_pairs, exp3_val_img, exp3_val_lbl)

    print(f"Written train positives: {pos_tr3}, negatives: {neg_tr3}")
    print(f"Written val   positives: {pos_va3}, negatives: {neg_va3}")

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