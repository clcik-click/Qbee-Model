import random
import shutil
from pathlib import Path
import re

IMG_EXTS = {".jpg", ".jpeg", ".png"}


ROOT = Path(".").resolve()

# Input folders
SRC_FOLDERS = [
    ROOT / "data/data-with-labels/Selected67",
    ROOT / "data/data-with-labels/from-online",
]

# Output
OUT_DIR = ROOT / "YOLOv8/Experiment1"

TRAIN_RATIO = 0.8
SEED = 42


def filter_to_queen_only(label_path: Path) -> str:
    """
    Keep ONLY class 0 (queen).
    If no queen boxes, return empty string (negative example).
    """
    if not label_path.exists():
        return ""

    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return ""

    lines_out = []

    for line in txt.splitlines():
        parts = re.split(r"\s+", line.strip())
        if not parts:
            continue

        try:
            cls = int(float(parts[0]))
        except Exception:
            continue

        if cls == 0:
            lines_out.append(" ".join(parts))

    return "\n".join(lines_out) + ("\n" if lines_out else "")


def collect_pairs():
    pairs = []

    for src in SRC_FOLDERS:
        img_dir = src / "images"
        lbl_dir = src / "labels"

        if not img_dir.exists():
            continue

        for img in img_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue

            lbl = lbl_dir / (img.stem + ".txt")
            pairs.append((img, lbl))

    print(f"Total images found: {len(pairs)}")
    return pairs


def reset_output():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def write_split(pairs, img_out, lbl_out):
    pos = 0
    neg = 0

    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)

        filtered = filter_to_queen_only(lbl)
        out_lbl = lbl_out / (img.stem + ".txt")
        out_lbl.write_text(filtered, encoding="utf-8")

        if filtered.strip():
            pos += 1
        else:
            neg += 1

    return pos, neg


def main():
    pairs = collect_pairs()

    if not pairs:
        print("No images found. Aborting.")
        return

    random.seed(SEED)
    random.shuffle(pairs)

    split_idx = int(TRAIN_RATIO * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    reset_output()

    train_img = OUT_DIR / "train/images"
    train_lbl = OUT_DIR / "train/labels"
    val_img = OUT_DIR / "val/images"
    val_lbl = OUT_DIR / "val/labels"

    pos_tr, neg_tr = write_split(train_pairs, train_img, train_lbl)
    pos_va, neg_va = write_split(val_pairs, val_img, val_lbl)

    print(f"\nTrain: {len(train_pairs)} | Pos: {pos_tr} | Neg: {neg_tr}")
    print(f"Val:   {len(val_pairs)} | Pos: {pos_va} | Neg: {neg_va}")
    print(f"\nOutput written to: {OUT_DIR}")


if __name__ == "__main__":
    main()