# collect_pictures.py
"""
Copy the images listed in selected_pictures.txt from a YOLOv8 dataset
(train/valid/test with images/ and labels/) into Selected<N>/.

This version copies ALL duplicates (all matches for each list line).

Key change (to avoid overly-broad matches like "bees"):
- Matching key is built from the FULL filename INCLUDING extension letters.
  Examples:
    "bees.jpg"      -> "beesjpg"
    "Bee SG 5.jpg"  -> "beesg5jpg"
    dataset "Bee-SG-5_jpg.rf....jpg" -> "beesg5jpgrf....jpg"

Matching:
1) exact key match
2) prefix key match (dataset_key.startswith(list_key))

Queen-only label option:
- If --queen_only is set AND --copy_labels is set:
  - Keep ONLY queen boxes (class == --queen_id_src)
  - Remap class id to --queen_id_new (default 0)
  - If no queen boxes -> write empty label file (negative example)
  - If label file missing -> write empty label file

Outputs (by default):
data/data-with-labels/Selected<N>/
  images/
  labels/ (optional)
  selected.csv
  all_matches.txt
  missing.txt
"""

# python collect_pictures.py --copy_labels --queen_only --queen_id_src 3 --queen_id_new 0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import re
import shutil
from typing import List, Tuple, Set


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS_DEFAULT = ["train", "valid", "test"]


@dataclass(frozen=True)
class Item:
    img: Path
    lbl: Path
    split: str


def normalize_full_filename(name: str) -> str:
    """
    Normalize using FULL filename (including extension letters):
    - lowercase
    - remove spaces, punctuation, brackets, underscores, hyphens, dots, etc.
    - keep letters+numbers only

    Examples:
      "Bee SG 5.jpg"              -> "beesg5jpg"
      "bees.jpg"                  -> "beesjpg"
      "Bee-SG-5_jpg.rf.x.jpg"     -> "beesg5jpgrfxjpg"
    """
    base = Path(name).name.lower()  # full filename, not stem
    return re.sub(r"[^a-z0-9]+", "", base)


def list_images(split_dir: Path) -> List[Path]:
    d = split_dir / "images"
    if not d.exists():
        return []
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def label_path(img: Path, split_dir: Path) -> Path:
    return split_dir / "labels" / (img.stem + ".txt")


def load_wanted(txt_path: Path) -> List[Tuple[str, str]]:
    """
    Read selected_pictures.txt -> [(original_line, normalized_key), ...]

    Tip: include extensions in your list lines to make matching more specific
    (e.g., "Bee SG 5.jpg" not just "Bee SG 5").
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing list file: {txt_path}")

    lines = [
        ln.strip().strip('"').strip("'")
        for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]
    return [(ln, normalize_full_filename(ln)) for ln in lines]


def build_index(yolo_dir: Path, splits: List[str]) -> List[Item]:
    items: List[Item] = []
    for split in splits:
        sd = yolo_dir / split
        for img in list_images(sd):
            items.append(Item(img=img, lbl=label_path(img, sd), split=split))
    return items


def precompute_keys(items: List[Item]) -> List[Tuple[Item, str]]:
    """Compute dataset normalized keys once for speed."""
    return [(it, normalize_full_filename(it.img.name)) for it in items]


def find_all_matches(prekeyed: List[Tuple[Item, str]], wanted_key: str) -> List[Item]:
    """
    Find ALL matches for a wanted key.

    1) exact key match
    2) prefix key match
    """
    exact = [it for (it, k) in prekeyed if k == wanted_key]
    if exact:
        return exact
    return [it for (it, k) in prekeyed if k.startswith(wanted_key)]


def filter_to_queen(label_path: Path, queen_id_src: int, queen_id_new: int) -> str:
    """
    Keep ONLY queen boxes (class == queen_id_src) and remap to queen_id_new.
    If no queen boxes (or missing/empty label file), return "" (empty) -> negative example.
    """
    if not label_path.exists():
        return ""

    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return ""

    lines_out: List[str] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if not parts:
            continue

        # Robust if class written like "3.0"
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue

        if cls == queen_id_src:
            parts[0] = str(queen_id_new)
            lines_out.append(" ".join(parts))

    return "\n".join(lines_out) + ("\n" if lines_out else "")


def clear_output(out_dir: Path, out_base: str) -> None:
    # Safety: only delete folders that start with out_base (default: "Selected")
    assert out_dir.name.lower().startswith(out_base.lower()), "Refusing to delete unexpected folder"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)


def clear_previous_selected(out_parent: Path, out_base: str) -> None:
    """
    Delete previous outputs like Selected67, Selected120, etc. inside out_parent.
    """
    base = out_base.lower()
    for p in out_parent.iterdir():
        if p.is_dir() and p.name.lower().startswith(base):
            shutil.rmtree(p)


def copy_selected(
    sel: List[Item],
    out_dir: Path,
    copy_labels: bool,
    queen_only: bool,
    queen_id_src: int,
    queen_id_new: int,
) -> None:
    """
    Copy all selected items.
    Prevent copying the exact same file path twice (if your list repeats lines).

    If copy_labels is True:
      - If queen_only is True: write filtered labels (queen-only, remapped)
      - Else: copy label if exists, otherwise write empty label file
    """
    oi, ol = out_dir / "images", out_dir / "labels"
    copied_paths: Set[str] = set()

    for it in sel:
        key = str(it.img.resolve())
        if key in copied_paths:
            continue
        copied_paths.add(key)

        # Always copy image
        shutil.copy2(it.img, oi / it.img.name)

        if not copy_labels:
            continue

        out_lbl = ol / (it.img.stem + ".txt")

        if queen_only:
            new_txt = filter_to_queen(it.lbl, queen_id_src, queen_id_new)
            out_lbl.write_text(new_txt, encoding="utf-8")
        else:
            if it.lbl.exists():
                shutil.copy2(it.lbl, out_lbl)
            else:
                out_lbl.write_text("", encoding="utf-8")


def write_selected_csv(sel: List[Item], out_dir: Path) -> None:
    p = out_dir / "selected.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "img_path", "lbl_path", "label_exists"])
        for it in sel:
            w.writerow([it.img.name, it.split, str(it.img), str(it.lbl), int(it.lbl.exists())])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--yolo_dir",
        default="data/data-with-labels/Honey Bee Detection Model.v1i.yolov8",
        help="Path (relative to --root) to YOLO dataset folder containing train/valid/test.",
    )
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--list_txt", default="selected_pictures.txt")
    ap.add_argument(
        "--out_parent",
        default="data/data-with-labels",
        help="Parent folder where Selected<N>/ will be created.",
    )
    ap.add_argument("--out_base", default="Selected")
    ap.add_argument("--copy_labels", action="store_true")

    # Queen-only label filtering/remap
    ap.add_argument(
        "--queen_only",
        action="store_true",
        help="If set (and --copy_labels), keep ONLY queen boxes and remap to --queen_id_new.",
    )
    ap.add_argument("--queen_id_src", type=int, default=3, help="Queen class index in source labels.")
    ap.add_argument("--queen_id_new", type=int, default=0, help="New class id for queen in output labels.")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    yolo = (root / args.yolo_dir).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] or SPLITS_DEFAULT

    wanted = load_wanted(root / args.list_txt)
    items = build_index(yolo, splits)
    if not items:
        print("No images found under the provided YOLO directory.")
        print(f"Checked: {yolo}")
        print(f"Splits:   {splits}")
        print("Expected: <split>/images/*.jpg (and optional <split>/labels/*.txt)")
        return

    prekeyed = precompute_keys(items)

    chosen: List[Item] = []
    all_matches_lines: List[str] = []
    missing_lines: List[str] = []

    for original, key in wanted:
        matches = find_all_matches(prekeyed, key)

        if not matches:
            missing_lines.append(original)
            continue

        all_matches_lines.append(f"{original}  (key={key})")
        for m in matches:
            all_matches_lines.append(f"  - {m.split}: {m.img}")
            chosen.append(m)  # copy ALL duplicates
        all_matches_lines.append("")

    out_parent = (root / args.out_parent).resolve()
    out_parent.mkdir(parents=True, exist_ok=True)

    # delete old Selected* outputs in out_parent so the new run replaces them
    clear_previous_selected(out_parent, args.out_base)

    out_dir = out_parent / f"{args.out_base}{len(chosen)}"
    clear_output(out_dir, out_base=args.out_base)

    copy_selected(
        chosen,
        out_dir,
        copy_labels=args.copy_labels,
        queen_only=args.queen_only,
        queen_id_src=args.queen_id_src,
        queen_id_new=args.queen_id_new,
    )
    write_selected_csv(chosen, out_dir)

    (out_dir / "all_matches.txt").write_text("\n".join(all_matches_lines), encoding="utf-8")
    (out_dir / "missing.txt").write_text("\n".join(missing_lines), encoding="utf-8")

    print(f"List size: {len(wanted)}")
    print(f"Matched:   {len(chosen)}  (all duplicates included)")
    print(f"Missing:   {len(missing_lines)}")
    print(f"Output:    {out_dir}")

    if args.copy_labels and args.queen_only:
        print(f"Queen-only labels: src={args.queen_id_src} -> new={args.queen_id_new} (empty = negative)")

    elif args.copy_labels:
        print("Labels copied as-is (missing labels written as empty .txt).")


if __name__ == "__main__":
    main()