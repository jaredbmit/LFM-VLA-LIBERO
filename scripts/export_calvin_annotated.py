"""Copy only language-annotated episode files from CALVIN D to a smaller directory.

The output has the same structure as the original, so CALVINDataset works unchanged.

Usage:
    python export_calvin_annotated.py \
        --src /path/to/task_D_D \
        --dst /path/to/task_D_D_annotated
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def export_split(src_dir: Path, dst_dir: Path):
    ann_path = src_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        print(f"  Skipping {src_dir.name}: no annotations found")
        return

    ann = np.load(ann_path, allow_pickle=True).item()
    indx = ann["info"]["indx"]

    # Collect all frame IDs covered by annotations
    frame_ids = set()
    for start, end in indx:
        for f in range(start, end + 1):
            frame_ids.add(f)

    # Filter to files that exist
    frame_ids = sorted(
        f for f in frame_ids if (src_dir / f"episode_{f:07d}.npz").exists()
    )
    print(f"  {len(frame_ids)} frames to copy")

    # Copy annotation file
    dst_dir.mkdir(parents=True, exist_ok=True)
    ann_dst = dst_dir / "lang_annotations"
    ann_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ann_path, ann_dst / "auto_lang_ann.npy")

    # Copy episode files
    for fid in tqdm(frame_ids, desc=f"  {src_dir.name}"):
        fname = f"episode_{fid:07d}.npz"
        shutil.copy2(src_dir / fname, dst_dir / fname)

    print(f"  Saved {len(frame_ids)} episodes to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export annotated CALVIN frames")
    parser.add_argument("--src", required=True, help="Path to full CALVIN dataset (e.g. task_D_D)")
    parser.add_argument("--dst", required=True, help="Output path")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    for split in ["training", "validation"]:
        split_src = src / split
        if not split_src.exists():
            continue
        print(f"Exporting {split}...")
        export_split(split_src, dst / split)

    print("Done!")


if __name__ == "__main__":
    main()
