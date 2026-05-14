import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from utils.svd_filter import svd_fil


def list_npy_files(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted([path.name for path in folder.iterdir() if path.suffix == ".npy"])


def make_output_dirs(output_folder):
    output_folder = Path(output_folder)
    data_dir = output_folder / "data"
    label_dir = output_folder / "label"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, label_dir


def process_npy_files(
    target_folder,
    background_folder,
    output_folder,
    seed=42,
    samples_per_target=5,
    split_name="",
    mapping_writer=None,
):

    target_folder = Path(target_folder)
    background_folder = Path(background_folder)
    data_dir, label_dir = make_output_dirs(output_folder)

    rng = random.Random(seed)
    target_files = list_npy_files(target_folder)
    background_files = list_npy_files(background_folder)

    if len(background_files) < samples_per_target:
        raise ValueError(
            f"Background folder {background_folder} contains only {len(background_files)} .npy files; "
            f"at least {samples_per_target} files are required"
        )

    generated_count = 0
    for target_file in target_files:
        target_path = target_folder / target_file
        data1 = np.load(target_path)
        data1 = data1 / np.max(data1) * (-1)
        data1_gau = gaussian_filter(data1, sigma=2)

        selected_files = rng.sample(background_files, samples_per_target)

        for selected_file in selected_files:
            background_path = background_folder / selected_file
            data2 = np.load(background_path)
            data2 = data2 / np.max(data2) * 0.3

            if data2.shape != data1.shape:
                print(f"Warning: file {selected_file} has a different shape from {target_file}; skipping")
                continue

            summed_data = data1_gau + data2
            summed_data = svd_fil(summed_data)

            output_file = f"{target_path.stem}_{background_path.stem}.npy"
            data_output_path = data_dir / output_file
            np.save(data_output_path, summed_data)

            # Preserve the original logic: use the SVD-filtered target data as the label.
            data1 = svd_fil(data1)
            label_output_path = label_dir / output_file
            np.save(label_output_path, data1)

            if mapping_writer is not None:
                mapping_writer.writerow(
                    [
                        split_name,
                        os.path.abspath(target_path),
                        os.path.abspath(background_path),
                        os.path.abspath(data_output_path),
                        os.path.abspath(label_output_path),
                    ]
                )

            generated_count += 1
            print(
                f"[{split_name}] Processed target {target_file}, "
                f"superimposed background {selected_file}, saved to {data_output_path}"
            )

    return generated_count


def process_dataset(
    target_root,
    background_root,
    output_folder,
    seed=42,
    samples_per_target=5,
):
    """
    Generate training and validation sets from pre-split target and background libraries.

    Input directory structure:
        target_root/train/*.npy
        target_root/val/*.npy
        background_root/train/*.npy
        background_root/val/*.npy

    Output directory structure:
        output_folder/train/data/*.npy
        output_folder/train/label/*.npy
        output_folder/val/data/*.npy
        output_folder/val/label/*.npy
    """
    target_root = Path(target_root)
    background_root = Path(background_root)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    mapping_path = output_folder / "file_mapping.csv"
    counts = {"train": 0, "val": 0}

    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Type", "Target File", "Background File", "Generated Data File", "Generated Label File"]
        )

        for split_name in ("train", "val"):
            counts[split_name] = process_npy_files(
                target_folder=target_root / split_name,
                background_folder=background_root / split_name,
                output_folder=output_folder / split_name,
                seed=seed if split_name == "train" else seed + 1000,
                samples_per_target=samples_per_target,
                split_name=split_name,
                mapping_writer=writer,
            )

    print(f"Generation complete. Training samples: {counts['train']}")
    print(f"Generation complete. Validation samples: {counts['val']}")
    print(f"File mapping has been saved to: {mapping_path}")
    return counts


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate a target-background superposition dataset from pre-split target and background libraries."
    )
    parser.add_argument("--target-root", default="target", help="Target library root directory with train/val subdirectories")
    parser.add_argument("--background-root", default="bk", help="Background library root directory with train/val subdirectories")
    parser.add_argument("--output-folder", default="data_mix", help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples-per-target", type=int, default=5, help="Number of backgrounds randomly superimposed per target")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    process_dataset(
        target_root=args.target_root,
        background_root=args.background_root,
        output_folder=args.output_folder,
        seed=args.seed,
        samples_per_target=args.samples_per_target,
    )
