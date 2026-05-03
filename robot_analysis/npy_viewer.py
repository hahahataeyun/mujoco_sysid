#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a .npy file and its size.")
    parser.add_argument("--path", required=True, help="Path to the .npy file to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    array = np.load(path, allow_pickle=True)

    print(array)
    print(f"shape: {array.shape}")
    print(f"dtype: {array.dtype}")
    print(f"array bytes: {array.nbytes}")
    print(f"file bytes: {path.stat().st_size}")


if __name__ == "__main__":
    main()
