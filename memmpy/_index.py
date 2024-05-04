import numpy as np

from typing import Generator, Literal
import math


def batch_slices(
    size: int,
    batch_size: int,
    drop_remainder: bool = False,
) -> Generator[tuple[int, int], None, None]:
    for start in range(0, size, batch_size):
        stop = start + batch_size

        if stop >= size:
            if drop_remainder:
                break

            stop = size

        yield start, stop


def batch_indicies(
    size: int,
    batch_size: int,
    drop_remainder: bool = False,
) -> Generator[np.ndarray, None, None]:
    for start, stop in batch_slices(size, batch_size, drop_remainder):
        yield np.arange(start, stop)


def batch_indicies_split(
    size: int,
    batch_size: int,
    split_index: Literal["train", "valid", "test"] = "train",
    valid_part: int = 10,
    kfold_index: int = 0,
    drop_remainder: bool = False,
) -> Generator[np.ndarray, None, None]:
    split_int = {"train": 0, "valid": 1, "test": 2}[split_index]
    cuts = [
        0,
        math.ceil(size * (1 - 2 / valid_part)),
        math.ceil(size * (1 - 1 / valid_part)),
        size,
    ]
    start, stop = cuts[split_int], cuts[split_int + 1]
    kfold_size = size // valid_part

    for indicies in batch_indicies(stop - start, batch_size, drop_remainder):
        indicies = indicies + start
        indicies = indicies + kfold_size * kfold_index

        indicies = indicies % size
        yield indicies
