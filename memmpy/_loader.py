import math
import os
from typing import Any, Callable, Generator, Hashable, Self

import numpy as np
import typeguard

from . import _jagged, _labels, _vector


@typeguard.typechecked
def load_memmaps(
    path: str,
    keys: set[str] | None = None,
) -> dict[str, np.memmap | _jagged.ReadJagged | _jagged.ReadShaped]:
    """
    Load memmaps from a memmpy directory.

    Parameters
    ---
    path : str
        Path to the memmpy directory.

    keys : set[str], optional
        Keys to load. If None, all keys are loaded.

    Returns
    ---
    dict[str, np.memmap | _jagged.ReadJagged | _jagged.ReadShaped]
        The loaded data. Has the same keys as requested.

    Raises
    ---
    ValueError
        If the path is not found or not a directory.

    KeyError
        If a requested key is not found in the metadata.
    """
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory.")

    meta = _labels.safe_load(path)
    keys = keys or meta["arrays"].keys()
    data = {}

    for k in keys:
        if _labels.get_index_name(k) in meta["arrays"]:
            data[k] = _jagged.ReadJagged(path, k)
            continue

        if _labels.get_shape_name(k) in meta["arrays"]:
            data[k] = _jagged.ReadShaped(path, k)
            continue

        if _labels.get_structured_name(k) in meta["arrays"]:
            # Skipping structured data.
            # This is because we do not know how to reconstruct it here.
            continue

        if k not in meta["arrays"]:
            raise KeyError(f"Key '{k}' not found in metadata.")

        data[k] = _vector.read_vector(path, k)

    return data


@typeguard.typechecked
def permutation_bands(length: int) -> Callable[[np.ndarray], np.ndarray]:
    bands_total = int(math.log2(length) ** 2)
    band_offsets = np.random.uniform(size=bands_total)
    band_offsets = (band_offsets * length).astype(np.int64)

    @typeguard.typechecked
    def permutation_func(indicies: np.ndarray) -> np.ndarray:
        indicies_offsets = indicies // bands_total
        indicies_remainders = indicies % bands_total

        indicies_offsets += band_offsets[indicies_remainders]
        indicies_offsets %= length // bands_total

        return indicies_offsets * bands_total + indicies_remainders

    return permutation_func


@typeguard.typechecked
def permutation_modulus(
    length: int,
    modulus: int = int(2**17 - 1),
) -> Callable[[np.ndarray], np.ndarray]:
    modulus = min(int(modulus), length)
    subpermutation = np.random.uniform(size=modulus).argsort()

    perm = np.random.uniform(size=modulus)
    perm = np.argsort(perm)

    @typeguard.typechecked
    def permute(indicies: np.ndarray) -> np.ndarray:
        indicies_offsets = indicies // modulus
        indicies_remainders = indicies % modulus

        indicies_remainders = subpermutation[indicies_remainders]

        result = indicies_offsets * modulus + indicies_remainders
        return result % length

    return permute


@typeguard.typechecked
def permutation_combination(length: int) -> Callable[[np.ndarray], np.ndarray]:
    perm1 = permutation_modulus(length)
    perm2 = permutation_bands(length)

    @typeguard.typechecked
    def perm(x: np.ndarray) -> np.ndarray:
        x = perm1(x)
        x = perm2(x)
        return x

    return perm


@typeguard.typechecked
def generate_batch_indicies(
    length: int,
    batch_size: int,
    drop_remainder: bool = True,
) -> Generator[np.ndarray, None, None]:
    for i in range(0, length - batch_size + 1, batch_size):
        yield np.arange(i, i + batch_size, dtype=np.int64)

    if not drop_remainder:
        remainder = length % batch_size
        yield np.arange(length - remainder, length, dtype=np.int64)


def _safe_take_set(s) -> Any:
    s = set(s)

    if len(s) != 1:
        raise ValueError("Set does not have size 1.")

    return next(iter(s))


@typeguard.typechecked
class SimpleLoader:
    """
    Iterates over a dictionary of numpy arrays or memmaps in batches.

    Attributes
    ---
    data : dict[Hashable, np.ndarray | np.memmap]
        The data to iterate over.

    batch_size : int
        The batch size.
    """

    data: dict[Hashable, np.ndarray | np.memmap]
    batch_size: int

    def __init__(
        self: Self,
        data: dict[Hashable, np.ndarray | np.memmap],
        batch_size: int = 32,
    ) -> None:
        """
        Initialize an itereable class which can generate batches from a
        dictionary of numpy arrays or memmaps.

        Parameters
        ---
        data : dict[Hashable, np.ndarray | np.memmap]
            The data to iterate over.

        batch_size : int, optional, default: 32
            The batch size.

        Raises
        ---
        ValueError
            If the data is empty or if the length is not the same for all
            arrays.

        ValueError
            If the batch size is not positive.

        ValueError
            If the batch size is larger than the data length.

        KeyError
            If the key '_index' is present in the data. This key is used
            internally already.

        TypeError
            If the given arguments do not match the type annotations.
        """
        if len(data) == 0:
            raise ValueError("Data is empty.")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if "_index" in data:
            raise KeyError("Key '_index' is already used internally.")

        self.data = data
        self.data_len: int = _safe_take_set(len(v) for v in self.data.values())

        if batch_size > self.data_len:
            raise ValueError("Batch size is larger than the data length.")

        self.batch_size = batch_size

    def __iter__(
        self: Self,
    ) -> Generator[dict[Hashable, np.ndarray | np.memmap], None, None]:
        """
        Iterate over the data in batches. The last batch may be smaller than
        the batch size.

        Yields
        ---
        dict[Hashable, np.ndarray | np.memmap]
            The next batch. The keys are the same as the data given to the
            constructor. A new key '_index' is added which contains the
            indicies of the batch.

        Raises
        ---
        TypeError
            If the given arguments do not match the type annotations.
        """
        data_len_rounded = self.data_len - self.batch_size + 1

        for i in range(0, data_len_rounded, self.batch_size):
            indicies = np.arange(i, i + self.batch_size, dtype=np.int64)
            batch = {k: v[indicies] for k, v in self.data.items()}
            yield batch | {"_index": indicies}

        remainder = self.data_len % self.batch_size
        if remainder != 0:
            indicies = np.arange(
                self.data_len - remainder,
                self.data_len,
                dtype=np.int64,
            )
            batch = {k: v[indicies] for k, v in self.data.items()}
            yield batch | {"_index": indicies}

    def __len__(self: Self) -> int:
        """
        Get the length of the iterator.

        Returns
        ---
        int
            The length of the iterator.
        """
        result = self.data_len // self.batch_size

        if self.data_len % self.batch_size != 0:
            result += 1

        return result


class FixedSeed:
    def __init__(self, seed: int):
        self.seed = seed

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.set_state(self.state)


@typeguard.typechecked
class SplitLoader:
    """
    Loads batches from a dictionary of numpy arrays or memmaps. This loader is
    designed to be used in machine learning training loops. It supports
    shuffling, splitting the data and subindexing, i.e. only loading a subset
    of the data.

    Attributes
    ---
    data : dict[Hashable, np.ndarray | np.memmap]
        The data to iterate over.

    batch_size : int
        The batch size.

    split : int
        The split to use. This is an index into the fractions tuple.

    fractions : tuple[float, ...]
        The fractions of the data to use for training, validation and testing.
        The fractions must sum to 1.

    subindex : np.memmap | None
        A memmap containing the indicies to use. If None, all indicies are
        used.

    split_seed : int
        The seed to use for splitting the data.

    shuffle : bool
        Whether to shuffle the data before splitting.

    data_len : int
        The length of the data.

    split_length : int
        The length of the split.

    split_start : int
        The start index of the split in the split permutation
    """

    data: dict[Hashable, np.ndarray | np.memmap]
    batch_size: int
    split: int
    fractions: tuple[float, ...]
    subindex: np.memmap | np.ndarray | None
    split_seed: int
    shuffle: bool
    data_len: int
    split_length: int
    split_start: int

    # TODO if batchsize is larger than length, repeat data

    def __init__(
        self: Self,
        data: dict[Hashable, np.ndarray | np.memmap],
        batch_size: int = 32,
        *,
        split: int = 0,
        fractions: tuple[float, ...] = (0.8, 0.1, 0.1),
        subindex: np.memmap | np.ndarray | None = None,
        split_seed: int = 0,
        shuffle=True,
    ) -> None:
        """
        Initialize an itereable class which can generate batches from a
        dictionary of numpy arrays or memmaps for machine learning training
        loops.

        Parameters
        ---
        data : dict[Hashable, np.ndarray | np.memmap]
            The data to iterate over.

        batch_size : int, optional, default: 32
            The batch size.

        split : int, optional, default: 0
            The split to use. This is an index into the fractions tuple.

        fractions : tuple[float, ...], optional, default: (0.8, 0.1, 0.1)
            The fractions of the data to use for training, validation and
            testing. The fractions must sum to 1.

        subindex : np.memmap | np.ndarray | None, optional, default: None
            A memmap containing the indicies to use. If None, all indicies are
            used.

        split_seed : int, optional, default: 0
            The seed to use for splitting the data.

        shuffle : bool, optional, default: True
            Whether to shuffle the data before splitting.

        Raises
        ---
        ValueError
            If the fractions do not sum to 1, or the batch size is not positive.

        TypeError
            If the given arguments do not match the type annotations.
        """

        if abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("Fractions must sum to 1.")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        self.data = data
        self.data_len: int = _safe_take_set(len(v) for v in self.data.values())

        if batch_size > self.data_len:
            factor = math.ceil(batch_size / self.data_len)
            for k, v in self.data.items():
                self.data[k] = np.tile(v, factor)

            self.data_len *= factor

            # this is not optimal for memeory usage but the simplest
            # implementation, it would be better to not tile the data
            # if the subindes is used anyway
            if subindex is not None:
                subindex = np.tile(subindex, factor)

        self.batch_size = batch_size
        self.split = split
        self.fractions = fractions
        self.subindex = subindex
        self.shuffle = shuffle

        if self.subindex is not None:
            self.data_len = len(self.subindex)

        with FixedSeed(split_seed):
            self.split_perm = permutation_combination(self.data_len)

        self.split_length = int(self.data_len * fractions[split])
        self.split_start = int(
            sum(self.data_len * f for f in fractions[:split]),
        )

    def __iter__(
        self: Self,
    ) -> Generator[dict[Hashable, np.ndarray | np.memmap], None, None]:
        """
        Iterate over the data in batches. If the last batch would be smaller
        than the batch size, it is dropped.

        Yields
        ---
        dict[Hashable, np.ndarray | np.memmap]
            The next batch. The keys are the same as the data given to the
            constructor. A new key '_index' is added which contains the
            indicies of the batch.
        """
        data_len_rounded = self.split_length - self.batch_size + 1

        if self.shuffle:
            perm_shuffle = permutation_combination(self.split_length)
        else:
            perm_shuffle = lambda x: x

        for i in range(0, data_len_rounded, self.batch_size):
            indicies = np.arange(i, i + self.batch_size, dtype=np.int64)
            indicies = self.split_perm(indicies + self.split_start)
            indicies = perm_shuffle(indicies)

            if self.subindex is not None:
                indicies = self.subindex[indicies]

            batch = {k: v[indicies] for k, v in self.data.items()}
            yield batch | {"_index": indicies}

    def __len__(self) -> int:
        """
        Get the length of the iterator.

        Returns
        ---
        int
            The length of the iterator.
        """
        return self.split_length // self.batch_size
