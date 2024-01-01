import math
from typing import Any, Callable, Generator, Self

import numpy as np
import typeguard

from . import _jagged, _labels, _vector


@typeguard.typechecked
def load_memmaps(
    path: str,
    keys: set[str] | None = None,
) -> dict[str, np.memmap | _jagged.ReadJagged | _jagged.ReadShaped]:
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
    def __init__(
        self: Self,
        data: dict,
        batch_size: int = 32,
    ) -> None:
        assert "_index" not in data, "Key '_index' is already used internally."
        self.data = data
        self.data_len: int = _safe_take_set(len(v) for v in self.data.values())
        self.batch_size = batch_size

    def __iter__(self: Self) -> Generator[dict[str, np.ndarray], None, None]:
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
    def __init__(
        self,
        data,
        batch_size: int = 32,
        *,
        split: int = 0,
        fractions: tuple[float, ...] = (0.8, 0.1, 0.1),
        subindex: np.memmap | None = None,
        split_seed: int = 0,
        shuffle=True,
    ) -> None:
        self.data = data
        self.data_len: int = _safe_take_set(len(v) for v in self.data.values())
        self.batch_size = batch_size

        assert abs(sum(fractions) - 1.0) < 1e-6, "Fractions must sum to 1."
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
    ) -> Generator[dict[str, np.ndarray | np.memmap], None, None]:
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
        return self.split_length // self.batch_size
