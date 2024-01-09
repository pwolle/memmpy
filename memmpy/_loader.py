import collections.abc
import math
from typing import (
    Any,
    Callable,
    Hashable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Self,
    overload,
    runtime_checkable,
)

import numpy as np
import typeguard

__all__ = [
    "Dict",
    "unwrap",
    "Sliced",
    "Indexed",
    "Shuffled",
    "Batched",
    "split",
]

# TODO implement kfold, and mapping, also concat?


@runtime_checkable
class NumpySequence(Protocol):
    """
    Similar to collections.abc.Sequence, but also allows for indexing with
    numpy arrays, to get multiple elements at once.
    """

    @overload
    def __getitem__(self: Self, index: int) -> Any:
        ...

    @overload
    def __getitem__(self: Self, index: slice) -> "NumpySequence":
        ...

    @overload
    def __getitem__(self: Self, index: np.ndarray) -> "NumpySequence":
        ...

    def __getitem__(self: Self, index: int | slice | np.ndarray) -> Any:
        ...

    def __iter__(self: Self) -> Iterator:
        ...

    def __len__(self: Self) -> int:
        ...


def unwrap(x):
    """
    Try to use the `_unwrap` method of an object recursively, if it exists,
    otherwise return the object itself.
    This is usefull for getting the underlying data out of lazy sequence
    manipulations.
    """
    if hasattr(x, "_unwrap"):
        return unwrap(x._unwrap())

    return x


@typeguard.typechecked
class Dict(collections.abc.Sequence):
    """
    Comnine multiple sequences into a single sequence using a dictionary.
    """

    def __init__(
        self: Self,
        _sequences: dict,
    ) -> None:
        self._sequences = _sequences

        if len(set(map(len, _sequences.values()))) != 1:
            raise ValueError("All sequences must have the same length.")

        self._length = len(next(iter(_sequences.values())))

    @overload
    def __getitem__(self: Self, index: int) -> dict[Hashable, Any]:
        ...

    @overload
    def __getitem__(self: Self, index: slice) -> "NumpySequence":
        ...

    @overload
    def __getitem__(self: Self, index: np.ndarray) -> "NumpySequence":
        ...

    def __getitem__(
        self: Self,
        index: int | slice | np.ndarray,
    ) -> Any:
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of bounds.")

            return {k: v[index] for k, v in self._sequences.items()}

        indexed = {}
        for k, v in self._sequences.items():
            indexed[k] = v[index]

        return Dict(indexed)

    def __len__(self: Self) -> int:
        return self._length

    def _unwrap(
        self: Self,
    ) -> Mapping[Hashable, NumpySequence | np.memmap | np.ndarray]:
        return self._sequences


@typeguard.typechecked
class _Sliced(collections.abc.Sequence):
    """
    Lalazily slice of a sequence by wrapping the __getitem__ method.
    """

    def __init__(
        self,
        _sequence: NumpySequence | np.memmap | np.ndarray,
        _slice: slice,
        /,
    ) -> None:
        self._sequence = _sequence

        self._start = _slice.start or 0
        self._stop = _slice.stop or len(_sequence)
        self._step = _slice.step or 1

        if self._start < 0:
            self._start += len(_sequence)

        if self._stop < 0:
            self._stop += len(_sequence)

        # check if start or stop are out of bounds
        if not 0 <= self._start < len(_sequence):
            raise IndexError(f"Start index {self._start} out of bounds.")

        if not 0 <= self._stop <= len(_sequence):
            raise IndexError(f"Stop index {self._stop} out of bounds.")

    def __getitem__(self: Self, index: int | slice | np.ndarray) -> Any:
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of bounds.")

            return self._sequence[self._start + self._step * index]

        if isinstance(index, slice):
            return Sliced(self, index)

        if isinstance(index, np.ndarray):
            index[index < 0] += len(self)

            if any(index < 0) or any(index >= len(self)):
                raise IndexError(f"Index {index} out of bounds.")

            return self._sequence[self._start + self._step * index]

        raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self: Self) -> int:
        return (self._stop - self._start + self._step - 1) // self._step

    def _unwrap(self: Self) -> NumpySequence | np.memmap | np.ndarray:
        index = np.arange(len(self))
        return self[index]


def Sliced(
    x: NumpySequence | np.memmap | np.ndarray,
    s: slice,
) -> _Sliced | np.memmap | np.ndarray:
    """
    Slice a sequence. If it is a numpy array, the slicing is done directly,
    otherwise a lazy wrapper is returned.
    """
    if isinstance(x, (np.ndarray, np.memmap, list, tuple, range)):
        return x[s]

    return _Sliced(x, s)


@typeguard.typechecked
class _Indexed(collections.abc.Sequence):
    """
    Restrict a sequence to a subset of indicies.
    """

    def __init__(
        self: Self,
        _sequence: NumpySequence | np.memmap | np.ndarray,
        _subindex: np.ndarray | np.memmap,  # TODO also a NumpySequence (?)
    ) -> None:
        if _subindex.ndim != 1:
            raise ValueError("Subindex must have ndim=1.")

        if not np.issubdtype(_subindex.dtype, np.integer):
            raise ValueError("Subindex must have integer dtype.")

        # do not test whether subindex is in bounds, the sequence and
        # subindex might be very large and only small parts of this index
        # might be used, so the wait might be much longer than the actual
        # time this index would be used for
        self._sequence = _sequence
        self._subindex = _subindex.astype(np.intp)

    def __getitem__(self, index: int | slice | np.ndarray) -> Any:
        if isinstance(index, int):
            return self._sequence[self._subindex[index]]

        if isinstance(index, slice):
            return Indexed(self._sequence, self._subindex[index])

        if isinstance(index, np.ndarray):
            # print(index)
            return self._sequence[self._subindex[index]]

        raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        return len(self._subindex)

    def _unwrap(self: Self) -> NumpySequence | np.memmap | np.ndarray:
        print(self._sequence)
        return self._sequence[self._subindex]


def Indexed(
    x: NumpySequence | np.memmap | np.ndarray,
    i: np.ndarray | np.memmap,
) -> _Indexed | np.memmap | np.ndarray:
    """
    Restrict a sequence to a subset of indicies. If it is a numpy array, the
    indexing is done directly, otherwise a lazy wrapper is returned.
    """
    if isinstance(x, (np.ndarray, np.memmap)):
        return x[i]

    return _Indexed(x, i)


@typeguard.typechecked
def permutation_bands(length: int) -> Callable[[np.ndarray], np.ndarray]:
    bands_total = int(math.log2(length) ** 2)
    bands_total = min(bands_total, length // 2)

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
def permutation_blocks(
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


class FixedSeed:
    def __init__(self, seed: int | None) -> None:
        self._seed = seed

    def __enter__(self: Self) -> None:
        if self._seed is None:
            return

        self.state = np.random.get_state()
        np.random.seed(self._seed)

    def __exit__(self: Self, *_) -> None:
        if self._seed is None:
            return

        np.random.set_state(self.state)


@typeguard.typechecked
def permutation_fast(length: int) -> Callable[[np.ndarray], np.ndarray]:
    perm1 = permutation_blocks(length)
    perm2 = permutation_bands(length)

    @typeguard.typechecked
    def perm(x: np.ndarray) -> np.ndarray:
        x = perm1(x)
        x = perm2(x)
        return x

    return perm


@typeguard.typechecked
class Shuffled(collections.abc.Sequence):
    """
    Lazily shuffle a sequence using a fast permutation function.
    """

    def __init__(
        self,
        sequence: NumpySequence | np.memmap | np.ndarray,
        /,
        *,
        seed: int | None = None,
    ) -> None:
        self._sequence = sequence

        with FixedSeed(seed):
            self._permutation = permutation_fast(len(sequence))

    def __getitem__(self, index: int | slice | np.ndarray) -> Any:
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of bounds.")

            index_arr = np.array([index])
            index_int = self._permutation(index_arr)[0]
            return self._sequence[index_int]

        if isinstance(index, slice):
            return Sliced(self, index)

        if isinstance(index, np.ndarray):
            index[index < 0] += len(self)

            if any(index < 0) or any(index >= len(self)):
                raise IndexError(f"Index {index} out of bounds.")

            return self._sequence[self._permutation(index)]

        raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        return len(self._sequence)

    def _unwrap(self) -> NumpySequence | np.memmap | np.ndarray:
        index = np.arange(len(self))
        return self[index]


@typeguard.typechecked
class Batched(collections.abc.Sequence):
    """
    Lazily collect a sequence into batches.
    """

    def __init__(
        self,
        _sequence: NumpySequence | np.memmap | np.ndarray,
        /,
        batch_size: int,
        drop_remainder: bool = False,
    ) -> None:
        self._sequence = _sequence
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder

    def __getitem__(self, index: int | slice | np.ndarray) -> Any:
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of bounds.")

            start = index * self._batch_size
            stop = (index + 1) * self._batch_size

            if not self._drop_remainder:
                stop = min(stop, len(self._sequence))

            return self._sequence[start:stop]

        if isinstance(index, slice):
            return Sliced(self, index)

        if isinstance(index, np.ndarray):
            index[index < 0] += len(self)

            if any(index < 0) or any(index >= len(self)):
                raise IndexError(f"Index {index} out of bounds.")

            result = []

            for i in index:
                start = i * self._batch_size
                stop = (i + 1) * self._batch_size

                if not self._drop_remainder:
                    stop = min(stop, len(self._sequence))

                result.append(self._sequence[start:stop])

            return result

        raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        if self._drop_remainder:
            return len(self._sequence) // self._batch_size

        return (len(self._sequence) + self._batch_size - 1) // self._batch_size

    def _unwrap(self) -> NumpySequence | np.memmap | np.ndarray:
        index = np.arange(len(self))
        return self[index]


@typeguard.typechecked
def split(
    sequence: NumpySequence,
    split_index: int | Literal["train", "valid", "test"] = "train",
    split_fracs: tuple[float, ...] = (0.8, 0.1, 0.1),
    shuffle: bool = True,
    seed: int | None = 42,
) -> _Sliced | np.memmap | np.ndarray:
    """
    Get a slice of a sequence according to a split fraction. This is useful
    for splitting a dataset into training, validation and test set.
    """
    if abs(sum(split_fracs) - 1) > 1e-12:
        raise ValueError("Split fractions must sum to 1.")

    if isinstance(split_index, str):
        split_index = {"train": 0, "valid": 1, "test": 2}[split_index]

    if split_index >= len(split_fracs):
        raise ValueError("Split index out of bounds.")

    if shuffle:
        sequence = Shuffled(sequence, seed=seed)

    start = int(sum(len(sequence) * f for f in split_fracs[:split_index]))
    length = int(len(sequence) * split_fracs[split_index])
    return Sliced(sequence, slice(start, start + length))


class Map(collections.abc.Sequence):
    def __init__(
        self,
        _sequence: NumpySequence | np.memmap | np.ndarray,
        _func: Callable[[Any], Any],
    ) -> None:
        self._sequence = _sequence
        self._func = _func

    def __getitem__(self, index: int | slice | np.ndarray) -> Any:
        if isinstance(index, int):
            if index < 0:
                index += len(self)

            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of bounds.")

            return self._func(self._sequence[index])

        if isinstance(index, slice):
            return Sliced(self, index)

        if isinstance(index, np.ndarray):
            index[index < 0] += len(self)

            if any(index < 0) or any(index >= len(self)):
                raise IndexError(f"Index {index} out of bounds.")

            return self._func(self._sequence[index])

        raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self) -> int:
        return len(self._sequence)

    def _unwrap(self) -> NumpySequence | np.memmap | np.ndarray:
        index = np.arange(len(self))
        return self[index]
