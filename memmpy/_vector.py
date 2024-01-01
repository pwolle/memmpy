import numpy as np

import tempfile
import math
import os

import typeguard

from numpy.typing import DTypeLike
from typing import Self

from . import _labels


@typeguard.typechecked
class WriteVector:
    def __init__(
        self,
        path: str,
        name: str,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        check: str | None = None,
    ) -> None:
        self.path = path
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.check = check
        self.enter = False
        self.fixed = False

    def fix_shape_and_dtype(self, shape: tuple[int, ...], dtype: DTypeLike):
        if self.fixed:
            return

        self.shape = self.shape or shape
        self.dtype = self.dtype or dtype

        self._file = tempfile.NamedTemporaryFile()
        self._mmap = np.memmap(
            self._file,
            self.dtype,
            "w+",
            shape=(1, *self.shape),
        )

        self._index = 0
        self.fixed = True

    def __enter__(self: Self) -> Self:
        self.enter = True
        return self

    def extend(self: Self, value: np.ndarray) -> Self:
        self.fix_shape_and_dtype(value.shape[1:], value.dtype)
        assert self.enter, "WriteVector should be used as context manager."

        if value.dtype != self._mmap.dtype:
            error = f"Got dtype {value.dtype} != {self._mmap.dtype}."
            raise ValueError(error)

        if value.shape[1:] != self._mmap.shape[1:]:
            error = f"Got shape {value.shape[1:]} != {self._mmap.shape[1:]}."
            raise ValueError(error)

        length = self._index + value.shape[0]
        length = 2 ** math.ceil(math.log2(length))

        self._file = tempfile.NamedTemporaryFile()
        mmap = np.memmap(
            self._file,
            self.dtype,
            "w+",
            shape=(length, *self._mmap.shape[1:]),
        )

        mmap[: self._index] = self._mmap[: self._index]
        self._mmap = mmap

        self._mmap[self._index : self._index + value.shape[0]] = value
        self._mmap.flush()

        self._index += value.shape[0]
        return self

    def append(self: Self, value: np.ndarray) -> Self:
        return self.extend(value[None])

    def __exit__(self: Self, exc_type, *_) -> None:
        if exc_type is not None:
            return

        mmap_path = _labels.get_array_path(self.path, self.name)

        os.makedirs(self.path, exist_ok=True)
        mmap = np.memmap(
            mmap_path,
            self.dtype,
            "w+",
            shape=(self._index, *self._mmap.shape[1:]),
        )
        mmap[:] = self._mmap[: self._index]
        mmap.flush()

        _labels.save_array_metadata(
            self.path,
            self.name,
            shape=mmap.shape,
            dtype=mmap.dtype,
            check=self.check or _labels.get_timestamp(),
        )


@typeguard.typechecked
class WriteVectorDict:
    def __init__(self: Self, path: str, check=None):
        self.path = path
        self.check = check
        self.fixed = False

    def fix_keys(self, keys):
        if self.fixed:
            return

        self.vectors = {}
        for k in keys:
            self.vectors[k] = WriteVector(self.path, k, check=self.check)
            self.vectors[k].__enter__()

        self.fixed = True

    def __enter__(self: Self) -> Self:
        self.enter = True
        return self

    def expand(self: Self, data: dict[str, np.ndarray]) -> Self:
        if not self.fixed:
            self.fix_keys(data.keys())

        assert set(data.keys()) == set(self.vectors.keys())
        assert len(set(v.shape[0] for v in data.values())) == 1

        for key, value in data.items():
            self.vectors[key].extend(value)

        return self

    def __exit__(self: Self, *args) -> None:
        for vector in self.vectors.values():
            vector.__exit__(*args)


@typeguard.typechecked
def read_vector(path: str, name: str) -> np.memmap:
    mmap_path, dtype, shape = _labels.read_array_metadata(path, name)
    return np.memmap(mmap_path, dtype, "r", shape=shape)
