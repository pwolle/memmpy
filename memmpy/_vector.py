import math
import os
import tempfile
from typing import Self

import numpy as np
from numpy.typing import DTypeLike


__all__ = ["Vector"]


class Vector:
    mmap: None | np.memmap

    def __init__(self: Self) -> None:
        self.mmap = None
        self._len = 0

    def __len__(self: Self) -> int:
        return self._len if self.mmap is not None else 0

    def _init_mmap(self: Self, shape: tuple[int, ...], dtype: DTypeLike) -> None:
        self.file = tempfile.NamedTemporaryFile(dir=".")
        self.mmap = np.memmap(
            self.file,
            dtype,
            "w+",
            shape=(1, *shape),
        )

    def extend(self: Self, value: np.ndarray) -> None:
        if self.mmap is None:
            self._init_mmap(value.shape[1:], value.dtype)

        assert self.mmap is not None, "Memory-mapped array is not initialized"

        if value.dtype != self.mmap.dtype:
            error = f"Got dtype {value.dtype} != {self.mmap.dtype}."
            raise ValueError(error)

        if value.shape[1:] != self.mmap.shape[1:]:
            error = f"Got shape {value.shape[1:]} != {self.mmap.shape[1:]}."
            raise ValueError(error)

        length = self._len + value.shape[0]
        length = 2 ** math.ceil(math.log2(length))

        if length > self.mmap.shape[0]:
            file = tempfile.NamedTemporaryFile(dir=".")
            mmap = np.memmap(
                file,
                self.mmap.dtype,
                "w+",
                shape=(length, *self.mmap.shape[1:]),
            )
            mmap[: self._len] = self.mmap[: self._len]
            mmap.flush()

            self.file.close()
            self.file = file
            self.mmap = mmap

        self.mmap[self._len : self._len + value.shape[0]] = value
        self._len += value.shape[0]

        self.mmap.flush()

    def append(self: Self, value: np.ndarray) -> None:
        self.extend(value[np.newaxis])

    def save(self: Self, path: str) -> None:
        if self.mmap is None:
            raise ValueError("No data to save.")

        self.mmap.flush()

        try:
            mmap = np.memmap(
                path,
                self.mmap.dtype,
                "w+",
                shape=(self._len, *self.mmap.shape[1:]),
            )
            mmap[:] = self.mmap[: self._len]
            mmap.flush()

        # if an error occurs, remove the file
        except Exception as e:
            os.remove(path)
            raise e
