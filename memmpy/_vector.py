import math
import os
import tempfile
from typing import Self

import numpy as np
import typeguard
from numpy.typing import DTypeLike

from . import _labels


@typeguard.typechecked
class WriteVector:
    """
    Write numpy arrays into a memory mapped file. The vector increases in size
    as needed. The shape and dtypes are fixed on the first call to `append` or
    `extend`.
    `WriteVector` must be used as context manager, so that the file is flushed
    and closed properly. The data can be accessed via `read_vector`.

    Attributes
    ---
    path: str
        Path to the directory where the memory mapped file is stored.

    name: str
        Name of the data. This is used as key to retrieve the data via
        `read_vector`.

    shape: tuple[int, ...] | None
        Shape of the data. `None` if the shape is not yet fixed.

    dtype: DTypeLike | None
        Dtype of the data. `None` if the dtype is not yet fixed.

    check: str | None
        Checksum of the data. `None` if a timestamp at the time of saving will
        be used as checksum.

    enter: bool
        Whether the context manager was entered.

    fixed: bool
        Whether the shape and dtype are fixed.
    """

    path: str
    name: str
    shape: tuple[int, ...] | None
    dtype: DTypeLike | None
    check: str | None
    enter: bool
    fixed: bool

    def __init__(
        self: Self,
        path: str,
        name: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        check: str | None = None,
    ) -> None:
        """
        Initializes the context manager for writing a memory mapped vector.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This is used as key to retrieve the data via
            `read_vector`.

        shape: tuple[int, ...] | None, optional, default: None
            Shape of the data. If not given, the shape is inferred from the
            first call to `append` or `extend`.

        dtype: DTypeLike | None, optional, default: None
            Dtype of the data. If not given, the dtype is inferred from the
            first call to `append` or `extend`.

        check: str | None, optional, default: None
            Checksum of the data. If not given a timestamp at the time of
            saving is used as checksum. Usually this would be a hash of the
            source files, such that the data can be recreated if the source
            files change.

        Raises
        ---
        ValueError
            If the shape is not positive.

        TypeError
            If any arguments python type does not match the type hints.
        """

        if shape is not None and any(s <= 0 for s in shape):
            raise ValueError("Shape must be positive.")

        self.path = path
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.check = check
        self.enter = False
        self.fixed = False

    def _fix_shape_and_dtype(
        self: Self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> None:
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

    def extend(self: Self, value: np.ndarray) -> None:
        """
        Extend the memory mapped vector by `value`. The first axis of `value`
        is treated as the number of elements to append. All other axes must
        match the shape of the vector. The dtypes must also match.

        Parameters
        ---
        value: np.ndarray
            Array to append to the vector.

        Raises
        ---
        ValueError
            If the dtypes or shapes do not match.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self._fix_shape_and_dtype(value.shape[1:], value.dtype)
        assert self.enter, "WriteVector should be used as context manager."

        if value.dtype != self._mmap.dtype:
            error = f"Got dtype {value.dtype} != {self._mmap.dtype}."
            raise ValueError(error)

        if value.shape[1:] != self._mmap.shape[1:]:
            error = f"Got shape {value.shape[1:]} != {self._mmap.shape[1:]}."
            raise ValueError(error)

        length = self._index + value.shape[0]

        if length > self._mmap.shape[0]:
            length = 2 ** math.ceil(math.log2(length))
            print(f"Resizing to {length}.")

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

    def append(self: Self, value: np.ndarray) -> None:
        """
        Append `value` to the memory mapped vector. The shape of `value` must
        match all but the first axis of the vector. The dtypes must also match.

        Parameters
        ---
        value: np.ndarray
            Array to append to the vector.

        Raises
        ---
        ValueError
            If the dtypes or shapes do not match.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.extend(value[None])

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
    """
    Write a dictionary of numpy arrays into memory mapped files. Works similar
    to `WriteVector`, but the arrays also have keys, similar to python dicts.
    Furthermore the number of appended elements must be the same for all keys.

    Attributes
    ---
    path: str
        Path to the directory where the memory mapped file is stored.

    check: str | None
        Checksum of the data. `None` if a timestamp at the time of saving will
        be used as checksum.

    fixed: bool
        Whether the keys are fixed.

    enter: bool
        Whether the context manager was entered.
    """

    path: str
    check: str | None
    fixed: bool
    enter: bool
    vectors: dict[str, WriteVector]

    def __init__(self: Self, path: str, check=None) -> None:
        """
        Initializes the context manager for writing a dictionary of memory
        mapped vectors.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        check: str | None, optional, default: None
            Checksum of the data. If not given a timestamp at the time of
            saving is used as checksum. Usually this would be a hash of the
            source files, such that the data can be recreated if the source
            files change.
            The `check` will be passed on to the attribute `vectors`.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.path = path
        self.check = check
        self.fixed = False

    def _fix_keys(self, keys):
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
        """
        Append the data to the memory mapped vectors. The keys, dtypes and all
        but the first axes of the arrays must be the same on all calls to
        `expand`. The first call fixes the keys dtypes and shapes.

        Parameters
        ---
        data: dict[str, np.ndarray]
            Data to append to the vectors.

        Raises
        ---
        ValueError
            If the keys, dtypes or shapes do not match to the previous calls to
            `expand`.

        ValueError
            If the length of the first axis of the data is not the same for all
            keys.

        TypeError
            If any arguments python type does not match the type hints.
        """
        if not self.fixed:
            self._fix_keys(data.keys())

        if set(data.keys()) != set(self.vectors.keys()):
            error = (
                "The keys of the data must match the keys of the vectors. "
                f"Got keys {set(data.keys())}, which do not match the fixed "
                f"keys {set(self.vectors.keys())}."
            )
            raise ValueError(error)

        lengths = {k: v.shape[0] for k, v in data.items()}
        if len(set(lengths.values())) != 1:
            error = (
                "The length of the first axis of the data must be the same for"
                f" all keys. Got different lengths {lengths}."
            )
            raise ValueError(error)

        for key, value in data.items():
            self.vectors[key].extend(value)

        return self

    def __exit__(self: Self, *args) -> None:
        for vector in self.vectors.values():
            vector.__exit__(*args)


@typeguard.typechecked
def read_vector(path: str, name: str) -> np.memmap:
    """
    Read a memory mapped vector from a memmpy directory.

    Parameters
    ---
    path: str
        Path to the directory where the memory mapped file is stored.

    name: str
        Name of the data. This name was used when writing the data via
        `WriteVector`.

    Returns
    ---
    mmap: np.memmap
        Memory mapped array. The dtype and shape are read from the metadata,
        which was written when the data was saved. The array is read only.

    Raises
    ---
    KeyError
        If the name is not found in the metadata.

    FileNotFoundError
        If the file for the name is not found. This could be because the data
        was not saved properly, or the user deleted/moved/renamed the underlying
        file.

    TypeError
        If any arguments python type does not match the type hints.
    """
    mmap_path, dtype, shape = _labels.read_array_metadata(path, name)
    mmap_path = os.path.abspath(mmap_path)

    if not os.path.exists(mmap_path):
        error = (
            f"The file for '{name}' at '{path}' was not found, perhaps it was "
            f"there was an error when saving. The full path is {mmap_path}."
        )
        raise FileNotFoundError(error)

    return np.memmap(mmap_path, dtype, "r", shape=shape)


@typeguard.typechecked
def read_vectors(path: str, keys: set[str] | None = None) -> dict[str, np.memmap]:
    """
    Read a dictionary of memory mapped vectors from a memmpy directory.

    Parameters
    ---
    path: str
        Path to the memmpy directory.

    keys : set[str], optional
        Keys to load. If None, all keys are loaded.

    Raises
    ---
    ValueError
        If the path is not found or not a directory.

    KeyError
        If a requested key is not found in the metadata.

    FileNotFoundError
        If the file for the name is not found. This could be because the data
        was not saved properly, or the user deleted/moved/renamed the underlying
        file.

    TypeError
        If any arguments python type does not match the type hints.
    """

    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory.")

    meta = _labels.safe_load(path)
    keys = keys or set(meta["arrays"].keys())
    data = {}

    for k in keys:
        data[k] = read_vector(path, k)

    return data
