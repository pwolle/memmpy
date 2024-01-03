from typing import Any, Callable, Generator, Self, overload

import numpy as np
import typeguard
from numpy.typing import DTypeLike

from . import _labels, _vector


@typeguard.typechecked
class WriteJagged:
    """
    Write numpy arrays with one variable length axis into a memory mapped file.
    The underlying file is a vector of the concatenated arrays. The index of
    the start of each array is stored in a second vector. This allows for fast
    random access of the arrays.
    The resulting memory mapped array is called jagged because along the first
    axis the arrays can have different lengths.

    Attributes
    ---
    vector: _vector.WriteVector
        The vector of the concatenated arrays.

    _index: _vector.WriteVector
        The vector of the start indices of the arrays.
    """

    vector: _vector.WriteVector
    _index: _vector.WriteVector

    def __init__(
        self: Self,
        path: str,
        name: str,
        *,
        dtype: DTypeLike | None = None,
        shape: tuple[int, ...] | None = None,
        check: str | None = None,
    ) -> None:
        """
        Initializes the context manager for writing a arrays with one variable
        length axis into a memory mapped file.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This is used as key to retrieve the data via
            `ReadJagged`.

        shape: tuple[int, ...] | None, optional, default: None
            Shape of the data. `None` if the shape is not yet fixed.

        dtype: DTypeLike | None, optional, default: None
            Dtype of the data. `None` if the dtype is not yet fixed.

        check: str | None, optional, default: None
            Checksum of the data. `None` if a timestamp at the time of saving will
            be used as checksum.
        """
        self.vector = _vector.WriteVector(
            path,
            name,
            shape=shape,
            dtype=dtype,
            check=check,
        )
        self._index = _vector.WriteVector(
            path,
            _labels.get_index_name(name),
            shape=(),
            dtype=np.int64,
            check=check,
        )

    def __enter__(self: Self) -> Self:
        self.vector.__enter__()
        self._index.__enter__()
        self._index.append(np.array(0, dtype=np.int64))
        return self

    def __exit__(self: Self, *args) -> None:
        self.vector.__exit__(*args)
        self._index.__exit__(*args)

    def append(self: Self, value: np.ndarray) -> None:
        """
        Append a new array to the memory mapped file. The first axis of the
        given array is treated as the variable length axis. All other axes are
        fixed by the first call of this method.

        Parameters
        ---
        value: np.ndarray
            The array to append.

        Raises
        ---
        ValueError
            If dtype or all but the first axis of the given array do not match
            the previously appended arrays.
        """
        self.vector.extend(value)
        self._index.append(np.array(self.vector._index, dtype=np.int64))


@typeguard.typechecked
class ReadJagged:
    """
    Read a memory mapped array of arrays with one variable length axis, i.e. a
    memory mapped, jagged array. This array can be indexed or iterated over.

    Attributes
    ---
    vector: np.memmap
        The memory mapped array of the concatenated arrays.

    _index: np.memmap
        The memory mapped array of the start indices of the arrays.
    """

    vector: np.memmap
    _index: np.memmap

    def __init__(self: Self, path: str, name: str) -> None:
        """
        Loads a memory mapped array of arrays with one variable length axis,
        i.e. a memory mapped, jagged array.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This name was used as a key to store the data via
            the `WriteJagged` context manager.

        Raises
        ---
        KeyError
            If the name is not found in the metadata.

        FileNotFoundError
            If the file for the name is not found. This could be because the
            data was not saved properly, or the user deleted/moved/renamed the
            underlying file.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.vector = _vector.read_vector(path, name)
        self._index = _vector.read_vector(path, _labels.get_index_name(name))

    @overload
    def __getitem__(self: Self, index: int) -> np.memmap:
        ...

    @overload
    def __getitem__(self: Self, index: np.ndarray) -> list[np.memmap]:
        ...

    def __getitem__(
        self: Self,
        index: int | np.ndarray,
    ) -> np.memmap | list[np.memmap]:
        """
        Index the jagged array. Indexing with an integer `i` returns the `i`-th
        array. Indexing with an array of integers returns a list of the
        corresponding arrays.

        Parameters
        ---
        index: int | np.ndarray
            The index or indices to retrieve.

        Returns
        ---
        np.memmap | list[np.memmap]
            The retrieved array or list of arrays.

        Raises
        ---
        IndexError
            If the index is out of bounds.

        ValueError
            If the index is not valid, i.e. not positive or one-dimensional.

        TypeError
            If any arguments python type does not match the type hints.
        """
        if isinstance(index, int):
            index = np.array([index])
            return self[index][0]

        assert isinstance(index, np.ndarray)

        if index.ndim == 0:
            error = "Index must be one dimensional."
            raise ValueError(error)

        if index.dtype == bool:
            index = np.where(index)[0]

        if np.any(index < 0):
            error = "Index must be positive."
            raise ValueError(error)

        if index.max() >= len(self._index) - 1:
            error = "Index out of bounds."
            raise IndexError(error)

        indicies = zip(self._index[index], self._index[index + 1])

        # preallocate result, this should be a bit faster
        result = [None] * len(index)

        for i, (s, e) in enumerate(indicies):
            result[i] = self.vector[s:e]  # type: ignore

        return result  # type: ignore

    def __len__(self) -> int:
        """
        The length of the jagged array, i.e. how often the `append` method was
        called on the corresponding `WriteJagged` context manager.

        Returns
        ---
        int
            The number of subarrays. in the jagged array.
        """
        return len(self._index) - 1

    def __iter__(self) -> Generator[np.memmap, None, None]:
        """
        Iterate over the jagged array, i.e. iterate over the subarrays.

        Returns
        ---
        Generator[np.memmap, None, None]
            A generator yielding the subarrays as `np.memmap`s.
        """
        for i in range(len(self)):
            yield self[i]


@typeguard.typechecked
class WriteShaped:
    """
    Write numpy arrays with arbitrary shapes into a memory mapped file. The
    underlying data is stored in a jagged array of flattened arrays. The shapes
    are stored in a second jagged array.

    Attributes
    ---
    vector: WriteJagged
        The jagged array of flattened arrays.

    _shape: WriteJagged
        The jagged array of shapes.
    """

    vector: WriteJagged
    _shape: WriteJagged

    def __init__(
        self: Self,
        path: str,
        name: str,
        *,
        dtype: DTypeLike | None = None,
        check: str | None = None,
    ) -> None:
        """
        Initializes the context manager for writing a arrays with arbitrary
        shapes into a memory mapped file.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This is used as key to retrieve the data via
            `ReadShaped`.

        dtype: DTypeLike | None, optional, default: None
            Dtype of the data. `None` if the dtype is inferred from the first
            call of `append`.

        check: str | None, optional, default: None
            Checksum of the data. `None` if a timestamp at the time of saving
            will be used as checksum.

        Raises
        ---
        TypeError
            If any arguments python type does not match the type hints.
        """
        self.vector = WriteJagged(
            path,
            name,
            shape=(),
            dtype=dtype,
            check=check,
        )
        self._shape = WriteJagged(
            path,
            _labels.get_shape_name(name),
            shape=(),
            dtype=np.int64,
        )

    def __enter__(self: Self) -> Self:
        self.vector.__enter__()
        self._shape.__enter__()
        return self

    def __exit__(self: Self, *args) -> None:
        self.vector.__exit__(*args)
        self._shape.__exit__(*args)

    def append(self: Self, value: np.ndarray) -> None:
        """
        Append a new array to the memory mapped file. The dtype of the given
        array must match the dtype of the previously appended arrays.

        Parameters
        ---
        value: np.ndarray
            The array to append.

        Raises
        ---
        ValueError
            If dtype of the given array does not match the previously appended
            arrays.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.vector.append(value.reshape(-1))
        self._shape.append(np.array(value.shape, dtype=np.int64))


@typeguard.typechecked
class ReadShaped:
    """
    Read a memory mapped array of arrays with arbitrary shapes. This array can
    be indexed or iterated over.

    Attributes
    ---
    vector: ReadJagged
        The jagged array of flattened arrays.

    _shape: ReadJagged
        The jagged array of shapes.
    """

    vector: ReadJagged
    _shape: ReadJagged

    def __init__(self: Self, path: str, name: str) -> None:
        """
        Loads a memory mapped array of arrays with arbitrary shapes.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This name was used as a key to store the data via
            the `WriteShaped` context manager.

        Raises
        ---
        KeyError
            If the name is not found in the metadata.

        FileNotFoundError
            If the file for the name is not found. This could be because the
            data was not saved properly, or the user deleted/moved/renamed the
            underlying file.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.vector = ReadJagged(path, name)
        self._shape = ReadJagged(path, _labels.get_shape_name(name))

    @overload
    def __getitem__(self: Self, index: int) -> np.memmap:
        ...

    @overload
    def __getitem__(self: Self, index: np.ndarray) -> list[np.memmap]:
        ...

    def __getitem__(
        self: Self,
        index: int | np.ndarray,
    ) -> np.memmap | list[np.memmap]:
        """
        Get the array(s) at the given index or indices.

        Parameters
        ---
        index: int | np.ndarray
            The index or indices to retrieve.

        Returns
        ---
        np.memmap | list[np.memmap]
            If an integer is given, the array at the given index is returned.
            If an array of integers is given, a list of the corresponding
            arrays is returned.

        Raises
        ---
        IndexError
            If the index is out of bounds.

        ValueError
            If the index is not valid, i.e. not positive or one-dimensional.

        TypeError
            If any arguments python type does not match the type hints.
        """
        if isinstance(index, int):
            index = np.array([index])
            return self[index][0]

        assert isinstance(index, np.ndarray)

        if index.ndim == 0:
            error = "Index must be one dimensional."
            raise ValueError(error)

        if index.dtype == bool:
            index = np.where(index)[0]

        if np.any(index < 0):
            error = "Index must be positive."
            raise ValueError(error)

        shape = self._shape[index]
        arrays = self.vector[index]

        for i, array in enumerate(arrays):
            arrays[i] = array.reshape(shape[i])  # type: ignore

        return arrays

    def __len__(self: Self) -> int:
        """
        Get the length of the array of heterogeneous arrays, i.e. how often the
        `append` method was called on the corresponding `WriteShaped` context
        manager.

        Returns
        ---
        int
            The number of arrays in the array of heterogeneous arrays.
        """
        return len(self.vector)

    def __iter__(self: Self) -> Generator[np.memmap, None, None]:
        """
        Iterate over the array of heterogeneous arrays.

        Returns
        ---
        Generator[np.ndarray, None, None]
            A generator yielding the arrays as `np.ndarray`s.
        """
        for i in range(len(self)):
            yield self[i]


@typeguard.typechecked
class WriteStructured:
    """
    Write arbitrary structured data into a memory mapped file.

    Attributes
    ---
    data: WriteShaped
        Where the flattened data is stored.

    structure: WriteShaped
        Where the structure of the data is stored.

    get_structure: Callable[[np.ndarray], np.ndarray]
        A function that takes an array and returns its structure. This structure
        will later be used in the `ReadStructured` class to reconstruct the
        data.

    """

    get_structure: Callable[[np.ndarray], np.ndarray]
    data: WriteShaped
    structure: WriteShaped

    def __init__(
        self: Self,
        path: str,
        name: str,
        *,
        get_structure: Callable[[np.ndarray], np.ndarray],
        check: str | None = None,
    ) -> None:
        """
        Initializes the context manager for writing a arrays with arbitrary
        structure into a memory mapped file.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This is used as key to retrieve the data via
            `ReadStructured`.

        get_structure: Callable[[np.ndarray], np.ndarray]
            A function that takes an array and returns its structure. This
            structure will later be used in the `ReadStructured` class to
            reconstruct the data.

        check: str | None, optional, default: None
            Checksum of the data. `None` if a timestamp at the time of saving
            will be used as checksum.

        Raises
        ---
        TypeError
            If any arguments python type does not match the type hints.
        """
        self.get_structure = get_structure
        self.data = WriteShaped(path, name, check=check)
        self.structure = WriteShaped(
            path,
            _labels.get_structured_name(name),
            check=check,
        )

    def __enter__(self: Self) -> Self:
        self.data.__enter__()
        self.structure.__enter__()
        return self

    def __exit__(self: Self, *args) -> None:
        self.data.__exit__(*args)
        self.structure.__exit__(*args)

    def append(self, value: np.ndarray) -> None:
        """
        Append a new array to the memory mapped file. The dtype of the given
        array must match the dtype of the previously appended arrays.

        Parameters
        ---
        value: np.ndarray
            The array to append.

        Raises
        ---
        ValueError
            If dtype of the given array does not match the previously appended
            arrays.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.data.append(value)
        self.structure.append(self.get_structure(value))


@typeguard.typechecked
class ReadStructured:
    """
    Read a memory mapped array of arrays with arbitrary structure. This array
    can be indexed or iterated over.

    Attributes
    ---
    reconstruct: Callable[[np.ndarray, np.ndarray], Any]
        A function that takes an array and its structure and returns the
        original data.

    data: ReadShaped
        Where the flattened data is stored.

    structure: ReadShaped
        Where the structure of the data is stored.
    """

    reconstruct: Callable[[np.ndarray, np.ndarray], Any]
    data: ReadShaped
    structure: ReadShaped

    def __init__(
        self: Self,
        path: str,
        name: str,
        reconstruct: Callable[[np.ndarray, np.ndarray], Any],
    ) -> None:
        """
        Loads a memory mapped array of arrays with arbitrary structure.

        Parameters
        ---
        path: str
            Path to the directory where the memory mapped file is stored.

        name: str
            Name of the data. This name was used as a key to store the data via
            the `WriteStructured` context manager.

        reconstruct: Callable[[np.ndarray, np.ndarray], Any]
            A function that takes an array and its structure and returns the
            original data.

        Raises
        ---
        KeyError
            If the name is not found in the metadata.

        FileNotFoundError
            If the file for the name is not found. This could be because the
            data was not saved properly, or the user deleted/moved/renamed the
            underlying file.

        TypeError
            If any arguments python type does not match the type hints.
        """
        self.reconstruct = reconstruct
        self.data = ReadShaped(path, name)
        self.structure = ReadShaped(path, _labels.get_structured_name(name))

    @overload
    def __getitem__(self: Self, index: int) -> Any:
        ...

    @overload
    def __getitem__(self: Self, index: np.ndarray) -> list[Any]:
        ...

    def __getitem__(self: Self, index: int | np.ndarray) -> Any | list[Any]:
        """
        Index the array of arrays with arbitrary structure. Indexing with an
        integer `i` returns the `i`-th array. Indexing with an array of
        integers returns a list of the corresponding arrays.

        Parameters
        ---
        index: int | np.ndarray
            The index or indices to retrieve.

        Returns
        ---
        Any | list[Any]
            The retrieved array or list of arrays.

        Raises
        ---
        IndexError
            If the index is out of bounds.

        ValueError
            If the index is not valid, i.e. not positive or one-dimensional.

        TypeError
            If any arguments python type does not match the type hints.
        """
        if isinstance(index, int):
            index = np.array([index])
            return self[index][0]

        data = self.data[index]
        structure = self.structure[index]

        for i, (d, s) in enumerate(zip(data, structure)):
            data[i] = self.reconstruct(d, s)

        return data
