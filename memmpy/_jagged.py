import numpy as np

import typeguard

from numpy.typing import DTypeLike
from typing import Self, Callable, Any, overload

from . import _vector
from . import _labels


@typeguard.typechecked
class WriteJagged:
    def __init__(
        self: Self,
        path: str,
        name: str,
        dtype: DTypeLike | None = None,
        shape: tuple[int, ...] | None = None,
        check: str | None = None,
    ) -> None:
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

    def append(self, value: np.ndarray) -> None:
        # first axis is treated as variable length
        self.vector.extend(value)
        self._index.append(np.array(self.vector._index, dtype=np.int64))


@typeguard.typechecked
class ReadJagged:
    def __init__(self: Self, path: str, name: str) -> None:
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
        return len(self._index) - 1

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]


@typeguard.typechecked
class WriteShaped:
    def __init__(
        self: Self,
        path: str,
        name: str,
        dtype: DTypeLike | None = None,
        check: str | None = None,
    ) -> None:
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

    def append(self, value: np.ndarray) -> None:
        self.vector.append(value.reshape(-1))
        self._shape.append(np.array(value.shape, dtype=np.int64))


@typeguard.typechecked
class ReadShaped:
    def __init__(self: Self, path: str, name: str) -> None:
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

    def __len__(self) -> int:
        return len(self.vector)

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]


@typeguard.typechecked
class WriteStructured:
    def __init__(
        self: Self,
        path: str,
        name: str,
        get_structure: Callable[[np.ndarray], np.ndarray],
        check: str | None = None,
    ) -> None:
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
        self.data.append(value)
        self.structure.append(self.get_structure(value))


@typeguard.typechecked
class ReadStructured:
    def __init__(
        self: Self,
        path: str,
        name: str,
        reconstruct: Callable[[np.ndarray, np.ndarray], Any],
    ) -> None:
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
        if isinstance(index, int):
            index = np.array([index])
            return self[index][0]

        data = self.data[index]
        structure = self.structure[index]

        for i, (d, s) in enumerate(zip(data, structure)):
            data[i] = self.reconstruct(d, s)

        return data
