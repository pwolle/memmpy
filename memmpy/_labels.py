import datetime
import hashlib
import json
import os

import numpy as np
import typeguard
from numpy.typing import DTypeLike


@typeguard.typechecked
def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


@typeguard.typechecked
def hash_64bit(b: bytes) -> int:
    """
    Hash a byte string to a 64 bit integer.

    Parameters
    ---
    b: bytes
        The byte string to hash.

    Returns
    ---
    int
        The 64 bit integer hash.

    Raises
    ---
    TypeError
        If the input is not a byte string.
    """
    hashed = hashlib.sha256(b)
    hashed = int(hashed.hexdigest(), 16)
    return hashed % 2**64


@typeguard.typechecked
def string_pad_64bit(integer) -> str:
    return f"{integer:0>16x}"


@typeguard.typechecked
def get_metadata_path(path: str) -> str:
    return os.path.join(path, "metadata_memmpy.json")


@typeguard.typechecked
def safe_load(path: str) -> dict:
    meta_path = get_metadata_path(path)

    if not os.path.exists(meta_path):
        return {
            "arrays": {},
            "hashes": {},
        }

    with open(meta_path, "r") as file:
        return json.load(file)


@typeguard.typechecked
def add_hash(path: str, name: str) -> None:
    meta = safe_load(path)

    hashed = hash_64bit(name.encode())
    hashed = string_pad_64bit(hashed)

    if hashed in meta["hashes"]:
        if meta["hashes"][hashed] != name:
            error = f"Hash collision {name}, with hash {hashed}."
            raise ValueError(error)

        return

    meta["hashes"][hashed] = name

    with open(get_metadata_path(path), "w") as file:
        json.dump(meta, file, indent=4)


@typeguard.typechecked
def save_array_metadata(
    path: str,
    name: str,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    check: str | None = None,
) -> None:
    meta = safe_load(path)
    meta["arrays"][name] = {
        "check": check,
        "dtype": np.dtype(dtype).str,
        "shape": list(shape),
    }

    hashed = hash_64bit(name.encode())
    hashed = string_pad_64bit(hashed)
    meta["hashes"][hashed] = name

    with open(get_metadata_path(path), "w") as file:
        json.dump(meta, file, indent=4)


@typeguard.typechecked
def get_array_path(path: str, name: str) -> str:
    hashed = hash_64bit(name.encode())
    hashed = string_pad_64bit(hashed)
    return os.path.join(path, f"{hashed}.npy")


@typeguard.typechecked
def read_array_metadata(
    path: str,
    name: str,
) -> tuple[str, DTypeLike, tuple[int, ...]]:
    meta = safe_load(path)

    if name not in meta["arrays"]:
        error = f"Array '{name}' not found in '{path}'."
        raise KeyError(error)

    apath = get_array_path(path, name)
    dtype = np.dtype(meta["arrays"][name]["dtype"])
    shape = tuple(meta["arrays"][name]["shape"])
    return apath, dtype, shape


@typeguard.typechecked
def get_index_name(name: str) -> str:
    return f"{name}:index"


@typeguard.typechecked
def get_shape_name(name: str) -> str:
    return f"{name}:shape"


@typeguard.typechecked
def get_structured_name(name: str) -> str:
    return f"{name}:structured"
