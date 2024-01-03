try:
    import uproot
except ImportError:
    error = "'Uproot' is a requirement for 'memmpy.reroot', but was not found."
    raise ImportError(error)

import dataclasses
import hashlib
import os
from typing import Any

import awkward as ak
import numpy as np
import tqdm
import typeguard

from . import _labels, _loader, _subset, _vector

__all__ = [
    "RFileConfig",
    "memmap_vectors_root",
    "load_root",
]


@dataclasses.dataclass(frozen=True)
class RFileConfig:
    path: str
    tree: str = dataclasses.field(default="nominal_Loose")
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)


def encode_metadata(
    metadata: dict[str, str],
    size: int,
) -> dict[str, np.ndarray]:
    metadata_encoded = {}

    for k, v in metadata.items():
        hashed = _labels.hash_64bit(v.encode())
        metadata_encoded[k] = np.full(
            size,
            hashed,
            dtype="uint64",
        )

    return metadata_encoded


@typeguard.typechecked
def memmap_vectors_root(
    root_files: list[RFileConfig],
    path_mmap: str,
    keys: set[str],
    keys_padded: dict[str, tuple[int, Any]] | None = None,
    aliases: dict[str, str] | None = None,
    recreate: bool = False,
) -> None:
    keys_padded = keys_padded or {}
    aliases = aliases or {}
    keys = keys | set(keys_padded.keys())

    hasher = hashlib.sha256()
    hasher.update(str(aliases).encode())

    for rfile in root_files:
        hasher.update(os.path.basename(rfile.path).encode())
        hasher.update(str(os.path.getmtime(rfile.path)).encode())
        hasher.update(str(os.path.getsize(rfile.path)).encode())
        hasher.update(str(rfile.tree).encode())
        hasher.update(str(rfile.metadata).encode())

    hashed = hasher.hexdigest()

    if not recreate:
        meta = _labels.safe_load(path_mmap)

        for k in keys:
            if k not in meta["arrays"]:
                break

            if meta["arrays"][k]["check"] != hashed:
                break

            if k in keys_padded:
                if meta["arrays"][k]["shape"][-1] != keys_padded[k][0]:
                    break

        else:
            path_mmap = os.path.abspath(path_mmap)
            print(f"Using cached files in '{path_mmap}' for {list(keys)}.")
            return

    with _vector.WriteVectorDict(path_mmap, check=hashed) as vectors:
        for rfile in tqdm.tqdm(root_files):
            tqdm.tqdm.write(f"processing '{rfile.path}'")

            with uproot.open(rfile.path) as file:  # type: ignore
                data_root = file[rfile.tree].arrays(  # type: ignore
                    keys,
                    aliases=aliases,
                    library="ak",
                    how=dict,
                )
                data_numpy = {}

                for k, v in data_root.items():
                    if k in keys_padded:
                        assert v.ndim == 2
                        pad_size, pad_value = keys_padded[k]
                        array = ak.pad_none(v, pad_size, clip=True)
                        array = ak.fill_none(array, pad_value)
                        data_numpy[k] = array.to_numpy()  # type: ignore
                        continue

                    assert v.ndim == 1
                    data_numpy[k] = v.to_numpy()

                size = next(iter(data_numpy.values())).shape[0]

                data_path = encode_metadata(rfile.metadata, size)
                vectors.expand(data_numpy | data_path)

        print("Saving...")

    # add hahses
    metadata_values = set()

    for rfile in root_files:
        metadata_values |= set(rfile.metadata.values())

    for value in metadata_values:
        _labels.add_hash(path_mmap, value)

    print("Saving done.")


@typeguard.typechecked
def load_root(
    root_files: list[RFileConfig],
    path_mmap: str,
    keys: set[str],
    keys_padded: dict[str, tuple[int, Any]] | None = None,
    tcut: str | None = None,
    aliases: dict[str, str] | None = None,
    recreate: bool = False,
    batch_size: int = 32,
    split: int = 0,
    fractions: tuple[float, ...] = (0.8, 0.1, 0.1),
    split_seed: int = 0,
    shuffle=True,
    cut_constants: dict[str, Any] | None = None,
) -> _loader.SplitLoader:
    keys_mmemap = keys
    if tcut is not None:
        keys_mmemap = keys_mmemap | _subset.get_symbols(tcut)

    memmap_vectors_root(
        root_files,
        path_mmap,
        keys_mmemap,
        keys_padded,
        aliases,
        recreate,
    )
    keys_metadata = set()
    for rfile in root_files:
        keys_metadata |= set(rfile.metadata.keys())

    data = _loader.load_memmaps(path_mmap, keys | keys_metadata)

    subindex = None
    if tcut is not None:
        _subset.compute_cut_batched(path_mmap, tcut, constants=cut_constants)
        subindex = _vector.read_vector(path_mmap, tcut)

    return _loader.SplitLoader(
        data,
        batch_size,
        split=split,
        fractions=fractions,
        subindex=subindex,
        split_seed=split_seed,
        shuffle=shuffle,
    )
