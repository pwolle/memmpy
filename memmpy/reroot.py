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

from . import _labels, _vector

__all__ = [
    "RFileConfig",
    "memmap_root",
]


@dataclasses.dataclass(frozen=True)
class RFileConfig:
    """
    Stores the configuration for a single ROOT file.

    Parameters
    ---
    path: str
        Path to the ROOT file.

    tree: str, optional, default: "nominal_Loose"
        Name of the tree to read from.

    metadata: dict[str, str], optional, default: {}
        Metadata to store in the memmap file. This can be used to store
        information about the what is stored in the file, i.e. from which
        process the data originates, or which run it is from.
    """

    path: str
    tree: str = dataclasses.field(default="nominal_Loose")
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)

    def __lt__(self, other: "RFileConfig") -> bool:
        return repr(self) < repr(other)


def _encode_metadata(
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
def memmap_root(
    root_files: list[RFileConfig],
    path_mmap: str,
    keys: set[str],
    keys_padded: dict[str, tuple[int, Any]] | None = None,
    aliases: dict[str, str] | None = None,
    recreate: bool = False,
) -> dict[str, np.memmap]:
    """
    Store the data from the given ROOT files in a memmap files.

    Parameters
    ---
    root_files: list[RFileConfig]
        List of ROOT file configurations to read from, i.e. the path to the file,
        the name of the tree to read from, and metadata to store in memmaps.

    path_mmap: str
        Path to the memmap directory.

    keys: set[str]
        Set of keys to read from the ROOT files. The keys must be scalar valued.

    keys_padded: dict[str, tuple[int, Any]], optional, default: None
        Dictionary of 1d vector valued keys to read from the ROOT files. The
        dictionary maps the key to a tuple of the size of the vector and the
        value to pad with.

    aliases: dict[str, str], optional, default: None
        Dictionary of aliases to use for the keys.

    recreate: bool, optional, default: False
        If True, the memmap files are recreated even if they already exist.

    Raises
    ---
    FileNotFoundError
        If a ROOT file is not found.

    TypeError
        If the arguments do not match the type hints.
    """
    keys_padded = keys_padded or {}
    aliases = aliases or {}
    keys = keys | set(keys_padded.keys())

    hasher = hashlib.sha256()
    for k in sorted(aliases.keys()):
        hasher.update(str(k).encode())
        hasher.update(aliases[k].encode())

    for rfile in sorted(root_files):
        hasher.update(os.path.basename(rfile.path).encode())
        hasher.update(str(os.path.getmtime(rfile.path)).encode())
        hasher.update(str(os.path.getsize(rfile.path)).encode())
        hasher.update(str(rfile.tree).encode())
        hasher.update(str(rfile.metadata).encode())

    hashed = hasher.hexdigest()

    metadata_values = set()
    metadata_keys = set()

    for rfile in root_files:
        metadata_values |= set(rfile.metadata.values())
        metadata_keys |= set(rfile.metadata.keys())

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
            return _vector.read_vectors(path_mmap, keys | metadata_keys)

    with _vector.WriteVectorDict(path_mmap, check=hashed) as vectors:
        for rfile in tqdm.tqdm(root_files):
            tqdm.tqdm.write(f"processing '{rfile.path}'")

            if not os.path.exists(rfile.path):
                raise FileNotFoundError(f"File '{rfile.path}' not found.")

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

                data_path = _encode_metadata(rfile.metadata, size)
                vectors.expand(data_numpy | data_path)

        print("Saving...")

    # add hahses to metadata
    for value in metadata_values:
        _labels.add_hash(path_mmap, value)

    print("Saving done.")
    return _vector.read_vectors(path_mmap, keys | metadata_keys)
