import ast
import hashlib
from typing import Any

import numpy as np
import tqdm
import typeguard

from . import _labels, _vector, _loader


@typeguard.typechecked
def _get_symbols(expression: str) -> set[str]:
    tree = ast.parse(expression, mode="eval")
    walk = ast.walk(tree)

    symbols = set()
    for node in walk:
        if isinstance(node, ast.Name):
            symbols.add(node.id)

    return symbols


@typeguard.typechecked
def compute_cut_batched(
    path: str,
    expression: str,
    *,
    batch_size: int = 1024 * 128,
    constants: dict[str, Any] | None = None,
    recreate: bool = False,
) -> np.memmap:
    """
    Computes indicies for which the given expression is true. The result is
    stored in a memmap file.

    Parameters
    ---
    path: str
        Path to the memmap file.

    expression: str
        Expression to evaluate. Can be any valid Python expression that returns
        a boolean value.

    batch_size: int, optional, default: 1024 * 128
        Number of events to process at once.

    write_type: type, optional, default: WriteVector
        Type of the memmap file to write to.

    constants: dict[str, Any], optional, default: None
        Constant values to use in the expression.

    recreate: bool, optional, default: False
        If True, the cut is recomputed even if it already exists.

    Returns
    ---
    np.memmap
        Array of indicies for which the expression is true.
    """
    constants = constants or {}

    keys = _get_symbols(expression)
    keys = keys - set(constants.keys())

    meta = _labels.safe_load(path)

    hasher = hashlib.sha256()
    for key in sorted(keys):
        if key in meta["arrays"]:
            hasher.update(meta["arrays"][key]["check"].encode())
        else:
            raise KeyError(f"Key {key} not found in memmap file.")

    hashed = hasher.hexdigest()

    if not recreate:
        if expression in meta["arrays"]:
            if meta["arrays"][expression]["check"] == hashed:
                print(f"Using cached cut: {expression}.")
                return _vector.read_vector(path, expression)

    print(f"Computing cut: {expression}.")
    with _vector.WriteVector(path=path, name=expression, check=hashed) as writer:
        batch_loader = _loader.Batched(
            _loader.Dict(_vector.read_vectors(path, keys)),  # type: ignore
            batch_size,
        )

        for i, batch in enumerate(tqdm.tqdm(batch_loader)):
            batch = _loader.unwrap(batch)
            result = eval(expression, batch | constants, {})

            index = np.arange(len(result)) + i * batch_size
            index = index[result]

            if len(index) == 0:
                continue

            writer.extend(index)

    return _vector.read_vector(path, expression)
