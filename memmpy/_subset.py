import ast
import hashlib
from typing import Any

import numpy as np
import tqdm
import typeguard

from . import _labels, _loader, _vector


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
    write_type=_vector.WriteVector,
    constants: dict[str, Any] | None = None,
    recreate: bool = False,
) -> np.memmap:
    constants = constants or {}

    keys = _get_symbols(expression)
    keys = keys - set(constants.keys())

    hasher = hashlib.sha256()
    for key in sorted(keys):
        hasher.update(key.encode())

    hashed = hasher.hexdigest()

    if not recreate:
        meta = _labels.safe_load(path)

        if expression in meta["arrays"]:
            if meta["arrays"][expression]["check"] == hashed:
                print(f"Using cached cut: {expression}.")
                return _vector.read_vector(path, expression)

    print(f"Computing cut: {expression}.")
    with write_type(path=path, name=expression, check=hashed) as writer:
        batch_loader = _loader.SimpleLoader(
            _loader.load_memmaps(path, keys),
            batch_size,
        )

        for batch in tqdm.tqdm(batch_loader):
            result = eval(expression, batch | constants, {})
            index = batch["_index"][result]
            writer.extend(index)

    return _vector.read_vector(path, expression)
