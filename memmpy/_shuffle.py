from typing import Callable, Self

import numpy as np


class FixedSeed:
    def __init__(self, seed: int | None) -> None:
        self._seed = seed

    def __enter__(self: Self) -> None:
        if self._seed is None:
            return

        self.state = np.random.get_state()
        np.random.seed(self._seed)

    def __exit__(self: Self, *_) -> None:
        if self._seed is None:
            return

        np.random.set_state(self.state)


def shuffle_blocks(
    size: int,
    size_perm: int = 1024,
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    # permutation for most of the array
    with FixedSeed(seed):
        perm_tot = np.random.uniform(size=size_perm).argsort()

    # threshold for the remaining elements
    c = size // size_perm * size_perm

    # number of remaining elements, at least one to ensure no no null division
    n = max(1, size % size_perm)

    # permutation for the remaining elements
    perm_rem = np.random.uniform(size=n).argsort()

    def apply_perm(x):
        div = x // size_perm
        rem = x % size_perm
        return np.where(
            x < c,
            perm_tot[rem] + div * size_perm,
            perm_rem[rem % n] + div * size_perm,
        )

    return apply_perm


def shuffle_bands(
    size: int,
    size_perm: int = 8,
) -> Callable[[np.ndarray], np.ndarray]:
    # permutation for offsets in modulus equivalance classes
    perm = np.arange(size_perm)

    # has to be multiplied by size_perm to stay in the same class
    perm = perm * size_perm
    # controll the distance between the consecutive elements
    perm = perm * (size // (size_perm - 1))

    # size of the largest multiple of size_perm smaller than size
    size_floor = size // size_perm * size_perm

    def apply_perm(x):
        return np.where(
            x < size_floor,
            (x + perm[x % size_perm]) % size_floor,
            x,
        )

    return apply_perm


def shuffle_fast(
    size: int,
    size_perm_band: int = 8,
    size_perm_block: int = 1024,
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    shuffle1 = shuffle_blocks(size, size_perm_block, seed)
    shuffle2 = shuffle_bands(size, size_perm_band)

    def apply(x):
        x = shuffle1(x)
        x = (x + size // 2) % size
        x = shuffle2(x)
        return x

    return apply
