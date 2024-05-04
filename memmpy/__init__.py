from beartype.claw import beartype_this_package

beartype_this_package()


from ._vector import Vector
from ._shuffle import shuffle_fast, shuffle_blocks, shuffle_bands
from ._index import batch_slices, batch_indicies, batch_indicies_split

__version__ = "0.1.10"

__all__ = [
    "__version__",
    "Vector",
    "shuffle_fast",
    "shuffle_blocks",
    "shuffle_bands",
    "batch_slices",
    "batch_indicies",
    "batch_indicies_split",
]
