# Memmpy
Memmpy is a Python library for storing datasets in, and loading datasets from, [memory-mapped](https://en.wikipedia.org/wiki/Memory-mapped_file) files. This is particularly useful for large datasets that do not fit in memory and therefore need to be processed in batches. Memmpy is based on the [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) implementation.

## Installation
Memmpy can be directly installed from PyPI using `pip`.
```bash
pip install memmpy
```

If you want to use `memmpy` to process `.root` files, you will also need to install `uproot`.
```bash
pip install uproot
```

## Usage
A simple memory mapped file can be created as follows:
```python
with memmpy.WriteVector(path="data.memmpy", key="testdata") as memfile:
    # Append a single numpy array.
    # The shape and dtype will be inferred from the array.
    memfile.append(np.array([1, 2, 3]))
    
    # Append another numpy array of the same shape and dtype
    memfile.append(np.array([4, 5, 6]))

    # Extend the file by an array with an additional axis.
    memfile.extend(np.array([[7, 8, 9], [10, 11, 12]]))

memmap_data = memmpy.read_vector(path="data.memmpy", key="testdata")
```

The `mempy` library also provides functionality to store jagged arrays or arrays with arbitrary shape using the `WriteJagged`, `ReadJagged`, `WriteShaped` and `ReadShaped` classes.


## See also
- [numpy.memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [uproot](https://uproot.readthedocs.io/en/latest/index.html)
