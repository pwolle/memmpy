# Memmpy
Memmpy is a Python library for storing datasets in, and loading datasets from, [memory mapped](https://en.wikipedia.org/wiki/Memory-mapped_file) files. This is particularly useful for large datasets that do not fit in memory and therefore need to be processed in batches. Memmpy is based on the [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) implementation.

## Installation
Memmpy can be installed directly from PyPI using `pip`. Memmpy requires Python 3.10 or higher.
If you want to process `.root` files, `uproot` is required. This can also be installed using `pip`.
```bash
pip install memmpy
```

## Usage
A simple memory mapped file can be created as follows:
```python
with WriteVector(path="data.mmpy", key="testdata") as memfile:
    # Append a single numpy array.
    # The shape and dtype will be inferred from the array.
    memfile.append(np.array([1, 2, 3]))
    
    # Append another numpy array of the same shape and dtype
    memfile.append(np.array([4, 5, 6]))

    # Extend the file by an array with an additional axis.
    memfile.extend(np.array([[7, 8, 9], [10, 11, 12]]))

memmap_data = read_vector(path="data.mmpy", key="testdata")
```

The `mempy` library also provides functionality to store jagged arrays or arrays with arbitrary shape using the `WriteJagged`, `ReadJagged`, `WriteShaped` and `ReadShaped` classes.

### Loading
A collection of memory mapped files can be loaded in batches using the `SimpleLoader` and `SplitLoader`. The `SplitLoader` also provides functionality for shuffling the data and splitting it into training and validation sets.
```python
loader = SplitLoader(
    # provide a dict of memmap, ReadJagged or ReadShaped
    data={"first_memmap": memmap, ...},  
    batch_size=128,
    shuffle=True,
)
    
for batch in loader:
    ...
```

### Filtering
Data can be filtered using the `compute_cut_batched` function.
```python
subindicies = compute_cut_batched(
    path="data.mmpy",
    expression="testdata > 5"
)
```
The subindicies can be used to load only the filtered data, by passing them to the `SplitLoader`. All computed cuts are automtatically cached.

### Processing ROOT files
To use `memmpy` with ROOT files, the `uproot` module is required. The `uproot` library can be installed using `pip`.
```bash
pip install uproot
```
The `load_root` function provides all-in-one functionality to load (multiple) ROOT files into memory. An example is shown below.
```python
loader = load_root(
    root_files=[
        RFileConfig(
            path="ttbb_mc16a.root",
            tree="tree1",
            metadata={"process": "ttbb", "year": "mc16a"},
        ),
        RFileConfig(
            path="ttH_mc16d.root",
            tree="tree2",
            metadata={"process": "ttH", "year": "mc16d"},
        ),
    ],
    path_mmap="data.mmpy",
    keys={"nJets", "nBTags_77"},
    # variable length arrays can be padded to the same length with a given value
    keys_padded={"jet_pt": (22, float("nan"))}, 
    batch_size=128,
    tcut="nJets >= 6 & nBTags_77 >= 2",
)
```
All results are cached, so that the next time the function is called, the data is loaded from the cache instead of the ROOT files. The metadata is stored in hashed form, making it possible to apply cuts on the metadata as well.


## See also
- [numpy.memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [uproot](https://uproot.readthedocs.io/en/latest/index.html)
