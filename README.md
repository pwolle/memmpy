# Memmpy
Memmpy is a Python library for working with memory mapped numpy arrays. It supports
- Saving to temporary or permanent memory mapped files
- Appending and extending arrays in constant time
- Loading arrays in batches
- Train-validation-test-splits and k-fold cross validation
- Fast & lazy shuffling by shuffling in blocks and bands


## Who should use Memmpy?
Memmpy is primarly intended for use in data science and machine learning applications where the dataset is too large to fit into memory at once.
Memmpy is not intended for use in small applications where the entire dataset fits into memory and can be loaded at once. It is also not intended for use in very large applications where training is massively distributed.


## Installation
Memmpy can be installed directly from PyPI using `pip`. It requires Python 3.10 or higher.
```bash
pip install memmpy
```

## Usage
Writing to a memory mapped file.

```python
# Create a memory mapped array
file = memmpy.Vector()

for i in range(4):
    file.append(np.random.rand(100))  # O(1)


file.extend(np.random.rand(32, 100))

# access the array
assert file.array.shape == (4 + 32, 100)

# save to non-temporary file
file.save("data.npy")
```

Loading random batches from a memory mapped file.

```python
array = np.memmap(data.npy, dtype=np.float64, mode='r', shape=(36, 100))

# Load the array in batches
batch_indicies = memmpy.batch_indicies_split(
    array.shape[0],
    4,
    "train",
    valid_part=10,  # size of the validation and train set
    kfold_index=2,  # take the second out of 10 folds
)
shuffle = memmpy.shuffle_fast(array.shape[0], seed=42)  # O(1)

for indicies in batch_indicies:
    indicies = shuffle(indicies)
    batch = array[indicies]
```