# NumPy

NumPy stands for Numerical Python. It's the backbone of all kinds of scientific and numerical computing in Python. Since machine learning is all about turning data into numbers and then figuring out the patterns, NumPy often comes into play.

## Why NumPy?

You can do numerical calculations using pure Python. In the beginning, you might think Python is fast, but once your data gets large, you'll start to notice slowdowns. One of the main reasons you use NumPy is because it's fast. Behind the scenes, the code has been optimized to run using C, which can do things much faster than Python. The benefit of this being behind the scenes is you don't need to know any C to take advantage of it. You can write your numerical computations in Python using NumPy and get the added speed benefits.

What causes this speed benefit? It's a process called **vectorization**. Vectorization aims to do calculations by avoiding loops, as loops can create potential bottlenecks. NumPy achieves vectorization through a process called **broadcasting**.

## Importing NumPy

To get started using NumPy, the first step is to import it. The most common way is to import NumPy as the abbreviation `np`. If you see the letters `np` used anywhere in machine learning or data science, it's probably referring to the NumPy library.

```python
import numpy as np
# Check the version
print(np.__version__)
```

## Data Structure and Attributes

In NumPy, the main data structure is the **ndarray** (N-dimensional array), but it's more than just an array—it enables efficient operations on large datasets, supports broadcasting, and is foundational for scientific computing.

```python
# 1-dimensional array, also referred to as a vector
a1 = np.array([1, 2, 3])

# 2-dimensional array, also referred to as a matrix
a2 = np.array([[1, 2.0, 3.3],
               [4, 5, 6.5]])

# 3-dimensional array, also referred to as a matrix
a3 = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]]])
```

### Attributes
| Attribute         | Meaning                        | Example                     |
|-------------------|--------------------------------|-----------------------------|
| `arr.ndim`        | Number of dimensions           | 2 for a 2D array            |
| `arr.shape`       | Dimensions (rows, cols, ...)   | `(2, 3)` for a 2x3 array    |
| `arr.size`        | Total number of elements      | 6 for a 2x3 array           |
| `arr.dtype`       | Data type of elements         | `int64`, `float32`, etc.    |
| `arr.itemsize`    | Size in bytes of one element  | 4 bytes for `float32`       |

## Pandas DataFrame out of NumPy Arrays

This is to exemplify how NumPy is the backbone of many other libraries.

```python
import pandas as pd
df = pd.DataFrame(np.random.randint(10, size=(5, 3)), columns=['a', 'b', 'c'])
print(df)
```

**Output**
```
   a  b  c
0  5  8  0
1  3  3  2
2  1  6  7
3  7  3  9
4  6  6  7
```

## Creating Arrays

- `np.array()` – Create an array from a list or tuple
- `np.ones()` – Create an array filled with ones
- `np.zeros()` – Create an array filled with zeros
- `np.random.random((5, 3))` – Create a 5x3 array of random floats between 0 and 1
- `np.random.rand(5, 3)` – Same as `random.random` but with a simpler syntax
- `np.random.randint(10, size=5)` – Create an array of 5 random integers between 0 and 9
- `np.random.seed()` – Set the seed for NumPy's random number generator to ensure reproducibility. It makes sure that every time you run your code, you get the same sequence of random numbers.

### Example: Without and With Seed
```python
# Without seed
import numpy as np
print(np.random.rand(3))  # Output changes every time

# With seed
np.random.seed(0)
print(np.random.rand(3))  # Always get the same numbers: [0.5488135 0.71518937 0.60276338]
```

## Manipulating Arrays

### Arithmetic
- `np.exp()` – Exponential function
- `np.log()` – Natural logarithm
- Dot product – `np.dot()`
- Broadcasting – See below for details
- Aggregation:
  - `np.sum()` – Faster than Python's `sum()` for NumPy arrays
  - `np.mean()` – Compute the mean
  - `np.std()` – Compute the standard deviation
  - `np.var()` – Compute the variance
  - `np.min()` – Find the minimum value
  - `np.max()` – Find the maximum value
  - `np.argmin()` – Find the index of the minimum value
  - `np.argmax()` – Find the index of the maximum value
- These work on all `ndarray`s, e.g., `a4.min(axis=0)` – you can use `axis` as well

### Reshaping
- `np.reshape()` – Change the shape (dimensions) of an array without changing the data

### Transposing
- `a3.T` – Reverses the order of the axes (e.g., shape `(2, 3)` becomes `(3, 2)`)

### Comparison Operators
- `>=`, `<=`, `==`, `!=`
- Example: `np.sum(x > 3)` – Count elements greater than 3

### Broadcasting
Broadcasting is a feature of NumPy that performs an operation across multiple dimensions of data without replicating the data. This saves time and space. For example, if you have a 3x3 array (A) and want to add a 1x3 array (B), NumPy will add the row of (B) to every row of (A).

#### Rules of Broadcasting
1. If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.
2. If the shape of the two arrays does not match in entlang dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
3. If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

**Example of Broadcasting Error**
```python
a2 * a3  # This will error as the arrays have different shapes: (2, 3) vs. (2, 3, 3)
```
**Error**: Operands could not be broadcast together with shapes `(2, 3) (2, 3, 3)`

#### The Broadcasting Rule
In order to broadcast, the size of the trailing axes for both arrays in an operation must be either the same size or one of them must be one.

### What's Mean?
Mean is the same as average. You can find the average of a set of numbers by adding them up and dividing them by how many there are.

### What's Standard Deviation?
Standard deviation is a measure of how spread out numbers are.

### What's Variance?
The variance is the averaged squared differences from the mean. To work it out, you:
1. Work out the mean
2. For each number, subtract the mean and square the result
3. Find the average of the squared differences

## Reshaping
```python
arr.reshape(shape)
```
Reshaping means changing the shape (dimensions) of a NumPy `ndarray` without changing the data. This is useful to:
- Prepare data for machine learning models
- Convert between 1D, 2D, or higher dimensions
- Meet input shape requirements for operations (like matrix multiplication)

## Transpose
A transpose reverses the order of the axes. For example, an array with shape `(2, 3)` becomes `(3, 2)`.

```python
a3.T
```

## Dot Product
The main two rules for dot product to remember are:
1. The **inner dimensions** must match:
   - `(3, 2) @ (3, 2)` won't work
   - `(2, 3) @ (3, 2)` will work
   - `(3, 2) @ (2, 3)` will work
2. The resulting matrix has the shape of the **outer dimensions**:
   - `(2, 3) @ (3, 2)` → `(2, 2)`
   - `(3, 2) @ (2, 3)` → `(3, 3)`

**Note**: In NumPy, `np.dot()` and `@` can be used to achieve the same result for 1-2 dimensional arrays. However, their behavior begins to differ for arrays with 3+ dimensions.

## Sorting Arrays
- `np.sort()` – Sort values in a specified dimension of an array
- `np.argsort()` – Return the indices to sort the array on a given axis
- `np.argmax()` – Return the index/indices which give the highest value(s) along an axis
- `np.argmin()` – Return the index/indices which give the lowest value(s) along an axis