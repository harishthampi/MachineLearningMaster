# Pandas

Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool, built on top of the Python programming language.

## Why Pandas?

Pandas provides a simple to use but very capable set of functions you can use on your data.  
It's integrated with many other data science and machine learning tools which use Python, so having an understanding of it will be helpful throughout your journey.  
One of the main use cases you'll come across is using pandas to transform your data in a way which makes it usable with machine learning algorithms.

## Importing Pandas

To get started using Pandas, the first step is to import it.  
The most common way (and method you should use) is to import pandas as the abbreviation `pd`.  
If you see the letters `pd` used anywhere in machine learning or data science, it's probably referring to the pandas library.

```python
import pandas as pd
# Print the version
print(f"pandas version: {pd.__version__}")
```

## Data Structures in Pandas

Pandas has two main datatypes, **Series** and **DataFrame**.  
- **pandas.Series** - a 1-dimensional column of data.  
- **pandas.DataFrame** (most common) - a 2-dimensional table of data with rows and columns.

### Series

A **Series** is a one-dimensional labeled array capable of holding any data type (integers, strings, floats, Python objects, etc.).

```python
import pandas as pd
data = [10, 20, 30, 40]
s = pd.Series(data)
print(s)
```

**Output**
```
0    10
1    20
2    30
3    40
dtype: int64
```

#### With Custom Index
```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
```

**Output**
```
a    10
b    20
c    30
dtype: int64
```

#### Accessing Elements
```python
print(s['a'])     # By label
print(s[0])       # By position
```

#### Useful Attributes and Methods
- `s.index` – shows index labels  
- `s.values` – shows underlying data  
- `s.head(n)` / `s.tail(n)` – first/last n elements  
- `s.mean()`, `s.sum()`, `s.max()`, `s.min()`

### DataFrame

A **DataFrame** is a two-dimensional, tabular data structure (like an Excel spreadsheet or SQL table) with labeled rows and columns.

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Toronto', 'London']
}

df = pd.DataFrame(data)
print(df)
```

**Output**
```
      Name  Age      City
0    Alice   25  New York
1      Bob   30   Toronto
2  Charlie   35    London
```

#### Accessing Elements
```python
df['Name']        # Access a column (as Series)
df[['Name', 'Age']]  # Multiple columns
df.loc[0]         # Row by index label
df.iloc[1]        # Row by position
df.loc[0, 'Age']  # Specific cell
```

#### Basic Operations
```python
df.head()         # First 5 rows
df.tail(2)        # Last 2 rows
df.shape          # (rows, columns)
df.columns        # Column names
df.dtypes         # Data types
df.info()         # Overview
```

#### Adding/Modifying Columns
```python
df['Salary'] = [50000, 60000, 70000]  # New column
df['Age'] = df['Age'] + 1            # Modify column
```

#### Dropping Rows/Columns
```python
df.drop('City', axis=1, inplace=True)    # Drop column
df.drop(0, axis=0, inplace=True)         # Drop row
```

#### Iterating Over Rows
```python
for index, row in df.iterrows():
    print(row['Name'], row['Age'])
```

## Axis in Pandas
- `axis=0` → rows → operations down the rows  
- `axis=1` → columns → operations down the columns  

## Importing (Reading) a CSV File
```python
import pandas as pd
# Basic read
df = pd.read_csv('filename.csv')
```

### Optional Parameters
```python
pd.read_csv('file.csv', delimiter=',')        # custom delimiter
pd.read_csv('file.csv', index_col=0)          # set first column as index
pd.read_csv('file.csv', usecols=['Name', 'Age'])  # load only specific columns
```

## Exporting (Writing) to a CSV File
```python
# Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)  # index=False to avoid writing row numbers
```

### Optional Parameters
```python
df.to_csv('output.csv', sep=';')      # use semicolon instead of comma
df.to_csv('output.csv', header=False) # exclude column name
```

## Data Types
Data types (`dtypes`) define the kind of data a Series or DataFrame column holds. These are built on NumPy types, but pandas has its own higher-level types too.

| pandas dtype      | Python Equivalent | Usage Example            |
|-------------------|-------------------|--------------------------|
| `int64`           | `int`             | Whole numbers            |
| `float64`         | `float`           | Decimal numbers          |
| `object`          | `str`, mixed types| Text or mixed types      |
| `bool`            | `bool`            | True/False               |
| `datetime64[ns]`  | `datetime`        | Timestamps               |
| `timedelta[ns]`   | `timedelta`       | Differences between dates|
| `category`        | categorical       | Fixed set of values      |

### Checking Data Types
```python
df.dtypes          # Data type of each column
df['Age'].dtype    # Data type of specific column
```

### Changing Data Types
```python
df['Age'] = df['Age'].astype(float)        # Convert to float
df['IsActive'] = df['IsActive'].astype(bool)  # Convert to boolean
```

## Attributes vs Functions
Attributes and functions (methods) are used to interact with and analyze Series and DataFrames.

- **Attributes** → `.shape`, `.columns`, `.dtypes` → To get metadata/structure info  
- **Functions** → `.head()`, `.mean()`, `.drop()` → To perform actions or calculations  

### Common Attributes in Pandas
These do not require parentheses (they are properties, not functions).
- `df.shape`       # (rows, columns)
- `df.columns`     # Index of column names
- `df.index`       # Index of row labels
- `df.dtypes`      # Data types of each column
- `df.size`        # Total number of elements
- `df.ndim`        # Number of dimensions (usually 2 for DataFrame)
- `df.values`      # NumPy array of the data

### Common Functions/Methods in Pandas
These require parentheses, even if no arguments are passed.

#### Viewing and Exploring
- `df.head(n=5)`       # First n rows
- `df.tail(n=5)`       # Last n rows
- `df.info()`          # Summary (non-null counts, dtypes)
- `df.describe()`      # Stats summary for numeric columns

#### Selection
- `df['column']`              # Access a column
- `df[['col1', 'col2']]`      # Multiple columns
- `df.loc[0]`                 # Row by label
- `df.iloc[0]`                # Row by position
- `df.loc[0, 'column']`       # Specific cell

#### Modification
- `df['new_col'] = df['Age'] * 2`
- `df.drop('column', axis=1)`
- `df.rename(columns={'old': 'new'})`
- `df.fillna(0)`              # Fill NaNs
- `df.replace('N/A', 0)`

#### Aggregation
- `df.sum()`
- `df.mean()`
- `df.min()`
- `df.max()`
- `df.count()`
- `df.value_counts()`         # For Series

## .loc & .iloc
`.loc[]` and `.iloc[]` are two powerful indexers in pandas for selecting data from a DataFrame or Series.

- `.loc[]` → stands for **Location** → Uses → Labels (index/column names)  
- `.iloc[]` → stands for **Integer Location** → Uses → Integer positions (0, 1, 2)

### .loc[] — Label-Based Selection
```python
df.loc[<row_label>, <column_label>]
```

```python
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}, index=['a', 'b', 'c'])

print(df.loc['b'])              # Row with label 'b'
print(df.loc['a', 'Name'])      # Cell at row 'a' and column 'Name'
print(df.loc[['a', 'c'], ['Age']])  # Multiple rows/columns
```

### .iloc[] — Position-Based Selection
```python
df.iloc[<row_position>, <column_position>]
```

```python
print(df.iloc[1])               # Row at position 1
print(df.iloc[0, 1])            # Cell at row 0, column 1
print(df.iloc[0:2, 0:1])        # Slicing: first 2 rows, first column
```

#### Slicing with .iloc[] (Integer-location based)
```python
df.iloc[start:stop, start:stop]
```

- Works like Python list slicing  
- Stop is excluded  
- Uses integer positions (0-based index)  
- `.iloc[]` is like Python slices — `df.iloc[0:2]` excludes row 2.

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40]
})

print(df.iloc[0:2, 0:2])  # Rows 0 and 1, Columns 0 and 1
```

**Output**
```
    Name  Age
0  Alice   25
1    Bob   30
```

#### Slicing with .loc[] (Label-based)
```python
df.loc[start_label:stop_label, start_col:stop_col]
```

- Stop is included  
- Uses row and column labels (names, not positions)  
- `.loc[]` is inclusive on both ends in slicing: `df.loc['a':'c']` includes `'c'`.

```python
df.index = ['a', 'b', 'c', 'd']
print(df.loc['a':'c', 'Name':'Age'])  # Includes 'c' and 'Age'
```

**Output**
```
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35
```

## Pandas Plotting Functions

In pandas, you can easily create visualizations using built-in plotting functions, which are wrappers around Matplotlib. These are super handy for quick data exploration.

First, make sure to import matplotlib:
```python
import matplotlib.pyplot as plt
```

### Basic Syntax
```python
df.plot(kind='plot_type')
# or
df['column'].plot(kind='plot_type')
```

### Plot Types
| Plot Type       | kind value   | Best For                     |
|-----------------|--------------|------------------------------|
| Line            | 'line'       | Time series or trends        |
| Bar             | 'bar'        | Comparing categories         |
| Horizontal Bar  | 'barh'       | Horizontal bar charts        |
| Histogram       | 'hist'       | Distribution of values       |
| Box Plot        | 'box'        | Spread and outliers          |
| Area            | 'area'       | Cumulative totals            |
| Pie             | 'pie'        | Proportions                  |
| Scatter         | 'scatter'    | Relationship between 2 variables |
| KDE             | 'kde'        | Smooth distribution          |

### Extra Options
You can pass arguments like:
- `title='Chart Title'`
- `color='green'`
- `figsize=(8, 6)`
- `grid=True`

### Example
```python
df['Age'].plot(kind='hist', title='Age Distribution', color='skyblue', grid=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```