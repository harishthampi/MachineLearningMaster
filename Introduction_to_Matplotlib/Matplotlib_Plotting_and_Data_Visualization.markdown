# Matplotlib: Plotting and Data Visualization

Matplotlib is the most popular data visualization library in Python. It allows you to create charts, plots, and graphs to better understand your data or present your results. Matplotlib is part of the standard Python data stack (pandas, NumPy, Matplotlib, Jupyter). It has terrific integration with many other Python libraries. Pandas uses Matplotlib as a backend to help visualize data in DataFrames.

## 1. Importing Matplotlib

We'll start by importing `matplotlib.pyplot`.  
**Why pyplot?**  
Because `pyplot` is a submodule for creating interactive plots programmatically, `pyplot` is often imported as the alias `plt`.

```python
# Import matplotlib and matplotlib.pyplot
import matplotlib
import matplotlib.pyplot as plt
```

## 2. Two Ways of Creating Plots

There are two main ways of creating plots in Matplotlib:

- **`matplotlib.pyplot.plot()`** - Recommended for simple plots (e.g., x and y).
```python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.title("Simple Line Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
```

- **Object-Oriented API (e.g., `plt.subplots()`)** - Recommended for more complex plots, such as creating multiple plots on the same Figure.
```python
import matplotlib.pyplot as plt

# Create Figure and Axes objects
fig, ax = plt.subplots()

# Now use ax to plot
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_title("Simple Line Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.show()
```

## 3. Anatomy of a Matplotlib Plot

- **Figure**: The whole canvas/window.
- **Axes**: Can be multiple; the actual plot area where data goes.
- **Axis**: x-axis, y-axis.
- **Titles**: Titles for plot and axes.
- **Labels**: Labels for axes.
- **Ticks**: Markers on axes.
- **Legends**: Descriptions of plot elements.
- **Plot elements**: Lines, markers, bars, etc.

### Matplotlib Workflow
1. Import Matplotlib
```python
import matplotlib.pyplot as plt
```
2. Prepare Data
```python
x = [1, 2, 3, 4]
y = [1, 22, 33, 44]
```
3. Set up Plot (Figure and Axes)
```python
fig, ax = plt.subplots(figsize=(10, 10))  # width, height of the figure
```
4. Plot Data
```python
ax.plot(x, y)
```
5. Customization
```python
ax.set(title="Sample Data", xlabel="x-axis", ylabel="y-axis")
```
6. Save the Image
```python
fig.savefig("images/simple-plot.png")
```

## 4. Making the Most Common Types of Plots Using NumPy Arrays

### Creating a Line Plot
Line is the default type of visualization in Matplotlib. Unless specified otherwise, your plots will start out as lines. Line plots are great for seeing trends over time.

```python
# The default plot is line
x = np.linspace(0, 10, 100)
fig, ax = plt.subplots()
ax.plot(x, x**2)
```

### Creating a Scatter Plot
Scatter plots are great when you have many different individual data points and want to see how they interact with each other without being connected.

```python
fig, ax = plt.subplots()
ax.scatter(x, np.exp(x))
```

### Creating Bar Plots
Bar plots are great for visualizing different amounts of similar-themed items. For example, the sales of items at a Nut Butter Store. You can create vertical bar plots with `ax.bar()` and horizontal bar plots with `ax.barh()`.

```python
# You can make plots from a dictionary
nut_butter_prices = {'Almond butter': 10,
                     "Peanut butter": 8,
                     "Cashew butter": 12}
fig, ax = plt.subplots()
ax.bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax.set(title="Dan's Nut Butter Store", ylabel="Price ($)")
```

### Creating a Histogram Plot
Histogram plots are excellent for showing the distribution of data, such as the distribution of ages of a population or wages in a city.

```python
# Make some data from a normal distribution
x = np.random.randn(1000)  # pulls data from a normal distribution
fig, ax = plt.subplots()
ax.hist(x)
```

## 5. Creating Figures with Multiple Axes with Subplots

Subplots allow you to create multiple Axes on the same Figure (multiple plots within the same plot). Subplots are helpful because you start with one plot per Figure but can scale it up to more when necessary. The `nrow` and `ncol` parameters are multiplicative, meaning `plt.subplots(nrows=2, ncols=2)` will create a 2 × 2 = 4 total Axes.

### Option 1: Create 4 Subplots with Each Axes Having Its Own Variable Name
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
# Plot data to each axis
ax1.plot(x, x / 2)
ax2.scatter(np.random.random(10), np.random.random(10))
ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax4.hist(np.random.randn(1000))
```

**What's happening?**  
`plt.subplots(nrows=2, ncols=2)` creates a 2 × 2 grid of subplots (4 subplots in total). It returns:
- `fig`: The Figure object.
- `axes`: A 2D NumPy array of Axes objects shaped like the subplot layout, i.e., `axes[0,0]`, `axes[0,1]`, `axes[1,0]`, `axes[1,1]`.

So, axes are:
```python
array([[ax1, ax2],
       [ax3, ax4]])
```

### Option 2: Create 4 Subplots with a Single `ax` Variable
```python
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
# Index the ax variable to plot data
ax[0, 0].plot(x, x / 2)
ax[0, 1].scatter(np.random.random(10), np.random.random(10))
ax[1, 0].bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax[1, 1].hist(np.random.randn(1000))
```

## 6. Plotting Data Directly with Pandas

If you have a DataFrame:
```python
import pandas as pd
data = pd.DataFrame({
    'Age': [45, 50, 55, 60],
    'Cholesterol': [200, 220, 210, 250],
    'Target': [1, 0, 1, 0]
})
```

You can plot directly:
```python
data.plot(x='Age', y='Cholesterol', kind='scatter')
```

`.plot()` can do many kinds of plots, including:
- `'hexbin'` – Hexbin plot (for density)

If you want to color by `Target`:
```python
data.plot.scatter(x='Age', y='Cholesterol', c='Target', colormap='winter')
```

## 7. xlim and ylim

These control the view limits of your axes:
- `xlim` = x-axis range
- `ylim` = y-axis range

### Pyplot Style
```python
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
```

### Object-Oriented API
```python
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
```

### Example
```python
ax.set_xlim(40, 80)  # x-axis only shows ages from 40 to 80
ax.set_ylim(100, 400)  # y-axis only shows cholesterol from 100 to 400
```