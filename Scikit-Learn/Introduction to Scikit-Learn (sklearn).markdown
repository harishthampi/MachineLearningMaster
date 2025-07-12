# Introduction to Scikit-Learn (sklearn)

Scikit-learn, often referred to as sklearn, is an open-source machine learning library for the Python programming language. It is built on top of the NumPy, SciPy, and Matplotlib libraries.

## An End-to-End Scikit-Learn Workflow

1. Getting the data ready
2. Choose the right estimator/algorithm for our problems
3. Fit the model/algorithm and use it to make predictions on our data
4. Evaluating a model
5. Improve a model
6. Save and load a trained model

## 1. Getting Our Data Ready to Be Used with Machine Learning

Before you can train any machine learning model with scikit-learn, you need to have your data in the right shape and format. This step is often called data preprocessing or data cleaning.

### Three Main Things We Have to Do:
1. Split the data into features and labels (usually `X` & `y`)
2. Filling (also called imputing) or disregarding missing values
3. Converting non-numerical values to numerical values (also called feature encoding)

### Typical Steps in Data Preparation

#### a) Loading the Data
You load data from CSV, Excel, SQL, etc.

```python
import pandas as pd
data = pd.read_csv('data.csv')
```

#### b) Splitting Features and Target
You separate your inputs and outputs.

```python
X = data.drop('target', axis=1)  # Example using heart_disease csv
y = data['target']
```

#### c) Encoding Categorical Variables
Scikit-learn models need numeric data. Strings/categories must be converted.

- **Label Encoding** (for target)
- **One-Hot Encoding** (for features)

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]  # List the columns to transform
one_hot = OneHotEncoder()  # Turns each category into a separate binary column
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)],
                               remainder="passthrough")
transformed_X = transformer.fit_transform(X)
```

**Example**: Using OneHotEncoding + ColumnTransformer to convert categorical features into numbers.

```python
categorical_features = ["Make", "Colour", "Doors"]
```

For example, "Colour" = ["Red", "Blue"] becomes:
- "Red" → [1, 0]
- "Blue" → [0, 1]

#### When to Treat Integer Columns as Categorical
You treat integer columns as categorical when:
- The numbers represent labels, not measurements
- No natural ordering or scaling

**Examples**:
- Number of doors (e.g., 4)
- Zip codes

**When to NOT Encode Integers**:
Do not one-hot encode if the numbers have meaningful order or scale.

| Feature       | Meaning        | Encoding?         |
|---------------|----------------|-------------------|
| Age           | Years          | Keep as numeric   |
| Price         | In dollars     | Keep as numeric   |
| Horsepower    | Power output   | Keep as numeric   |
| Doors (2 vs 4)| Category label | Encode as categorical |

#### d) Handling Missing Values
Machine learning models can't handle missing values directly. You can:
- Remove them
- Fill (impute) them

##### Option 1: Fill Missing Data with Pandas

```python
# Fill the "Make" column
car_sales_missing["Make"].fillna("missing", inplace=True)
# Fill the "Colour" column
car_sales_missing["Colour"].fillna("missing", inplace=True)
# Fill the "Odometer" column
car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean(), inplace=True)
# Fill the "Doors" column
car_sales_missing["Doors"].fillna(4.0, inplace=True)
# Remove rows with missing Price value
car_sales_missing.dropna(inplace=True)
```

##### Option 2: Fill Missing Data with Scikit-Learn

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with 'missing' & numeric values with mean
categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
door_imputer = SimpleImputer(strategy='constant', fill_value=4)
odometer_imputer = SimpleImputer(strategy='mean')

# Define columns
categorical_features = ["Make", "Colour"]
door_feature = ["Doors"]
odometer_feature = ["Odometer (KM)"]

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([
    ('categorical_imputer', categorical_imputer, categorical_features),
    ('door_imputer', door_imputer, door_feature),
    ('odometer_imputer', odometer_imputer, odometer_feature)
])

# Transform the data
filled_X = imputer.fit_transform(X)
```

**Import the Classes**:
- `from sklearn.impute import SimpleImputer`: Automatically fills in (imputes) missing values in data, using a chosen strategy (like mean, median, or constant).
- `from sklearn.compose import ColumnTransformer`: Lets you apply different preprocessing steps to different columns in your data.

**Example**:
- **Categorical Imputer**:
  ```python
  categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
  ```
  - `strategy='constant'`: Always fills missing entries with the same value.
  - `fill_value='missing'`: Replaces any NaN with the string 'missing'.
  - Example:
    - Before: `['Red', NaN, 'Blue']`
    - After: `['Red', 'missing', 'Blue']`

- **Door Imputer**:
  ```python
  door_imputer = SimpleImputer(strategy='constant', fill_value=4)
  ```
  - Fills missing values with the constant 4.
  - Example:
    - Before: `[2, NaN, 5]`
    - After: `[2, 4, 5]`

- **Odometer Imputer**:
  ```python
  odometer_imputer = SimpleImputer(strategy='mean')
  ```
  - Calculates the mean of non-missing values in the column and fills missing entries with that mean.
  - Example:
    - Before: `[100000, NaN, 120000]`
    - Mean = 110000
    - After: `[100000, 110000, 120000]`

**Apply the Transformation**:
```python
filled_X_train = imputer.fit_transform(X_train)
filled_X_test = imputer.transform(X_test)
```

- **Fit Step**:
  - For "Odometer (KM)" with mean strategy: Calculates the mean of the Odometer (KM) column (ignoring NaNs).
  - For "Doors" with constant strategy: No learning needed (uses 4).
  - For "Make" and "Colour" with constant strategy: No learning needed (uses "missing").
  - These "learned" values (like the mean) are stored in the imputer object.

- **Transform Step**:
  - Applies rules to `X_train`:
    - "Make" and "Colour": Any missing → filled with "missing".
    - "Doors": Any missing → filled with 4.
    - "Odometer (KM)": Any missing → filled with the calculated mean.
  - Result: No missing values remain in `X_train`. Output is a NumPy array with all values filled.

- **Test Set**:
  ```python
  filled_X_test = imputer.transform(X_test)
  ```
  - Only transform! No fitting.
  - Uses the same rules learned from training data to prevent data leakage.

**Summary of Operations on car-sales-extended-missing-data.csv**:
1. Read the CSV file.
2. Checked the missing data count:
   ```python
   car_sales_missing.isna().sum()
   # Make: 49
   # Colour: 50
   # Odometer (KM): 50
   # Doors: 50
   # Price: 50
   ```
3. Drop rows with no labels (Price):
   ```python
   car_sales_missing.dropna(subset=["Price"], inplace=True)
   ```
   Updated missing data count:
   ```python
   # Make: 47
   # Colour: 48
   # Odometer (KM): 48
   # Doors: 47
   # Price: 0
   ```
4. Split the dataset into training and testing sets:
   ```python
   X = car_sales_missing.drop("Price", axis=1)
   y = car_sales_missing["Price"]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```
5. Create imputer and ColumnTransformer (as shown above).
6. Get transformed data arrays back into DataFrames:
   ```python
   car_sales_filled_train = pd.DataFrame(filled_X_train, columns=["Make", "Colour", "Doors", "Odometer (KM)"])
   car_sales_filled_test = pd.DataFrame(filled_X_test, columns=["Make", "Colour", "Doors", "Odometer (KM)"])
   ```
7. Convert to numeric values:
   ```python
   categorical_features = ["Make", "Colour", "Doors"]
   one_hot = OneHotEncoder()
   transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder='passthrough')
   transformed_X_train = transformer.fit_transform(car_sales_filled_train)
   transformed_X_test = transformer.transform(car_sales_filled_test)
   ```
8. Fit a model:
   ```python
   np.random.seed(42)
   from sklearn.ensemble import RandomForestRegressor
   model = RandomForestRegressor(n_estimators=100)
   model.fit(transformed_X_train, y_train)
   model.score(transformed_X_test, y_test)
   ```

## 2. Choosing the Right Estimator/Algorithm for a Problem

**Notes**:
- Scikit-learn refers to machine learning models/algorithms as **estimators**.
- **Classification problem**: Predicting a category (e.g., heart disease or not).
  - `clf` (short for classifier) is used as a classification estimator.
- **Regression problem**: Predicting a number (e.g., selling price of a car).
- Use the [Scikit-Learn Estimator Selection Map](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) to decide:
  - What kind of problem you have.
  - What model is a good starting point.
  - Whether you have a lot of data, whether it's linear, etc.

**Tidbit**:
1. If you have **structured data** (in the form of tables), use ensemble methods.
2. If you have **unstructured data** (like images, audio, text), use deep learning or transfer learning.

## 3. Fitting the Model/Algorithm on Our Data and Use It to Make Predictions

**Fitting the Model**:
- Train the model on `X_train` and `y_train` to learn patterns.
- Use the trained model to predict on `X_test` or new, unseen data.

**Two Ways to Make Predictions**:
1. `predict()`:
   ```python
   y_preds = model.predict(X_test)
   ```
   - Returns the most likely class label (for classification) or value (for regression).
   - **Note**: Data you predict on should be in the same shape and format as the training data.
   - Example of incorrect usage:
     ```python
     y_preds = model.predict(np.array([0, 2, 3, 5, 6]))  # Incorrect shapes
     ```
     - This causes a `ValueError: Expected 2D array, got 1D array instead`. Reshape data appropriately.

2. `predict_proba()`:
   - Returns probabilities for each class (for classification).
   - Example for binary classification:
     ```python
     # [0.2, 0.8] → 20% chance of class 0, 80% chance of class 1
     # [0.89, 0.11] → 89% chance of class 0, 11% chance of class 1
     ```
   - For multiclass classification (>2 classes):
     ```python
     # Example: [0.1, 0.3, 0.6] for 3 classes (sums to 1)
     ```

## 4. Evaluating a Machine Learning Model

**Three Ways to Evaluate Scikit-Learn Models/Estimators**:
1. Estimator's built-in `score()` method.
2. The `scoring` parameter.
3. Problem-specific metric functions.

### 4.1 Evaluating a Model with `score()` Method
- Most estimators have a `.score()` method, summarizing model performance.
- **Regression**: Returns **R²** (coefficient of determination).
  - R² ranges from negative infinity to 1.
  - Closer to 1 → better fit.
  - Closer to 0 → worse fit.
  - Negative R² → model is worse than predicting the mean.
- **Classification**: Returns **accuracy** (fraction of correct predictions).

### 4.2 Evaluating a Model Using the `scoring` Parameter
- Use `cross_val_score(model, X, y, cv=5)`:
  - Splits data into `k` folds (default 5).
  - Trains on `(k-1)` folds, tests on the remaining fold.
  - Returns an array of scores (one per fold).
  - Helps evaluate how well the model generalizes to unseen data.

**Default Metrics**:
- Classification: Accuracy
- Regression: R²

**Change the Metric**:
- Specify the `scoring` parameter in `cross_val_score()` to use any supported metric.

#### 4.2.1 Evaluation Metrics - Classification Model
1. **Accuracy**:
   - Proportion of correct predictions among total predictions.
   - Use when classes are balanced (roughly equal positives and negatives).
2. **Area under ROC (Receiver Operating Characteristic) Curve (AUC)**:
   - ROC curves compare true positive rate (TPR) vs. false positive rate (FPR) at different thresholds.
   - AUC tells how well the model distinguishes between classes (e.g., heart disease or not).
   - AUC = 1 → perfect model.
3. **Confusion Matrix**:
   - A table showing actual vs. predicted values.
   - Calculate:
     - **Precision** = TP / (TP + FP)
     - **Recall** = TP / (TP + FN)
     - **F1-Score** = Harmonic mean of precision and recall.
   - Use when you want detailed insight into types of errors.
4. **Classification Report**:
   - Summarizes multiple metrics per class:
     - **Precision**: Proportion of positive predictions that were correct.
     - **Recall**: Proportion of actual positives correctly classified.
     - **F1-Score**: Harmonic mean of precision and recall.
     - **Support**: Number of samples per class.
     - **Accuracy**: Proportion of correct predictions.
     - **Macro Avg**: Average precision, recall, and F1-score across classes (no class imbalance adjustment).
     - **Weighted Avg**: Weighted average by number of samples per class (favors majority class).

#### 4.2.2 Evaluation Metrics - Regression Model
1. **R² (Coefficient of Determination)**:
   - Compares model predictions to the mean of the targets.
   - Ranges from negative infinity to 1.
   - R² = 0 → model predicts the mean.
   - R² = 1 → perfect predictions.
   - Higher is better.
2. **Mean Absolute Error (MAE)**:
   - Average of absolute differences between predicted and actual values.
   - Lower is better.
3. **Mean Squared Error (MSE)**:
   - Average of squared differences between predicted and actual values.
   - Amplifies outliers (larger errors).
   - Lower is better.

**Which Regression Metric to Use?**:
- **R²**: Quick indication of model performance (like accuracy).
- **MAE**: Better for understanding average prediction error.
- **MSE**: Pay more attention when larger errors are significantly worse.
  - Example (house price prediction):
    - Use MAE if being $10,000 off is twice as bad as $5,000.
    - Use MSE if being $10,000 off is more than twice as bad as $5,000.

#### 4.2.3 Evaluating a Model Using the `scoring` Parameter
- Control which metric to compute during cross-validation with `scoring`.

**Popular Scoring Values**:
- **Classification**:
  - `'accuracy'`
  - `'roc_auc'`
  - `'f1'`
  - `'precision'`
  - `'recall'`
  - `'log_loss'`
- **Regression**:
  - `'r2'`
  - `'neg_mean_absolute_error'`
  - `'neg_mean_squared_error'`
  - `'neg_root_mean_squared_error'`

## 5. Improving a Machine Learning Model

The first predictions and evaluation metrics are **baseline predictions** and **baseline metrics**. The goal is to improve upon these.

### Two Main Methods:
1. **From a Data Perspective**:
   - Collect more data (more data generally improves model performance).
   - Improve data quality (e.g., better encoding, filling missing values).
2. **From a Model Perspective**:
   - Use a better model (e.g., move from simple to complex, like ensemble methods).
   - Tune hyperparameters to improve the current model.

**Note**: Machine learning models find **parameters** in data automatically, while **hyperparameters** are settings you adjust to improve pattern-finding.

### Three Ways to Adjust Hyperparameters:
1. **By Hand** (Manual Tuning):
   - Try different hyperparameter values and evaluate on a validation set.
2. **Randomly with RandomizedSearchCV**:
   - Randomly searches hyperparameter combinations.
   - Create a dictionary of parameter distributions:
     ```python
     from sklearn.model_selection import RandomizedSearchCV
     grid = {
         "n_estimators": [10, 100, 200, 500, 1000, 1200],
         "max_depth": [None, 5, 10, 20, 30]
     }
     ```
   - Specify the number of iterations to control combinations tested.
   - More efficient for large search spaces.
3. **Exhaustively with GridSearchCV**:
   - Searches every possible combination of hyperparameters.
   - Example:
     ```python
     from sklearn.model_selection import GridSearchCV
     grid_2 = {
         "n_estimators": [100, 200, 500],
         "max_depth": [None],
         "max_features": ["sqrt", "log2"],
         "min_samples_leaf": [6],
         "min_samples_split": [4]
     }
     gs_clf = GridSearchCV(estimator=clf, param_grid=grid_2, cv=5, verbose=2)
     ```

## 6. Saving and Loading Trained Machine Learning Models

**Two Ways to Save and Load Models**:
1. **With Python's `pickle` Module**:
   ```python
   import pickle
   pickle.dump(gs_clf, open("gs_random_forest_model.pkl", "wb"))
   ```
2. **With `joblib` Module**:
   ```python
   from joblib import dump
   dump(gs_clf, filename="gs_random_forest_model_joblib.joblib")
   ```

## 7. Pipeline

A **Pipeline** chains multiple steps into one object, typically:
- Data preprocessing (e.g., scaling, encoding)
- Feature selection
- A model

**Why Use a Pipeline?**:
- Keeps code clean and reproducible.
- Prevents data leakage (e.g., accidentally scaling using test data).
- Easier to cross-validate the entire workflow.
- Supports hyperparameter tuning for all steps.

**Example Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Define different features and transformer pipelines
categorical_features = ["Make", "Colour"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehotencoder", OneHotEncoder(handle_unknown="ignore"))
])

door_feature = ["Doors"]
door_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))
])

odometer_feature = ["Odometer (KM)"]
odometer_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

# Create a preprocessing and modeling pipeline
preprocessor = ColumnTransformer([
    ("categorical_transformer", categorical_transformer, categorical_features),
    ("door_transformer", door_transformer, door_feature),
    ("odometer_transformer", odometer_transformer, odometer_feature)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])
```