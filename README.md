# CMIM Feature Selector
A  scikit-learn friendly python implementation of Conditional Mutual Information Maximization (CMIM) feature selection

[![PyPI version](https://badge.fury.io/py/cmim-feature-selector.svg)](https://badge.fury.io/py/cmim-feature-selector)

An efficient, scikit-learn compatible implementation of the **Conditional Mutual Information Maximization (CMIM)** algorithm for feature selection in machine learning, supporting both classification and regression tasks with continuous and discrete variables.

CMIM (Conditional Mutual Information Maximization) is a feature selection algorithm that iteratively selects features by maximizing the conditional mutual information between each candidate feature and the target variable, conditioned on the features already selected. It aims to choose features that provide the most additional information about the target, reducing redundancy by considering the information overlap with previously selected features.
---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Using with scikit-learn Pipeline](#using-with-scikit-learn-pipeline)
  - [Analyzing Feature Scores](#analyzing-feature-scores)
- [Parameters](#parameters)
- [Examples](#examples)
  - [Classification Example with Titanic Dataset](#classification-example-with-titanic-dataset)
- [Caveats and Considerations](#caveats-and-considerations)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction

**CMIM Feature Selector** is a Python library that implements the Conditional Mutual Information Maximization (CMIM) algorithm for feature selection. Feature selection is a crucial step in machine learning pipelines to enhance model performance, reduce overfitting, and improve interpretability.

The CMIM algorithm selects features that provide the most additional information about the target variable, conditioned on previously selected features. This implementation is designed to be:

- **Efficient**: Optimized for performance with caching and vectorization.
- **Flexible**: Supports continuous and discrete variables without the need for discretization.
- **Compatible**: Integrates seamlessly with scikit-learn estimators and pipelines.
- **Informative**: Provides mutual information (MI) and conditional mutual information (CMI) scores for each feature for deeper analysis.

---

## Key Features

- **Scikit-learn Compatibility**: Seamless integration with scikit-learn transformers and pipelines.
- **Continuous and Discrete Data Support**: Handles various data types without explicit discretization.
- **Efficient Mutual Information Estimation**: Utilizes k-nearest neighbors (KNN) estimators for accurate mutual information computation. i.e., you can use continous and discrete input features without needing to preprocess/discretize.
- **Feature Analysis**: Access to MI and CMI scores for all features.

---

## Installation

You can install the package via cloning the repository and install manually:

```bash
git clone https://github.com/ddofer/cmim.git
cd cmim
python setup.py install
```

---

## Usage

### Basic Example

```python
from cmim import CMIMFeatureSelector

# Assuming X is your feature matrix and y is the target vector
selector = CMIMFeatureSelector(n_features_to_select=5, task='classification')
selector.fit(X, y)
X_transformed = selector.transform(X)

# Get selected feature indices
selected_indices = selector.get_support(indices=True)

# Get mutual information and conditional mutual information scores
mi_scores = selector.mi_scores_
cmi_scores = selector.cmi_scores_
```

### Using with scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from cmim import CMIMFeatureSelector

pipeline = Pipeline([
    ('feature_selection', CMIMFeatureSelector(n_features_to_select=5, task='classification')),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)
```

### Analyzing Feature Scores

```python
import pandas as pd

# Assuming feature_names is a list of feature names
analysis_df = pd.DataFrame({
    'Feature': feature_names,
    'MI Score': selector.mi_scores_,
    'CMI Score': selector.cmi_scores_
})

# Sort features by their MI scores
analysis_df_sorted = analysis_df.sort_values(by='MI Score', ascending=False)
print(analysis_df_sorted)
```

---

## Parameters

- **`n_features_to_select`**: `int`, default=None  
  The number of features to select. If `None`, all features are selected.

- **`task`**: `str`, default='classification'  
  The type of task to perform:
  - `'classification'` for classification tasks.
  - `'regression'` for regression tasks.

- **`n_neighbors`**: `int`, default=3  
  Number of neighbors to use for mutual information estimation. Affects the bias-variance trade-off.

- **`n_jobs`**: `int`, default=-1  
  Number of jobs to run in parallel during mutual information estimation. `-1` means using all processors.

- **`random_state`**: `int` or `RandomState` instance, default=None  
  Determines random number generation for neighbor searches. Pass an int for reproducible results.

---

## Examples

### Classification Example with Titanic Dataset

```python
import pandas as pd
import seaborn as sns
from cmim import CMIMFeatureSelector

# Load the Titanic dataset
data = sns.load_dataset('titanic')

# Drop rows with missing target values
data = data.dropna(subset=['survived'])

# Separate features and target
X = data.drop(columns=['survived'])
y = data['survived']

# Fill missing numerical values with median
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

# Fill missing categorical values with mode
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=False)

# Instantiate and fit the selector
selector = CMIMFeatureSelector(n_features_to_select=5, task='classification')
selector.fit(X, y)

# Transform the dataset
X_transformed = selector.transform(X)

# Get selected feature names
selected_features = selector.get_feature_names_out(input_features=X.columns)
print("Selected features:", selected_features)

# Analyze feature scores
analysis_df = pd.DataFrame({
    'Feature': X.columns,
    'MI Score': selector.mi_scores_,
    'CMI Score': selector.cmi_scores_
}).sort_values(by='MI Score', ascending=False)

print(analysis_df)
```

---

## Caveats and Considerations

- **Computational Complexity**: Estimating mutual information using KNN can be computationally intensive for large datasets. Consider subsampling or dimensionality reduction techniques for very large datasets.

- **Choice of `n_neighbors`**: The `n_neighbors` parameter affects the bias-variance trade-off in mutual information estimation:
  - **Smaller values** capture finer details but may introduce variance.
  - **Larger values** provide smoother estimates but may miss subtle relationships.

- **Data Preprocessing**: Ensure that missing values are appropriately handled, and data types are compatible (numerical). The algorithm assumes that `X` and `y` do not contain missing values.

- **Random State**: For reproducible results, set the `random_state` parameter.

- **Dependencies**: The package relies on scikit-learn's mutual information estimators and nearest neighbors implementation.


## License

This project is licensed under the Open Apache License. See the [LICENSE](LICENSE) file for details.
Please cite me (Dan Ofer) if you use it!

---

## References

- **Fleuret, F.** (2004). [Fast Binary Feature Selection with Conditional Mutual Information](http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf). *Journal of Machine Learning Research*, 5, 1531-1555.
- **Kraskov, A., Stögbauer, H., & Grassberger, P.** (2004). [Estimating mutual information](https://arxiv.org/abs/cond-mat/0305641). *Physical Review E*, 69(6), 066138.
- **scikit-learn Documentation**:
  - [Mutual Information in Classification](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
  - [Mutual Information in Regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)
