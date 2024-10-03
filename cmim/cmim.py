"""
This is a implementation of the Conditional Mutual Information Maximisation (cmim) feature selection algorithm.
cmim iteratively selects features by maximizing MI with a target variable, conditioned on previously selected
features.



Related cmim implementations:
https://github.com/CharlesGaydon/Fast-CMIM   (I based this off of that)
https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/CMIM.py

For the rationale of this algorithm and of the fast implementation, see:
http://www.ams.jhu.edu/~yqin/cvrg/Feature_Selection.htm#ch1_2
Original method article : 
http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors

def conditional_mutual_info(X, y, z, n_neighbors=3, random_state=None):
    """
    Estimate the conditional mutual information I(X; y | z) using the KSG estimator.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Feature variable. Continuous or discrete data.
    y : array-like, shape (n_samples,)
        Target variable. Continuous or discrete data.
    z : array-like, shape (n_samples,) or (n_samples, n_features)
        Conditioning variable(s). Continuous or discrete data.
    n_neighbors : int, default=3
        Number of nearest neighbors to use in the KNN density estimation.
        A small number captures fine-grained dependencies; a larger number
        provides smoother estimates.
    random_state : int or RandomState instance, default=None
        Determines random number generation for neighbor searches.
        Pass an int for reproducible results across multiple function calls.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information I(X; y | z).

    Notes
    -----
    - The estimation is based on the Kraskov-Stögbauer-Grassberger (KSG) estimator,
      suitable for continuous variables.
    - This function can handle mixed data types (continuous and discrete) without
      explicit discretization.

    References
    ----------
    - A. Kraskov, H. Stögbauer, and P. Grassberger. "Estimating mutual information."
      Physical Review E 69, 066138 (2004).
    """

    # Ensure inputs are numpy arrays
    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    z = np.asarray(z)

    # Concatenate variables
    xyz = np.hstack((X, y, z))
    xz = np.hstack((X, z))
    yz = np.hstack((y, z))
    z = np.asarray(z)

    # Number of samples
    n_samples = X.shape[0]

    # Set up nearest neighbors
    nn_xyz = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_xz = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_yz = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_z = NearestNeighbors(n_neighbors=n_neighbors + 1)

    # Fit the models
    nn_xyz.fit(xyz)
    nn_xz.fit(xz)
    nn_yz.fit(yz)
    nn_z.fit(z)

    # Distances
    dists_xyz, _ = nn_xyz.kneighbors(xyz, n_neighbors=n_neighbors + 1)
    dists_xz, _ = nn_xz.kneighbors(xz, n_neighbors=n_neighbors + 1)
    dists_yz, _ = nn_yz.kneighbors(yz, n_neighbors=n_neighbors + 1)
    dists_z, _ = nn_z.kneighbors(z, n_neighbors=n_neighbors + 1)

    # Exclude the point itself (distance zero)
    eps = np.finfo(float).eps
    dists_xyz = dists_xyz[:, n_neighbors] - eps
    dists_xz = dists_xz[:, n_neighbors] - eps
    dists_yz = dists_yz[:, n_neighbors] - eps
    dists_z = dists_z[:, n_neighbors] - eps

    # Counts
    nxz = np.sum(nn_xz.radius_neighbors_graph(xz, radius=dists_xyz, mode='connectivity'), axis=1) - 1
    nyz = np.sum(nn_yz.radius_neighbors_graph(yz, radius=dists_xyz, mode='connectivity'), axis=1) - 1
    nz = np.sum(nn_z.radius_neighbors_graph(z, radius=dists_xyz, mode='connectivity'), axis=1) - 1

    # Avoid log of zero
    nxz = np.maximum(nxz, 1)
    nyz = np.maximum(nyz, 1)
    nz = np.maximum(nz, 1)

    # Compute conditional mutual information
    psi = lambda x: np.log(x)
    cmi = np.mean(psi(nxz) + psi(nyz) - psi(n_neighbors) - psi(nz))

    return cmi



class CMIMFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Conditional Mutual Information Maximization (cmim) feature selector.

    This transformer selects features based on the cmim algorithm, which iteratively
    selects features that provide the most additional information about the target
    variable, conditioned on previously selected features.

    Parameters
    ----------
    n_features_to_select : int, default=None
        The number of features to select. If None, all features are selected.
    task : str, default='classification'
        The type of task to perform:
        - 'classification' for classification tasks.
        - 'regression' for regression tasks.
    n_neighbors : int, default=3
        Number of neighbors to use for mutual information estimation.
        Affects the bias-variance trade-off of the estimator.
    n_jobs : int, default=1
        Number of jobs to run in parallel during mutual information estimation.
        -1 means using all processors.
    random_state : int or RandomState instance, default=None
        Determines random number generation for neighbor searches.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    selected_features_ : ndarray of shape (n_features_to_select,)
        Indices of the selected features.
    mi_scores_ : ndarray of shape (n_features,)
        Mutual information between each feature and the target variable.
    cmi_scores_ : ndarray of shape (n_features,)
        Conditional mutual information scores for each feature.
    feature_scores_ : ndarray of shape (n_features,)
        Scores of features during the selection process (minimal CMI).

    Examples
    --------
    >>> from cmim import CMIMFeatureSelector
    >>> selector = CMIMFeatureSelector(n_features_to_select=5, task='classification')
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)

    Notes
    -----
    - The selector supports both continuous and discrete data without explicit discretization.
    - Integrates seamlessly with scikit-learn pipelines and estimators.

    References
    ----------
    - F. Fleuret. "Fast Binary Feature Selection with Conditional Mutual Information."
      Journal of Machine Learning Research, 5, 1531-1555, 2004.
    - scikit-learn mutual information estimators:
      https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    """
    def __init__(self, n_features_to_select=None, task='classification', n_neighbors=3, n_jobs=-1, random_state=None):
        self.n_features_to_select = n_features_to_select
        self.task = task
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the cmim feature selector to the data.

        Parameters
        ----------
        X : array-like or pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_samples_, self.n_features_ = X.shape

        if self.n_features_to_select is None:
            self.n_features_to_select = self.n_features_

        if self.n_features_to_select > self.n_features_:
            raise ValueError("Cannot select more features than available")

        self.selected_features_ = []
        self.feature_scores_ = np.zeros(self.n_features_)  # Stores CMI scores for all features
        self.mi_scores_ = np.zeros(self.n_features_)       # Stores initial MI scores for all features
        self.m_ = -np.ones(self.n_features_, dtype=int)
        self.t1_ = np.zeros(self.n_features_)

        # Initialize mutual information between each feature and the target
        if self.task == 'classification':
            mi = mutual_info_classif(X, y, n_neighbors=self.n_neighbors, random_state=self.random_state)
        elif self.task == 'regression':
            mi = mutual_info_regression(X, y, n_neighbors=self.n_neighbors, random_state=self.random_state)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

        self.t1_ = mi.copy()
        self.mi_scores_ = mi.copy()

        # Cache for conditional mutual information
        self.cmi_cache_ = {}

        # Main cmim algorithm
        for k in range(self.n_features_to_select):
            # Select the feature with the highest t1
            idx = np.argmax(self.t1_)
            self.selected_features_.append(idx)
            self.feature_scores_[idx] = self.t1_[idx]
            self.t1_[idx] = -np.inf  # Exclude this feature from further selection

            # Update t1 for remaining features
            features_remaining = [i for i in range(self.n_features_) if i not in self.selected_features_]

            for i in features_remaining:
                while self.t1_[i] > self.feature_scores_[idx] and self.m_[i] < len(self.selected_features_) - 1:
                    self.m_[i] += 1
                    j = self.selected_features_[self.m_[i]]

                    # Compute conditional mutual information I(f_i; y | f_j)
                    key = (i, j)
                    if key in self.cmi_cache_:
                        cmi = self.cmi_cache_[key]
                    else:
                        cmi = conditional_mutual_info(
                            X[:, i], y, X[:, j],
                            n_neighbors=self.n_neighbors,
                            random_state=self.random_state
                        )
                        self.cmi_cache_[key] = cmi

                    # Update t1 with the minimal CMI
                    self.t1_[i] = min(self.t1_[i], cmi)

                # Update feature scores for analysis
                self.feature_scores_[i] = self.t1_[i]

        self.selected_features_ = np.array(self.selected_features_)

        # Store the final CMI scores
        self.cmi_scores_ = self.feature_scores_.copy()

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like or pandas DataFrame of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_to_select)
            The input samples with only the selected features.
        """
        # Validate input
        X = check_array(X, accept_sparse=False)
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape [# input features],
            in which an element is True iff its corresponding feature is selected for retention.
            If `indices` is True, this is an integer array of shape [# output features]
            whose values are indices into the input feature vector.
        """
        mask = np.zeros(self.n_features_, dtype=bool)
        mask[self.selected_features_] = True
        if indices:
            return self.selected_features_
        else:
            return mask

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. If None, default names are used (e.g., "x0", "x1", ...).

        Returns
        -------
        output_feature_names : ndarray of shape (n_features_to_select,)
            Transformed feature names.
        """
        if input_features is None:
            input_features = ['x%d' % i for i in range(self.n_features_)]
        return np.array(input_features)[self.selected_features_]


