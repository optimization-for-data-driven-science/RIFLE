import collections
import numbers
from itertools import chain
from itertools import combinations_with_replacement as combinations_w_r
import numpy as np
from scipy.special import comb

from .validation import _check_feature_names_in


class PolyFeatures:
    """ Generate interaction and polynomial features. Altered version of
    sklearn.preprocessing.PolynomialFeatures to preserve NaN values.

    Parameters
    ----------
    degree : int, default=2
    Maximum degree of the polynomial features.

    include_bias : bool, default=True
    If 'True', then include the bias column, the feature in which all
    polynomial powers are zero (acts as an intercept term in a linear
    model.
    """

    def __init__(self, degree=2, *, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    @staticmethod
    def _combinations(n_features, min_degree, max_degree, include_bias):
        comb = combinations_w_r
        start = max(1, min_degree)
        iter = chain.from_iterable(
            comb(range(n_features), i) for i in range(start, max_degree + 1)
        )
        if include_bias:
            iter = chain(comb(range(n_features), 0), iter)
        return iter

    @staticmethod
    def _num_combinations(n_features, min_degree, max_degree, include_bias):
        """
        Calculate number of terms in polynomial expansion.

        """
        combinations = comb(n_features + max_degree, max_degree, exact=True) - 1
        if min_degree > 0:
            d = min_degree - 1
            combinations -= comb(n_features + d, d, exact=True) - 1

        if include_bias:
            combinations += 1

        return combinations

    @property
    def powers_(self):
        """
        Exponent for each of the inputs in the output.

        """
        combinations = self._combinations(
            n_features=self.n_features_in_,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            include_bias=self.include_bias,
        )
        return np.vstack(
            [np.bincount(c, minlength=self.n_features_in_) for c in combinations]
        )

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array of str objects or None, default=None
        Input features.

        Returns
        -------
        feature_names : ndarray of str objects
        Transformed feature names.
        """
        powers = self.powers_
        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    "%s^%d" % (input_features[ind], exp)
                    if exp != 1
                    else input_features[ind]
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.asarray(feature_names, dtype=object)

    def fit(self, X):
        """
        Compute number of output features.

        Parameters
        ----------
        X : array-like matrix of shape (n_samples, n_features)
        The data.

        Returns
        -------
        self : object
        Fitted transformer.
        """
        _, n_features = X.shape
        self.n_features_in_ = n_features
        if isinstance(self.degree, numbers.Integral):
            if self.degree < 0:
                raise ValueError(
                    f"degree must be a non-negative integer, got {self.degree}."
                )
            elif self.degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting degree to zero and include_bias to False would result in"
                    " an empty output array."
                )

            self._min_degree = 0
            self._max_degree = self.degree
        elif (
                isinstance(self.degree, collections.abc.Iterable) and len(self.degree) == 2
        ):
            self._min_degree, self._max_degree = self.degree
            if not (
                    isinstance(self._min_degree, numbers.Integral)
                    and isinstance(self._max_degree, numbers.Integral)
                    and self._min_degree >= 0
                    and self._min_degree <= self._max_degree
            ):
                raise ValueError(
                    "degree=(min_degree, max_degree) must "
                    "be non-negative integers that fulfil "
                    "min_degree <= max_degree, got "
                    f"{self.degree}."
                )
            elif self._max_degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting both min_deree and max_degree to zero and include_bias to"
                    " False would result in an empty output array."
                )
        else:
            raise ValueError(
                "degree must be a non-negative int or tuple "
                "(min_degree, max_degree), got "
                f"{self.degree}."
            )

        self.n_output_features_ = self._num_combinations(
            n_features=n_features,
            min_degree=self._min_degree,
            max_degree=self._max_degree,
            include_bias=self.include_bias,
        )
        # We also record the number of output features for
        # _max_degree = 0
        self._n_out_full = self._num_combinations(
            n_features=n_features,
            min_degree=0,
            max_degree=self._max_degree,
            include_bias=self.include_bias,
        )

        return self

    def transform(self, X):
        """
        Transform data to polynomial features.

        Parameters
        ----------
        X : array-like matrix of shape (n_samples, n_features)
        The data to transform.

        Returns
        -------
        XP : ndarray matrix of shape (n_samples, NP)
        The matrix of features, where NP is the number of polynomial features
        generated from the combination of inputs.
        """
        n_samples, n_features = X.shape
        # Do as if _min_degree = 0 and cut down array after the
        # computation, i.e. use _n_out_full instead of n_output_features_.

        XP = np.empty(shape=(n_samples, self._n_out_full),
                      dtype=X.dtype)

        # degree 0 term
        if self.include_bias:
            XP[:, 0] = 1
            current_col = 1
        else:
            current_col = 0

        if self._max_degree == 0:
            return XP

        # degree 1 term
        XP[:, current_col: current_col + n_features] = X
        index = list(range(current_col, current_col + n_features))
        current_col += n_features
        index.append(current_col)

        # loop over degree >= 2 terms
        for _ in range(2, self._max_degree + 1):
            new_index = []
            end = index[-1]
            for feature_idx in range(n_features):
                start = index[feature_idx]
                new_index.append(current_col)
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                # multiply
                np.multiply(
                    XP[:, start:end],
                    X[:, feature_idx: feature_idx + 1],
                    out=XP[:, current_col:next_col],
                    casting="no",
                )
                # print(XP[:, start:end])
                # print(X[:, feature_idx: feature_idx + 1])
                # print(XP[:, current_col:next_col])
                # print('-----')
                current_col = next_col

            new_index.append(current_col)
            index = new_index

        if self._min_degree > 1:
            n_XP, n_Xout = self._n_out_full, self.n_output_features_
            if self.include_bias:
                Xout = np.empty(
                    shape=(n_samples, n_Xout), dtype=XP.dtype, order=self.order
                )
                Xout[:, 0] = 1
                Xout[:, 1:] = XP[:, n_XP - n_Xout + 1:]
            else:
                Xout = XP[:, n_XP - n_Xout:].copy()
            XP = Xout

        return XP
