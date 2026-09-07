from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    f_oneway,
    linregress,
    iqr,
    median_abs_deviation,
    kurtosis,
    skew,
    poisson,
    binom,
    laplace,
)

import numpy as np


class Statistical:
    """Statistical analysis methods for layer-wise model data."""

    def pearson_correlation(self, *, data, kind="dual_weights"):
        """
        Calculate Pearson correlation between corresponding layers.

        Each layer is flattened independently before correlation is
        calculated, allowing N-dimensional layer arrays to be analyzed
        while preserving layer boundaries.

        Parameters
        ----------
        data : dict
            Layer-wise model data containing parameter groups.
        kind : str, default="dual_weights"
            Parameter groups to compare. Supported values are
            "dual_weights" and "dual_biases".

        Returns
        -------
        list[dict]
            Pearson correlation coefficient and p-value for each layer.

        Raises
        ------
        ValueError
            If no layer data is provided, the number of layers differs
            between parameter groups, or corresponding layers contain
            different numbers of elements.
        """
        x_values = []
        y_values = []
        result = []

        match kind:
            case "dual_weights":
                for key, _ in data.items():
                    if key == "weights_1":
                        for _, value in data[key].items():
                            x_values.append(value)
                    elif key == "weights_2":
                        for _, value in data[key].items():
                            y_values.append(value)

            case "dual_biases":
                for key, _ in data.items():
                    if key == "bias_1":
                        for _, value in data[key].items():
                            x_values.append(value)
                    elif key == "bias_2":
                        for _, value in data[key].items():
                            y_values.append(value)

            case _:
                raise ValueError(
                    'Expected kind to be "dual_weights" or "dual_biases".'
                )

        if not x_values or not y_values:
            raise ValueError(
                "Expected data to contain two non-empty parameter groups."
            )

        if len(x_values) != len(y_values):
            raise ValueError(
                "Expected both parameter groups to contain the same number of layers."
            )

        for layer, (x_layer, y_layer) in enumerate(
            zip(x_values, y_values)
        ):
            x_layer = np.asarray(x_layer).ravel()
            y_layer = np.asarray(y_layer).ravel()

            if x_layer.size != y_layer.size:
                raise ValueError(
                    f"Expected corresponding layers to contain the same "
                    f"number of elements. Layer {layer} contains "
                    f"{x_layer.size} and {y_layer.size} elements."
                )

            correlation, p_value = pearsonr(x_layer, y_layer)

            result.append(
                {
                    "layer": layer,
                    "R_value": correlation,
                    "p_value": p_value,
                }
            )

        return result

    def covariance(self, *, data, kind="dual_weights"):
        """
        Calculate covariance between corresponding layers.

        Each layer is flattened independently before covariance is
        calculated, allowing N-dimensional layer arrays to be analyzed
        while preserving layer boundaries.

        Parameters
        ----------
        data : dict
            Layer-wise model data containing parameter groups.
        kind : str, default="dual_weights"
            Parameter groups to compare. Supported values are
            "dual_weights" and "dual_biases".

        Returns
        -------
        list[dict]
            Covariance matrix for each layer.

        Raises
        ------
        ValueError
            If no layer data is provided, the number of layers differs
            between parameter groups, or corresponding layers contain
            different numbers of elements.
        """
        x_values = []
        y_values = []
        result = []

        match kind:
            case "dual_weights":
                for key, _ in data.items():
                    if key == "weights_1":
                        for _, value in data[key].items():
                            x_values.append(value)
                    elif key == "weights_2":
                        for _, value in data[key].items():
                            y_values.append(value)

            case "dual_biases":
                for key, _ in data.items():
                    if key == "bias_1":
                        for _, value in data[key].items():
                            x_values.append(value)
                    elif key == "bias_2":
                        for _, value in data[key].items():
                            y_values.append(value)

            case _:
                raise ValueError(
                    'Expected kind to be "dual_weights" or "dual_biases".'
                )

        if not x_values or not y_values:
            raise ValueError(
                "Expected data to contain two non-empty parameter groups."
            )

        if len(x_values) != len(y_values):
            raise ValueError(
                "Expected both parameter groups to contain the same number of layers."
            )

        for layer, (x_layer, y_layer) in enumerate(
            zip(x_values, y_values)
        ):
            x_layer = np.asarray(x_layer).ravel()
            y_layer = np.asarray(y_layer).ravel()

            if x_layer.size != y_layer.size:
                raise ValueError(
                    f"Expected corresponding layers to contain the same "
                    f"number of elements. Layer {layer} contains "
                    f"{x_layer.size} and {y_layer.size} elements."
                )

            covariance_value = np.cov(x_layer, y_layer)

            result.append(
                {
                    "layer": layer,
                    "Result_value": covariance_value,
                }
            )

        return result

    def rank_correlation(
        self,
        *,
        kind="spearman",
        data,
        rel_kind="dual_weights",
    ):
        """
        Calculate Spearman or Kendall rank correlation layer-wise.

        Each layer is flattened independently before correlation is
        calculated, allowing N-dimensional layer arrays to be analyzed
        while preserving layer boundaries.

        Parameters
        ----------
        kind : str, default="spearman"
            Correlation method. Supported values are "spearman" and
            "kendall".
        data : dict
            Layer-wise model data.
        rel_kind : str, default="dual_weights"
            Parameter groups to compare. Supported values are
            "dual_weights" and "dual_biases".

        Returns
        -------
        list[dict]
            Rank correlation result for each layer.

        Raises
        ------
        ValueError
            If an unsupported correlation method or relationship is
            provided, the parameter groups are empty, the number of
            layers differs, or corresponding layers contain different
            numbers of elements.
        """
        x_values = []
        y_values = []
        result = []

        match rel_kind:
            case "dual_weights":
                for key in data.keys():
                    if key == "weights_1":
                        for value in data[key].values():
                            x_values.append(value)
                    elif key == "weights_2":
                        for value in data[key].values():
                            y_values.append(value)

            case "dual_biases":
                for key in data.keys():
                    if key == "bias_1":
                        for value in data[key].values():
                            x_values.append(value)
                    elif key == "bias_2":
                        for value in data[key].values():
                            y_values.append(value)

            case _:
                raise ValueError(
                    'Expected rel_kind to be "dual_weights" or "dual_biases".'
                )

        if not x_values or not y_values:
            raise ValueError(
                "Expected data to contain two non-empty parameter groups."
            )

        if len(x_values) != len(y_values):
            raise ValueError(
                "Expected both parameter groups to contain the same number of layers."
            )

        match kind:
            case "spearman":
                for layer, (x_layer, y_layer) in enumerate(
                    zip(x_values, y_values)
                ):
                    x_layer = np.asarray(x_layer).ravel()
                    y_layer = np.asarray(y_layer).ravel()

                    if x_layer.size != y_layer.size:
                        raise ValueError(
                            f"Expected corresponding layers to contain the "
                            f"same number of elements. Layer {layer} contains "
                            f"{x_layer.size} and {y_layer.size} elements."
                        )

                    correlation = spearmanr(x_layer, y_layer)

                    result.append(
                        {
                            "layer": layer,
                            "Result_value": correlation,
                        }
                    )

                return result

            case "kendall":
                for layer, (x_layer, y_layer) in enumerate(
                    zip(x_values, y_values)
                ):
                    x_layer = np.asarray(x_layer).ravel()
                    y_layer = np.asarray(y_layer).ravel()

                    if x_layer.size != y_layer.size:
                        raise ValueError(
                            f"Expected corresponding layers to contain the "
                            f"same number of elements. Layer {layer} contains "
                            f"{x_layer.size} and {y_layer.size} elements."
                        )

                    correlation = kendalltau(x_layer, y_layer)

                    result.append(
                        {
                            "layer": layer,
                            "Result_value": correlation,
                        }
                    )

                return result

            case _:
                raise ValueError(
                    'Expected kind to be "spearman" or "kendall".'
                )

    def anova(self, *, data):
        """
        Perform one-way ANOVA across multiple groups.

        Parameters
        ----------
        data : list or tuple
            Each element represents one statistical group.

        Returns
        -------
        tuple
            F-statistic and p-value.
        """
        f_statistic, p_value = f_oneway(*data)
        return f_statistic, p_value

    def linear_regression(self, *, data, kind="dual_weights"):
        """
        Perform linear regression between corresponding layers.

        Each layer is flattened independently before regression is
        calculated, allowing N-dimensional layer arrays to be analyzed
        while preserving layer boundaries.

        Parameters
        ----------
        data : dict
            Layer-wise model data.
        kind : str, default="dual_weights"
            Parameter groups to compare. Supported values are
            "dual_weights" and "dual_biases".

        Returns
        -------
        list[dict]
            Linear regression result for each layer.

        Raises
        ------
        ValueError
            If no layer data is provided, the number of layers differs
            between parameter groups, or corresponding layers contain
            different numbers of elements.
        """
        x_values = []
        y_values = []
        result = []

        match kind:
            case "dual_weights":
                for key in data.keys():
                    if key == "weights_1":
                        for value in data[key].values():
                            x_values.append(value)
                    elif key == "weights_2":
                        for value in data[key].values():
                            y_values.append(value)

            case "dual_biases":
                for key in data.keys():
                    if key == "bias_1":
                        for value in data[key].values():
                            x_values.append(value)
                    elif key == "bias_2":
                        for value in data[key].values():
                            y_values.append(value)

            case _:
                raise ValueError(
                    'Expected kind to be "dual_weights" or "dual_biases".'
                )

        if not x_values or not y_values:
            raise ValueError(
                "Expected data to contain two non-empty parameter groups."
            )

        if len(x_values) != len(y_values):
            raise ValueError(
                "Expected both parameter groups to contain the same number of layers."
            )

        for layer, (x_layer, y_layer) in enumerate(
            zip(x_values, y_values)
        ):
            x_layer = np.asarray(x_layer).ravel()
            y_layer = np.asarray(y_layer).ravel()

            if x_layer.size != y_layer.size:
                raise ValueError(
                    f"Expected corresponding layers to contain the same "
                    f"number of elements. Layer {layer} contains "
                    f"{x_layer.size} and {y_layer.size} elements."
                )

            regression_result = linregress(x_layer, y_layer)

            result.append(
                {
                    "layer": layer,
                    "Result_value": regression_result,
                }
            )

        return result

    def interquartile_range(self, *, data):
        """
        Calculate the interquartile range of the input data.

        Parameters
        ----------
        data : array-like
            Input observations.

        Returns
        -------
        np.ndarray or float
            Interquartile range.
        """
        return iqr(data)

    def median_absolute_deviation(self, *, data):
        """
        Calculate the median absolute deviation.

        Parameters
        ----------
        data : array-like
            Input observations.

        Returns
        -------
        np.ndarray or float
            Median absolute deviation.
        """
        return median_abs_deviation(data)

    def kurtosis(self, *, data):
        """
        Calculate the kurtosis of the input data.

        Parameters
        ----------
        data : array-like
            Input observations.

        Returns
        -------
        np.ndarray or float
            Kurtosis value.
        """
        return kurtosis(data)

    def skewness(self, *, data):
        """
        Calculate the skewness of the input data.

        Parameters
        ----------
        data : array-like
            Input observations.

        Returns
        -------
        np.ndarray or float
            Skewness value.
        """
        return skew(data)

    def poisson_distribution(self, *, data, mu):
        """
        Calculate Poisson distribution probabilities and samples.

        Parameters
        ----------
        data : array-like
            Values at which probabilities are evaluated.
        mu : float
            Expected number of events.

        Returns
        -------
        tuple
            Poisson CDF, PMF, and random samples.
        """
        cdf = poisson.cdf(data, mu)
        pmf = poisson.pmf(data, mu)
        samples = poisson.rvs(mu, size=len(data))

        return cdf, pmf, samples

    def binomial_distribution(self, *, data, n, p):
        """
        Calculate Binomial distribution probabilities and samples.

        Parameters
        ----------
        data : array-like
            Values at which probabilities are evaluated.
        n : int
            Number of trials.
        p : float
            Probability of success.

        Returns
        -------
        tuple
            Binomial CDF, PMF, and random samples.
        """
        cdf = binom.cdf(data, n, p)
        pmf = binom.pmf(data, n, p)
        samples = binom.rvs(n, p, size=len(data))

        return cdf, pmf, samples

    def laplace_fit(self, *, data):
        """
        Fit a Laplace distribution to one-dimensional data.

        Parameters
        ----------
        data : array-like
            One-dimensional observations.

        Returns
        -------
        tuple
            Location and scale parameters.
        """
        loc, scale = laplace.fit(data)
        return loc, scale
