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
        Calculate Pearson correlation for corresponding layers.

        Parameters
        ----------
        data : dict
            Layer-wise model data containing parameter groups.
        kind : str, default="dual_weights"
            Parameter groups to compare. One of "dual_weights" or
            "dual_biases".

        Returns
        -------
        list[dict]
            Pearson correlation coefficient and p-value for each layer.
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

        if isinstance(x_values[0], np.ndarray):
            for layer, (x_layer, y_layer) in enumerate(
                zip(x_values, y_values)
            ):
                correlation, p_value = pearsonr(x_layer, y_layer)

                result.append(
                    {
                        "layer": layer,
                        "R_value": correlation,
                        "p_value": p_value,
                    }
                )
        else:
            raise AttributeError(
                "You need to pass at least two layers of arrays, not a single one."
            )

        return result

    def covariance(self, *, data, kind="dual_weights"):
        """
        Calculate covariance for corresponding layers.

        Parameters
        ----------
        data : dict
            Layer-wise model data containing parameter groups.
        kind : str, default="dual_weights"
            Parameter groups to compare. One of "dual_weights" or
            "dual_biases".

        Returns
        -------
        list[dict]
            Covariance matrix for each layer.
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

        if isinstance(x_values[0], np.ndarray):
            for layer, (x_layer, y_layer) in enumerate(
                zip(x_values, y_values)
            ):
                covariance_value = np.cov(x_layer, y_layer)

                result.append(
                    {
                        "layer": layer,
                        "Result_value": covariance_value,
                    }
                )
        else:
            raise AttributeError(
                "You need to pass at least two layers of arrays, not a single one."
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

        Parameters
        ----------
        kind : str, default="spearman"
            Correlation method. Either "spearman" or "kendall".
        data : dict
            Layer-wise model data.
        rel_kind : str, default="dual_weights"
            Parameter groups to compare. One of "dual_weights" or
            "dual_biases".

        Returns
        -------
        list[dict]
            Correlation coefficient for each layer.
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

        match kind:
            case "spearman":
                for layer, (x_layer, y_layer) in enumerate(
                    zip(x_values, y_values)
                ):
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
                    correlation = kendalltau(x_layer, y_layer)

                    result.append(
                        {
                            "layer": layer,
                            "Result_value": correlation,
                        }
                    )

                return result

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
        Perform linear regression for corresponding layers.

        Parameters
        ----------
        data : dict
            Layer-wise model data.
        kind : str, default="dual_weights"
            Parameter groups to compare. One of "dual_weights" or
            "dual_biases".

        Returns
        -------
        list[dict]
            Regression result for each layer.
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

        if isinstance(x_values[0], np.ndarray):
            for layer, (x_layer, y_layer) in enumerate(
                zip(x_values, y_values)
            ):
                regression_result = linregress(x_layer, y_layer)

                result.append(
                    {
                        "layer": layer,
                        "Result_value": regression_result,
                    }
                )
        else:
            raise AttributeError(
                "You need to pass at least two layers of arrays, not a single one."
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
