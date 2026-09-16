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
    laplace
)
from typing import Literal
import numpy as np


class Statistical:
    """Statistical analysis methods for layer-wise model data."""

    def pearson_correlation(self, *, data, kind:Literal["dual_weights","dual_biases","non_grouped"]="dual_weights"):
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
                    if key == "weight_1":
                        for _, value in data[key].items():
                            x_values.append(value)
                    elif key == "weight_2":
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
            case "non_grouped":
                pass
            case _:
                raise ValueError(
                    'Expected kind to be "dual_weights" , "dual_biases" or "NON GROUPED".'
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

    def covariance(self, *, data, kind:Literal["dual_weights","dual_bias"]="dual_weights"):
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
                    if key == "weight_1":
                        for _, value in data[key].items():
                            x_values.append(value)
                    elif key == "weight_2":
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
                    if key == "weight_1":
                        for value in data[key].values():
                            x_values.append(value)
                    elif key == "weight_2":
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

    def anova(self, *, data,kind:Literal["one_way","two_way"]):
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
        f_val=[]
        p_val=[]
        match kind:
            case  "one_way":
              for k,v in data.items():
                if k!="sup_title":
                    f_statistic, p_value = f_oneway(*v.values())
                    f_val.append
        return None

    def linear_regression(self, *, data):
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
        parameter_groups = [key for key in ("weights", "biases") if key in data]
        if len(parameter_groups) != 1:
            raise ValueError("Expected exactly one of 'weights' or 'biases'.")

        values = data[parameter_groups[0]]
        if not hasattr(values, "values"):
            raise ValueError("Expected the parameter group to be a layer mapping.")
        if len(values) < 2:
            raise ValueError("Linear regression requires at least two layers.")

        layer_values = [float(np.asarray(value).mean()) for value in values.values()]
        layer_positions = np.arange(len(layer_values), dtype=float)
        regression_result = linregress(layer_positions, layer_values)
        fitted_values = regression_result.intercept + regression_result.slope * layer_positions

        return [
            {"Layer": layer, "Value": value}
            for layer, value in enumerate(fitted_values)
        ]

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
        return {
            group: {
                layer: np.asarray(median_abs_deviation(values,axis=None,nan_policy="omit"),dtype=float).item()
                for layer, values in layers.items()
            }
            for group, layers in data.items()
            if group!="sup_title"
        }

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
        result={}
        data_x={}
        data_y={}

        key_map=[]
        for i in data:
            if i =="weight_1" or i=="weight_2":
               key_map.append("weights")
            else:
                key_map.append("biases")
        if key_map[0]=="weights":
            var_key_1="weight_1"
            var_key_2="weight_2"
            result[var_key_1]={}
            result[var_key_2]={}
            for k_1,v_1 in data[var_key_1].items():
                data_x[k_1]=v_1
            for k_2,v_2 in data[var_key_2].items():
                data_y[k_2]=v_2
            for k_sx,v_sx in data_x.items():
              skew_val=kurtosis(v_sx,axis=None)
              result[var_key_1][k_sx]=np.asarray(skew_val,dtype=float).item()
            for k_sy,v_sy in data_y.items():
                skew_val=kurtosis(v_sy,axis=None)
                result[var_key_2][k_sy]=np.asarray(skew_val,dtype=float).item()
        elif key_map[0]=="biases":
            var_key_1="bias_1"
            var_key_2="bias_2"
            result[var_key_1]={}
            result[var_key_2]={}
            for k_1,v_1 in data[var_key_1].items():
                data_y[k_1]=v_1
            for k_2,v_2 in data[var_key_2].items():
                data_y[k_2]=v_2
            for k_sx,v_sx in data_x.items():
                skew_val=kurtosis(v_sx,axis=None)
                result[var_key_1][k_sx]=np.asarray(skew_val,dtype=float).item()
            for k_sy,v_sy in data_y.items():
                skew_val=kurtosis(v_sy,axis=None)
                result[var_key_2][k_sy]=np.asarray(skew_val,dtype=float).item()
        return result

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
        result={}
        data_x={}
        data_y={}

        key_map=[]
        for i in data:
            if i =="weight_1" or i=="weight_2":
               key_map.append("weights")
            else:
                key_map.append("biases")
        if key_map[0]=="weights":
            var_key_1="weight_1"
            var_key_2="weight_2"
            result[var_key_1]={}
            result[var_key_2]={}
            for k_1,v_1 in data[var_key_1].items():
                data_x[k_1]=v_1
            for k_2,v_2 in data[var_key_2].items():
                data_y[k_2]=v_2
            for k_sx,v_sx in data_x.items():
              skew_val=skew(v_sx,axis=None)
              result[var_key_1][k_sx]=np.asarray(skew_val,dtype=float).item()
            for k_sy,v_sy in data_y.items():
                skew_val=skew(v_sy,axis=None)
                result[var_key_2][k_sy]=np.asarray(skew_val,dtype=float).item()
        elif key_map[0]=="biases":
            var_key_1="bias_1"
            var_key_2="bias_2"
            result[var_key_1]={}
            result[var_key_2]={}
            for k_1,v_1 in data[var_key_1].items():
                data_y[k_1]=v_1
            for k_2,v_2 in data[var_key_2].items():
                data_y[k_2]=v_2
            for k_sx,v_sx in data_x.items():
                skew_val=skew(v_sx,axis=None)
                result[var_key_1][k_sx]=np.asarray(skew_val,dtype=float).item()
            for k_sy,v_sy in data_y.items():
                skew_val=skew(v_sy,axis=None)
                result[var_key_2][k_sy]=np.asarray(skew_val,dtype=float).item()
        return result
