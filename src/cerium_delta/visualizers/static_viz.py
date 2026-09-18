import matplotlib.pyplot as plt
from typing import cast
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal, Mapping
from .stats_method import Statistical
import datashader as ds
import datashader.transfer_functions as tf


class controller:
    """Parameter-preparation utilities shared by all validators.

    Caches results of :meth:`sudo_control` on ``control_memory`` so repeat
    calls for the same operation avoid recomputation.
    """

    def __init__(self):
        """Initialize an empty result cache."""
        self.control_memory = {}

    def sudo_control(self, *, sens_pad: bool = False, parameters: dict | None, diff_params: dict | None = None) -> dict | object:
        """Pad sensitivity layers or split parameters into train/test groups.

        Parameters
        ----------
        sens_pad : bool, default False
            If True, pad ``parameters``' sensitivity-score layers with a
            synthetic ``"forced_layer"`` derived from the mean/median of
            the existing normalized values.
        parameters : dict or None
            Full NVS parameter dictionary. Required when ``sens_pad`` is True.
        diff_params : dict or None, optional
            Weight/bias parameters to split into ``trainable_parameters``
            and ``test_parameters``. Accepts flat keys (``"weights"``,
            ``"weights_train"``, ``"bias"``, ``"bias_train"``) or pre-nested
            ``"train_parameters"`` / ``"test_parameters"``.

        Returns
        -------
        dict
            Padded sensitivity dict (if ``sens_pad``) or the
            train/test-split parameter dict (if ``diff_params``).
        """
        if sens_pad:
            if parameters is not None:
                padded_sensitivity = {
                    "weights": {"norm_values": {}, "raw_values": {}, "ranks_weights": {}},
                    "biases": {"norm_values": {}, "raw_values": {}, "ranks_weights": {}, "ranks_biases": {}},
                }
                weight_norm_values = [v for v in parameters["sensitivity_score"]["weights"]["norm_values"].values()]
                bias_norm_values = [v for v in parameters["sensitivity_score"]["biases"]["norm_values"].values()]
                weight_mean = np.mean(weight_norm_values)
                bias_mean = np.mean(bias_norm_values)
                weight_median = np.median(weight_norm_values)
                bias_median = np.median(bias_norm_values)
                final_layer_key = list(parameters["sensitivity_score"]["weights"]["norm_values"].keys())[-1]

                for layer_name, layer_values in parameters["sensitivity_score"]["weights"]["norm_values"].items():
                    padded_sensitivity["weights"]["norm_values"][layer_name] = layer_values
                    if layer_name == final_layer_key:
                        padded_sensitivity["weights"]["raw_values"]["forced_layer"] = np.full(layer_values.shape, np.float64((weight_mean / weight_median)))
                        padded_sensitivity["weights"]["ranks_weights"]["forced_layer"] = np.float64(weight_mean * 100)
                        padded_sensitivity["weights"]["norm_values"]["forced_layer"] = np.full(layer_values.shape, np.float64((weight_mean / weight_median)))

                for layer_name, layer_values in parameters["sensitivity_score"]["biases"]["norm_values"].items():
                    padded_sensitivity["biases"]["norm_values"][layer_name] = layer_values
                    if layer_name == final_layer_key:
                        padded_sensitivity["biases"]["raw_values"]["forced_layer"] = np.full(layer_values.shape, np.float64((bias_mean / bias_median)))
                        padded_sensitivity["biases"]["ranks_biases"]["forced_layer"] = np.float64((bias_mean / bias_median) * 100)
                        padded_sensitivity["biases"]["norm_values"]["forced_layer"] = np.full(layer_values.shape, np.float64((bias_mean / bias_median)))

                self.control_memory["sudo_control_sens"] = padded_sensitivity
                return padded_sensitivity

        if diff_params is not None:
            split_parameters = {"trainable_parameters": {}, "test_parameters": {}}

            if (
                "weights" in diff_params.keys() and "weights_train" in diff_params.keys()
                and "bias" in diff_params.keys() and "bias_train" in diff_params.keys()
            ):
                split_parameters["trainable_parameters"]["weights"] = diff_params["weights_train"]
                split_parameters["test_parameters"]["weights"] = diff_params["weights"]
                split_parameters["trainable_parameters"]["biases"] = diff_params["bias_train"]
                split_parameters["test_parameters"]["biases"] = diff_params["bias"]
                self.control_memory["sudo_control_diff_params"] = split_parameters
                return split_parameters

            elif "train_parameters" in diff_params.keys() and "test_parameters" in diff_params.keys():
                split_parameters["trainable_parameters"]["weights"] = diff_params["train_parameters"]["weights"]
                split_parameters["trainable_parameters"]["biases"] = diff_params["train_parameters"]["biases"]
                split_parameters["test_parameters"]["weights"] = diff_params["test_parameters"]["weights"]
                split_parameters["test_parameters"]["biases"] = diff_params["test_parameters"]["biases"]

            self.control_memory["sudo_control_diff_params"] = split_parameters
            return split_parameters


class validator(controller):
    """Selects and shapes the data each plot type needs from an NVS parameter set.

    Each ``*_validator`` method reads ``family[family_idx]`` — a code
    naming the score family / tensor type to plot (e.g. ``"sens_weight"``,
    ``"lcs_bias"``, ``"lsens_w"``) — and returns a small dict with only the
    arrays and title the matching :class:`visualizer` method needs.
    """

    def bar_plot_validator(self, *, family_idx, sens_pad: bool = True, parameters, anot, family):
        """Select ranking values + title for a bar plot.

        Parameters
        ----------
        family_idx : int
            Index into ``family``.
        sens_pad : bool, default True
            Pad sensitivity layers via :meth:`controller.sudo_control` first.
        parameters : dict
            Full NVS parameter dictionary.
        anot : list or None
            Annotation values, passed through unchanged.
        family : list of str
            Ordered score/tensor family codes.

        Returns
        -------
        dict
            ``"anot"``, one or two value arrays, and ``"sup_title"``.
        """
        selected_values = {"anot": anot}
        if family[family_idx] == "sens_weight":
            if sens_pad:
                padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameters)
                selected_values["weights"] = padded_sensitivity["weights"]["ranks_weights"]
                selected_values["sup_title"] = "Sensitivity weight Ranking"
            else:
                selected_values["weights"] = parameters["sensitivity_score"]["weights"]["ranks_weights"]
                selected_values["sup_title"] = "Sensitivity Weight Ranking"
        elif family[family_idx] == "sens_bias":
            if sens_pad:
                padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameters)
                selected_values["biases"] = padded_sensitivity["biases"]["ranks_biases"]
                selected_values["sup_title"] = "sensitivity Bias Ranking"
        elif family[family_idx] == "lcs_bias":
            selected_values["biases"] = parameters["layer_contribution_score"]["biases"]["ranks_biases"]
            selected_values["sup_title"] = "Layer Contribution Bias Ranking"
        elif family[family_idx] == "lcs_weight":
            selected_values["weights"] = parameters["layer_contribution_score"]["weights"]["ranks_weights"]
            selected_values["sup_title"] = "Layer Contribution Weight Ranking"
        elif family[family_idx] == "evol_weight":
            selected_values["weights"] = parameters["evolution_score"]["weights"]["ranks_weights"]
            selected_values["sup_title"] = "Layer Contribution Weight Ranking"
        elif family[family_idx] == "evol_bias":
            selected_values["biases"] = parameters["evolution_score"]["biases"]["ranks_biases"]
            selected_values["sup_title"] = "Layer Contribution Bias Ranking"
        elif family[family_idx] == "senvolution_b":
            padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameters)
            selected_values["bias_1"] = padded_sensitivity["biases"]["ranks_biases"]
            selected_values["bias_2"] = parameters["evolution_score"]["biases"]["rank_biases"]
            selected_values["sup_title"] = "Sensitivity & Evolution Contribution Bias Ranking"
        elif family[family_idx] == "sensvolution_w":
            padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameters)
            selected_values["weight_1"] = padded_sensitivity["weights"]["ranks_weights"]
            selected_values["weight_2"] = parameters["evolution_score"]["weights"]["rank_weights"]
            selected_values["sup_title"] = "Sensitivity & Evolution Contribution Weight Ranking"
        elif family[family_idx] == "lsens_b":
            selected_values["bias_1"] = parameters["layer_contribution_score"]["biases"]["ranks_biases"]
            padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameters)
            selected_values["bias_2"] = padded_sensitivity["biases"]["ranks_biases"]
            selected_values["sup_title"] = "Layer & Sensitivity Contribution Bias Ranking"
        elif family[family_idx] == "lsens_w":
            selected_values["weight_1"] = parameters["layer_contribution_score"]["weights"]["ranks_weights"]
            padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameters)
            selected_values["weight_2"] = padded_sensitivity["weights"]["ranks_weights"]
            selected_values["sup_title"] = "Layer & Sensitivity Contribution Weight Ranking"
        elif family[family_idx] == "lvolution_w":
            selected_values["weight_1"] = parameters["layer_contribution_score"]["weights"]["ranks_weights"]
            selected_values["weight_2"] = parameters["evolution_score"]["weights"]["rank_weights"]
            selected_values["sup_title"] = "Layer & Evolution Contribution Weight Ranking"
        elif family[family_idx] == "lvolution_b":
            selected_values["bias_1"] = parameters["layer_contribution_score"]["biases"]["ranks_biases"]
            selected_values["bias_2"] = parameters["evolution_score"]["biases"]["rank_biases"]
            selected_values["sup_title"] = "Layer & Evolution Contribution Bias Ranking"
        return selected_values

    def scatter_validator(self, *, parameter, sens_pad: bool = True, family, family_idx, choice: Literal["non_grouped", "grouped_bias", "grouped_weight"]):
        """Select paired layer values for a correlation/covariance scatter plot.

        Parameters
        ----------
        parameter : dict
            Full NVS parameter dictionary.
        sens_pad : bool, default True
            Pad sensitivity layers before selecting values.
        family : list of str
            Ordered score/tensor family codes.
        family_idx : int
            Index into ``family``.
        choice : {"non_grouped", "grouped_bias", "grouped_weight"}
            Single family (non-grouped) or a paired weight/bias comparison.

        Returns
        -------
        dict
            Selected value array(s) plus ``"sup_title"``.
        """
        selected_values = {}
        match choice:
            case "non_grouped":
                if family[family_idx] == "lcs_weight":
                    selected_values["weights"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layers_weights"}
                    selected_values["sup_title"] = "Layer Contribution Weight Relation"
                elif family[family_idx] == "lcs_bias":
                    selected_values["biases"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layers_biases"}
                    selected_values["sup_title"] = "Layer Contribution Bias Relation"
                elif family[family_idx] == "sens_weight":
                    selected_values["weights"] = parameter["sensitivity_score"]["raw_values"]
                    selected_values["sup_title"] = "Layer & Sensitivity Contribution Weight Relation"
                elif family[family_idx] == "sens_bias":
                    selected_values["biases"] = parameter["sensitivity_score"]["raw_values"]
                    selected_values["sup_title"] = "Sensitivity Contribution Bias Relation"
                elif family[family_idx] == "evol_bias":
                    selected_values["biases"] = parameter["evolution_score"]["raw_values"]
                    selected_values["sup_title"] = "Evolution Contribution Bias Relation"
                elif family[family_idx] == "evol_weight":
                    selected_values["weights"] = parameter["evolution_score"]["raw_values"]
                    selected_values["sup_title"] = "Evolution Contribution Weight Relation"
            case "grouped_weight":
                if family[family_idx] == "lsens_w":
                    selected_values["weight_1"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layer_weights"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["weight_2"] = padded_sensitivity["weights"]["raw_values"]
                    selected_values["sup_title"] = "Layer & Sensitivity Contribution Weight Relation"
                elif family[family_idx] == "sensvolution_w":
                    selected_values["weight_1"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layer_weights"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["weight_2"] = padded_sensitivity["weigths"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Sensitivity Contribution Weight Relation"
                elif family[family_idx] == "lvolution_w":
                    selected_values["weight_1"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layer_weights"}
                    selected_values["weight_2"] = parameter["evolution_score"]["weights"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Layer Contribution weight Relation"
            case "grouped_bias":
                if family[family_idx] == "lsens_b":
                    selected_values["bias_1"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layer_biases"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["bias_2"] = padded_sensitivity["biases"]["raw_values"]
                    selected_values["sup_title"] = "Layer & Sensitivity Contribution Bias Relation"
                elif family[family_idx] == "sensvolution_b":
                    selected_values["bias_1"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layer_biases"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["bias_2"] = padded_sensitivity["biases"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Sensitivity Contribution Bias Relation"
                elif family[family_idx] == "lvolution_b":
                    selected_values["bias_1"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layer_biases"}
                    selected_values["bias_2"] = parameter["evolution_score"]["biases"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Layer Contribution Bias Relation"

        return selected_values

    def box_plot_validator(self, *, family_idx, family, parameter):
        """Select per-layer arrays for a box plot.

        Parameters
        ----------
        family_idx : int
            Index into ``family``.
        family : list of str
            Ordered score/tensor family codes.
        parameter : dict
            Full NVS parameter dictionary.

        Returns
        -------
        dict
            ``"weights"`` or ``"biases"`` (layer -> array) plus ``"sup_title"``.
        """
        selected_values = {}
        if family[family_idx] == "lcs_weight":
            selected_values["weights"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items()}
            selected_values["sup_title"] = "Layer Contribution Weight"
        elif family[family_idx] == "lcs_bias":
            selected_values["biases"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"]["filtered_layer_biases"].items()}
            selected_values["sup_title"] = "Layer Contribution Bias"
        elif family[family_idx] == "sens_weight":
            selected_values["weights"] = parameter["sensitivity_score"]["norm_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Weight"
        elif family[family_idx] == "sens_bias":
            selected_values["biases"] = parameter["sensitivity_score"]["norm_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Bias"
        elif family[family_idx] == "evol_bias":
            selected_values["biases"] = parameter["evolution_score"]["norm_values"]
            selected_values["sup_title"] = "Evolution Contribution Bias"
        elif family[family_idx] == "evol_weight":
            selected_values["weights"] = parameter["evolution_score"]["norm_values"]
            selected_values["sup_title"] = "Evolution Contribution Weight"
        return selected_values

    def hist_plot_validator(self, *, family_idx, family, parameter, kind: Literal["grouped_weight", "grouped_bias"], sens_pad: bool = True):
        """Select paired weight/bias arrays for a skewness/kurtosis histogram.

        Parameters
        ----------
        family_idx : int
            Index into ``family``.
        family : list of str
            Ordered score/tensor family codes.
        parameter : dict
            Full NVS parameter dictionary.
        kind : {"grouped_weight", "grouped_bias"}
            Which paired tensor type to select.
        sens_pad : bool, default True
            Pad sensitivity layers before selecting values.

        Returns
        -------
        dict
            Paired value arrays plus ``"sup_title"``.
        """
        selected_values = {}
        match kind:
            case "grouped_weight":
                if family[family_idx] == "lsens_w":
                    selected_values["weight_1"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layer_weights"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["weight_2"] = padded_sensitivity["weights"]["raw_values"]
                    selected_values["sup_title"] = "Layer & Sensitivity Contribution Weight Relation"
                elif family[family_idx] == "sensvolution_w":
                    selected_values["weight_1"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layer_weights"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["weight_2"] = padded_sensitivity["weigths"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Sensitivity Contribution Weight Relation"
                elif family[family_idx] == "lvolution_w":
                    selected_values["weight_1"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"].items() if k != "filtered_layer_weights"}
                    selected_values["weight_2"] = parameter["evolution_score"]["weights"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Layer Contribution weight Relation"
            case "grouped_bias":
                if family[family_idx] == "lsens_b":
                    selected_values["bias_1"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layer_biases"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["bias_2"] = padded_sensitivity["biases"]["raw_values"]
                    selected_values["sup_title"] = "Layer & Sensitivity Contribution Bias Relation"
                elif family[family_idx] == "sensvolution_b":
                    selected_values["bias_1"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layer_biases"}
                    padded_sensitivity = self.sudo_control(sens_pad=sens_pad, parameters=parameter)
                    selected_values["bias_2"] = padded_sensitivity["biases"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Sensitivity Contribution Bias Relation"
                elif family[family_idx] == "lvolution_b":
                    selected_values["bias_1"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layer_biases"}
                    selected_values["bias_2"] = parameter["evolution_score"]["biases"]["raw_values"]
                    selected_values["sup_title"] = "Evolution & Layer Contribution Bias Relation"
        return selected_values

    def fitting_plot_validator(self, *, parameter, family_idx, family):
        """Select per-layer values for a linear-regression fitting plot.

        Parameters
        ----------
        parameter : dict
            Full NVS parameter dictionary.
        family_idx : int
            Index into ``family``.
        family : list of str
            Ordered score/tensor family codes.

        Returns
        -------
        dict
            ``"weights"`` or ``"biases"`` plus ``"sup_title"``.
        """
        selected_values = {}
        if family[family_idx] == "lcs_weight":
            selected_values["weights"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items() if k != "filtered_layers_weights"}
            selected_values["sup_title"] = "Layer Contribution Weight"
        elif family[family_idx] == "lcs_bias":
            selected_values["biases"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layers_biases"}
            selected_values["sup_title"] = "Layer Contribution Bias"
        elif family[family_idx] == "sens_weight":
            selected_values["weights"] = parameter["sensitivity_score"]["raw_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Weight"
        elif family[family_idx] == "sens_bias":
            selected_values["biases"] = parameter["sensitivity_score"]["raw_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Bias"
        elif family[family_idx] == "evol_bias":
            selected_values["biases"] = parameter["evolution_score"]["raw_values"]
            selected_values["sup_title"] = "Evolution Contribution Bias"
        elif family[family_idx] == "evol_weight":
            selected_values["weights"] = parameter["evolution_score"]["raw_values"]
            selected_values["sup_title"] = "Evolution Contribution Weight"
        return selected_values

    def mad_plot_validator(self, *, parameter, family, family_idx):
        """Select per-layer arrays for a median-absolute-deviation plot.

        Parameters
        ----------
        parameter : dict
            Full NVS parameter dictionary.
        family : list of str
            Ordered score/tensor family codes.
        family_idx : int
            Index into ``family``.

        Returns
        -------
        dict
            ``"weights"`` or ``"biases"`` plus ``"sup_title"``.
        """
        selected_values = {}
        if family[family_idx] == "lcs_weight":
            selected_values["weights"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items() if k != "filtered_layers_weights"}
            selected_values["sup_title"] = "Layer Contribution Weight"
        elif family[family_idx] == "lcs_bias":
            selected_values["biases"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layers_biases"}
            selected_values["sup_title"] = "Layer Contribution Bias"
        elif family[family_idx] == "sens_weight":
            selected_values["weights"] = parameter["sensitivity_score"]["raw_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Weight"
        elif family[family_idx] == "sens_bias":
            selected_values["biases"] = parameter["sensitivity_score"]["raw_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Bias"
        elif family[family_idx] == "evol_bias":
            selected_values["biases"] = parameter["evolution_score"]["raw_values"]
            selected_values["sup_title"] = "Evolution Contribution Bias"
        elif family[family_idx] == "evol_weight":
            selected_values["weights"] = parameter["evolution_score"]["raw_values"]
            selected_values["sup_title"] = "Evolution Contribution Weight"
        return selected_values

    def large_dist_plot_validator(self, *, family, family_idx, parameter):
        """Select per-layer arrays for the large-distribution Datashader plot.

        Parameters
        ----------
        family : list of str
            Ordered score/tensor family codes.
        family_idx : int
            Index into ``family``.
        parameter : dict
            Full NVS parameter dictionary.

        Returns
        -------
        dict
            ``"weights"`` or ``"biases"`` plus ``"sup_title"``.
        """
        selected_values = {}
        if family[family_idx] == "lcs_weight":
            selected_values["weights"] = {k: v for k, v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items() if k != "filtered_layers_weights"}
            selected_values["sup_title"] = "Layer Contribution Weight"
        elif family[family_idx] == "lcs_bias":
            selected_values["biases"] = {k: v for k, v in parameter["layer_contribution_score"]["biases"].items() if k != "filtered_layers_biases"}
            selected_values["sup_title"] = "Layer Contribution Bias"
        elif family[family_idx] == "sens_weight":
            selected_values["weights"] = parameter["sensitivity_score"]["raw_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Weight"
        elif family[family_idx] == "sens_bias":
            selected_values["biases"] = parameter["sensitivity_score"]["raw_values"]
            selected_values["sup_title"] = "Sensitivity Contribution Bias"
        elif family[family_idx] == "evol_bias":
            selected_values["biases"] = parameter["evolution_score"]["raw_values"]
            selected_values["sup_title"] = "Evolution Contribution Bias"
        elif family[family_idx] == "evol_weight":
            selected_values["weights"] = parameter["evolution_score"]["raw_values"]
            selected_values["sup_title"] = "Evolution Contribution Weight"
        return selected_values


class visualizer:
    """Public plotting API for NVS/NVB diagnostic score families.

    Bound to a fixed ``family`` list of score/tensor codes that every
    plotting method indexes into via ``family_idx``. Each method delegates
    data selection to a matching :class:`validator` method, then renders
    with matplotlib/seaborn (or Datashader for the large-distribution case).

    Parameters
    ----------
    family : list of str
        Ordered score/tensor family codes this visualizer indexes into.
    """

    def __init__(self, *, family: list, plot_style: Mapping | None = None) -> None:
        """Bind this visualizer to a fixed list of family codes and plot style."""
        self.family = family
        self.plot_style = dict(plot_style or {})

    def _style(self, plot_style: Mapping | None = None, **overrides) -> dict:
        """Merge instance and per-plot styling options."""
        style = {
            "context": "notebook",
            "theme": "whitegrid",
            "palette": "deep",
            "fig_size": (9, 5),
            "dpi": 120,
            "title_size": 16,
            "label_size": 11,
            "tick_rotation": 35,
            "grid_alpha": 0.25,
            "despine": True,
            "show": True,
        }
        style.update(self.plot_style)
        if plot_style:
            style.update(plot_style)
        style.update({key: value for key, value in overrides.items() if value is not None})
        return style

    def _figure(self, plot_style: Mapping | None = None, **overrides):
        """Create a consistently styled figure and axes."""
        style = self._style(plot_style, **overrides)
        sns.set_theme(
            context=style["context"],
            style=style["theme"],
            palette=style["palette"],
            rc={"grid.alpha": style["grid_alpha"]},
        )
        figure, axes = plt.subplots(figsize=style["fig_size"], dpi=style["dpi"])
        return style, figure, axes

    @staticmethod
    def _primary_color(style: dict):
        """Resolve a palette option to one valid matplotlib color."""
        palette = style["palette"]
        if isinstance(palette, str):
            return sns.color_palette(palette, 1)[0]
        return palette[0] if palette else None

    def _finish(self, axes, style: dict, *, title: str, xlabel: str | None = None, ylabel: str | None = None):
        """Apply shared labels and layout, then optionally display the figure."""
        axes.set_title(style.get("title", title), fontsize=style["title_size"], weight="bold", pad=14)
        if xlabel is not None:
            axes.set_xlabel(style.get("xlabel", xlabel), fontsize=style["label_size"])
        if ylabel is not None:
            axes.set_ylabel(style.get("ylabel", ylabel), fontsize=style["label_size"])
        axes.tick_params(axis="both", labelsize=style["label_size"] - 1)
        if style["despine"]:
            sns.despine(ax=axes)
        axes.figure.tight_layout()
        if style["show"]:
            plt.show()

    def bar_plot(
        self, *, family_idx: int = 0, parameters: dict, sens_pad: bool = True,
        choice: Literal["non_grouped", "grouped_biases", "grouped_weights"] = "non_grouped",
        anot: None | list = None, range: str | None = None, fig_size: tuple | None = None,
        orient: Literal["v", "h", "x", "y"] = "v", kind: Literal["normal", "iqr"] = "normal",
        plot_style: Mapping | None = None,
    ):
        """Plot per-layer ranking scores as a bar chart.

        Parameters
        ----------
        family_idx : int, default 0
            Index into ``self.family``.
        parameters : dict
            Full NVS parameter dictionary.
        sens_pad : bool, default True
            Pad sensitivity layers before selecting values.
        choice : {"non_grouped", "grouped_biases", "grouped_weights"}, default "non_grouped"
            Single-series bar plot vs. grouped (hue-split) comparison.
        anot : list or None, optional
            Annotation values passed through the validator.
        range : str or None, optional
            Layer key to drop from every group before plotting, if given.
        fig_size : tuple, default (6, 4)
            Matplotlib figure size in inches.
        orient : {"v", "h", "x", "y"}, default "v"
            Bar orientation passed to :func:`seaborn.barplot`.
        kind : {"normal", "iqr"}, default "normal"
            Plot the mean of each layer's array (``"normal"``) or its
            interquartile range (``"iqr"``).

        Returns
        -------
        None
            Displays the plot via ``plt.show()``.
        """
        obj = validator()
        selected_values = obj.bar_plot_validator(family_idx=family_idx, parameters=parameters, anot=anot, sens_pad=sens_pad, family=self.family)
        groups = {k: v for k, v in selected_values.items() if k not in "sup_title" and k not in "anot"}
        if range is not None and isinstance(range, str):
            for group_key in groups.keys():
                groups[group_key].pop(range)
        else:
            groups = groups

        rows = []
        for group_name, layer_scores in groups.items():
            for layer_name, layer_score in layer_scores.items():
                if kind == "iqr":
                    layer_score = Statistical().interquartile_range(data=np.asarray(layer_score).ravel())
                else:
                    score_array = np.asarray(layer_score)
                    layer_score = score_array.item() if score_array.ndim == 0 else np.mean(score_array)
                rows.append({"Layers": layer_name, "Scores": layer_score, "Group": group_name})
        data = pd.DataFrame(rows)

        match choice:
            case "non_grouped":
                style, _, axes = self._figure(plot_style, fig_size=fig_size)
                sns.barplot(data=data, x="Layers", y="Scores", orient=orient, ax=axes, color=self._primary_color(style))
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=selected_values["sup_title"], xlabel="Layer", ylabel="Score")

            case "grouped_weights":
                style, _, axes = self._figure(plot_style, fig_size=fig_size)
                sns.barplot(data=data, x="Layers", y="Scores", hue="Group", palette=style["palette"], ax=axes)
                axes.legend(title=None, frameon=False)
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=selected_values["sup_title"], xlabel="Layer", ylabel="Score")

            case "grouped_biases":
                style, _, axes = self._figure(plot_style, fig_size=fig_size)
                sns.barplot(data=data, x="Layers", y="Scores", hue="Group", palette=style["palette"], ax=axes)
                axes.legend(title=None, frameon=False)
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=selected_values["sup_title"], xlabel="Layer", ylabel="Score")

    def scatter_plot(self, *, family_index: int = 0, parameters: dict, plot_choice: Literal["covariance", "corelation"], choice: Literal["grouped_bias", "grouped_weight"], sens_pad: bool = True, plot_style: Mapping | None = None):
        """Plot per-layer Pearson correlation or covariance between two score families.

        Parameters
        ----------
        family_index : int, default 0
            Index into ``self.family``.
        parameters : dict
            Full NVS parameter dictionary.
        plot_choice : {"covariance", "corelation"}
            Statistic to compute and plot per layer.
        choice : {"grouped_bias", "grouped_weight"}
            Which paired tensor type to compare.
        sens_pad : bool, default True
            Pad sensitivity layers before selecting values.

        Returns
        -------
        None
            Displays the plot via ``plt.show()``.
        """
        obj = validator()
        selected_values = obj.scatter_validator(parameter=parameters, sens_pad=sens_pad, family=self.family, family_idx=family_index, choice=choice)
        pairing_kind = {"grouped_weight": "dual_weights", "grouped_bias": "dual_biases"}[choice]
        match plot_choice:
            case "corelation":
                correlation_results = Statistical().pearson_correlation(data=selected_values, kind=pairing_kind)
                df = pd.DataFrame(
                    [{"Layer": item["layer"], "Value": item["R_value"]} for item in correlation_results]
                )
                style, _, axes = self._figure(plot_style)
                sns.scatterplot(data=df, x="Layer", y="Value", color=self._primary_color(style), s=70, alpha=0.85, ax=axes)
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=selected_values["sup_title"], xlabel="Layer", ylabel="Correlation")
            case "covariance":
                covariance_results = Statistical().covariance(data=selected_values, kind=pairing_kind)
                df = pd.DataFrame(
                    [
                        {"Layer": item["layer"], "Value": np.asarray(item["Result_value"])[0, 1]}
                        for item in covariance_results
                    ]
                )
                style, _, axes = self._figure(plot_style)
                sns.scatterplot(data=df, x="Layer", y="Value", color=self._primary_color(style), s=70, alpha=0.85, ax=axes)
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=selected_values["sup_title"], xlabel="Layer", ylabel="Covariance")

    def large_dist_plot(self, *, family_index: int = 0, parameters: dict, plot_style: Mapping | None = None) -> object | None:
        """Aggregate large per-layer value arrays with Datashader and show a heatmap.

        Parameters
        ----------
        family_index : int, default 0
            Index into ``self.family``.
        parameters : dict
            Full NVS parameter dictionary.

        Returns
        -------
        datashader.transfer_functions.Image or None
            The shaded Datashader image that was displayed.

        Raises
        ------
        ValueError
            If no layer values are available to aggregate.
        """
        obj = validator()
        selected_values = obj.large_dist_plot_validator(parameter=parameters, family=self.family, family_idx=family_index)
        rows = []
        layer_labels = []
        for group_name, layers in selected_values.items():
            if group_name == "sup_title":
                continue
            for layer_name, layer_values in layers.items():
                if layer_name not in layer_labels:
                    layer_labels.append(layer_name)
                layer_index = layer_labels.index(layer_name)
                rows.extend(
                    {"Layer": layer_index, "Value": element, "Group": group_name}
                    for element in np.asarray(layer_values).ravel()
                )
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("large_dist_plot requires at least one layer value")

        canvas = ds.Canvas(
            plot_width=max(1, min(1200, len(layer_labels) * 20)),
            plot_height=600,
            x_range=(-0.5, max(len(layer_labels) - 0.5, 0.5)),
        )
        aggregate = canvas.points(df, x="Layer", y="Value", agg=ds.count())
        image = tf.shade(
            aggregate,
            cmap=["#000004", "#2c115f", "#721f81", "#b73779", "#f1605d", "#feb078", "#fcfdbf"],
            how="eq_hist",
        )
        style, _, axes = self._figure(plot_style, fig_size=(12, 6))
        axes.imshow(image.to_pil(), aspect="auto", origin="lower")
        axes.set_xticks(
            np.linspace(0, len(layer_labels) - 1, min(len(layer_labels), 10), dtype=int),
            [layer_labels[index] for index in np.linspace(0, len(layer_labels) - 1, min(len(layer_labels), 10), dtype=int)],
            rotation=style["tick_rotation"],
            ha="right"
        )
        self._finish(axes, style, title=selected_values.get("sup_title", "Large Distribution"), xlabel="Layer", ylabel="Value")
        return image

    def box_plot(self, *, family_index: int = 0, parameters: dict, plot_style: Mapping | None = None):
        """Plot the distribution of element values per layer as a box plot.

        Parameters
        ----------
        family_index : int, default 0
            Index into ``self.family``.
        parameters : dict
            Full NVS parameter dictionary.

        Returns
        -------
        None
            Displays the plot via ``plt.show()``.
        """
        obj = validator()
        selected_values = obj.box_plot_validator(family_index=family_index, family=self.family, parameter=parameters)
        df = pd.DataFrame(
            [
                {"Layer": layer_name, "Value": element, "Group": group_name}
                for group_name, layers in selected_values.items()
                if group_name != "sup_title"
                for layer_name, layer_values in layers.items()
                for element in np.asarray(layer_values).ravel()
            ]
        )
        style, _, axes = self._figure(plot_style)
        sns.boxplot(data=df, x="Layer", y="Value", hue="Group", palette=style["palette"], ax=axes)
        axes.legend(title=None, frameon=False)
        axes.tick_params(axis="x", rotation=style["tick_rotation"])
        self._finish(axes, style, title=selected_values["sup_title"], xlabel="Layer", ylabel="Value")

    def hist_plot(self, *, family_index: int = 0, parameters: dict, kind: Literal["grouped_weight", "grouped_bias"] = "grouped_weight", statistics_method: Literal["skewness", "kurtosis"], bins: int = 30, plot_style: Mapping | None = None):
        """Plot per-layer skewness or kurtosis as a histogram.

        Parameters
        ----------
        family_index : int, default 0
            Index into ``self.family``.
        parameters : dict
            Full NVS parameter dictionary.
        kind : {"grouped_weight", "grouped_bias"}, default "grouped_weight"
            Which paired tensor type to compute the statistic for.
        statistics_method : {"skewness", "kurtosis"}
            Distribution statistic to compute per layer.
        bins : int, default 30
            Bin count passed to :func:`seaborn.histplot`.

        Returns
        -------
        None
            Displays the plot via ``plt.show()``.
        """
        obj = validator()
        selected_values = obj.hist_plot_validator(family_idx=family_index, family=self.family, parameter=parameters, kind=kind)
        match statistics_method:
            case "kurtosis":
                kurtosis_results = Statistical().kurtosis(data=selected_values)
                df = pd.DataFrame(
                    [
                        {"Layer": layer_name, "Value": score, "Group": group_name}
                        for group_name, layers in kurtosis_results.items()
                        for layer_name, score in layers.items()
                    ]
                )
                style, _, axes = self._figure(plot_style)
                sns.histplot(df, x="Layer", y="Value", hue="Group", kde=True, bins=bins, palette=style["palette"], ax=axes)
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=f"{selected_values['sup_title']} Kurtosis", xlabel="Layer", ylabel="Kurtosis")
            case "skewness":
                skewness_results = Statistical().skewness(data=selected_values)
                df = pd.DataFrame(
                    [
                        {"Layer": layer_name, "Value": score, "Group": group_name}
                        for group_name, layers in skewness_results.items()
                        for layer_name, score in layers.items()
                    ]
                )
                style, _, axes = self._figure(plot_style)
                sns.histplot(df, x="Layer", y="Value", hue="Group", kde=True, bins=bins, palette=style["palette"], ax=axes)
                axes.tick_params(axis="x", rotation=style["tick_rotation"])
                self._finish(axes, style, title=f"{selected_values['sup_title']} Skewness", xlabel="Layer", ylabel="Skewness")

    def fitting_plot(self, *, family_index: int = 0, parameters: dict, plot_style: Mapping | None = None):
        """Plot a fitted regression line for the selected layer values.

        Parameters
        ----------
        family_index : int, default 0
            Index into ``self.family``.
        parameters : dict
            Full NVS parameter dictionary.

        Returns
        -------
        None
            Displays the plot via ``plt.show()``.
        """
        obj = validator()
        selected_values = obj.fitting_plot_validator(family=self.family, family_idx=family_index, parameter=parameters)
        regression_results = Statistical().linear_regression(data=selected_values)
        regression_df = pd.DataFrame(regression_results)
        style, _, axes = self._figure(plot_style)
        sns.lineplot(data=regression_df, x="Layer", y="Value", marker="o", linewidth=2.5, color=self._primary_color(style), ax=axes)
        axes.tick_params(axis="x", rotation=style["tick_rotation"])
        self._finish(axes, style, title=selected_values["sup_title"] + " Fit", xlabel="Layer", ylabel="Fitted Value")

    def mad_plot(self, *, parameters: dict, family_index: int = 0, plot_style: Mapping | None = None):
        """Plot the median-absolute-deviation distribution by score group.

        Parameters
        ----------
        parameters : dict
            Full NVS parameter dictionary.
        family_index : int, default 0
            Index into ``self.family``.

        Returns
        -------
        None
            Displays the plot via ``plt.show()``.
        """
        obj = validator()
        selected_values = obj.mad_plot_validator(parameter=parameters, family=self.family, family_idx=family_index)
        mad_results = Statistical().median_absolute_deviation(data=selected_values)
        df = pd.DataFrame(
            [
                {"Layer": layer_name, "Value": score, "Group": group_name}
                for group_name, layers in mad_results.items()
                for layer_name, score in layers.items()
            ]
        )
        style, _, axes = self._figure(plot_style)
        sns.kdeplot(data=df, x="Value", hue="Group", fill=True, alpha=0.25, palette=style["palette"], ax=axes)
        self._finish(axes, style, title=selected_values["sup_title"] + " MAD Distribution", xlabel="Value", ylabel="Density")
