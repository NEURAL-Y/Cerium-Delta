import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal, Mapping
from .stats_method import Statistical
import datashader as ds
import datashader.transfer_functions as tf
from scipy.stats import percentileofscore

class controller:
    """Parameter-preparation utilities shared by all validators.

    Caches results of :meth:`sudo_control` on ``control_memory`` so repeat
    calls for the same operation avoid recomputation.
    """

    def __init__(self):
        """Initialize an empty result cache."""
        self.control_memory = {}

    @staticmethod
    def _metric_values(parameters: dict, metric: str, parameter_type: str, statistic: str = "raw") -> dict:
        """Return one metric's layer values across current and legacy NVS layouts."""
        if not isinstance(parameters, dict):
            return {}

        section = parameters.get(metric, {})
        group_name = "biases" if parameter_type == "biases" else "weights"
        group = section.get(group_name, {}) if isinstance(section, dict) else {}
        if not isinstance(group, dict):
            group = {}

        if metric == "layer_contribution_score":
            filtered_name = f"filtered_layers_{group_name}"
            filtered = group.get(filtered_name, {})
            if statistic == "raw":
                return {key: value for key, value in group.items() if key != filtered_name and not isinstance(value, dict)}
            if not isinstance(filtered, dict):
                return {}
            if statistic == "norm":
                return filtered.get("norm_values", {})
            if statistic == "rank":
                rank_name = "ranks_biases" if group_name == "biases" else "ranks_weights"
                return filtered.get(rank_name, {})
            if statistic == "coefficient_of_variation":
                return {
                    key: value for key, value in filtered.items()
                    if key not in {"norm_values", "ranks_weights", "ranks_biases"}
                    and not isinstance(value, dict)
                }
            return {}

        statistic_name = {
            "raw": "raw_values",
            "norm": "norm_values",
            "rank": "ranks_biases" if group_name == "biases" else "ranks_weights",
        }.get(statistic)
        if statistic_name is None:
            return {}
        if statistic_name in group and isinstance(group[statistic_name], dict):
            return group[statistic_name]

        # Support older outputs where score dictionaries were not grouped by
        # parameter type. Do not confuse a grouped current-format result.
        legacy_group = section if isinstance(section, dict) else {}
        values = legacy_group.get(statistic_name, {})
        return values if isinstance(values, dict) else {}

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
                for parameter_type in ("weights", "biases"):
                    norm_values = self._metric_values(
                        parameters, "sensitivity_score", parameter_type, "norm"
                    )
                    raw_values = self._metric_values(
                        parameters, "sensitivity_score", parameter_type, "raw"
                    )
                    ranks = self._metric_values(
                        parameters, "sensitivity_score", parameter_type, "rank"
                    )
                    padded_group = padded_sensitivity[parameter_type]
                    padded_group["norm_values"].update(norm_values)
                    padded_group["raw_values"].update(raw_values)
                    rank_name = "ranks_biases" if parameter_type == "biases" else "ranks_weights"
                    padded_group[rank_name].update(ranks)

                    if norm_values:
                        values = np.asarray(list(norm_values.values()), dtype=float)
                        mean_value = float(np.mean(values))
                        median_value = float(np.median(values))
                        forced_value = mean_value / median_value if median_value != 0 else mean_value
                        padded_group["norm_values"]["forced_layer"] = np.asarray(forced_value/100)
                        padded_group["raw_values"]["forced_layer"] = np.asarray(forced_value/100)
                        padded_group[rank_name]["forced_layer"] = percentileofscore([norm_val for norm_val in norm_values.values()],np.asarray(forced_value/100) )

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
        code = family[family_idx]
        padded = self.sudo_control(sens_pad=True, parameters=parameters) if sens_pad else None
        if code in {"sens_weight", "sens_bias"}:
            parameter_type = "weights" if code == "sens_weight" else "biases"
            rank_key = "ranks_weights" if parameter_type == "weights" else "ranks_biases"
            selected_values[parameter_type] = (
                padded[parameter_type][rank_key] if padded is not None
                else self._metric_values(parameters, "sensitivity_score", parameter_type, "rank")
            )
            selected_values["sup_title"] = f"Sensitivity {parameter_type[:-1].capitalize()} Ranking"
        elif code in {"lcs_weight", "lcs_bias", "evol_weight", "evol_bias"}:
            parameter_type = "biases" if code.endswith("bias") else "weights"
            metric = "layer_contribution_score" if code.startswith("lcs") else "evolution_score"
            selected_values[parameter_type] = self._metric_values(parameters, metric, parameter_type, "rank")
            selected_values["sup_title"] = f"{metric.replace('_', ' ').title()} {parameter_type[:-1].capitalize()} Ranking"
        elif code in {"senvolution_b", "sensvolution_b", "sensvolution_w", "lsens_b", "lsens_w", "lvolution_w", "lvolution_b"}:
            parameter_type = "biases" if code.endswith("_b") or code.endswith("b") else "weights"
            if code.startswith("senvolution") or code.startswith("sensvolution"):
                rank_key = "ranks_biases" if parameter_type == "biases" else "ranks_weights"
                selected_values[f"{'bias' if parameter_type == 'biases' else 'weight'}_1"] = (
                    padded[parameter_type][rank_key] if padded is not None
                    else self._metric_values(parameters, "sensitivity_score", parameter_type, "rank")
                )
                selected_values[f"{'bias' if parameter_type == 'biases' else 'weight'}_2"] = self._metric_values(parameters, "evolution_score", parameter_type, "rank")
                title = "Sensitivity & Evolution"
            elif code.startswith("lsens"):
                selected_values[f"{'bias' if parameter_type == 'biases' else 'weight'}_1"] = self._metric_values(parameters, "layer_contribution_score", parameter_type, "rank")
                rank_key = "ranks_biases" if parameter_type == "biases" else "ranks_weights"
                selected_values[f"{'bias' if parameter_type == 'biases' else 'weight'}_2"] = (
                    padded[parameter_type][rank_key] if padded is not None
                    else self._metric_values(parameters, "sensitivity_score", parameter_type, "rank")
                )
                title = "Layer & Sensitivity"
            else:
                selected_values[f"{'bias' if parameter_type == 'biases' else 'weight'}_1"] = self._metric_values(parameters, "layer_contribution_score", parameter_type, "rank")
                selected_values[f"{'bias' if parameter_type == 'biases' else 'weight'}_2"] = self._metric_values(parameters, "evolution_score", parameter_type, "rank")
                title = "Layer & Evolution"
            selected_values["sup_title"] = f"{title} Contribution {parameter_type[:-1].capitalize()} Ranking"
        return selected_values

    def scatter_validator(self, *, parameter, family, family_idx, choice: Literal["non_grouped", "grouped_bias", "grouped_weight"], sens_pad: bool = True):
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
        code = family[family_idx]
        if choice == "non_grouped":
            family_map = {
                "lcs_weight": ("layer_contribution_score", "weights", "raw", "Layer Contribution Weight Relation"),
                "lcs_bias": ("layer_contribution_score", "biases", "raw", "Layer Contribution Bias Relation"),
                "sens_weight": ("sensitivity_score", "weights", "raw", "Layer & Sensitivity Contribution Weight Relation"),
                "sens_bias": ("sensitivity_score", "biases", "raw", "Sensitivity Contribution Bias Relation"),
                "evol_bias": ("evolution_score", "biases", "raw", "Evolution Contribution Bias Relation"),
                "evol_weight": ("evolution_score", "weights", "raw", "Evolution Contribution Weight Relation"),
            }
            if code in family_map:
                metric, parameter_type, statistic, title = family_map[code]
                selected_values[parameter_type] = self._metric_values(parameter, metric, parameter_type, statistic)
                selected_values["sup_title"] = title
        elif choice in {"grouped_weight", "grouped_bias"}:
            parameter_type = "weights" if choice == "grouped_weight" else "biases"
            prefix = "weight" if parameter_type == "weights" else "bias"
            if code in {"lsens", "sensvolution", "lvolution"}:
                code = f"{code}_{'w' if parameter_type == 'weights' else 'b'}"
            if code in {"lsens_w", "lsens_b"}:
                left_metric, right_metric, title = "layer_contribution_score", "sensitivity_score", "Layer & Sensitivity"
            elif code in {"sensvolution_w", "sensvolution_b"}:
                left_metric, right_metric, title = "evolution_score", "sensitivity_score", "Evolution & Sensitivity"
            elif code in {"lvolution_w", "lvolution_b"}:
                left_metric, right_metric, title = "layer_contribution_score", "evolution_score", "Evolution & Layer Contribution"
            else:
                return selected_values
            left_values = self._metric_values(
                parameter, left_metric, parameter_type, "norm"
            )
            if right_metric == "sensitivity_score" and sens_pad:
                padded_sensitivity = self.sudo_control(
                    sens_pad=True, parameters=parameter
                )
                right_values = padded_sensitivity[parameter_type]["norm_values"]
            else:
                right_values = self._metric_values(
                    parameter, right_metric, parameter_type, "norm"
                )
            shared_layers = [layer for layer in left_values if layer in right_values]
            if len(shared_layers) < 2:
                raise ValueError(
                    f"Grouped {parameter_type} correlation for {family[family_idx]!r} "
                    "requires at least two layers with values in both metrics."
                )
            selected_values[f"{prefix}_1"] = {
                "layer_summaries": np.asarray([left_values[layer] for layer in shared_layers])
            }
            selected_values[f"{prefix}_2"] = {
                "layer_summaries": np.asarray([right_values[layer] for layer in shared_layers])
            }
            selected_values["sup_title"] = f"{title} Contribution {parameter_type[:-1].capitalize()} Relation"
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
        family_map = {
            "lcs_weight": ("layer_contribution_score", "weights", "raw"),
            "lcs_bias": ("layer_contribution_score", "biases", "raw"),
            "sens_weight": ("sensitivity_score", "weights", "norm"),
            "sens_bias": ("sensitivity_score", "biases", "norm"),
            "evol_bias": ("evolution_score", "biases", "norm"),
            "evol_weight": ("evolution_score", "weights", "norm"),
        }
        code = family[family_idx]
        if code in family_map:
            metric, parameter_type, statistic = family_map[code]
            selected_values[parameter_type] = self._metric_values(parameter, metric, parameter_type, statistic)
            selected_values["sup_title"] = f"{metric.replace('_', ' ').title()} {parameter_type[:-1].capitalize()}"
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
        choice = "grouped_weight" if kind == "grouped_weight" else "grouped_bias"
        return self.scatter_validator(
            parameter=parameter,
            sens_pad=sens_pad,
            family=family,
            family_idx=family_idx,
            choice=choice,
        )

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
        return self._single_family_values(parameter, family[family_idx], "raw")

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
        return self._single_family_values(parameter, family[family_idx], "raw")

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
        return self._single_family_values(parameter, family[family_idx], "raw")

    def _single_family_values(self, parameter, code, statistic):
        """Select one named metric family for the array-oriented validators."""
        family_map = {
            "lcs_weight": ("layer_contribution_score", "weights"),
            "lcs_bias": ("layer_contribution_score", "biases"),
            "sens_weight": ("sensitivity_score", "weights"),
            "sens_bias": ("sensitivity_score", "biases"),
            "evol_weight": ("evolution_score", "weights"),
            "evol_bias": ("evolution_score", "biases"),
        }
        selected_values = {}
        if code in family_map:
            metric, parameter_type = family_map[code]
            selected_values[parameter_type] = self._metric_values(
                parameter, metric, parameter_type, statistic
            )
            selected_values["sup_title"] = f"{metric.replace('_', ' ').title()} {parameter_type[:-1].capitalize()}"
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
        if data.empty:
            selected_family = self.family[family_idx]
            if selected_family in {"evol_weight", "evol_bias"}:
                raise ValueError(
                    "No evolution scores are available. Evolution plots require matching current and trained/reference weights in parameters."
                )
            raise ValueError(f"No score data is available for family {selected_family!r}.")

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
        style_palette = self._style(plot_style)["palette"]
        if isinstance(style_palette, Mapping):
            style_palette = list(style_palette.values())
        cmap = sns.color_palette(style_palette, n_colors=7).as_hex()
        image = tf.shade(
            aggregate,
            cmap=cmap,
            how="eq_hist",
        )
        fig_size = (plot_style or {}).get(
            "fig_size", self.plot_style.get("fig_size", (12, 6))
        )
        style, _, axes = self._figure(plot_style, fig_size=fig_size)
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
        selected_values = obj.box_plot_validator(family_idx=family_index, family=self.family, parameter=parameters)
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
