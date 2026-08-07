from scipy.stats import percentileofscore
import numpy as np


class NVS:
    """
    Neural Vitality System (NVS)

    A lightweight neural network observability module for measuring
    layer contribution, sensitivity, and parameter evolution during
    training.

    Parameters
    ----------
    state_memory : dict
        Dictionary containing model parameters and training metadata.

        Required keys
        -------------
        weights : dict
            Initial or reference model weights.

        weights_train : dict
            Current/trained model weights.

        bias : dict
            Initial or reference model biases.

        bias_train : dict
            Current/trained model biases.

        epochs : int
            Number of training epochs used for normalization.
    """

    def __init__(self, model_state: dict) -> None:
        # Stores all model states required by the metric pipeline.
        self.model_state = model_state

    def compute(self, choose_metrics="all"):
        """
        Compute one or all NVS metrics.

        Parameters
        ----------
        choose_metrics : str
            Metric to compute.

            Available options:
                - "lcs"
                - "sensitivity"
                - "evolution"
                - "lcs_bias"
                - "sensitivity_bias"
                - "evolution_bias"
                - "all"

        Returns
        -------
        dict or tuple
            Requested metric(s).
        """

        if choose_metrics == "lcs":
            return self.compute_lcs()

        elif choose_metrics == "sensitivity":
            return self.compute_sensitivity()

        elif choose_metrics == "evolution":
            return self.compute_evolution()

        elif choose_metrics == "lcs_bias":
            return self.compute_lcs_bias()

        elif choose_metrics == "sensitivity_bias":
            return self.compute_sensitivity_bias()

        elif choose_metrics == "evolution_bias":
            return self.compute_evolution_bias()

        else:
            return (
                self.compute_lcs(),
                self.compute_sensitivity(),
                self.compute_evolution(),
                self.compute_lcs_bias(),
                self.compute_sensitivity_bias(),
                self.compute_evolution_bias(),
            )

    # ------------------------------------------------------------------
    # WEIGHT-BASED METRICS (unchanged)
    # ------------------------------------------------------------------

    def compute_lcs(self):
        """
        Layer Contribution Score (LCS)

        Measures the contribution of each layer by amplifying its
        parameter values using an adaptive power derived from the
        dominant eigenvalue of the layer.

        Larger scores generally indicate stronger parameter influence.
        """

        self.lcs = {}

        # Reference weights.
        x = self.model_state["weights"]

        # Compute eigenvalues for every layer.
        p = np.linalg.eigvals([x[layer] for layer in x])

        # Adaptive exponent derived from the largest eigenvalue.
        self.layer_powers = []
        self.layer_powers.extend(
            [
                np.log(np.abs(np.max(i))) + 1e-12
                for i in p
            ]
        )

        for i,(k, v) in enumerate(x.items()):

            powered = np.sign(v) * np.power(np.abs(v), self.layer_powers[i])
            self.lcs.update(
                {
                    k: v * powered
                }
            )

        return self.lcs

    def compute_sensitivity(self):
        """
        Sensitivity Score

        Estimates how responsive each layer is to parameter changes by
        combining the derivative of the LCS formulation with the layer
        magnitude (L2 norm).

        Higher values indicate that small parameter variations have
        larger influence.
        """

        self.sensitivity_score = {}

        x = self.model_state["weights"]

        f = {}

        # Compute derivative of x^p.
        for i,(k, v) in enumerate(x.items()):

            if k != "layer 1":
                f[k] = (
                    self.layer_powers[i] *
                    (np.sign(v) * np.power(np.abs(v), self.layer_powers[i] - 1))
                )
            else:
                continue

        for k, v in x.items():

            if k not in f:
                continue

            # L2 norm summarizes layer sensitivity into a scalar.
            self.sensitivity_score.update(
                {
                    k: np.linalg.norm(v * f[k])
                }
            )

        return self.sensitivity_score

    def compute_evolution(self):
        """
        Evolution Score

        Measures how much each layer has changed throughout training.

        Evolution is normalized by the number of epochs to provide
        comparable scores across different training durations.
        """

        self.evolution_scores = {}

        trained_weights = self.model_state["weights_train"]
        reference_weights= self.model_state["weights"]

        epochs = self.model_state["epochs"]

        for k in trained_weights:

            if k in reference_weights:

                self.evolution_scores[k] = (
                    np.linalg.norm(trained_weights[k] - reference_weights[k]) / epochs
                )

        return self.evolution_scores

    def threshold_lcs(self):
        """
        Filter LCS values using coefficient of variation (CV).

        Layers with excessive variance are discarded before percentile
        ranking to reduce unstable contribution estimates.

        Remaining layers are ranked using percentile statistics.
        """

        filtered_layer = {}

        eps = 1e-12

        for k, v in self.lcs.items():

            std = np.std(self.lcs[k])
            mean = np.mean(self.lcs[k])

            # Coefficient of Variation.
            cv = std / (abs(mean) + eps)

            if cv <= 0.75:
                filtered_layer.update({k: v})

        norm = {}
        norm_values = []

        # Convert tensors into scalar scores.
        for k, v in filtered_layer.items():

            norm[k] = np.linalg.norm(v)
            norm_values.append(norm[k])

        Ranking = {}

        # Percentile ranking.
        for k, v in norm.items():

            Ranking[k] = percentileofscore(
                norm_values,
                v,
                kind="rank",
            )

        filtered_layer.update({"ranks": Ranking})
        self.lcs.update({"filtered_layers": filtered_layer})

        return None

    def threshold_sens(self):
        """
        Convert sensitivity scores into percentile rankings.

        Percentiles provide a model-independent interpretation of
        sensitivity without relying on fixed thresholds.
        """

        scores = list(self.sensitivity_score.values())

        percentile = {}

        for layer, score in self.sensitivity_score.items():

            percentile[layer] = percentileofscore(
                scores,
                score,
                kind="rank",
            )

        self.sensitivity_score.update(
            {"rank": percentile}
        )

        return None

    def threshold_evolution(self):
        """
        Convert evolution scores into percentile rankings.

        Higher percentiles indicate layers that experienced greater
        parameter updates during training.
        """

        rankings = {}

        scores = list(self.evolution_scores.values())

        rankings.update(
            {
                k: percentileofscore(
                    scores,
                    v,
                    kind="rank",
                )
                for k, v in self.evolution_scores.items()
            }
        )

        self.evolution_scores.update({"ranks": rankings})

        return None

    # ------------------------------------------------------------------
    # BIAS-BASED METRICS (mirrors weight logic exactly, "bias"/"bias_train")
    # ------------------------------------------------------------------

    def compute_lcs_bias(self):
        """
        Layer Contribution Score (LCS) — bias variant.

        Mirrors compute_lcs exactly, but operates on
        self.model_state["bias"] instead of ["weights"].

        Step by step
        ------------
        1. x = self.model_state["bias"]
           Pull the reference (pre-training) bias arrays, one per layer.

        2. p = np.linalg.eigvals([x[layer] for layer in x])
           Stack every layer's bias array and compute eigenvalues.
           NOTE: eigvals requires a square 2D array per layer. Weight
           matrices are naturally square-ish; bias vectors are usually
           1D (one value per neuron), so this step is the most likely
           place this breaks if your bias tensors aren't shaped like
           your weight tensors. Reshape/pad bias into a square form
           upstream if needed.

        3. self.layer_powers_bias[i] = log(|max(eigvals_i)|) + 1e-12
           For each layer, take the largest eigenvalue by magnitude,
           log-transform it, and add an epsilon floor so log(0) never
           happens. This becomes that layer's adaptive exponent —
           layers with a dominant/large eigenvalue get a larger power.

        4. self.lcs_bias[k] = v * (v ** layer_powers_bias[i])
           Raise each layer's bias values to its own spectral-derived
           power (effectively v^(p+1)) and store as that layer's LCS.
           Larger layer_powers_bias -> more amplification -> higher
           apparent "contribution" for that layer's bias.

        Returns
        -------
        dict
            {layer_name: lcs_bias_array} for every layer in "bias".
        """

        self.lcs_bias = {}

        # Reference biases.
        x = self.model_state["bias"]

        # Compute eigenvalues for every layer.
        p = np.linalg.eigvals([x[layer] for layer in x])

        # Adaptive exponent derived from the largest eigenvalue.
        self.layer_powers_bias = []
        self.layer_powers_bias.extend(
            [
                np.log(np.abs(np.max(i))) + 1e-12
                for i in p
            ]
        )

        for i,(k, v) in enumerate(x.items()):

            # Apply adaptive power scaling.
            powered = np.sign(v) * np.power(np.abs(v), self.layer_powers_bias[i])
            self.lcs_bias.update(
                {
                    k: v * powered
                }
            )

        return self.lcs_bias

    def compute_sensitivity_bias(self):
        """
        Sensitivity Score — bias variant.

        Mirrors compute_sensitivity exactly, but operates on
        self.model_state["bias"] and self.layer_powers_bias (from
        compute_lcs_bias) instead of the weight equivalents.

        Step by step
        ------------
        1. x = self.model_state["bias"]
           Same reference bias arrays used in compute_lcs_bias.

        2. f[i] = layer_powers_bias[i] * v ** (layer_powers_bias[i] - 1)
           This is just the power rule applied to the LCS expression:
               d/dv [ v^p ] = p * v^(p-1)
           So f holds, per layer, the derivative of that layer's LCS
           output with respect to its own bias values -- i.e. "how
           much would LCS change for a small nudge in this bias."
           The "layer 1" key is skipped here exactly as in the weight
           version (carried over as-is; only meaningful if one of your
           bias dict keys is literally the string "layer 1").

        3. self.sensitivity_score_bias[k] = norm(v * f[index])
           Multiply the bias values by their derivative and L2-norm
           the result down to a single scalar per layer: a compact
           "how sensitive is this layer to small bias perturbations"
           score.

        Known caveat (inherited from compute_sensitivity)
        ---------------------------------------------------
        Because one layer's derivative is skipped via "continue" but
        the second loop still iterates over every layer in x, f[index]
        can become misaligned by one position for every layer after
        the skipped one. This is a pre-existing issue in the weight
        version, reproduced identically here.

        Returns
        -------
        dict
            {layer_name: sensitivity_scalar} for every layer in "bias".
        """

        self.sensitivity_score_bias = {}

        x = self.model_state["bias"]

        # Fix: dict keyed by layer name instead of positional list,
        # same reasoning as compute_sensitivity.
        f = {}

        # Compute derivative of x^p.
        for i,(k, v) in enumerate(x.items()):

            if k != "layer 1":
                f[k] = (
                    self.layer_powers_bias[i] *
                    (np.sign(v) * np.power(np.abs(v), self.layer_powers_bias[i] - 1))
                )
            else:
                continue

        for k, v in x.items():

            # "layer 1" excluded consistently, same as compute_sensitivity.
            if k not in f:
                continue

            # L2 norm summarizes layer sensitivity into a scalar.
            self.sensitivity_score_bias.update(
                {
                    k: np.linalg.norm(v * f[k])
                }
            )

        return self.sensitivity_score_bias

    def compute_evolution_bias(self):
        """
        Evolution Score — bias variant.

        Mirrors compute_evolution exactly, but operates on
        self.model_state["bias_train"] and ["bias"] instead of the
        weight equivalents.

        Step by step
        ------------
        1. trained_bias = self.model_state["bias_train"]
           reference_bias = self.model_state["bias"]
           Grab the post-training and pre-training bias snapshots.

        2. self.evolution_scores_bias[k] =
               norm(trained_bias[k] - reference_bias[k]) / epochs
           Take the raw difference between trained and reference bias
           for each layer, L2-norm it into a single "how much did this
           layer's bias move" magnitude, then divide by epoch count so
           models trained for different numbers of epochs are still
           comparable (a longer run naturally accumulates more change,
           this normalizes it out). Matches the "velocity as
           cross-checkpoint difference" convention used for weights.

        Returns
        -------
        dict
            {layer_name: evolution_scalar} for every layer present in
            both "bias_train" and "bias".
        """

        self.evolution_scores_bias = {}

        trained_bias = self.model_state["bias_train"]
        reference_bias = self.model_state["bias"]

        epochs = self.model_state["epochs"]

        for k in trained_bias:

            if k in reference_bias:

                self.evolution_scores_bias[k] = (
                    np.linalg.norm(trained_bias[k] - reference_bias[k]) / epochs
                )

        return self.evolution_scores_bias

    def threshold_lcs_bias(self):
        """
        Filter bias LCS values using coefficient of variation (CV).

        Mirrors threshold_lcs exactly, but operates on self.lcs_bias
        (populated by compute_lcs_bias) instead of self.lcs.

        Step by step
        ------------
        1. CV filter
           For each layer's LCS array: cv = std(v) / (|mean(v)| + eps)
           If cv > 0.75, the layer is dropped before ranking. Rationale:
           a layer whose LCS values are internally very inconsistent
           (high spread relative to its mean) is treated as too noisy
           to trust for a reliable contribution ranking.

        2. Scalar reduction
           Every surviving layer's LCS array is collapsed to a single
           scalar via L2 norm (norm[k] = ||v||).

        3. Percentile ranking
           Each surviving layer's scalar is converted into a percentile
           rank (0-100) relative to the other surviving layers using
           percentileofscore(..., kind="rank"). This makes contribution
           scores comparable across layers regardless of raw scale.

        4. self.lcs_bias["filtered_layers"] is set to a dict containing
           the surviving layers plus a "ranks" sub-dict of percentiles.

        Returns
        -------
        None
            Mutates self.lcs_bias in place.
        """

        filtered_layer = {}

        eps = 1e-12

        for k, v in self.lcs_bias.items():

            std = np.std(self.lcs_bias[k])
            mean = np.mean(self.lcs_bias[k])

            # Coefficient of Variation.
            cv = std / (abs(mean) + eps)

            if cv <= 0.75:
                filtered_layer.update({k: v})

        norm = {}
        norm_values = []

        # Convert tensors into scalar scores.
        for k, v in filtered_layer.items():

            norm[k] = np.linalg.norm(v)
            norm_values.append(norm[k])

        Ranking = {}

        # Percentile ranking.
        for k, v in norm.items():

            Ranking[k] = percentileofscore(
                norm_values,
                v,
                kind="rank",
            )

        filtered_layer.update({"ranks": Ranking})
        self.lcs_bias.update({"filtered_layers": filtered_layer})

        return None

    def threshold_sens_bias(self):
        """
        Convert bias sensitivity scores into percentile rankings.

        Mirrors threshold_sens exactly, but operates on
        self.sensitivity_score_bias (populated by
        compute_sensitivity_bias) instead of self.sensitivity_score.

        Step by step
        ------------
        1. scores = list of all layers' sensitivity scalars.

        2. For each layer, compute its percentile rank among `scores`
           via percentileofscore(..., kind="rank"). No CV filtering
           step here (unlike threshold_lcs_bias) — every layer is
           ranked, none are dropped.

        3. self.sensitivity_score_bias["rank"] is set to a dict of
           {layer_name: percentile}.

        Returns
        -------
        None
            Mutates self.sensitivity_score_bias in place.
        """

        scores = list(self.sensitivity_score_bias.values())

        percentile = {}

        for layer, score in self.sensitivity_score_bias.items():

            percentile[layer] = percentileofscore(
                scores,
                score,
                kind="rank",
            )

        self.sensitivity_score_bias.update(
            {"rank": percentile}
        )

        return None

    def threshold_evolution_bias(self):
        """
        Convert bias evolution scores into percentile rankings.

        Mirrors threshold_evolution exactly, but operates on
        self.evolution_scores_bias (populated by
        compute_evolution_bias) instead of self.evolution_scores.

        Step by step
        ------------
        1. scores = list of all layers' evolution scalars (the
           per-epoch-normalized bias movement from compute_evolution_bias).

        2. For each layer, compute its percentile rank among `scores`
           via percentileofscore(..., kind="rank"). Higher percentile
           means that layer's bias moved more, relative to the other
           layers, over the course of training.

        3. self.evolution_scores_bias["ranks"] is set to a dict of
           {layer_name: percentile}.

        Returns
        -------
        None
            Mutates self.evolution_scores_bias in place.
        """

        rankings = {}

        scores = list(self.evolution_scores_bias.values())

        rankings.update(
            {
                k: percentileofscore(
                    scores,
                    v,
                    kind="rank",
                )
                for k, v in self.evolution_scores_bias.items()
            }
        )

        self.evolution_scores_bias.update({"ranks": rankings})

        return None
