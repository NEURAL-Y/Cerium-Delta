from scipy.stats import percentileofscore
import numpy as np
from numpy.typing import NDArray

class NVS:
    """
    Neural Vitality System (NVS) is internally uses NVB(NEURAL VITALITY BENCHMARK)

    A lightweight neural network observability module for measuring
    layer contribution, sensitivity, and parameter evolution during
    training and give score to architecture and training weights and bias per layer.

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

    def compute(self, choose_metrics="all")->object:
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
            self.compute_lcs()
            self.threshold_lcs()
            return self.lcs

        elif choose_metrics == "sensitivity":
            self.compute_sensitivity()
            self.threshold_sens()
            return self.sensitivity_score

        elif choose_metrics == "evolution":
            self.compute_evolution()
            self.threshold_evolution()
            return self.evolution_scores

        elif choose_metrics == "lcs_bias":
            self.compute_lcs_bias()
            self.threshold_lcs_bias()
            return self.lcs_bias

        elif choose_metrics == "sensitivity_bias":
            self.compute_sensitivity_bias()
            self.threshold_sens_bias()
            return self.sensitivity_score_bias

        elif choose_metrics == "evolution_bias":
            self.compute_evolution_bias()
            self.threshold_evolution_bias()
            return self.evolution_scores_bias

        else:
            self.compute_lcs()
            self.compute_sensitivity()
            self.compute_evolution()
            self.compute_lcs_bias()
            self.compute_sensitivity_bias()
            self.compute_evolution_bias()
            self.threshold_evolution()
            self.threshold_lcs_bias()
            self.threshold_sens_bias()
            self.threshold_evolution_bias()
            self.threshold_lcs()
            self.threshold_sens()
            return (self.evolution_scores_bias,self.sensitivity_score_bias,self.lcs_bias,self.lcs,self.sensitivity_score,self.evolution_scores
            )
    def adaptive_transformation_lcs(
        self,
        p,
        weight,
        max_loop=500
    ) -> NDArray[np.float64]:
        """
        Adaptively stabilize the LCS power transformation.

        Applies:

            sign(W) * |W|^|p|

        and checks whether the result is finite. If the transformation
        produces inf or NaN, the exponent is compressed using:

            p = log(|p| + eps)

        The process repeats until the transformation is finite or
        ``max_loop`` is reached.

        Parameters
        ----------
        p : float
            Initial exponent derived from the layer's spectral property.

        weight : NDArray[np.float64]
            Weight array to which the power transformation is applied.

        max_loop : int, default=500
            Maximum number of adaptive transformation attempts. This
            prevents an unbounded loop for numerically extreme inputs.

        Returns
        -------
        NDArray[np.float64]
            Power-transformed weight array after numerical stabilization.

        Notes
        -----
        The weight values themselves are not modified. Only the exponent
        is adaptively transformed when the power operation becomes
        non-finite. Layers that are already numerically stable exit
        without additional transformations.
        """

        current_p = p

        powered = (
            np.sign(weight)
            * np.power(
                np.abs(weight),
                np.abs(current_p)
            )
        )

        for _ in range(max_loop):
            try:
                with np.errstate(
                    over="raise",
                    invalid="raise"
                ):
                    powered = powered

                    if not np.all(np.isfinite(powered)):
                        raise FloatingPointError(
                            "not finite value"
                        )
                    else:
                        break

            except FloatingPointError:
                current_p = np.log(
                    np.abs(current_p) + 1e-12
                )

        return powered

    def adaptive_transformation_sens(
    self,
    p,
    weight,
    max_loop=500
) -> NDArray[np.float64]:
        """
        Adaptive transformation for the sensitivity calculation.

        Applies the sensitivity power transformation:

            p * sign(W) * |W|^(|p| - 1)

        where W represents the layer weights and p is the
        layer-specific power derived from the spectral property.

        The transformation is designed to handle numerical overflow
        or non-finite values that may occur when extreme weight values
        interact with the power term.

        If the transformed values are not finite, the current exponent
        is compressed using:

            p = log(|p| + 1e-12)

        This adaptive transformation can be repeated up to ``max_loop``
        iterations. The loop can be stopped once a finite result is
        obtained.

        Parameters
        ----------
        p : float
            Initial power associated with the current layer.

        weight : NDArray[np.float64]
            Weight array used in the sensitivity transformation.

        max_loop : int, default=500
            Maximum number of adaptive transformation iterations.

        Returns
        -------
        NDArray[np.float64]
            Sensitivity-transformed weight array.

        Notes
        -----
        The original weight values are not modified. Only the exponent
        is adaptively transformed when the result is non-finite.
        """

        current_p = p

        powered = (
            current_p
            * np.sign(weight)
            * np.power(
                np.abs(weight),
                np.abs(current_p - 1)
            )
        )

        for _ in range(max_loop):
            try:
                with np.errstate(
                    invalid="raise",
                    over="raise"
                ):
                    powered = powered

                    if not np.all(np.isfinite(powered)):
                        raise FloatingPointError(
                            "Value is not finite"
                        )
                    else:
                        break

            except FloatingPointError:
                current_p = np.log(
                    np.abs(current_p) + 1e-12
                )

        return powered
    def compute_lcs(self) -> dict[str, np.ndarray]:
        """
        Layer Contribution Score (LCS)
        """

        self.lcs = {}
        self.layer_powers = {}

        weights = self.model_state["weights"]

        for name, weight in weights.items():

            weight = weight.astype(np.float64)

            # Gram matrix
            matrix = weight @ weight.T

            # Dominant eigenvalue
            eigenvalues = np.linalg.eigvalsh(matrix)

            lambda_max = np.max(np.abs(eigenvalues))

            # Initial spectral power
            power = np.log(
                np.sqrt(lambda_max) + 1e-12
            )

        # Adaptive transformation
            powered= self.adaptive_transformation_lcs(
                p=power,
                weight=weight
            )

            # Store the actual exponent used
            # self.layer_powers[name] = final_power

            # LCS
            self.lcs[name] = weight * powered

        return self.lcs
    
    def compute_sensitivity(self)->dict[str,NDArray]:
        """
        Sensitivity Score

        Estimates how responsive each layer is to parameter changes by
        combining the derivative of the LCS formulation with the layer
        magnitude (L2 norm).

        Higher values indicate that small parameter variations have
        larger influence.
        """

        self.sensitivity_score = {}
        x = {
            k: v.astype(np.float64)
            for k, v in self.model_state["weights"].items()
        }
        self.layer_powers=[]
        layers = list(x.keys())
        for v in x.values():
            gram = v.T @ v
            eig = np.sqrt(np.linalg.eigvals(gram))

            spectral = np.sqrt(
                np.max(np.abs(eig))
            )

            self.layer_powers.append(
                np.log(spectral + 1e-12)
    )
        for i in range(len(layers) - 1):

            current_weight = x[layers[i]]
            next_weight = x[layers[i + 1]]

            p_next = self.layer_powers[i + 1]

            # Adaptive transformation - protects the (p_next - 1)
            # exponent used in the jacobian, validated against the
            # actual downstream matmul with current_weight.
            jac_powered= self.adaptive_transformation_sens(
                p=p_next,
                weight=next_weight
            )


            sensitivity = np.linalg.norm(
                current_weight @ jac_powered
            )

            self.sensitivity_score[layers[i]] = sensitivity
        return self.sensitivity_score

    def compute_evolution(self)->dict[str,NDArray]:
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

    def threshold_lcs(self)->None:
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
            filtered_layer.update({k: cv})

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

        filtered_layer.update({"ranks_weights": Ranking})
        self.lcs.update({"filtered_layers_weights": filtered_layer})

        return None

    def threshold_sens(self)->None:
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
            {"ranks_weights": percentile}
        )

        return None

    def threshold_evolution(self)->None:
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

        self.evolution_scores.update({"ranks_weights": rankings})

        return None

    # ------------------------------------------------------------------
    # BIAS-BASED METRICS (mirrors weight logic exactly, "bias"/"bias_train")
    # ------------------------------------------------------------------

    def compute_lcs_bias(self)->dict[str,NDArray]:
        """
        Layer Contribution Score (LCS) — bias variant.

        Mirrors compute_lcs exactly, but operates on
        self.model_state["bias"] instead of ["weights"].

        Step by step
        ------------
        1. bias = self.model_state["bias"]
           Pull the reference (pre-training) bias arrays, one per layer.

        2. For each layer, build a square matrix for eigenvalue
           calculation: b @ b.T if 2D, otherwise np.outer(b, b) — this
           is what actually handles 1D bias vectors correctly (unlike
           the old stacked-eigvals version).

        3. power = log(|max(eigvals(matrix))|) + 1e-12
           Take the largest eigenvalue by magnitude, log-transform it,
           and add an epsilon floor so log(0) never happens. This
           becomes that layer's adaptive exponent — layers with a
           dominant/large eigenvalue get a larger power. Stored per
           layer name in self.layer_powers_bias (dict, keyed like the
           weight version's self.layer_powers).

        4. self.lcs_bias[name] = b * (sign(b) * |b|^power)
           Sign-preserving power scaling identical to compute_lcs,
           then multiplied back by the original bias values.

        Returns
        -------
        dict
            {layer_name: lcs_bias_array} for every layer in "bias".
        """

        self.lcs_bias = {}
        self.layer_powers_bias = {}

        bias = self.model_state["bias"]

        for name, b in bias.items():

            b = b.astype(np.float64)

            # Make square matrix for eigenvalue calculation
            if b.ndim == 2:
                matrix = b @ b.T
            else:
                matrix = np.outer(b, b)

            eigenvalues = np.linalg.eigvalsh(matrix)
            lambda_max = np.max(np.abs(eigenvalues))

            power = np.log(np.sqrt(lambda_max) + 1e-12)

            # Adaptive transformation - same protection as compute_lcs
            powered= self.adaptive_transformation_lcs(
                p=power,
                weight=b,
            )

            self.lcs_bias[name] = b * powered

        return self.lcs_bias

    def compute_sensitivity_bias(self)->dict[str,NDArray]:
        """
        Sensitivity Score — bias variant.

        Mirrors compute_sensitivity exactly (own-layer spectral power,
        then next-layer jacobian chained via current_bias @
        next_jacobian), but operates on self.model_state["bias"]
        instead of ["weights"]. self.layer_powers_bias is recomputed
        here locally, same as compute_sensitivity recomputes its own
        self.layer_powers rather than reusing compute_lcs's.

        Step by step
        ------------
        1. x = self.model_state["bias"]

        2. Per layer, build gram = outer(v, v) (1D bias) or v @ v.T
           (2D bias), take spectral = sqrt(max(|eigvals(gram)|)), and
           store log(spectral + eps) in self.layer_powers_bias, in
           layer order (list, not dict — matches compute_sensitivity).

        3. For each consecutive layer pair (i, i+1): build
           next_jacobian from next_bias and layer_powers_bias[i+1],
           then sensitivity_score_bias[layers[i]] =
               norm(current_bias @ next_jacobian)

        Caveat carried over from compute_sensitivity
        ----------------------------------------------
        current_bias @ next_jacobian is a plain dot product for 1D
        arrays, which requires current_bias and next_bias to be the
        SAME length. Weight matrices chain dimensions (out_i ==
        in_{i+1}) so this always lines up; consecutive bias vectors
        generally do NOT share a length (e.g. an 8-unit layer feeding
        a 16-unit layer), so this will raise a shape error on most
        real models. Flagging this rather than silently reshaping —
        let me know if you want it changed to elementwise/broadcast
        instead of a literal mirror of the weight version.

        Returns
        -------
        dict
            {layer_name: sensitivity_scalar} for every layer in "bias".
        """

        self.sensitivity_score_bias = {}
        x = {
            k: v.astype(np.float64)
            for k, v in self.model_state["bias"].items()
        }
        self.layer_powers_bias = []
        layers = list(x.keys())

        for v in x.values():

            if v.ndim == 2:
                gram = v @ v.T
            else:
                gram = np.outer(v, v)

            eig = np.linalg.eigvals(gram)

            spectral = np.max(np.abs(eig))

            self.layer_powers_bias.append(
                np.log(spectral + 1e-12)
            )

        for i in range(len(layers) - 1):

            current_bias = x[layers[i]]
            next_bias = x[layers[i + 1]]

            p_next = self.layer_powers_bias[i + 1]

            # Adaptive transformation - same protection as
            # compute_sensitivity, but validated against the outer
            # product (bias vectors don't share a contraction
            # dimension the way weight matrices do).
            jac_powered = self.adaptive_transformation_sens(
                p=p_next,
                weight=next_bias,
            )

            sensitivity = np.linalg.norm(
                np.outer(current_bias, jac_powered)
            )

            self.sensitivity_score_bias[layers[i]] = sensitivity

        return self.sensitivity_score_bias

    def compute_evolution_bias(self)->dict[str,NDArray]:
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

    def threshold_lcs_bias(self)->None:
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
            filtered_layer.update({k:cv})

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

        filtered_layer.update({"ranks_biases": Ranking})
        self.lcs_bias.update({"filtered_layers_biases": filtered_layer})

        return None

    def threshold_sens_bias(self)->None:
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
            {"ranks_biases": percentile}
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

        self.evolution_scores_bias.update({"ranks_biases": rankings})

        return None
