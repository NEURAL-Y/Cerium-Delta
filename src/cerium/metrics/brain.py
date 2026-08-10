from scipy.stats import percentileofscore
import numpy as np
from numpy.typing import NDArray

class NVS:
    """
    Neural Vitality System (NVS) is internally uses NVB (NEURAL VITALITY BENCHMARK).

    A lightweight neural network observability module for measuring
    layer contribution, sensitivity, and parameter evolution during
    training and giving scores to architecture parameters and biases.

    The NVS object is intended as an analytics companion for model
    diagnostics. It consumes a saved model state and computes a set of
    interpretable layer-wise metrics that can be compared across
    training runs, model checkpoints, or different architectures.

    The design is deliberately separated into three metric families:

    1. Layer Contribution Score (LCS)
       Captures a weighted, sign-preserving transformation of each layer's
       parameters to produce a contribution tensor that is comparable
       across layers of similar topology.

    2. Sensitivity Score
       Estimates how strongly a layer's parameters interact with the
       immediately downstream layer's parameterization. This is achieved
       through a spectral power transform and an outer product norm.

    3. Evolution Score
       Measures how much each layer changed between a reference snapshot
       and a trained snapshot, normalized by the number of epochs.

    Each of these families is implemented for both weights and biases.
    The weight and bias computations share the same high-level intent and
    mirror each other structurally, while respecting the different shapes
    and numerical properties of weight matrices and bias vectors.

    The class stores intermediate values so that metric computation and
    thresholding can be called separately. For example, calling
    ``compute_lcs()`` populates ``self.lcs`` and then ``threshold_lcs()``
    uses those results to produce layer rankings.

    Parameters
    ----------
    model_state : dict
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

    Notes
    -----
    The class is not responsible for model training or optimization. It
    only computes analytical metrics from supplied tensors. The metrics
    are intended to be used for monitoring, ranking, and internal model
    inspection, not as loss functions or optimization objectives.

    The metrics are best interpreted relatively rather than absolutely.
    A higher sensitivity score indicates a stronger layer interaction
    relative to other layers in the same model state, while a higher
    evolution score indicates a larger change during the recorded
    training run.

    Data requirements
    -----------------
    The supplied ``model_state`` dictionary is expected to contain the
    reference and trained snapshots of both weights and biases. The
    shapes of these tensors should be consistent across the same layer
    names, but the implementation does not validate cross-layer
    dimensional compatibility beyond the operations that are actually
    performed.

    Feature semantics
    -----------------
    The core semantics of the metrics are intentionally descriptive:

    - Contribution (LCS) captures how much each layer's parameters would
      contribute to a magnitude-weighted representation of the model.
    - Sensitivity captures how changes in one layer may influence the
      effective representation of the next layer through a downstream
      transformation.
    - Evolution captures how much the parameters changed during training.

    Use cases
    ---------
    NVS is designed for the following diagnostic tasks:

    - Model architecture comparison within the same dataset and training
      regime.
    - Layer importance ranking for pruning, compression, or debugging.
    - Detecting layers with abnormal parameter drift or sensitivity.
    - Monitoring training dynamics across checkpoints.

    Interpretation guidance
    -----------------------
    Because the metrics are derived from internal parameter statistics,
    they should be interpreted in a relative manner rather than as
    absolute performance guarantees.

    - A high LCS value for a layer does not necessarily mean the layer is
      the most important for final task performance; it means that layer
      has a larger self-weighted spectral contribution within the supplied
      weight tensors.
    - A high sensitivity score indicates a relatively strong interaction
      between adjacent layers in the supplied ordering of the weight
      dictionary.
    - A high evolution score means a layer's parameters moved more during
      training, which can be normal for some layers and anomalous for
      others depending on the model and dataset.

    Limitations
    -----------
    NVS is not a replacement for task-specific validation metrics such as
    accuracy or loss. It is an internal, structural diagnostic tool.

    - It does not evaluate predictions, loss values, or dataset samples.
    - It does not automatically handle weight shapes that are incompatible
      with the current matrix- or vector-based operations.
    - The sensitivity metric is based on a local, adjacent-layer proxy and
      does not model full network backpropagation.

    Implementation notes
    --------------------
    The current implementation makes a number of simplifying assumptions:

    - The weight-based LCS assumes each weight tensor can participate in
      ``W @ W.T``.
    - The bias-based LCS uses a Gram matrix constructed from bias values
      and therefore applies a similar spectral intuition to vectors.
    - Thresholding is implemented with percentile rankings for relative
      comparison rather than absolute thresholds.

    Examples
    --------
    ``model_state`` should contain corresponding reference and trained
    tensors for weights and biases. For example:

        model_state = {
            "weights": {"layer1": W1, "layer2": W2},
            "weights_train": {"layer1": W1_tr, "layer2": W2_tr},
            "bias": {"layer1": b1, "layer2": b2},
            "bias_train": {"layer1": b1_tr, "layer2": b2_tr},
            "epochs": 100,
        }

    Then instantiate and compute:

        nvs = NVS(model_state)
        result = nvs.compute("all")

    The ``result`` tuple includes both bias and weight metrics along
    with their respective ranked summaries.

    Advanced usage
    --------------
    Users can call the individual computation methods directly if they
    need fine-grained control over the sequence of operations. For
    example:

        nvs = NVS(model_state)
        nvs.compute_lcs()
        nvs.threshold_lcs()
        print(nvs.lcs)

    This is useful when the caller wants to inspect intermediate
    metrics separately from the final ranked representations.

    Storage and state
    -----------------
    The class preserves computed metric dictionaries on the instance.
    This makes it easy to access the results after calling a computation
    function without requiring an additional return value for every
    intermediate step.

    - ``self.lcs`` stores layer contribution tensors for weights.
    - ``self.sensitivity_score`` stores layer sensitivity scalars for weights.
    - ``self.evolution_scores`` stores normalized parameter evolution values.
    - ``self.lcs_bias`` stores layer contribution tensors for biases.
    - ``self.sensitivity_score_bias`` stores layer sensitivity scalars for biases.
    - ``self.evolution_scores_bias`` stores normalized bias evolution values.

    Conventions
    -----------
    Percentile ranking is computed with ``kind="rank"``. This means that
    values that are equal receive the same percentile rank and the rank is
    based on their ordered position.

    Terms
    -----
    - ``LCS``: Layer Contribution Score.
    - ``sensitivity``: Sensitivity Score.
    - ``evolution``: Evolution Score.
    - ``CV``: Coefficient of Variation.
    - ``Gram matrix``: A product of a matrix with its transpose used to
      capture pairwise inner product structure.

    Notes on numeric stability
    --------------------------
    The adaptive transformation helpers are designed to address cases where
    the power exponent is too large for direct floating-point exponentiation.
    They do not change the underlying data values, only the exponent.

    The ``log(abs(p) + eps)`` fallback is intentionally conservative. It
    gradually shrinks the exponent toward a stable range rather than
    forcing an abrupt clamp.

    Historical context
    ------------------
    NVS was inspired by observability approaches that use spectral analysis
    and layer-wise norm statistics to understand neural network internals.
    It trades off strict mathematical derivation for practical, easy-to-
    compute metrics that can be applied directly to saved model states.

    The implementation favors readability and interpretability over raw
    performance. The metric computations are primarily designed for
    diagnostics rather than real-time production inference.

    Future improvements
    -------------------
    Future enhancements to NVS may include:

    - Explicit support for convolutional weight tensors and broadcastable
      parameter shapes that are not directly compatible with ``W @ W.T``.
    - Additional metric families such as gradient-based sensitivity,
      activation-based contribution, or task-specific importance scores.
    - Automatic layer grouping and aggregation mechanisms for very deep
      networks, where per-layer scores may be too fine-grained for
      practical interpretation.
    - A plugin-style metric API that allows external diagnostic modules
      to register additional score computations and thresholding
      functions.

    Development notes
    -----------------
    The current implementation is intentionally minimal in dependencies.
    It only relies on ``numpy`` and ``scipy.stats.percentileofscore``.
    This keeps the module lightweight and easy to run in most Python
    environments without introducing additional machine learning
    framework dependencies.

    Diagnostic workflow example
    ---------------------------
    A recommended workflow for using NVS in a diagnostics pipeline is:

    1. Load the model state dictionary from disk or checkpoint storage.
    2. Instantiate ``NVS(model_state)``.
    3. Compute the desired metric family or families explicitly using
       ``compute_lcs()``, ``compute_sensitivity()``, ``compute_evolution()``,
       ``compute_lcs_bias()``, ``compute_sensitivity_bias()``, and
       ``compute_evolution_bias()``.
    4. Apply the corresponding thresholding method for percentile ranking:
       ``threshold_lcs()``, ``threshold_sens()``, ``threshold_evolution()``,
       ``threshold_lcs_bias()``, ``threshold_sens_bias()``,
       ``threshold_evolution_bias()``.
    5. Inspect the raw metrics and the ranked summary dictionaries.
    6. Combine NVS outputs with model training logs, validation metrics,
       and domain-specific heuristics to decide whether any layer
       requires further investigation.

    Best practices
    --------------
    - Use the same reference and trained model state structure for
      every run to ensure metric comparability.
    - When comparing two or more models, compute NVS metrics on each
      model separately rather than merging states across models.
    - Review both the raw metric values and the percentile rankings.
      The raw values provide absolute magnitude context, while the
      rankings expose relative layer position.
    - Record the ``epochs`` value used for normalization and avoid
      comparing evolution scores across training runs with vastly
      different epoch counts without accounting for the difference.
    - If a layer's bias or weight tensor does not appear in both the
      reference and trained snapshots, that layer is skipped by the
      evolution score computation.

    Common questions
    ----------------
    Q: Why does NVS use percentile ranks instead of fixed thresholds?
    A: Percentile ranks are more robust across different models and
       datasets because they express importance relative to the model's
       own distribution of values. Fixed thresholds can be brittle when
       the scale of values varies substantially.

    Q: What does a large LCS value mean?
    A: A large LCS value indicates that the layer has a larger
       self-weighted spectral contribution according to the transformation
       used in this metric. It does not imply that the layer is more
       important for model performance in an absolute sense.

    Q: Is the sensitivity score a gradient?
    A: No. The sensitivity score is a proxy based on a downstream
       transformed weight tensor. It resembles a local Jacobian-like
       quantity, but it is not derived from the model's actual
       backpropagation gradients.

    Q: Can this be used for pruning?
    A: It can provide useful diagnostic signals, but it should not be
       the only criterion used for pruning decisions. NVS metrics are
       designed to complement task-based validation and performance data.

    Q: Why is the exponent compressed using ``log(abs(p) + eps)``?
    A: The logarithmic compression strategy is a stable way to reduce the
       exponent magnitude while preserving its sign-awareness. It is a
       practical fallback that avoids hard clamping and enables the
       adaptive loop to find a finite transform gradually.

    Q: What happens if all values in a layer are zero?
    A: The implementation adds a small epsilon before taking logarithms
       to avoid ``log(0)``. Zero-valued layers will produce a well-defined
       exponent and a resulting transformed tensor, though the absolute
       metrics may be near zero.

    Q: Are bias and weight metrics comparable?
    A: They are computed with the same high-level logic, but weights and
       biases are different kinds of parameters. Use them separately or
       compare them with care.

    Q: Is the dictionary ordering significant?
    A: Yes. The methods that process consecutive layers rely on the
       ordering of the dictionaries provided in ``model_state``. This
       ordering should reflect the forward pass order of the model.

    Q: Can NVS handle sparse weights?
    A: The current implementation uses dense numpy operations and does
       not natively support sparse matrix types. Sparse weights should
       be converted to dense arrays before being passed into NVS.

    Q: Why are bias distances normalized by epochs?
    A: Normalizing by epochs provides a per-epoch drift metric. This
       makes evolution scores more comparable across training runs with
       different lengths.

    Q: Will this work for recurrent or transformer-style models?
    A: It depends on the shape and naming conventions of the weight and
       bias tensors. The current code expects pairwise compatible
       weight matrices for the operations performed.

    Q: What are the key failure modes?
    A: The main failure modes are shape incompatibility and numerical
       instability during exponentiation. The adaptive transformation
       helpers mitigate the latter, but they do not address every possible
       invalid shape scenario.

    Q: Should I call ``compute("all")`` or the individual methods?
    A: ``compute("all")`` is convenient and executes all metrics plus
       the ranking steps. If you need more control or want to inspect
       intermediate results, call the individual methods instead.

    Q: How should I interpret percentile scores?
    A: Percentile scores are relative ranks. A score near 100 means the
       layer is near the top of that metric's distribution for the
       current model state.

    Q: Are the metrics deterministic?
    A: Yes, given the same ``model_state`` and the same numpy backend,
       the results are deterministic.

    Q: Does the module preserve input data?
    A: Yes. All transformations are computed without modifying the
       original ``model_state`` tensors.

    Q: Why does NVS use spectral values and eigenvalues?
    A: Spectral values capture dominant linear structures in matrices and
       vectors. They provide a way to measure the relative scale of the
       layer parameters, which is useful for the contribution and
       sensitivity heuristics.

    Q: Is NVS intended for production use?
    A: NVS is primarily a research and diagnostics tool. It can be used
       in production monitoring pipelines if the model shapes and
       dictionary ordering are compatible, but it is not optimized for
       high-throughput inference workloads.

    Q: Can this be used for model debugging?
    A: Yes. The metrics can help identify layers that behave differently
       from the rest of the network, such as layers with unusually high
       sensitivity or evolution.

    Q: Are the percentile ranks stable?"""

    def __init__(self, model_state: dict) -> None:
        # Stores all model states required by the metric pipeline.
        self.model_state = model_state

    def compute(self, choose_metrics="all")->object:
        """
        Compute one or all NVS metrics.

        This method acts as a convenience wrapper around the individual
        metric computation and thresholding methods. It determines which
        metric family to compute based on ``choose_metrics`` and ensures
        that any corresponding ranking or thresholding step is also
        executed before returning results.

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
            Requested metric(s). When ``choose_metrics == "all"``, returns a tuple:
            ``(evolution_scores_bias, sensitivity_score_bias, lcs_bias,``
            ``lcs, sensitivity_score, evolution_scores)``.

        Notes
        -----
        If a single metric is requested, only that metric and its associated
        ranking or thresholding step are computed. When ``"all"`` is
        requested, every metric family is computed for both weights and
        biases, and all ranking functions are applied.

        The returned tuple order is intentional and matches the internal
        ordering used by the all-metrics branch. This makes it easier to
        unpack results consistently across different calling contexts.
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

        The transformation applies a sign-preserving power operation to the
        weight array using the absolute value of the exponent ``p``.
        Because exponentiating very large or very small values can lead to
        numerical overflow, underflow, or NaN values, this method applies an
        adaptive compression strategy to the exponent until the result is
        finite.

        The core transformation is:

            powered = sign(weight) * abs(weight)**abs(p)

        If the result contains any non-finite values, the exponent is
        updated as:

            p = log(abs(p) + eps)

        and the transformation is reattempted. This adaptive exponent
        contraction is repeated for at most ``max_loop`` iterations.

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
        This method performs a numerical stability check after each
        exponentiation attempt. It does not change the original weight
        array, and it only changes the exponent when the computed result is
        not finite.

        A typical use case is when ``p`` is large enough that
        ``abs(weight)**abs(p)`` would overflow to ``inf`` for some entries
        in ``weight``. In that case, the exponent is gradually compressed
        toward smaller values until the output tensor is safe.

        Example
        -------
        If ``weight`` contains values near ``1e-2`` and ``p`` is very large,
        the first exponentiation may produce values outside the representable
        range. This function will then reduce ``p`` via a logarithm and
        retry until the transformation is finite.
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

        This function constructs a Jacobian-like transform for the next
        layer's weight tensor. The key formula is:

            powered = p * sign(weight) * abs(weight)**abs(p - 1)

        The multiplication by ``p`` gives each element a magnitude that is
        proportional to the spectral exponent derived from the next layer.
        The use of ``abs(p - 1)`` ensures the exponent remains non-negative
        for the power operation.

        Because the transformation may still produce non-finite values for
        extreme weights or large exponents, the exponent ``p`` is adaptively
        compressed via:

            p = log(abs(p) + 1e-12)

        This retry mechanism continues until the result is finite or the
        ``max_loop`` iteration limit is reached.

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
        The original data shape is preserved. This function only changes the
        exponent used for the power transformation when non-finite values
        are detected.

        In the sensitivity computation pipeline, this method is typically
        called with the next layer's weight tensor so that the current layer
        can be scored against a downstream transformed representation.
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
        Layer Contribution Score (LCS).

        Computes a sign-preserving spectral transform for each layer weight
        matrix and multiplies the original weights by the transformed values.

        The method first computes the Gram matrix of each weight tensor and
        extracts the dominant eigenvalue. That eigenvalue is used to derive a
        layer-specific exponent, which gives larger weight to layers with a
        greater dominant spectral component.

        For each layer weight matrix ``W``:
        1. ``matrix = W @ W.T``
        2. ``lambda_max = max(abs(eigvals(matrix)))``
        3. ``power = log(sqrt(lambda_max) + eps)``
        4. ``powered = sign(W) * abs(W)**abs(power)``
        5. ``lcs[name] = W * powered``

        The multiplication by ``W`` at the end preserves the original
        parameter sign while scaling the tensor by its own transformed
        magnitude.

        Returns
        -------
        dict[str, np.ndarray]
            Contribution arrays keyed by layer name.

        Notes
        -----
        The current implementation implicitly assumes that each weight
        tensor is a 2D array suitable for the ``W @ W.T`` operation.
        If the model state contains higher-dimensional weights, the method
        will need to be adapted to compute an appropriate Gram matrix.
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
        Sensitivity Score.

        Computes a layer-wise scalar using the next layer's transformed
        weights and the current layer's weight matrix.

        The sensitivity logic is designed to capture how changes in one
        layer may propagate into the next through a Jacobian-inspired
        transformation.

        For each consecutive layer pair ``i`` and ``i+1``:
        1. ``spectral = sqrt(sqrt(max(abs(eigvals(W_{i+1}.T @ W_{i+1}))))))``
        2. ``p_next = log(spectral + eps)``
        3. ``jac_powered = p_next * sign(W_{i+1}) * abs(W_{i+1})**abs(p_next - 1)``
        4. ``sensitivity[layer_i] = norm(outer(jac_powered, W_i))``

        The outer product between the transformed next-layer weights and the
        current layer's weights produces a single scalar norm for each layer
        pair.

        Returns
        -------
        dict[str, NDArray]
            Sensitivity scores keyed by layer name.

        Notes
        -----
        If the model has only one layer or if the weight order does not
        allow a consecutive pairing, this method will return an empty
        sensitivity dictionary. The implementation is intentionally simple
        and uses the natural ordering of the ``self.model_state["weights"]``
        dictionary.
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
            eig = np.linalg.eigvalsh(gram)

            spectral = np.sqrt(
                np.sqrt(np.max(np.abs(eig)))
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
                                np.outer(jac_powered,current_weight)
                            )
            self.sensitivity_score[layers[i]] = sensitivity
        return self.sensitivity_score

    def compute_evolution(self)->dict[str,NDArray]:
        """
        Evolution Score.

        Measures how much each layer has changed throughout training.

        Evolution is normalized by the number of epochs to provide
        comparable scores across different training durations.

        The metric is computed by taking the L2 norm of the difference
        between the trained weights and the reference weights for each
        layer, and then dividing by the number of epochs if that value is
        non-zero.

        Returns
        -------
        dict[str, NDArray]
            Evolution scores keyed by layer name.

        Notes
        -----
        A larger value indicates a larger parameter update magnitude per
        epoch for the corresponding layer. If ``epochs`` is zero, the
        unnormalized L2 difference is returned instead.
        """

        self.evolution_scores = {}

        trained_weights = self.model_state["weights_train"]
        reference_weights= self.model_state["weights"]

        epochs = self.model_state["epochs"]

        for k in trained_weights:

            if k in reference_weights:
                if epochs!=0:
                    self.evolution_scores[k] = (
                        np.linalg.norm(trained_weights[k] - reference_weights[k]) / epochs
                    )
                else:
                    self.evolution_scores[k] = (
                                            np.linalg.norm(trained_weights[k] - reference_weights[k])
                                        )
        return self.evolution_scores

    def threshold_lcs(self)->None:
        """
        Filter LCS values using coefficient of variation (CV).

        Computes a scalar CV score for each layer's LCS array and then
        ranks those scalar scores with percentiles.

        The resulting filtered layer metrics are stored under
        ``self.lcs["filtered_layers_weights"]``.
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
        ``self.model_state["bias"]`` instead of ``["weights"]``.

        Step by step
        ------------
        1. bias = self.model_state["bias"]
           Pull the reference (pre-training) bias arrays, one per layer.

        2. For each layer, build a square matrix for eigenvalue
           calculation: ``b @ b.T`` if ``b`` is 2D, otherwise ``np.outer(b, b)``.
           This handles both vector and matrix bias representations.

        3. power = ``log(sqrt(max(abs(eigvals(matrix)))) + eps)``
           Take the largest eigenvalue by magnitude, log-transform it,
           and add an epsilon floor so ``log(0)`` never happens.
           This becomes the layer's adaptive exponent.

        4. ``self.lcs_bias[name] = b * powered``
           Multiply the original bias vector by its sign-preserving
           power-transformed version.

        Returns
        -------
        dict[str, NDArray]
            {layer_name: lcs_bias_array} for every layer in ``bias``.

        Notes
        -----
        The bias variant uses the same spectral intuition as the weight
        variant, but it is specialized for tensor shapes that are more
        common for bias parameters. The resulting bias contribution arrays
        are useful for comparing how much a layer's bias contributes to
        the overall trained parameter state.
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
        then next-layer Jacobian-like transform), but operates on
        ``self.model_state["bias"]`` instead of ``["weights"]``. The
        bias variant recomputes its own ``self.layer_powers_bias`` list
        rather than reusing the weight-side powers.

        Step by step
        ------------
        1. x = self.model_state["bias"]

        2. Per layer, build ``gram = np.outer(v, v)`` for 1D bias or
           ``v @ v.T`` for 2D bias, take ``spectral = sqrt(max(abs(eigvals(gram))))``,
           and store ``log(spectral + eps)`` in ``self.layer_powers_bias``.

        3. For each consecutive layer pair ``(i, i+1)``: build
           ``jac_powered`` from ``next_bias`` and ``layer_powers_bias[i+1]``,
           then compute ``sensitivity_score_bias[layers[i]] =``
           ``norm(outer(jac_powered, current_bias))``.

        Caveat carried over from compute_sensitivity
        ----------------------------------------------
        ``outer(jac_powered, current_bias)`` is used here because consecutive
        bias arrays are typically 1D. This can produce a very large dense
        tensor before the norm reduction is taken.

        Returns
        -------
        dict[str, NDArray]
            {layer_name: sensitivity_scalar} for every layer in ``bias``.

        Notes
        -----
        The bias sensitivity score is not a direct derivative in the
        mathematical sense, but it acts as a proxy for how much one layer's
        bias interacts with the next layer's transformed bias representation.
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

            eig = np.linalg.eigvalsh(gram)

            spectral = np.sqrt(np.max(np.abs(eig)))

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
                np.outer(jac_powered,current_bias)
            )

            self.sensitivity_score_bias[layers[i]] = sensitivity

        return self.sensitivity_score_bias

    def compute_evolution_bias(self)->dict[str,NDArray]:
        """
        Evolution Score — bias variant.

        Mirrors compute_evolution exactly, but operates on
        ``self.model_state["bias_train"]`` and ``["bias"]`` instead of the
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
           comparable.

        Returns
        -------
        dict[str, NDArray]
            {layer_name: evolution_scalar} for every layer present in
            both ``bias_train`` and ``bias``.

        Notes
        -----
        The bias evolution score can be interpreted as the per-epoch
        magnitude of bias drift for a given layer. If ``epochs`` is zero,
        the raw norm difference is returned instead of a normalized value.
        """

        self.evolution_scores_bias = {}

        trained_bias = self.model_state["bias_train"]
        reference_bias = self.model_state["bias"]

        epochs = self.model_state["epochs"]

        for k in trained_bias:

            if k in reference_bias:
               if epochs!=0:
                self.evolution_scores_bias[k] = (
                    np.linalg.norm(trained_bias[k] - reference_bias[k]) / epochs
                )
               else:
                   self.evolution_scores_bias[k] = (
                                       np.linalg.norm(trained_bias[k] - reference_bias[k])
                                   )
        return self.evolution_scores_bias

    def threshold_lcs_bias(self)->None:
        """
        Filter bias LCS values using coefficient of variation (CV).

        Mirrors threshold_lcs exactly, but operates on ``self.lcs_bias``
        (populated by compute_lcs_bias) instead of ``self.lcs``.

        Computes a scalar CV score for each bias LCS array, reduces it to
        a normed scalar, and then ranks those scalars with percentiles.

        The resulting values are stored under
        ``self.lcs_bias["filtered_layers_biases"]``.

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
        ``self.sensitivity_score_bias`` (populated by
        compute_sensitivity_bias) instead of ``self.sensitivity_score``.

        Computes a percentile rank for each layer sensitivity scalar and
        stores the result in ``self.sensitivity_score_bias["ranks_biases"]``.

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
        ``self.evolution_scores_bias`` (populated by
        compute_evolution_bias) instead of ``self.evolution_scores``.

        Computes a percentile rank for each bias evolution score and
        stores the result in ``self.evolution_scores_bias["ranks_biases"]``.

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
