from ..metrics.brain import NVS
from numpy.typing import NDArray
from typing import Literal
import numpy as np


class bridge:
    """Bridge class for converting framework-specific model metadata into one common format.

    This class acts as a compatibility layer between different DL and ML frameworks and the
    internal NVS data structure used later for analysis.

    Components handled here:
    - framework: the selected backend, such as torch, tensorflow, sklearn, or jax.
    - model: the trained or untrained model object supplied by the user.
    - optimizer: optional optimizer state for frameworks that expose it separately.
    - epoch: number of training epochs already completed.
    - save_model: optional saved checkpoint or parameter file.
    - compute_choice: the scoring mode selected for NVS computation.

    The bridge keeps the final output dictionary consistent so downstream code can read
    weights, biases, trained parameters, and epoch information without needing to know
    which framework produced the data.
    """

    def __init__(
        self,
        model: object,
        *,
        framework: Literal["torch", "tensorflow", "sklearn", "jax"],
        compute_choice: Literal[
            "lcs",
            "sensitivity",
            "evolution",
            "all",
            "lcs_bias",
            "lcs_weight",
            "sensitivity_weight",
            "evolution_bias",
            "evolution_weight",
            "sensitivity_bias",
        ] = "lcs",
        max_loop: int = 500,
        epoch: int = 0,
        device: str = "cpu",
        save_model: str | None = None,
        optimizer: object | None = None,
    ) -> None:

        self.framework = framework
        self.compute_choice = compute_choice
        self.model = model
        self.device = device
        self.epoch = epoch
        self.save_model = save_model
        self.optimizer = optimizer
        self.max_loop = max_loop

        self.nvs_memory = {
            "weights": {},
            "weights_train": {},
            "bias": {},
            "bias_train": {},
            "epochs": 0,
            "co_relations_layers": {},
        }

    def checker(self) -> None:
        """Choose and initialize the correct converter according to the selected framework."""

        if self.framework == "torch":

            from .torch_converter import converter_pytorch

            self.convert = converter_pytorch(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.epoch,
                device=self.device,
                save_model=self.save_model,
            )

        elif self.framework == "tensorflow":

            from .tensorflow_converter import converter_tensorflow

            self.convert = converter_tensorflow(
                self.model,
                self.epoch,
                self.optimizer,
                self.save_model,
                self.device,
            )

        elif self.framework == "sklearn":

            from .sklearn_converter import converter_sklearn

            self.convertsk = converter_sklearn(
                self.model,
                self.save_model,
            )

        elif self.framework == "jax":

            from .jax_converter import converter_jax

            self.convert = converter_jax(
                self.model,
                self.optimizer,
                self.epoch,
                self.save_model,
            )

        else:

            raise RuntimeError(
                "FRAMEWORK_FOUND_ERROR : framework is not found in our list. "
                "Please use one of [torch, tensorflow, sklearn, jax]. "
                "Read our documentation for more information: "
                "https://cerium-delta.pages.dev"
            )

    def information_extract(self) -> dict:
        """Convert framework-specific extracted values into the common NVS memory layout."""

        self.checker()

        self.nvs_memory = {
            "weights": {},
            "weights_train": {},
            "bias": {},
            "bias_train": {},
            "epochs": 0,
            "co_relations_layers": {},
        }

        self.layer_current = {
            "torch_weight_index": 0,
            "torch_bias_index": 0,
            "tensorflow_bias_index": 0,
            "tensorflow_weight_index": 0,
        }

        self.layer_train = {
            "torch_weight_index": 0,
            "torch_bias_index": 0,
            "tensorflow_bias_index": 0,
            "tensorflow_weight_index": 0,
        }

        if self.framework == "sklearn":

            self.infosk = self.convertsk.extractor_architecture()

            for i, (k, v) in enumerate(
                self.infosk["architecture_parameters"].items()
            ):

                self.nvs_memory["co_relations_layers"][f"layer {i}"] = k

                if k.endswith("bias"):
                    self.nvs_memory["bias"][k.removesuffix(" bias")] = v
                else:
                    self.nvs_memory["weights"][k.removesuffix(" weight")] = v

            for k, v in self.infosk["training_parameters"].items():

                if k.endswith("bias"):
                    self.nvs_memory["bias_train"][k.removesuffix(" bias")] = v
                else:
                    self.nvs_memory["weights_train"][k.removesuffix(" weight")] = v

            self.total_step = self.infosk["total_steps"]
            self.nvs_memory["epochs"] = self.total_step

        else:

            self.info = self.convert.extractor_architecture()

            self.architecture_info = self.info["architecture_parameters"]
            self.training_info = self.info["training_parameters"]

            self.total_step = self.info.get(
                "total_epochs",
                self.info.get("total_steps", 0),
            )

            self.nvs_memory["epochs"] = self.total_step

            for i, (k, v) in enumerate(self.architecture_info.items()):

                self.nvs_memory["co_relations_layers"][f"layer {i}"] = k

                if self.framework == "torch":

                    self.torch_parameters_classifier(
                        k,
                        v,
                        layer_idx=i,
                    )

                elif self.framework == "tensorflow":

                    self.tensorflow_parameters_classifier(
                        k,
                        v,
                        layer_idx=i,
                    )

                elif self.framework == "jax":

                    self.jax_parameters_classifier(
                        k,
                        v,
                        layer_idx=i,
                    )

            for k, v in self.training_info.items():

                if self.framework == "torch":

                    self.torch_parameters_classifier(
                        k,
                        v,
                        reference="train",
                    )

                elif self.framework == "tensorflow":

                    self.tensorflow_parameters_classifier(
                        k,
                        v,
                        reference="train",
                    )

                elif self.framework == "jax":

                    self.jax_parameters_classifier(
                        k,
                        v,
                        reference="train",
                    )

        return self.nvs_memory

    def nvs_export_info(self) -> object:

        self.nvs_mem = self.information_extract()

        nvs = NVS(
            self.nvs_mem,
            self.max_loop,
        )

        try:

            self.nvs_result = nvs.compute(
                self.compute_choice
            )

            return self.nvs_result

        except Exception as e:

            raise RuntimeError(
                "File_caught_bug : there is something which struck "
                f"the operations {e}\n"
                "You can report us on --> "
                "https://cerium-delta.pages.dev/feedback"
            )

    def pyarr_to_onnx(
        self,
        *,
        arr: NDArray,
        output_path: str,
        name_arr: str,
    ) -> None:

        """Convert a NumPy array into an ONNX model containing only that array."""

        import onnx
        from onnx import numpy_helper, helper, TensorProto

        arr = np.asarray(
            arr,
            dtype=np.float32,
        )

        onnx_tensor = numpy_helper.from_array(
            arr,
            name=name_arr,
        )

        graph = helper.make_graph(
            nodes=[],
            name="save_array_only",
            inputs=[],
            outputs=[
                helper.make_tensor_value_info(
                    name_arr,
                    TensorProto.FLOAT,
                    list(arr.shape),
                )
            ],
            initializer=[
                onnx_tensor
            ],
        )

        model = helper.make_model(
            graph
        )

        onnx.save(
            model,
            output_path,
        )

    def torch_parameters_classifier(
        self,
        name,
        parameters,
        reference="current",
        layer_idx=None,
    ) -> None:

        if layer_idx is None:
            layer_idx = 0

        layer_name = f"layer {layer_idx}"

        if reference == "current":

            if name.endswith(".weight"):

                self.nvs_memory["weights"][layer_name] = parameters

            elif name.endswith(".bias"):

                self.nvs_memory["bias"][layer_name] = parameters

        else:

            if name.endswith(".weight"):

                self.nvs_memory["weights_train"][layer_name] = parameters

            elif name.endswith(".bias"):

                self.nvs_memory["bias_train"][layer_name] = parameters

    def tensorflow_parameters_classifier(
        self,
        name,
        parameters,
        reference="current",
        layer_idx=None,
    ) -> None:

        if layer_idx is None:
            layer_idx = 0

        layer_name = f"layer {layer_idx}"

        if reference == "current":

            if "kernel" in name.lower():

                self.nvs_memory["weights"][layer_name] = parameters

            elif "bias" in name.lower():

                self.nvs_memory["bias"][layer_name] = parameters

        else:

            if "kernel" in name.lower():

                self.nvs_memory["weights_train"][layer_name] = parameters

            elif "bias" in name.lower():

                self.nvs_memory["bias_train"][layer_name] = parameters

    def jax_parameters_classifier(
        self,
        name,
        parameters,
        reference="current",
        layer_idx=None,
    ) -> None:

        name = str(name).lower()

        if layer_idx is not None:

            layer_name = f"layer {layer_idx}"

        else:

            layer_name = "layer 0"

            if "." in name:

                candidate = name.split(".")[0]

                if candidate.startswith("layer"):

                    layer_name = candidate

        target_key = (
            "weights"
            if reference == "current"
            else "weights_train"
        )

        bias_key = (
            "bias"
            if reference == "current"
            else "bias_train"
        )

        if "kernel" in name or name.endswith("weight"):

            self.nvs_memory[target_key][layer_name] = parameters

        if "bias" in name:

            self.nvs_memory[bias_key][layer_name] = parameters
