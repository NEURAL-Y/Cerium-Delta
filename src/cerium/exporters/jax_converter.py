import numpy as np
import jax
import joblib


class converter_jax:

    def __init__(
        self,
        model,
        optimizer=None,
        epoch=0,
        save_model=None
    ) -> None:
        """
        Initialize the JAX model converter.

        Parameters
        ----------
        model : PyTree
            JAX/Flax model parameters represented as a PyTree.

            The converter expects the model parameters to be supplied
            using the standard parameter structure defined by the
            Cerium Delta website.

        optimizer : PyTree, optional
            JAX optimizer state represented as a PyTree.

            The optimizer state is extracted separately from the model
            parameters.

        epoch : int, default=0
            Number of training epochs completed by the model.

        save_model : str, default="None"
            Path to the saved model or parameter file.

            The converter expects the saved data to follow the standard
            format defined by the Cerium Delta website.
        """

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.save_model = save_model

    @staticmethod
    def _flatten_named_tree(tree):
        flat = {}
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]

        for path, value in leaves:
            name_parts = []
            for part in path:
                if hasattr(part, "key"):
                    name_parts.append(str(part.key))
                elif hasattr(part, "idx"):
                    name_parts.append(str(part.idx))
                else:
                    name_parts.append(str(part))

            name = ".".join(part for part in name_parts if part not in {"", "None"})
            if not name:
                name = "root"
            flat[name] = np.asarray(value).copy()
        return flat

    @staticmethod
    def _flatten_saved_model(data):
        if isinstance(data, dict):
            flattened = {}
            for key, value in data.items():
                if isinstance(value, (dict, list, tuple)):
                    nested = converter_jax._flatten_saved_model(value)
                    for nested_key, nested_value in nested.items():
                        flattened[f"{key}.{nested_key}" if nested_key != "root" else key] = nested_value
                else:
                    flattened[key] = np.asarray(value).copy()
            return flattened

        if isinstance(data, (list, tuple)):
            flattened = {}
            for index, value in enumerate(data):
                if isinstance(value, (dict, list, tuple)):
                    nested = converter_jax._flatten_saved_model(value)
                    for nested_key, nested_value in nested.items():
                        flattened[f"layer {index}.{nested_key}"] = nested_value
                else:
                    flattened[f"layer {index}"] = np.asarray(value).copy()
            return flattened

        return {"root": np.asarray(data).copy()}

    def extractor_architecture(self) -> dict:
        """
        Extract JAX model parameters, optimizer state, and training
        information.

        Returns
        -------
        dict
            Dictionary containing the extracted JAX model information.
        """

        architecture_parameters = self._flatten_named_tree(self.model)
        optimizer_state = self._flatten_named_tree(self.optimizer) if self.optimizer is not None else {}

        self.culter = {
            "architecture_parameters": architecture_parameters,
            "trained_parameters": {},
            "training_parameters": {},
            "optimizer": optimizer_state,
            "total_layer": len(architecture_parameters),
            "total_epochs": self.epoch,
        }

        if self.save_model is not None:
            self.save_model=str(self.save_model)
            trained_parameters = joblib.load(self.save_model)

            if isinstance(trained_parameters, (dict, list, tuple)):
                flattened = self._flatten_saved_model(trained_parameters)
                self.culter["trained_parameters"] = flattened
                self.culter["training_parameters"] = flattened
            else:
                raise RuntimeError(
                    "save_model_standard_error : you use wrong standard to save your model weights and biases it should be in a list or a dictionary type learn more about--> https://cerium-delta.pages.dev "
                )
        else:
            self.save_model=None
            
        return self.culter
