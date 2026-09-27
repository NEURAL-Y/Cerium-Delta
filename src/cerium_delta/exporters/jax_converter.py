import numpy as np
import jax


class converter_jax:

    def __init__(
        self,*,
        model,
        optimizer=None,
        epoch:int=0,
        model_test:object|None=None
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

        model_test : object, default="None"
            Optional parameter PyTree from a test or trained model. Its
            leaves are flattened into ``training_parameters``.
        """

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.model_test=model_test
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
        training_parameters = (
            self._flatten_named_tree(self.model_test)
            if self.model_test is not None
            else {}
        )
        optimizer_state = self._flatten_named_tree(self.optimizer) if self.optimizer is not None else {}

        self.culter = {
            "architecture_parameters": architecture_parameters,
            "trained_parameters": {},
            "training_parameters": training_parameters,
            "optimizer": optimizer_state,
            "total_layer": len(architecture_parameters),
            "total_epochs": self.epoch,
        }
            
        return self.culter
