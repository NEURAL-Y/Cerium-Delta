import numpy as np
import jax
import joblib


class converter_jax:

    def __init__(
        self,
        model,
        optimizer=None,
        epoch=0,
        save_model="None"
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

        Attributes
        ----------
        model
            Stores the JAX model parameter PyTree.

        optimizer
            Stores the JAX optimizer state PyTree.

        epoch
            Stores the number of completed training epochs.

        save_model
            Stores the path to the saved model or parameter file.
        """

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.save_model = save_model

    def extractor_architecture(self) -> dict:
        """
        Extract JAX model parameters, optimizer state, and training
        information.

        Returns
        -------
        dict
            Dictionary containing the extracted JAX model information.

            ``architecture_parameters``
                Contains model parameters extracted from the JAX PyTree.
                Each JAX array is converted into an independent NumPy
                array.

            ``training_parameters``
                Contains parameters loaded from the saved model file
                according to the standard Cerium Delta format.

            ``optimizer``
                Contains optimizer state extracted from the optimizer
                PyTree and converted into NumPy arrays.

            ``total_layer``
                Number of parameter arrays contained in the model PyTree.

            ``total_epoch``
                Number of training epochs supplied to the converter.
        """

        self.culter = {
            "architecture_parameters": {},
            "training_parameters": {},
            "optimizer": {},
            "total_layer": 0,
            "total_epoch": self.epoch
        }

        # Extract model parameters
        model_parameters = jax.tree_util.tree_leaves(self.model)

        for index, parameter in enumerate(model_parameters):

            self.culter["architecture_parameters"][
                f"parameter_{index}"
            ] = np.asarray(parameter).copy()

        self.culter["total_layer"] = len(model_parameters)

        # Extract optimizer state
        if self.optimizer is not None:

            optimizer_parameters = jax.tree_util.tree_leaves(
                self.optimizer
            )

            for index, parameter in enumerate(optimizer_parameters):

                self.culter["optimizer"][
                    f"state_{index}"
                ] = np.asarray(parameter).copy()

        # Extract saved trained parameters
        if self.save_model != "None":

            trained_parameters = joblib.load(self.save_model)

            if isinstance(trained_parameters, dict):

                for key, value in trained_parameters.items():
                    self.culter["training_parameters"][key] = (
                        np.asarray(value).copy()
                    )

            elif isinstance(trained_parameters,list):
                            for i, weight in enumerate(trained_parameters):
                                self.culter["training_parameters"][f"layer {i} weights"] = np.asarray(weight).copy()
            else:
                            raise RuntimeError("save_model_standard_error : you use wrong standard to save your model weights and biases it should be in a list or a dictionary type learn more about--> https://cerium-delta.pages.dev ")
        return self.culter
