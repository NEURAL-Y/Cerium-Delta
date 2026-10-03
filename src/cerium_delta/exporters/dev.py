from ..metrics.brain import NVS
from numpy.typing import NDArray
from typing import Literal
import json as js
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
    weights, biases, trained parameters, and epoch information without needing to know which
    framework produced the data.
    """
    def __init__(self,model:object|None=None,*,framework:Literal["torch","tensorflow","sklearn","jax"],compute_choice:Literal["lcs","sensitivity","evolution","all","lcs_bias","lcs_weight","sensitivity_weight","evolution_bias","evolution_weight","sensitivity_bias"]="lcs",max_loop:int=500,epoch:int=0,optimizer:object|None=None,model_test:object|None=None)->None:
        """Initialize the bridge with the model, framework, and training metadata.

        Parameters
        ----------
        model : object
            Model or parameter object supplied by the selected framework.
        framework : str
            Name of the ML framework, such as "torch", "tensorflow", "sklearn", or "jax".
        compute_choice : str, optional
            The NVS compute mode selected for analysis.
        epoch : int, optional
            Number of completed training epochs.
        device : str, optional
            Device used for tensor movement, mostly relevant for PyTorch.
        model_test : object, optional
            to a saved model or checkpoint.
        optimizer : object, optional
            Optimizer object or optimizer state for the framework.
        """
        self.framework=framework
        self.compute_choice=compute_choice
        self.model=model
        self.epoch=epoch
        self.optimizer=optimizer
        self.model_test=model_test
        self.max_loop=max_loop
        self.nvs_memory={"weights":{},"weights_train":{},"bias":{},"bias_train":{},"epochs":0,"co_relations_layers":{}}
    
    def checker(self)->None:
        """Choose and initialize the correct converter according to the selected framework.

        This method stores the framework-specific converter in either ``self.convert`` or
        ``self.convertsk`` depending on the framework type. The actual logic for conversion is
        delegated to the converter classes rather than being implemented here.
        """
        if self.framework=="torch":
           from torch_converter import converter_pytorch
           # PyTorch model parameters are converted through the torch-specific extractor.
           self.convert=converter_pytorch(model=self.model,optimizer=self.optimizer,epoch=self.epoch,model_test=self.model_test)

        elif self.framework=="tensorflow":
            from tensorflow_converter import converter_tensorflow
           # TensorFlow variables use the framework's variable naming convention.
            self.convert=converter_tensorflow(model=self.model,epoch=self.epoch,optimizer=self.optimizer,model_test=self.model_test)

        elif self.framework=="sklearn":
            from sklearn_converter import converter_sklearn
           # sklearn models do not use a training optimizer in the same way as deep learning models.
            self.convertsk=converter_sklearn(model=self.model)

        elif self.framework=="jax":
            from jax_converter import converter_jax
           # JAX models are stored as PyTrees, so the JAX converter extracts named leaves.
            self.convert=converter_jax(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.epoch,
                model_test=self.model_test,
            )
        else:
            raise RuntimeError(
                "FRAMEWORK_FOUND_ERROR : framework is not found in our list please use this framework only from our list [torch,tensorflow,sklearn,jax] \n why we choose only this list read our docs for more information visit our website---> https://cerium-delta.pages.dev"
            )
        
    def information_extract(self)->dict:
      """Convert framework-specific extracted values into the common NVS memory layout.

      This method gathers all extracted parameter information from the selected framework,
      classifies it into the shared weight and bias buckets, and returns a standardized
      dictionary that downstream analysis code can consume.

      Returns
      -------
      dict
          Dictionary containing the following main entries:

          weights : dict
              Current weight values grouped by layer.
          weights_train : dict
              Trained or saved weight values grouped by layer.
          bias : dict
              Current bias values grouped by layer.
          bias_train : dict
              Trained or saved bias values grouped by layer.
          epochs : int
              Total number of training epochs.
          co_relations_layers : dict
              Mapping between each layer label and the original parameter names.

      Notes
      -----
      The classification logic remains framework-aware because PyTorch uses ".weight" while
      TensorFlow and JAX typically use "kernel". Even so, the final storage format is made
      consistent so the rest of the project can treat all frameworks in the same way.
      """
      self.checker()
      self.nvs_memory={"weights":{},"weights_train":{},"bias":{},"bias_train":{},"epochs":0,"co_relations_layers":{}}

      self.layer_current={"torch_weight_index":0,"torch_bias_index":0,"tensorflow_bias_index":0,"tensorflow_weight_index":0}

      self.layer_train={"torch_weight_index":0,"torch_bias_index":0,"tensorflow_bias_index":0,"tensorflow_weight_index":0}

      if self.framework=="sklearn":
            
            self.infosk=self.convertsk.extractor_architecture()

            for i,(k,v) in enumerate(self.infosk["architecture_parameters"].items()):

                self.nvs_memory["co_relations_layers"][f"layer {i}"]=k

                if k.endswith("bias"):

                    self.nvs_memory["bias"][k.removesuffix(" bias")]=v

                else:

                    self.nvs_memory["weights"][k.removesuffix(" weight")]=v

            for k,v in self.infosk["training_parameters"].items():
                            
                            if k.endswith("bias"):

                                self.nvs_memory["bias_train"][k.removesuffix(" bias")]=v

                            else:

                                self.nvs_memory["weights_train"][k.removesuffix(" weight")]=v

            self.total_step=self.infosk["total_steps"]
            self.nvs_memory["epochs"]=self.total_step

      else:
            
            self.info=self.convert.extractor_architecture()

            self.architecture_info=self.info["architecture_parameters"]

            self.training_info=self.info["training_parameters"]

            self.total_step=self.info.get("total_epochs", self.info.get("total_steps", 0))
            self.nvs_memory["epochs"]=self.total_step

            

            for i,(k,v) in enumerate(self.architecture_info.items()):

                self.nvs_memory["co_relations_layers"][f"layer {i}"]=k

                if self.framework=="torch":

                    self.torch_parameters_classifier(k,v)

                elif self.framework=="tensorflow":

                    self.tensorflow_parameters_classifier(k,v)

                elif self.framework=="jax":

                    self.jax_parameters_classifier(k,v)
            
            for k,v in self.training_info.items():
                if self.framework=="torch":

                    self.torch_parameters_classifier(k,v,"train")

                elif self.framework=="tensorflow":

                    self.tensorflow_parameters_classifier(k,v,"train")

                elif self.framework=="jax":

                    self.jax_parameters_classifier(k,v,"train")
                    
      return self.nvs_memory
    
    def nvs_export_info(self)->object:
        self.nvs_mem=self.information_extract()
        nvs=NVS(self.nvs_mem,self.max_loop)

        try:
            
            self.nvs_result=nvs.compute(self.compute_choice)
            return self.nvs_result

        except Exception as e:
           raise RuntimeError(f"File_caught_bug : there is something which struck the operations {e} \n you can report us on --> https://cerium-delta.pages.dev/feedback")

    def pyarr_to_onnx(self,*,arr:NDArray,output_path:str,name_arr:str)->None:

        """Convert a NumPy array into an ONNX model containing only that array.
        
        Parameters
        ----------
        arr : NDArray
            NumPy array to be converted into an ONNX initializer.
        output_path : str
            File path where the ONNX model will be saved.
        name_arr : str
            Name to assign to the ONNX tensor corresponding to the NumPy array.
        """
        import onnx
        from onnx import numpy_helper, helper, TensorProto
        
        onnx_tensor = numpy_helper.from_array(arr, name=name_arr)
        graph = helper.make_graph(
            nodes=[], # no computation
            name="save_array_only",
            inputs=[], # nothing to feed at runtime
            outputs=[
                helper.make_tensor_value_info(name_arr, TensorProto.FLOAT, arr.shape)
            ],
            initializer=[onnx_tensor] 
        )
        model = helper.make_model(graph)
        onnx.save(model, output_path)

    def nvs_output_to_json(self,*,nvs_dict:dict,filename:str,indent:int=4):
      """Write an NVS result dictionary to a formatted JSON file.

      Pass an NVS result dictionary, such as the result returned by
      ``nvs_export_info``. NumPy arrays and scalars are converted to their
      corresponding JSON-compatible Python values.
      The ``indent`` option controls the whitespace used to format the file.

      Parameters
      ----------
      nvs_dict : dict
          NVS metric results or other JSON-serializable output data. NumPy
          arrays and scalars are supported.
      filename : str
          Destination path. The extension must be ``.json``.
      indent : int, default=4
          Number of spaces used for each indentation level.

      Raises
      ------
      AttributeError
          If ``filename`` does not end in ``.json``.

      Example
      -------
      ``model_bridge.nvs_output_to_json(nvs_dict=result, filename="nvs.json")``
      """
      if filename.endswith(".json"):
        def json_default(value):
          if isinstance(value, np.ndarray):
              return value.tolist()
          if isinstance(value, np.generic):
              return value.item()
          raise TypeError(
              f"Object of type {type(value).__name__} is not JSON serializable"
          )

        with open(filename,"w") as f:
            js.dump(nvs_dict,f,indent=indent,default=json_default)
      else:
        raise AttributeError("FILENAME_INVALID : use <filename>.json")
    
    def show_hierarchy(self,d:dict, indent=0):
        """
        sees the hierarchy of the *NVS OUTPUT Dictionary*.

        Parameters
        ---------
        d: given the dict as an input

        indent:  default no need to configure
        
        """
        for key, value in d.items():
            print("  " * indent + str(key))
    
            if isinstance(value, dict):
                self.show_hierarchy(value, indent + 1)
    def _dataframe_validator(self,nvs_dict:dict):
        """Normalize NVS metric dictionaries for DataFrame construction.

        The input may be the combined result returned by ``NVS.compute("all")``,
        one named metric such as ``{"sensitivity_score": ...}``, or an individual
        metric dictionary containing layer values and optional ranking fields.
        Metric names and common aliases are normalized before extracting scalar
        summaries. Tensor-valued parameter data is represented by its norm rather
        than expanded into one DataFrame column per tensor element.

        Parameters
        ----------
        nvs_dict : dict
            Combined NVS output or a dictionary for one metric family.

        Returns
        -------
        dict
            Mapping from ``(metric, parameter_type, statistic)`` column keys to
            dictionaries of layer names and scalar values. This is an internal
            intermediate format used by ``nvs_output_to_dataframe``.

        Raises
        ------
        TypeError
            If ``nvs_dict`` is not a dictionary.
        """
        if not isinstance(nvs_dict, dict):
            raise TypeError("nvs_dict must be a dictionary")

        metric_aliases = {
            "layercontributionscore": "layer_contribution_score",
            "lcs": "layer_contribution_score",
            "sensitivity": "sensitivity_score",
            "sensitivityscore": "sensitivity_score",
            "evolution": "evolution_score",
            "evolutionscore": "evolution_score",
        }
        measurements = {}

        def compact(value):
            return "".join(character for character in str(value).lower() if character.isalnum())

        def put(metric, parameter_type, statistic, layer, value):
            array = np.asarray(value)
            if array.ndim != 0:
                return
            measurements.setdefault(
                (metric, parameter_type, statistic), {}
            )[str(layer)] = array.item()

        def ensure_metric_columns(metric, parameter_type):
            if metric == "layer_contribution_score":
                statistics = ("norm", "coefficient_of_variation", "rank")
            elif metric in {"sensitivity_score", "evolution_score"}:
                statistics = ("raw", "norm", "rank")
            else:
                statistics = ("value",)
            for statistic in statistics:
                measurements.setdefault((metric, parameter_type, statistic), {})

        def infer_metric(data):
            if not isinstance(data, dict):
                return "metric"
            keys = {compact(key) for key in data}
            if any("filteredlayers" in key for key in keys):
                return "layer_contribution_score"
            if "weights" in keys or "biases" in keys:
                for key in ("weights", "biases"):
                    if key in data:
                        inferred = infer_metric(data[key])
                        if inferred != "metric":
                            return inferred
            if "rawvalues" in keys:
                raw_values = data.get("raw_values", {})
                if isinstance(raw_values, dict) and any(
                    np.asarray(value).ndim > 0 for value in raw_values.values()
                ):
                    return "evolution_score"
                return "sensitivity_score"
            if any(np.asarray(value).ndim > 0 for value in data.values()):
                return "layer_contribution_score"
            return "metric"

        def collect_metric(metric, data, parameter_type):
            if not isinstance(data, dict):
                return

            if metric == "layer_contribution_score":
                for key, value in data.items():
                    normalized = compact(key)
                    if normalized.startswith("filteredlayers") and isinstance(value, dict):
                        for statistic_key, statistic_values in value.items():
                            statistic_name = compact(statistic_key)
                            if statistic_name == "normvalues" and isinstance(statistic_values, dict):
                                for layer, score in statistic_values.items():
                                    put(metric, parameter_type, "norm", layer, score)
                            elif statistic_name.startswith("ranks") and isinstance(statistic_values, dict):
                                for layer, score in statistic_values.items():
                                    put(metric, parameter_type, "rank", layer, score)
                            elif isinstance(statistic_values, dict):
                                continue
                            else:
                                put(metric, parameter_type, "coefficient_of_variation", statistic_key, statistic_values)
                    elif isinstance(value, dict):
                        continue
                    else:
                        array = np.asarray(value)
                        if array.ndim > 0:
                            key_columns = (metric, parameter_type, "norm")
                            layer_values = measurements.setdefault(key_columns, {})
                            layer_values.setdefault(str(key), np.linalg.norm(array))
                return

            if metric in {"sensitivity_score", "evolution_score"}:
                statistic_aliases = {
                    "rawvalues": "raw",
                    "normvalues": "norm",
                }
                for key, values in data.items():
                    normalized = compact(key)
                    if normalized.startswith("ranks"):
                        statistic = "rank"
                    else:
                        statistic = statistic_aliases.get(normalized)
                    if statistic is None or not isinstance(values, dict):
                        continue
                    for layer, score in values.items():
                        put(metric, parameter_type, statistic, layer, score)
                return

            for layer, value in data.items():
                array = np.asarray(value)
                if array.ndim == 0:
                    put(metric, parameter_type, "value", layer, value)
                else:
                    put(metric, parameter_type, "norm", layer, np.linalg.norm(array))

        recognized_metric = False
        for key, data in nvs_dict.items():
            metric = metric_aliases.get(compact(key))
            if metric is None:
                continue
            recognized_metric = True
            if isinstance(data, dict):
                groups = [
                    (group, data[group])
                    for group in ("weights", "biases")
                    if isinstance(data.get(group), dict)
                ]
                if groups:
                    for parameter_type, group_data in groups:
                        ensure_metric_columns(metric, parameter_type)
                        collect_metric(metric, group_data, parameter_type)
                else:
                    ensure_metric_columns(metric, "parameters")
                    collect_metric(metric, data, "parameters")

        if not recognized_metric:
            inferred_metric = infer_metric(nvs_dict)
            groups = [
                (group, nvs_dict[group])
                for group in ("weights", "biases")
                if isinstance(nvs_dict.get(group), dict)
            ]
            if groups:
                for parameter_type, group_data in groups:
                    metric = infer_metric(group_data)
                    ensure_metric_columns(metric, parameter_type)
                    collect_metric(metric, group_data, parameter_type)
            else:
                parameter_type = "biases" if any(
                    "bias" in compact(key) for key in nvs_dict
                ) else "weights"
                ensure_metric_columns(inferred_metric, parameter_type)
                collect_metric(inferred_metric, nvs_dict, parameter_type)

        return measurements

    def nvs_output_to_dataframe(
        self,
        *,
        nvs_dict: dict,
        filename: str | None = None,
    ):
        """Convert NVS metric output into a layer-oriented pandas DataFrame.

        This is a convenient way to inspect one or more NVS metric families in
        tabular form. Combined results and individual LCS, sensitivity, or
        evolution dictionaries are accepted. Tensor-valued layer results are
        summarized by their norm; scalar norms, ranks, and LCS coefficients of
        variation are kept as separate statistics. Metric families present in
        the input retain their columns even when their values are empty.

        The result uses MultiIndex columns named ``metric``, ``parameter_type``,
        and ``statistic``. Its MultiIndex rows are named ``parameter_type`` and
        ``layer``. Missing values, such as evolution scores without a trained
        checkpoint, appear as ``NaN``.

        Parameters
        ----------
        nvs_dict : dict
            NVS output dictionary, for example the result of
            ``bridge(...).nvs_export_info()``.
        filename : str, optional
            If provided, path ending in ``.csv`` where the DataFrame is saved.
            If omitted, the DataFrame is returned without writing a file.

        Returns
        -------
        pandas.DataFrame
            Layer summaries with metric/statistic MultiIndex columns.

        Raises
        ------
        TypeError
            If ``nvs_dict`` is not a dictionary.
        AttributeError
            If ``filename`` is provided and does not end in ``.csv``.

        Examples
        --------
        Convert all computed metric families and select a value by its labels::

            result = model_bridge.nvs_export_info()
            frame = model_bridge.nvs_output_to_dataframe(nvs_dict=result)
            score = frame.loc[
                ("weights", "layer 0"),
                ("sensitivity_score", "weights", "norm"),
            ]

        Save the same table as CSV::

            frame = model_bridge.nvs_output_to_dataframe(
                nvs_dict=result,
                filename="nvs_metrics.csv",
            )
        """
        import pandas as pd

        measurements = self._dataframe_validator(nvs_dict)
        column_keys = list(measurements)
        row_keys = []
        for (_, parameter_type, _), layer_values in measurements.items():
            for layer in layer_values:
                row_key = (parameter_type, layer)
                if row_key not in row_keys:
                    row_keys.append(row_key)

        rows = []
        for parameter_type, layer in row_keys:
            rows.append({
                column: values.get(layer)
                for column, values in measurements.items()
                if column[1] == parameter_type and layer in values
            })

        index = pd.MultiIndex.from_tuples(
            row_keys, names=["parameter_type", "layer"]
        )
        columns = pd.MultiIndex.from_tuples(
            column_keys, names=["metric", "parameter_type", "statistic"]
        )
        dataframe = pd.DataFrame(rows, index=index, columns=columns)

        if filename is not None:
            if not filename.lower().endswith(".csv"):
                raise AttributeError("FILENAME_INVALID : use <filename>.csv")
            dataframe.to_csv(filename)

        return dataframe
    def torch_parameters_classifier(self,name,parameters,reference="current"):
        """Route a PyTorch parameter name into the weights or bias storage bucket.

        The logic keeps each layer grouped under the same layer label instead of resetting
        the index on every parameter call. This preserves the original bridge behavior while
        keeping the weight and bias entries attached to the same layer.
        """
        if reference=="current":
            if name.endswith(".weight"):
                self.nvs_memory["weights"][f"layer {self.layer_current.get("torch_weight_index",0)}"] = parameters
                self.layer_current["torch_weight_index"]+=1
            elif name.endswith(".bias"):
                self.nvs_memory["bias"][f"layer {self.layer_current.get("torch_bias_index",0)}"] = parameters
                self.layer_current["torch_bias_index"]+=1
            else:
                return None
        else:
            if name.endswith(".weight"):
                            self.nvs_memory["weights_train"][f"layer {self.layer_train.get("torch_weight_index",0)}"] = parameters
                            self.layer_train["torch_weight_index"]+=1
            elif name.endswith(".bias"):
                            self.nvs_memory["bias_train"][f"layer {self.layer_train.get("torch_bias_index",0)}"] = parameters
                            self.layer_train["torch_bias_index"]+=1
            else:
                return None

        return None
    def tensorflow_parameters_classifier(self,name,parameters,reference="current")->object:
        """Route TensorFlow variable names into the shared weights/bias buckets.

        TensorFlow uses "kernel" for weights and "bias" for bias terms. The layer-aware
        grouping keeps the data attached to the correct layer instead of overwriting the
        dictionary with a fresh index each time.
        """
        if reference=="current":
            if "kernel" in name.lower():
                self.nvs_memory["weights"][f"layer {self.layer_current.get("tensorflow_weight_index",0)}"] = parameters
                self.layer_current["tensorflow_weight_index"]+=1
            elif "bias" in name.lower():
                self.nvs_memory["bias"][f"layer {self.layer_current.get("tensorflow_bias_index",0)}"] = parameters
                self.layer_current["tensorflow_bias_index"]+=1
            else:
                return None
        else:
           if "kernel" in name.lower():
                           self.nvs_memory["weights_train"][f"layer {self.layer_train.get("tensorflow_weight_index",0)}"] = parameters
                           self.layer_train["tensorflow_weight_index"]+=1
           elif "bias" in name.lower():
                           self.nvs_memory["bias_train"][f"layer {self.layer_train.get("tensorflow_bias_index",0)}"] = parameters
                           self.layer_train["tensorflow_bias_index"]+=1
           else:
                return None

        return None
    
    def jax_parameters_classifier(
        self,
        name,
        parameters,
        reference="current",
        layer_idx=None
    ) -> None:
        """Route JAX parameter names into the common weights/bias buckets.

        JAX names usually follow the TensorFlow convention, such as "layer1.kernel" and
        "layer1.bias". This keeps the same weighted/bias split while maintaining the correct
        layer grouping.
        """

        name = str(name).lower()

        if layer_idx is not None:
            layer_name = f"layer {layer_idx}"
        else:
            layer_name = "layer0"
            if "." in name:
                candidate = name.split(".")[0]
                if candidate.startswith("layer"):
                    layer_name = candidate

        target_key = "weights" if reference == "current" else "weights_train"
        bias_key = "bias" if reference == "current" else "bias_train"

        if "kernel" in name or name.endswith("weight"):
            self.nvs_memory[target_key][layer_name] = parameters

        if "bias" in name:
            self.nvs_memory[bias_key][layer_name] = parameters
