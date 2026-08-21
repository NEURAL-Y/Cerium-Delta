from brain import NVS

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
    def __init__(self,model,*,framework,compute_choice="lcs",epoch=0,device="cpu",save_model=None,optimizer=None)->None:
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
        save_model : str, optional
            File path to a saved model or checkpoint.
        optimizer : object, optional
            Optimizer object or optimizer state for the framework.
        """
        self.framework=framework
        self.compute_choice=compute_choice
        self.model=model
        self.device=device
        self.epoch=epoch
        self.save_model=save_model
        self.optimizer=optimizer
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
           self.convert=converter_pytorch(model=self.model,optimizer=self.optimizer,epoch=self.epoch,device=self.device,save_model=self.save_model)

        elif self.framework=="tensorflow":
            from tensorflow_converter import converter_tensorflow
           # TensorFlow variables use the framework's variable naming convention.
            self.convert=converter_tensorflow(self.model,self.epoch,self.optimizer,self.save_model,self.device)

        elif self.framework=="sklearn":
            from sklearn_converter import converter_sklearn
           # sklearn models do not use a training optimizer in the same way as deep learning models.
            self.convertsk=converter_sklearn(self.model,self.save_model)

        elif self.framework=="jax":
            from jax_converter import converter_jax
           # JAX models are stored as PyTrees, so the JAX converter extracts named leaves.
            self.convert=converter_jax(self.model,self.optimizer,self.epoch,self.save_model)
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
        nvs=NVS(self.nvs_mem)

        try:
            
            self.nvs_result=nvs.compute(self.compute_choice)
            return self.nvs_result

        except Exception as e:
           raise RuntimeError(f"File_error : there is something which struck the operations {e} \n report us on --> https://cerium-delta.pages.dev/feedback")
        
    def torch_parameters_classifier(self,name,parameters,reference="current")->object:
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

