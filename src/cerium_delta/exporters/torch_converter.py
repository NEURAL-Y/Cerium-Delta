import torch
import numpy
class converter_pytorch:
    """Extracts model parameters, optimizer state, and training metadata from a PyTorch model for downstream analysis. The converter creates a framework-independent representation by converting PyTorch tensors into independent NumPy arrays. 
    
    Parameters 
    ---------- 

    model : torch.nn.Module PyTorch model whose parameters will be extracted. epoch : int Number of epochs completed during training. optimizer : torch.optim.Optimizer Optimizer associated with the model. Its state is extracted separately from the model parameters. device : str, default="cpu" Device used when loading a saved model or checkpoint. save_model : str | None, default=None Optional path to a saved PyTorch model/checkpoint. If provided, the saved parameters are extracted as trained parameters.

    Attributes
     ---------- 

      model : torch.nn.Module Reference to the PyTorch model. optimizer : torch.optim.Optimizer Reference to the optimizer. device : str Device used for loading saved model data. epoch : int Number of completed training epochs. save_model : str | None Path to the saved model/checkpoint, if provided. culter : dict Extracted model information containing: - ``model_parameters``: Current model parameters represented as NumPy arrays. - ``optimizer_state``: Optimizer state dictionary. - ``total_parameter_tensors``: Number of parameter tensors in the model. - ``total_epoch``: Number of completed training epochs. - ``trained_parameters``: Parameters extracted from the optional saved checkpoint. 

      Notes 
      ----- 
      Model parameters and optimizer state are intentionally stored separately because they represent different aspects of training. Model parameters describe the learned state of the neural network, while optimizer state may contain quantities such as momentum, exponential moving averages, variance estimates, and optimization steps.
    """
    def __init__(self,model,epoch,device="cpu",optimizer=None,save_model=None)->None:

        self.model=model
        self.optimizer=optimizer
        self.device=device
        self.epoch=epoch
        self.save_model=save_model  

    def extractor_architecture(self)->dict:
       """ Extract model parameters, optimizer state, and training metadata. 
         Returns
         ------- 
         dict Dictionary containing the extracted information. """
       self.culter={"architecture_parameters":{},"training_parameters":{},"total_layer":0,"total_epochs":0}

       i=0

       for name,param in self.model.named_parameters():
         
         self.culter["architecture_parameters"][name] = param.detach().cpu().numpy().copy()
         i+=1
       self.culter["total_layer"]=i
       self.culter["total_epochs"]=self.epoch
       if self.optimizer is not None:
          self.culter["optimizer"]=self.optimizer.state_dict()
       else:
          self.culter["optimizer"]=None

       if self.save_model is not None:
          self.save_model=str(self.save_model)
          state=torch.load(self.save_model,map_location=self.device)

          for k,v in state.items():
             self.culter["trained_parameters"][k]=v.detach().cpu().numpy().copy()
       else:
          self.save_model=None
          
       return self.culter
