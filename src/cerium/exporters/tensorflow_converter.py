import numpy
import tensorflow as tf

class converter_tensorflow:
    """Initialize the TensorFlow model converter.

    Parameters 
    ---------- 
    model : tf.keras.Model TensorFlow/Keras model whose variables will be extracted. optimizer : tf.keras.optimizers.Optimizer Optimizer associated with the model. Its internal state variables will be extracted separately. epoch : int Number of training epochs completed by the model. save_model : str, default="None" Path to a saved TensorFlow/Keras model. If provided, the saved model will be loaded and its trained variables will be extracted. device : str, default="cpu" Device configuration used when loading the saved model. 

    Attributes 
    ---------- 
    model Stores the TensorFlow/Keras model. optimizer Stores the optimizer associated with the model. epoch Stores the number of completed training epochs. save_model Stores the path of the saved trained model. device Stores the device configuration.
    """
    def __init__(self,model,epoch,optimizer=None,save_model=None,device="cpu")->None:
        self.model=model
        self.optimizer=optimizer
        self.save_model=save_model
        self.device=device
        self.epoch=epoch

    def extractor_architecture(self)->dict:
        """Extract model variables, optimizer state, and trained variables. 

        Returns 
        ------- 
        dict A dictionary containing the extracted model information. ``architecture_parameters`` Stores the current model variables as independent NumPy arrays. These include trainable parameters such as kernels and biases. ``training_parameters`` Stores the variables extracted from the optional saved trained model. ``total_layer`` Stores the number of model variables extracted. ``total_epoch`` Stores the number of training epochs. ``optimizer`` Stores the optimizer variables as independent NumPy arrays.
        """
        self.culter={"architecture_parameters":{},"training_parameters":{},"total_layer":0,"total_epochs":0,"optimizer":{}}

        index=0

        for i in self.model.variables:

            self.culter["architecture_parameters"][f"{i.name}_{i.path if hasattr(i,'path') else id(i)}"] = i.numpy().copy()

            index+=1

        self.culter["total_layer"]=index

        self.culter["total_epochs"]=self.epoch
        if self.optimizer is not None:

            for i in self.optimizer.variables:

                self.culter["optimizer"][i.name]=i.numpy().copy()
        else:
            self.culter["optimizer"]=None

        if self.save_model is not None:

            if tf.config.list_physical_devices(self.device):
               device="/GPU:0"
            else:
                device = "/CPU:0"

            self.save_model=str(self.save_model)

            with tf.device(device):

                model = tf.keras.models.load_model(self.save_model)

            for var in model.variables:

                self.culter["training_parameters"][var.name]=var.numpy().copy()
        else:
            
            self.save_model=None

        return self.culter