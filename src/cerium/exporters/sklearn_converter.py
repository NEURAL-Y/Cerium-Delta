import joblib
import numpy
class converter_sklearn:
    """Initialize the scikit-learn model converter. 
    
            Parameters 
            ---------- 
            model : sklearn.neural_network.MLPClassifier or MLPRegressor Trained or initialized scikit-learn MLP model whose weights, biases, and training information will be extracted. save_model : str, default="None" Path to the saved model or parameter file. The converter expects the saved data to follow the standard format defined by the Cerium Delta website. If the loaded object is a dictionary, each key-value pair is stored separately in ``training_parameters``. If the loaded object is not a dictionary, the complete loaded object is stored under the ``"weights"`` key.
             
            Attributes 
            ---------- 
            model Stores the scikit-learn MLP model. save_model Stores the path to the saved model or parameter file."""
    def __init__(self,model,save_model=None)->None:
        self.model=model
        self.save_model=save_model
    def extractor_architecture(self)->dict:
        """ Extract model parameters and training information. 

        Returns 
        ------- 
        dict Dictionary containing the extracted model information. 
        ``architecture_parameters`` 
        Contains the weights and biases of each MLP layer. 
        The weights are extracted from ``model.coefs_`` and the biases are extracted from ``model.intercepts_``. Each parameter is copied into the output dictionary as an independent NumPy array. ``training_parameters`` Contains parameters loaded from the saved model file. The expected saved-model format follows the standard format defined by the Cerium Delta website. If the loaded object is a dictionary, each key-value pair is stored individually. If the loaded object is not a dictionary, the complete object is stored under the ``"weights"`` key. ``total_layer`` Number of parameter layers represented by the extracted MLP biases. ``total_steps`` 
        Number of training iterations completed by the scikit-learn MLP, obtained from ``model.n_iter_``. """

        self.culter={"architecture_parameters":{},"training_parameters":{},"total_layer":0,"total_steps":0}
        for i,weight in enumerate(self.model.coefs_):

            self.culter["architecture_parameters"][f"layer {i} weight"]=weight.copy()
 
        index=0

        for i,bias in enumerate(self.model.intercepts_):

            self.culter["architecture_parameters"][f"layer {i} bias"]=bias.copy()

            index+=1

        self.culter["total_layer"]=index

        self.culter["total_steps"]=self.model.n_iter_

        if self.save_model is not None:
            self.save_model=str(self.save_model)
            weights=joblib.load(self.save_model)

            if isinstance(weights,dict):

                for k,v in weights.items():

                    self.culter["training_parameters"][k]=numpy.asarray(v).copy()

            elif isinstance(weights,list):
                for i, weight in enumerate(weights):
                    self.culter["training_parameters"][f"layer {i} weights"] = numpy.asarray(weight).copy()
            else:
                raise RuntimeError("save_model_standard_error : you use wrong standard to save your model weights and biases it should be in a list or a dictionary type learn more about--> https://cerium-delta.pages.dev ")
        else:
            self.save_model=None
            
        return self.culter
