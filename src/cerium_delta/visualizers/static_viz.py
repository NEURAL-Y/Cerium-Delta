import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.container as BarContainer
from typing import cast
import numpy as np
import pandas as pd
import seaborn as sns
import datashader as ds
class visualizer:
    def __init__(self,*,family:set)->None: 
        self.family=list(family)
    def controller(self,alias=None,alias_value:set|None=None,*,sens_pad:bool=False,parameters:dict|None=None,diff_params:dict|None=None)->dict|None:

        self.alias_map={"senvolution":{"sensitivity score","evolution score"},"lvolution":{"layer contribution score","evolution"},"lsens":{"layer contribution score","sensitivity score"},"sensitivity":{"sens","sensitivity_score","sensitivity"},"layer contribution score":{"lcs","layer_contribution_score","layer_score"},"evolution score":{"evolution_score","evolution","evolve"}}
        if sens_pad:
            if parameters is not None:
                self.padded_sens={"weights":{},"biases":{}}
                valw=[v for k,v in parameters["sensitivity_score"]["weights"].items() if k.startswith("layer")]
                valb=[v for k,v in parameters["sensitivity_score"]["biases"].items() if k.startswith("layer")]
                meanw=np.mean(valw)
                meanb=np.mean(valb)
                last_key=list(parameters["sensitivity_score"]["weights"].keys())[-1]
                for k,v in parameters["sensitivity_score"]["weights"].items():
                    self.padded_sens["weights"][k]=v
                    if k.startswith("layer"):
                        if k==last_key:
                            self.padded_sens["weights"]["forced_layer"]=meanw
                            self.padded_sens["ranks_weights"]["forced_layer"]=np.float64(meanw*100)
                for k,v in parameters["sensitivity_score"]["biases"].items():
                                    self.padded_sens["biases"][k]=v
                                    if k.startswith("layer"):
                                        if k==last_key:
                                            self.padded_sens["biases"]["forced_layer"]=meanb
                                            self.padded_sens["ranks_biases"]["forced_layer"]=np.float64(meanb*100)
                return self.padded_sens

        if alias is not None and alias_value is not None:
                
                self.alias_map[alias]=alias_value
                return self.alias_map
        
        if diff_params is not None:
            
            self.parameters={"trainable_parameters":{},"test_parameters":{}}

            if "weights" in diff_params.keys() and "weights_train" in diff_params.keys() and "bias" in diff_params.keys() and "bias_train" in diff_params.keys():

                self.parameters["trainable_parameters"]["weights"]=diff_params["weights_train"]
                self.parameters["test_parameters"]["weights"]=diff_params["weights"]
                self.parameters["trainable_parameters"]["biases"]=diff_params["bias_train"]
                self.parameters["test_parameters"]["biases"]=diff_params["bias"]
                return self.parameters
            
            elif "train_parameters" in diff_params.keys() and "test_parameters" in diff_params.keys():

                self.parameters["trainable_parameters"]["weights"]=diff_params["train_parameters"]["weights"]
                self.parameters["trainable_parameters"]["biases"]=diff_params["train_parameters"]["biases"]
                self.parameters["test_parameters"]["weights"]=diff_params["test_parameters"]["weights"]
                self.parameters["test_parameters"]["biases"]=diff_params["test_parameters"]["biases"]
                return self.parameters
            


    def bar_plot(self,*,family_index:int=0,parameters:dict,sens_pad:bool=True,choice:str="non_grouped",alias:str|None=None,color_map:str|list[str]="blue",color_index:slice|None=None,anot:None|list=None,range:tuple|None=None,fig_size:tuple=(6,4),orient:str="v",font_weight:str="bold",font_size:int=8)->object|None:

          values={}
          if self.family[family_index]=="sensitivity" or  self.family[family_index]==alias or self.family[family_index]=="sens" or self.family[family_index]=="sensitivity_score":
                              if sens_pad:
                                   padded_sens=self.controller(sens_pad=True,parameters=parameters)
                                   values["weights"]=padded_sens["weights"]["ranks_weights"]#type:ignore
                                   values["biases"]=padded_sens["biases"]["ranks_biases"]#type:ignore
                                   values["sup_title"]="Sensitivity Score Ranking"
                                   if anot is not None:
                                        values["anot"]=anot
                                   if range is not None:
                                        values["range"]=range
                              else:
                                   values["weights"]=parameters["sensitivity_score"]["weights"]["ranks_weights"]
                                   values["biases"]=parameters["sensitivity_score"]["biases"]["ranks_biases"]
                                   values["sup_title"]="Sensitivity Score Ranking"
                                   if anot is not None:
                                        values["anot"]=anot
                                   if range is not None:
                                        values["range"]=range
          elif self.family[family_index]=="layer_score" or self.family[family_index]==alias or self.family[family_index]=="lcs" or self.family[family_index]=="layer_contribution_score":
                         values["weights"]=parameters["layer_contribution_score"]["weights"]["filtered_layers_weights"]["ranks_weights"]
                         values["biases"]=parameters["layer_contribution_score"]["biases"]["filtered_layers_biases"]["ranks_biases"]
          elif self.family[family_index]=="evolution" or self.family[family_index]==alias or self.family[family_index]=="evolution_score" or self.family[family_index]=="evolve":
                         values["weights"]=parameters["evolution_score"]["weights"]["ranks_weights"]
                         values["biases"]=parameters["evolution_score"]["biases"]["ranks_biases"]
          elif self.family[family_index]=="sensitivity_evolution" or self.family[family_index]==alias or self.family[family_index]=="senvolution":
                         if sens_pad:
                              padded_sens=self.controller(sens_pad=True,parameters=parameters)
                              values["sensitivity_weights"]=padded_sens["weights"]["ranks_weights"]#type:ignore
                              values["sensitivity_biases"]=padded_sens["biases"]["ranks_biases"]#type:ignore
                              values["sup_title"]="Sensitivity & Evolution Score Ranking"
                              values["evolution_weights"]=parameters["evolution_score"]["weights"]["ranks_weights"]
                              values["evolution_biases"]=parameters["evolution_score"]["biases"]["ranks_biases"]
                              if anot is not None:
                                   values["anot"]=anot
                              if range is not None:
                                   values["range"]=range 
                         else:
                               raise ValueError("Sensitivity padding is required for sensitivity_evolution family. Please set sens_pad=True.")        
          elif self.family[family_index]=="layer_sensitivity" or self.family[family_index]==alias or self.family[family_index]=="lsens":
                
                         if sens_pad:
                              padded_sens=self.controller(sens_pad=True,parameters=parameters)
                              values["sensitivity_weights"]=padded_sens["weights"]["ranks_weights"]#type:ignore
                              values["sensitivity_biases"]=padded_sens["biases"]["ranks_biases"]#type:ignore
                              values["sup_title"]="Layer Contribution & Sensitivity Score Ranking"
                              values["layer_weights"]=parameters["layer_contribution_score"]["weights"]["filtered_layers_weights"]["ranks_weights"]
                              values["layer_biases"]=parameters["layer_contribution_score"]["biases"]["filtered_layers_biases"]["ranks_biases"]
                              if anot is not None:
                                   values["anot"]=anot
                              if range is not None:
                                   values["range"]=range
                         else:
                              raise ValueError("Sensitivity padding is required for layer_sensitivity family. Please set sens_pad=True.")
          elif self.family[family_index]=="layer_evolution" or self.family[family_index]==alias or self.family[family_index]=="lvolution":
                         values["layer_weights"]=parameters["layer_contribution_score"]["weights"]["filtered_layers_weights"]["ranks_weights"]
                         values["layer_biases"]=parameters["layer_contribution_score"]["biases"]["filtered_layers_biases"]["ranks_biases"]
                         values["evolution_weights"]=parameters["evolution_score"]["weights"]["ranks_weights"]
                         values["evolution_biases"]=parameters["evolution_score"]["biases"]["ranks_biases"]
                         values["sup_title"]="Layer Contribution & Evolution Score Ranking"
                         if anot is not None:
                              values["anot"]=anot
                         if range is not None:
                              values["range"]=range
          value={k:v for k,v in values.items() if k not in ["sup_title","anot","range"]}
          data=pd.DataFrame.from_dict(value)
          match choice:
               case "non_grouped":
                    if color_index is not None and color_index.start is not None and color_index.stop is not None :
                          color_index=color_index
                    else: 
                         raise AttributeError("Invalid color_index")
                    plt.figure(figsize=fig_size)
                    ax=sns.barplot(data=data,palette=color_map[color_index],x="Layers",y="Scores")
                    for container in ax.containers:
                         container = cast(BarContainer, container)
                         ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore
                         
                    plt.title(values["sup_title"])
                    plt.show()
               case "grouped_stack":
                    if color_index is not None and color_index.start is not None and color_index.stop is not None :
                                              color_index=color_index
                    else: 
                         raise AttributeError("Invalid color_index")
                    plt.figure(figsize=fig_size)
                    a=sns.catplot(data=data,palette=color_map[color_index],x="Layers",y="Scores")
                    ax=a.ax
                    if "anot" in values.keys():
                         for container in ax.containers:
                                             container = cast(BarContainer, container)
                                             ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore
                                             
                    plt.title(values["sup_title"])
                    plt.show()
               case "grouped_weights":
                    if color_index is not None and color_index.start is not None and color_index.stop is not None :
                                              color_index=color_index
                    else: 
                         raise AttributeError("Invalid color_index")
                    plt.figure(figsize=fig_size)
                    a=sns.catplot(data=data,palette=color_map[color_index],x="Layers",y="Scores")
                    ax=a.ax
                    if "anot" in values.keys():
                         for container in ax.containers:
                              container = cast(BarContainer, container)
                              ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore     
                    plt.title(values["sup_title"])
                    plt.show()
               case "grouped_biases":
                    if color_index is not None and color_index.start is not None and color_index.stop is not None :
                          color_index=color_index
                    else: 
                         raise AttributeError("Invalid color_index")
                    plt.figure(figsize=fig_size)
                    a=sns.catplot(data=data,palette=color_map[color_index],x="Layers",y="Scores")
                    ax=a.ax
                    if "anot" in values.keys():
                         for container in ax.containers:
                              container = cast(BarContainer, container)
                              ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore   
                    plt.title(values["sup_title"])
                    plt.show()
               
                  
    










    def violin_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    










    def heatmap_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    











    def scatter_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    












    def hexabin_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    





    def kde_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    









    def cde_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    









    def linear_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    










    def large_dist_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    












    def pdf_distribution(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    








    def box_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    









    def spectral_analysis(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass










    def bivariate_analysis(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
            pass
    










    def diag_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass


    





    def face_grid_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass




    def tri_histo_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
            pass


    def benchmark_report(self,*,family_index:int=0,parameters:dict | None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object | None:
       if diff_params is not None:
        df=pd.DataFrame()
       else:
            pass

