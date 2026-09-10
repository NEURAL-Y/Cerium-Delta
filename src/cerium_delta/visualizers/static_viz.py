import matplotlib.pyplot as plt
import matplotlib.container as BarContainer
from typing import cast
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal
class controller:
     def __init__(self):
          self.control_memory={}
     def sudo_control(self,*,sens_pad:bool=False,parameters:dict|None,diff_params:dict|None=None)->dict|object:
          if sens_pad:
               if parameters is not None:
                         self.padded_sens={"weights":{},"biases":{}}
                         valw=[v for k,v in parameters["sensitivity_score"]["weights"].items() if k.startswith("layer")]
                         valb=[v for k,v in parameters["sensitivity_score"]["biases"].items() if k.startswith("layer")]
                         meanw=np.mean(valw)
                         meanb=np.mean(valb)
                         medw=np.median(valw)
                         medb=np.median(valb)
                         last_key=list(parameters["sensitivity_score"]["weights"].keys())[-1]
                         for k,v in parameters["sensitivity_score"]["weights"].items():
                              self.padded_sens["weights"][k]=v
                              if k.startswith("layer"):
                                  if k==last_key:
                                      self.padded_sens["weights"]["forced_layer"]=np.full(v.shape,np.float64((meanw/medw)))
                                      self.padded_sens["ranks_weights"]["forced_layer"]=np.float64(meanw*100)
                         for k,v in parameters["sensitivity_score"]["biases"].items():
                                              self.padded_sens["biases"][k]=v
                                              if k.startswith("layer"):
                                                  if k==last_key:
                                                      self.padded_sens["biases"]["forced_layer"]=np.full(v.shape,np.float64((meanb/medb)))
                                                      self.padded_sens["ranks_biases"]["forced_layer"]=np.float64(
                                                             (meanb/medb)*100
                                                             )
                         self.control_memory["sudo_control_sens"]=self.padded_sens
                         return self.padded_sens
          if diff_params is not None:
                      
                      self.parameters={"trainable_parameters":{},"test_parameters":{}}
          
                      if (
                             "weights" in diff_params.keys() and "weights_train" in diff_params.keys() and 
                             "bias" in diff_params.keys() and "bias_train" in diff_params.keys()
                          ):
          
                          self.parameters["trainable_parameters"]["weights"]=diff_params["weights_train"]
                          self.parameters["test_parameters"]["weights"]=diff_params["weights"]
                          self.parameters["trainable_parameters"]["biases"]=diff_params["bias_train"]
                          self.parameters["test_parameters"]["biases"]=diff_params["bias"]
                          self.control_memory["sudo_control_diff_params"]=self.parameters
                          return self.parameters
                      
                      elif "train_parameters" in diff_params.keys() and "test_parameters" in diff_params.keys():
          
                          self.parameters["trainable_parameters"]["weights"]=diff_params["train_parameters"]["weights"]
                          self.parameters["trainable_parameters"]["biases"]=diff_params["train_parameters"]["biases"]
                          self.parameters["test_parameters"]["weights"]=diff_params["test_parameters"]["weights"]
                          self.parameters["test_parameters"]["biases"]=diff_params["test_parameters"]["biases"]
                      self.control_memory["sudo_control_diff_params"]=self.parameters
                      return self.parameters
          
     def style_control(self,*,context:tuple|None=None,color_map:tuple[list],style:tuple|None=None,rc:dict|None=None,styling_idx:list,color_idx):
          self.style_memory={"context":{},"color_map":{},"style":{},"rc":{}}
          for i in styling_idx:
               for j,_ in enumerate(context):
                    self.style_memory["context"][i]=context[j]
                    for k in color_idx:
                         self.style_memory["color_map"][i]=color_map[k]
                    self.style_memory["style"][i]=style[j]
                    if rc is not None:
                         self.style_memory["rc"]=rc[i]
          self.control_memory["style_control"]=self.style_memory
          return self.style_memory
class validator(controller):
     def __init__(self,func_type:str):
          self.func_type=func_type
     def bar_plot_validator(self,*,family_idx,sens_pad:bool=True,parameters,anot):
          validator_val={"anot":anot}
          if self.family[family_idx]=="sens_weight":
                    if sens_pad:
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
                         validator_val["weights"]=padded_sens["weights"]["ranks_weights"]
                         validator_val["sup_title"]="Sensitivity weight Ranking"
                    else:
                         validator_val["weights"]=parameters["sensitivity_score"]["weights"]["ranks_weights"]
                         validator_val["sup_title"]="Sensitivity Weight Ranking"
          elif self.family[family_idx]=="sens_bias":
               if sens_pad:
                    padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
                    validator_val["biases"]=padded_sens["biases"]["ranks_biases"]
                    validator_val["sup_title"]="sensitivity Bias Ranking"
          elif self.family[family_idx]=="lcs_bias":
                validator_val["biases"]=parameters["layer_contribution_score"]["biases"]["ranks_biases"]
                validator_val["sup_title"]="Layer Contribution Bias Ranking"
          elif self.family[family_idx]=="lcs_weight":
                validator_val["weights"]=parameters["layer_contribution_score"]["weights"]["ranks_weights"]
                validator_val["sup_title"]="Layer Contribution Weight Ranking"
          elif self.family[family_idx]=="evol_weight":
                validator_val["weights"]=parameters["evolution_score"]["weights"]["ranks_weights"]
                validator_val["sup_title"]="Layer Contribution Weight Ranking"
          elif self.family[family_idx]=="evol_bias":
                    validator_val["biases"]=parameters["evolution_score"]["biases"]["ranks_biases"]
                    validator_val["sup_title"]="Layer Contribution Bias Ranking"
          elif self.family[family_idx]=="senvolution_b":
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["bias_1"]=padded_sens["biases"]["ranks_biases"]
               validator_val["bias_2"]=parameters["evolution_score"]["biases"]["rank_biases"]
               validator_val["sup_title"]="Sensitivity & Evolution Contribution Bias Ranking"
          elif self.family[family_idx]=="sensvolution_w":
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["weight_1"]=padded_sens["weights"]["ranks_weights"]
               validator_val["weight_2"]=parameters["evolution_score"]["weights"]["rank_weights"]
               validator_val["sup_title"]="Sensitivity & Evolution Contribution Weight Ranking"
          elif self.family[family_idx]=="lsens_b":
               validator_val["bias_1"]=parameters["layer_contribution_score"]["biases"]["ranks_biases"]
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["bais_2"]=padded_sens["biases"]["rank_biases"]
               validator_val["sup_title"]="Layer & Sensitivity Contribution Bias Ranking"

          elif self.family[family_idx]=="lsens_w":
               validator_val["weight_1"]=parameters["layer_contribution_score"]["weights"]["ranks_weights"]
               validator_val["weight_2"]=parameters["layer_contribution_score"]["weights"]["rank_weights"]
               validator_val["sup_title"]="Layer & Sensitivity Contribution Weight Ranking"

          elif self.family[family_idx]=="lvolution_w":
               validator_val["weight_1"]=parameters["layer_contribution_score"]["weights"]["ranks_weights"]
               validator_val["weight_2"]=parameters["evolution_score"]["weights"]["rank_weights"]
               validator_val["sup_title"]="Layer & Evolution Contribution Weight Ranking"

          elif self.family[family_idx]=="lvolution_b":
               validator_val["bias_1"]=parameters["layer_contribution_score"]["biases"]["ranks_biases"]
               validator_val["bias_2"]=parameters["evolution_score"]["biases"]["rank_biases"]
               validator_val["sup_title"]="Layer & Evolution Contribution Bias Ranking"
          
          return validator_val
class visualizer:
    def __init__(self,*,family:set)->None: 
        self.family=list(family)
    def bar_plot(
     self,*,family_idx:int=0,parameters:dict,sens_pad:bool=True,
     choice:Literal["non_grouped","grouped_biases","grouped_weights"]="non_grouped",anot:None|list=None,
     range:str|None=None,fig_size:tuple=(6,4),
     orient:Literal["v","h","x","y"]="v",font_weight:Literal["ultralight","light","normal","regular","book","medium","roman","semibold","demibold","demi","bold","heavy","extra bold","black"]="bold",
     font_size:int=8,kind:str|None=None)->object|None:
          obj=validator()
          values=obj.bar_plot_validator(family_idx=family_idx,parameters=parameters,anot=anot,sens_pad=sens_pad)
          value={k:v for k,v in values.items() if k not in "sup_title" and k not in "anot"}
          if range is not None and isinstance(range,str):
               for k in value.keys():
                    value[k].pop[range]
          else:
                 value=value
          
          data=pd.DataFrame.from_dict(value)
          match choice:
               case "non_grouped":
                    plt.figure(figsize=fig_size)
                    ax=sns.barplot(data=data,x="Layers",y="Scores",orient=orient)
                    for container in ax.containers:
                         container = cast(BarContainer, container)
                         ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore
                         
                    plt.title(values["sup_title"])
                    plt.show()

               case "grouped_weights":
                    plt.figure(figsize=fig_size)
                    a=sns.catplot(data=data,x="Layers",y="Scores",kind="bar")
                    ax=a.ax
                    if "anot" in values.keys():
                         for container in ax.containers:
                              container = cast(BarContainer, container)
                              ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore     
                    plt.title(values["sup_title"])
                    plt.show()

               case "grouped_biases":
                    plt.figure(figsize=fig_size)
                    a=sns.catplot(data=data,x="Layers",y="Scores")
                    ax=a.ax
                    if "anot" in values.keys():
                         for container in ax.containers:
                              container = cast(BarContainer, container)
                              ax.bar_label(container,labels=values["anot"],label_type="center",fontsize=font_size,fontweight=font_weight)#type:ignore   
                    plt.title(values["sup_title"])
                    plt.show()          

    def heatmap_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    def scatter_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    def hexabin_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    def linear_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    def large_dist_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    def box_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
         pass
    def bivariate_analysis(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
            pass
    def tri_histo_plot(self,*,family_index:int=0,parameters:dict|None=None,diff_params:dict|None=None,model_diff:dict|None=None,sens_pad:bool=True,gen_pdf:bool=False,obj_return:bool=False)->object|None:
            pass
