"""Score selection and rendering utilities for NVS layer diagnostic plots."""
import matplotlib.pyplot as plt
import matplotlib.container as BarContainer
from typing import cast
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal
from stats_method import Statistical
import datashader as ds
import datashader.transfer_functions as tf
class controller:
     """Shared helpers for padding sensitivity data and building style lookup tables."""
     def __init__(self):
          """Initialize the controller's in-memory result cache."""
          self.control_memory={}
     def sudo_control(self,*,sens_pad:bool=False,parameters:dict|None,diff_params:dict|None=None)->dict|object:
          """Pad sensitivity layers or split difference parameters into train/test groups."""
          if sens_pad:
               if parameters is not None:
                         padded_sens={"weights":{"norm_values":{},"raw_values":{},"ranks_weights":{}},"biases":{"norm_values":{},"raw_values":{},"ranks_weights":{},"ranks_biases":{}}}
                         valw=[v for v in parameters["sensitivity_score"]["weights"]["norm_values"].values()]
                         valb=[v for v in parameters["sensitivity_score"]["biases"]["norm_values"].values()]
                         meanw=np.mean(valw)
                         meanb=np.mean(valb)
                         medw=np.median(valw)
                         medb=np.median(valb)
                         last_key=list(parameters["sensitivity_score"]["weights"]["norm_values"].keys())[-1]
                         for k,v in parameters["sensitivity_score"]["weights"]["norm_values"].items():
                              padded_sens["weights"]["norm_values"][k]=v
                              if k==last_key:
                                      padded_sens["weights"]["raw_values"]["forced_layer"]=np.full(v.shape,np.float64((meanw/medw)))
                                      padded_sens["weights"]["ranks_weights"]["forced_layer"]=np.float64(meanw*100)
                                      padded_sens["weights"]["norm_values"]["forced_layer"]=np.full(v.shape,np.float64((meanw/medw)))
                         for k,v in parameters["sensitivity_score"]["biases"]["norm_values"].items():
                                              padded_sens["biases"]["norm_values"][k]=v
                                              if k==last_key:
                                                      padded_sens["biases"]["raw_values"]["forced_layer"]=np.full(v.shape,np.float64((meanb/medb)))
                                                      padded_sens["biases"]["ranks_biases"]["forced_layer"]=np.float64(
                                                             (meanb/medb)*100
                                                             )
                                                      padded_sens["biases"]["norm_values"]["forced_layer"]=np.full(v.shape,np.float64((meanb/medb)))
                         self.control_memory["sudo_control_sens"]=padded_sens
                         return padded_sens
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
          """Build per-index context, color-map, style, and rc lookup tables."""
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
     """Selects and shapes the score data required for each supported plot family."""
     def bar_plot_validator(self,*,family_idx,sens_pad:bool=True,parameters,anot,family):
          """Select ranking values and titles for the bar plot family."""
          validator_val={"anot":anot}
          if family[family_idx]=="sens_weight":
               if sens_pad:
                    padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
                    validator_val["weights"]=padded_sens["weights"]["ranks_weights"]
                    validator_val["sup_title"]="Sensitivity weight Ranking"
               else:
                    validator_val["weights"]=parameters["sensitivity_score"]["weights"]["ranks_weights"]
                    validator_val["sup_title"]="Sensitivity Weight Ranking"
          elif family[family_idx]=="sens_bias":
               if sens_pad:
                    padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
                    validator_val["biases"]=padded_sens["biases"]["ranks_biases"]
                    validator_val["sup_title"]="sensitivity Bias Ranking"
          elif family[family_idx]=="lcs_bias":
               validator_val["biases"]=parameters["layer_contribution_score"]["biases"]["ranks_biases"]
               validator_val["sup_title"]="Layer Contribution Bias Ranking"
          elif family[family_idx]=="lcs_weight":
               validator_val["weights"]=parameters["layer_contribution_score"]["weights"]["ranks_weights"]
               validator_val["sup_title"]="Layer Contribution Weight Ranking"
          elif family[family_idx]=="evol_weight":
               validator_val["weights"]=parameters["evolution_score"]["weights"]["ranks_weights"]
               validator_val["sup_title"]="Layer Contribution Weight Ranking"
          elif family[family_idx]=="evol_bias":
               validator_val["biases"]=parameters["evolution_score"]["biases"]["ranks_biases"]
               validator_val["sup_title"]="Layer Contribution Bias Ranking"
          elif family[family_idx]=="senvolution_b":
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["bias_1"]=padded_sens["biases"]["ranks_biases"]
               validator_val["bias_2"]=parameters["evolution_score"]["biases"]["rank_biases"]
               validator_val["sup_title"]="Sensitivity & Evolution Contribution Bias Ranking"
          elif family[family_idx]=="sensvolution_w":
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["weight_1"]=padded_sens["weights"]["ranks_weights"]
               validator_val["weight_2"]=parameters["evolution_score"]["weights"]["rank_weights"]
               validator_val["sup_title"]="Sensitivity & Evolution Contribution Weight Ranking"
          elif family[family_idx]=="lsens_b":
               validator_val["bias_1"]=parameters["layer_contribution_score"]["biases"]["ranks_biases"]
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["bias_2"]=padded_sens["biases"]["ranks_biases"]
               validator_val["sup_title"]="Layer & Sensitivity Contribution Bias Ranking"
          elif family[family_idx]=="lsens_w":
               validator_val["weight_1"]=parameters["layer_contribution_score"]["weights"]["ranks_weights"]
               padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameters)
               validator_val["weight_2"]=padded_sens["weights"]["ranks_weights"]
               validator_val["sup_title"]="Layer & Sensitivity Contribution Weight Ranking"
          elif family[family_idx]=="lvolution_w":
               validator_val["weight_1"]=parameters["layer_contribution_score"]["weights"]["ranks_weights"]
               validator_val["weight_2"]=parameters["evolution_score"]["weights"]["rank_weights"]
               validator_val["sup_title"]="Layer & Evolution Contribution Weight Ranking"
          elif family[family_idx]=="lvolution_b":
               validator_val["bias_1"]=parameters["layer_contribution_score"]["biases"]["ranks_biases"]
               validator_val["bias_2"]=parameters["evolution_score"]["biases"]["rank_biases"]
               validator_val["sup_title"]="Layer & Evolution Contribution Bias Ranking"
          return validator_val

     def scatter_validator(self,*,parameter,sens_pad:bool=True,family,family_idx,choice:Literal["non_grouped","grouped_bias","grouped_weight"]):
          """Select paired layer values for correlation or covariance plots."""
          validator_val={}
          match choice:
               case "non_grouped":
                    if family[family_idx]=="lcs_weight":
                         validator_val["weights"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layers_weights"}
                         validator_val["sup_title"]="Layer Contribution Weight Relation"
                    elif family[family_idx]=="lcs_bias":
                         validator_val["biases"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layers_biases"}
                         validator_val["sup_title"]="Layer Contribution Bias Relation"
                    elif family[family_idx]=="sens_weight":
                         validator_val["weights"]=parameter["sensitivity_score"]["raw_values"]
                         validator_val["sup_title"]="Layer & Sensitivity Contribution Weight Relation"
                    elif family[family_idx]=="sens_bias":
                         validator_val["biases"]=parameter["sensitivity_score"]["raw_values"]
                         validator_val["sup_title"]="Sensitivity Contribution Bias Relation"
                    elif family[family_idx]=="evol_bias":
                         validator_val["biases"]=parameter["evolution_score"]["raw_values"]
                         validator_val["sup_title"]="Evolution Contribution Bias Relation"
                    elif family[family_idx]=="evol_weight":
                         validator_val["weights"]=parameter["evolution_score"]["raw_values"]
                         validator_val["sup_title"]="Evolution Contribution Weight Relation"
               case "grouped_weight":
                    if family[family_idx]=="lsens_w":
                         validator_val["weight_1"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layer_weights"}
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                         validator_val["weight_2"]=padded_sens["weights"]["raw_values"]
                         validator_val["sup_title"]="Layer & Sensitivity Contribution Weight Relation"
                    elif family[family_idx]=="sensvolution_w":
                                             validator_val["weight_1"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layer_weights"}
                                             padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                                             validator_val["weight_2"]=padded_sens["weigths"]["raw_values"]
                                             validator_val["sup_title"]="Evolution & Sensitivity Contribution Weight Relation"
                    elif family[family_idx]=="lvolution_w":
                                             validator_val["weight_1"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layer_weights"}
                                             validator_val["weight_2"]=parameter["evolution_score"]["weights"]["raw_values"]
                                             validator_val["sup_title"]="Evolution & Layer Contribution weight Relation"
               case "grouped_bias":
                    if family[family_idx]=="lsens_b":
                         validator_val["bias_1"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layer_biases"}
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                         validator_val["bias_2"]=padded_sens["biases"]["raw_values"]
                         validator_val["sup_title"]="Layer & Sensitivity Contribution Bias Relation"
                    elif family[family_idx]=="sensvolution_b":
                         validator_val["bias_1"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layer_biases"}
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                         validator_val["bias_2"]=padded_sens["biases"]["raw_values"]
                         validator_val["sup_title"]="Evolution & Sensitivity Contribution Bias Relation"
                    elif family[family_idx]=="lvolution_b":
                         validator_val["bias_1"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layer_biases"}
                         validator_val["bias_2"]=parameter["evolution_score"]["biases"]["raw_values"]
                         validator_val["sup_title"]="Evolution & Layer Contribution Bias Relation"

          return validator_val
     def box_plot_validator(self,*,family_idx,family,parameter):
          """Select layer arrays for a box plot."""
               validator_val={}
               if family[family_idx]=="lcs_weight":
                                       validator_val["weights"]={k:v for k,v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items()}
                                       validator_val["sup_title"]="Layer Contribution Weight"
               elif family[family_idx]=="lcs_bias":
                                       validator_val["biases"]={k:v for k,v in parameter["layer_contribution_score"]["biases"]["filtered_layer_biases"].items()}
                                       validator_val["sup_title"]="Layer Contribution Bias"
               elif family[family_idx]=="sens_weight":
                                       validator_val["weights"]=parameter["sensitivity_score"]["norm_values"]
                                       validator_val["sup_title"]="Sensitivity Contribution Weight"
               elif family[family_idx]=="sens_bias":
                                       validator_val["biases"]=parameter["sensitivity_score"]["norm_values"]
                                       validator_val["sup_title"]="Sensitivity Contribution Bias"
               elif family[family_idx]=="evol_bias":
                                       validator_val["biases"]=parameter["evolution_score"]["norm_values"]
                                       validator_val["sup_title"]="Evolution Contribution Bias"
               elif family[family_idx]=="evol_weight":
                                       validator_val["weights"]=parameter["evolution_score"]["norm_values"]
                                       validator_val["sup_title"]="Evolution Contribution Weight"
               return validator_val
     def hist_plot_validator(self,*,family_idx,family,parameter,kind:Literal["grouped_weight","grouped_bias"],sens_pad:bool=True):
          """Select paired weight or bias arrays for a histogram."""
          validator_val={}
          match kind:
               case "grouped_weight":
                    if family[family_idx]=="lsens_w":
                         validator_val["weight_1"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layer_weights"}
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                         validator_val["weight_2"]=padded_sens["weights"]["raw_values"]
                         validator_val["sup_title"]="Layer & Sensitivity Contribution Weight Relation"
                    elif family[family_idx]=="sensvolution_w":
                                             validator_val["weight_1"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layer_weights"}
                                             padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                                             validator_val["weight_2"]=padded_sens["weigths"]["raw_values"]
                                             validator_val["sup_title"]="Evolution & Sensitivity Contribution Weight Relation"
                    elif family[family_idx]=="lvolution_w":
                                             validator_val["weight_1"]={k:v for k,v in parameter["layer_contribution_score"]["weights"].items() if k!="filtered_layer_weights"}
                                             validator_val["weight_2"]=parameter["evolution_score"]["weights"]["raw_values"]
                                             validator_val["sup_title"]="Evolution & Layer Contribution weight Relation"
               case "grouped_bias":
                    if family[family_idx]=="lsens_b":
                         validator_val["bias_1"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layer_biases"}
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                         validator_val["bias_2"]=padded_sens["biases"]["raw_values"]
                         validator_val["sup_title"]="Layer & Sensitivity Contribution Bias Relation"
                    elif family[family_idx]=="sensvolution_b":
                         validator_val["bias_1"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layer_biases"}
                         padded_sens=self.sudo_control(sens_pad=sens_pad,parameters=parameter)
                         validator_val["bias_2"]=padded_sens["biases"]["raw_values"]
                         validator_val["sup_title"]="Evolution & Sensitivity Contribution Bias Relation"
                    elif family[family_idx]=="lvolution_b":
                         validator_val["bias_1"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layer_biases"}
                         validator_val["bias_2"]=parameter["evolution_score"]["biases"]["raw_values"]
                         validator_val["sup_title"]="Evolution & Layer Contribution Bias Relation"
          return validator_val
     def fitting_plot_validator(self,*,parameter,family_idx,family):
          """Select layer values for the fitting plot."""
                         validator_val={}
                         if family[family_idx]=="lcs_weight":
                                                 validator_val["weights"]={k:v for k,v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items() if k!="filtered_layers_weights"}
                                                 validator_val["sup_title"]="Layer Contribution Weight"
                         elif family[family_idx]=="lcs_bias":
                                                 validator_val["biases"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layers_biases"}
                                                 validator_val["sup_title"]="Layer Contribution Bias"
                         elif family[family_idx]=="sens_weight":
                                                 validator_val["weights"]=parameter["sensitivity_score"]["raw_values"]
                                                 validator_val["sup_title"]="Sensitivity Contribution Weight"
                         elif family[family_idx]=="sens_bias":
                                                 validator_val["biases"]=parameter["sensitivity_score"]["raw_values"]
                                                 validator_val["sup_title"]="Sensitivity Contribution Bias"
                         elif family[family_idx]=="evol_bias":
                                                 validator_val["biases"]=parameter["evolution_score"]["raw_values"]
                                                 validator_val["sup_title"]="Evolution Contribution Bias"
                         elif family[family_idx]=="evol_weight":
                                                 validator_val["weights"]=parameter["evolution_score"]["raw_values"]
                                                 validator_val["sup_title"]="Evolution Contribution Weight"
                         return validator_val
     def mad_plot_validator(self,*,parameter,family,family_idx):
          """Select layer arrays for median absolute deviation analysis."""
          validator_val={}
          if family[family_idx]=="lcs_weight":
               validator_val["weights"]={k:v for k,v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items() if k!="filtered_layers_weights"}
               validator_val["sup_title"]="Layer Contribution Weight"
          elif family[family_idx]=="lcs_bias":
               validator_val["biases"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layers_biases"}
               validator_val["sup_title"]="Layer Contribution Bias"
          elif family[family_idx]=="sens_weight":
               validator_val["weights"]=parameter["sensitivity_score"]["raw_values"]
               validator_val["sup_title"]="Sensitivity Contribution Weight"
          elif family[family_idx]=="sens_bias":
               validator_val["biases"]=parameter["sensitivity_score"]["raw_values"]
               validator_val["sup_title"]="Sensitivity Contribution Bias"
          elif family[family_idx]=="evol_bias":
               validator_val["biases"]=parameter["evolution_score"]["raw_values"]
               validator_val["sup_title"]="Evolution Contribution Bias"
          elif family[family_idx]=="evol_weight":
               validator_val["weights"]=parameter["evolution_score"]["raw_values"]
               validator_val["sup_title"]="Evolution Contribution Weight"
          return validator_val
     
     def large_dist_plot_validator(self,*,family,family_idx,parameter):
          """Select layer arrays for the large distribution plot."""
          validator_val={}
          if family[family_idx]=="lcs_weight":
               validator_val["weights"]={k:v for k,v in parameter["layer_contribution_score"]["weights"]["filtered_layer_weights"].items() if k!="filtered_layers_weights"}
               validator_val["sup_title"]="Layer Contribution Weight"
          elif family[family_idx]=="lcs_bias":
                        validator_val["biases"]={k:v for k,v in parameter["layer_contribution_score"]["biases"].items() if k!="filtered_layers_biases"}
                        validator_val["sup_title"]="Layer Contribution Bias"
          elif family[family_idx]=="sens_weight":
                        validator_val["weights"]=parameter["sensitivity_score"]["raw_values"]
                        validator_val["sup_title"]="Sensitivity Contribution Weight"
          elif family[family_idx]=="sens_bias":
                        validator_val["biases"]=parameter["sensitivity_score"]["raw_values"]
                        validator_val["sup_title"]="Sensitivity Contribution Bias"
          elif family[family_idx]=="evol_bias":
                        validator_val["biases"]=parameter["evolution_score"]["raw_values"]
                        validator_val["sup_title"]="Evolution Contribution Bias"
          elif family[family_idx]=="evol_weight":
                        validator_val["weights"]=parameter["evolution_score"]["raw_values"]
                        validator_val["sup_title"]="Evolution Contribution Weight"
          return validator_val
                                      
class visualizer:
     """Renders diagnostic plots for a configured metric family using validator-selected data."""
     def __init__(self,*,family:list)->None: 
     """Create a visualizer configured for one supported family."""
        self.family=family
     def bar_plot(
     self,*,family_idx:int=0,parameters:dict,sens_pad:bool=True,
     choice:Literal["non_grouped","grouped_biases","grouped_weights"]="non_grouped",anot:None|list=None,
     range:str|None=None,fig_size:tuple=(6,4),
     orient:Literal["v","h","x","y"]="v",font_weight:Literal["ultralight","light","normal","regular","book","medium","roman","semibold","demibold","demi","bold","heavy","extra bold","black"]="bold",
     font_size:int=8,kind:Literal["normal","iqr"]="normal")->object|None:
          """Plot layer ranking scores as normal values or per-layer IQR values."""
          obj=validator()
          valueses=obj.bar_plot_validator(family_idx=family_idx,parameters=parameters,anot=anot,sens_pad=sens_pad,family=self.family)
          value={k:v for k,v in valueses.items() if k not in "sup_title" and k not in "anot"}
          if range is not None and isinstance(range,str):
               for k in value.keys():
                    value[k].pop(range)
          else:
                 value=value
          rows=[]
          for group, scores in value.items():
               for layer, score in scores.items():
                    if kind=="iqr":
                         score=Statistical().interquartile_range(data=np.asarray(score).ravel())
                    else:
                         score_array=np.asarray(score)
                         score=score_array.item() if score_array.ndim==0 else np.mean(score_array)
                    rows.append({"Layers": layer, "Scores": score, "Group": group})
          data=pd.DataFrame(rows)
          match choice:
               case "non_grouped":
                    plt.figure(figsize=fig_size)
                    sns.barplot(data=data,x="Layers",y="Scores",orient=orient)
                    plt.title(valueses["sup_title"])
                    plt.xticks(rotation=45,ha="right")
                    plt.tight_layout()
                    plt.show()

               case "grouped_weights":
                    plt.figure(figsize=fig_size)
                    sns.barplot(data=data,x="Layers",y="Scores",hue="Group",legend=False)
                    plt.title(valueses["sup_title"])
                    plt.xticks(rotation=45,ha="right")
                    plt.tight_layout()
                    plt.show()

               case "grouped_biases":
                    plt.figure(figsize=fig_size)
                    sns.barplot(data=data,x="Layers",y="Scores",hue="Group",legend=False)
                    plt.title(valueses["sup_title"])
                    plt.xticks(rotation=45,ha="right")
                    plt.tight_layout()
                    plt.show()          

     def scatter_plot(self,*,family_index:int=0,parameters:dict,plot_choice:Literal["covariance","corelation"],choice:Literal["grouped_bias","grouped_weight"],sens_pad:bool=True):
          """Plot per-layer Pearson correlation or covariance values."""
         obj=validator()
         value=obj.scatter_validator(parameter=parameters,sens_pad=sens_pad,family=self.family,family_idx=family_index,choice=choice)
         kind={"grouped_weight":"dual_weights","grouped_bias":"dual_biases"}[choice]
         match plot_choice:
               case "corelation":
                   stat_val=Statistical().pearson_correlation(data=value,kind=kind)
                   df=pd.DataFrame(
                        [{"Layer": item["layer"], "Value": item["R_value"]} for item in stat_val]
                   )
                   sns.scatterplot(data=df,x="Layer",y="Value")
                   plt.show()
               case "covariance":
                    stat_val=Statistical().covariance(data=value,kind=kind)
                    df=pd.DataFrame(
                         [
                              {"Layer": item["layer"], "Value": np.asarray(item["Result_value"])[0,1]}
                              for item in stat_val
                         ]
                    )
                    sns.scatterplot(data=df,x="Layer",y="Value")
                    plt.show()
     def large_dist_plot(self,*,family_index:int=0,parameters:dict)->object|None:
          """Aggregate large layer arrays with Datashader and display a heatmap."""
          obj=validator()
          value=obj.large_dist_plot_validator(parameter=parameters,family=self.family,family_idx=family_index)
          rows=[]
          layer_labels=[]
          for group, layers in value.items():
               if group=="sup_title":
                    continue
               for layer, layer_values in layers.items():
                    if layer not in layer_labels:
                         layer_labels.append(layer)
                    layer_index=layer_labels.index(layer)
                    rows.extend(
                         {"Layer": layer_index, "Value": element, "Group": group}
                         for element in np.asarray(layer_values).ravel()
                    )
          df=pd.DataFrame(rows)
          if df.empty:
               raise ValueError("large_dist_plot requires at least one layer value")
          canvas=ds.Canvas(
               plot_width=max(1,min(1200,len(layer_labels)*20)),
               plot_height=600,
               x_range=(-0.5,max(len(layer_labels)-0.5,0.5)),
          )
          aggregate=canvas.points(df,x="Layer",y="Value",agg=ds.count())
          image=tf.shade(
               aggregate,
               cmap=["#000004","#2c115f","#721f81","#b73779","#f1605d","#feb078","#fcfdbf"],
               how="eq_hist",
          )
          plt.figure(figsize=(12,6))
          plt.imshow(image.to_pil(),aspect="auto",origin="lower")
          plt.title(value.get("sup_title","Large Distribution"))
          plt.xlabel("Layer")
          plt.ylabel("Value")
          plt.xticks(
               np.linspace(0,len(layer_labels)-1,min(len(layer_labels),10),dtype=int),
               [layer_labels[index] for index in np.linspace(0,len(layer_labels)-1,min(len(layer_labels),10),dtype=int)],
               rotation=45,
               ha="right",
          )
          plt.tight_layout()
          plt.show()
          return image
     def box_plot(self,*,family_index:int=0,parameters:dict|None=None):
          """Plot the distribution of element values for each layer."""
          obj=validator()
          value=obj.box_plot_validator(family_idx=family_index,family=self.family,parameter=parameters)
          df=pd.DataFrame(
               [
                    {"Layer": layer, "Value": element, "Group": group}
                    for group, layers in value.items()
                    if group!="sup_title"
                    for layer, values in layers.items()
                    for element in np.asarray(values).ravel()
               ]
          )
          sns.boxplot(data=df,x="Layer",y="Value",hue="Group",legend=False)
          plt.show()



     
     def hist_plot(self,*,family_index:int=0,parameters:dict|None=None,kind:Literal["grouped_weight","grouped_bias"]="grouped_weight",statistics_method:Literal["skewness","kurtosis"],bins):
          """Plot per-layer skewness or kurtosis values."""
          obj=validator()
          value=obj.hist_plot_validator(family_idx=family_index,family=self.family,parameter=parameters,kind=kind)
          match statistics_method:
               case "kurtosis":
                    kurt_val=Statistical().kurtosis(data=value)
                    df=pd.DataFrame(
                         [
                              {"Layer": layer, "Value": score, "Group": group}
                              for group, layers in kurt_val.items()
                              for layer, score in layers.items()
                         ]
                    )
                    sns.histplot(df,x="Layer",y="Value",kde=True,bins=bins)
                    plt.show()
               case "skewness":
                    skew_val=Statistical().skewness(data=value)
                    df=pd.DataFrame(
                         [
                              {"Layer": layer, "Value": score, "Group": group}
                              for group, layers in skew_val.items()
                              for layer, score in layers.items()
                         ]
                    )
                    sns.histplot(df,x="Layer",y="Value",kde=True,bins=bins)
                    plt.show()

     def fitting_plot(self,*,family_index:int=0,parameters:dict|None):
          """Plot a fitted line for the selected layer values."""
         obj=validator()
         value=obj.fitting_plot_validator(family=self.family,family_idx=family_index,parameter=parameters)
         reg=Statistical().linear_regression(data=value)
         reg_df=pd.DataFrame(reg)
         sns.lineplot(data=reg_df,x="Layer",y="Value")
         plt.show()


     def mad_plot(self,*,parameters:dict|None,family_index:int=0):
          """Plot the median absolute deviation distribution by group."""
          obj=validator()
          value=obj.mad_plot_validator(parameter=parameters,family=self.family,family_idx=family_index)
          mad_val=Statistical().median_absolute_deviation(data=value)
          df=pd.DataFrame(
               [
                    {"Layer": layer, "Value": score, "Group": group}
                    for group, layers in mad_val.items()
                    for layer, score in layers.items()
               ]
          )
          sns.kdeplot(data=df,x="Value",hue="Group")
          plt.show()
