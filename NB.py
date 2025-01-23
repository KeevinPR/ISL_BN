import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
import networkx as nx
import numpy as np
import pyAgrum as gum
import pyAgrum.skbn as skbn
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
from rpy2.robjects import r, pandas2ri

from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from contextlib import contextmanager

robjects.r('.libPaths(c("/home/ubuntu/miniconda3/lib/R/library", .libPaths()))')

# Activate the pandas to R conversion and set it globally
pandas2ri.activate()

@contextmanager
def rpy2_context():
    # Activar pandas2ri y conversiones locales
    pandas2ri.activate()
    robjects.conversion.set_conversion(default_converter + pandas2ri.converter)
    
    with localconverter(default_converter + pandas2ri.converter):
        #try:
            yield
        #finally:
            # Desactivar pandas2ri después del uso
            #pandas2ri.deactivate()

# Import R packages at module-level, once
utils = importr('utils')
bnclassify = importr('bnclassify')

def convert_to_r_df(df):
    with localconverter(pandas2ri.converter):
        r_df = pandas2ri.py2rpy(df)
    return r_df

def cross_val_to_number(cross_list):
    return "{:.4f}".format(sum(cross_list) / len(cross_list) * 100)

def obtain_weight_features_mi(features_list, class_var):
    weight_list = []
    for f in features_list:
        order = "cmi('" + f + "','" + class_var + "', r_from_pd_df)"
        weight = float(robjects.r(order)[0])
        weight_list.append(weight)
    return weight_list

def obtain_weight_features_score(bn, df, df2, y, features_list, class_var):
    weight_list = []
    for feature in features_list:
        bn_aux = gum.BayesNet(bn)
        bn_aux.add(feature)
        bn_aux.addArc(class_var, feature)

        df3 = df2.copy(deep=True)
        df3[feature] = df[feature]
        
        kf = KFold(n_splits=4, shuffle=True)
        scores = []

        for k in kf.split(df3):
            df_train = df3.iloc[k[0]]
            learner = gum.BNLearner(df_train)
            learner.useSmoothingPrior()
            bn2 = learner.learnParameters(bn_aux.dag())
            
            bnc = skbn.BNClassifier()
            bnc.fromTrainedModel(bn2, targetAttribute=class_var)

            yTest = y.iloc[k[1]]
            scoreCSV1 = bnc.score(df.iloc[k[1]], y=yTest)
            scores.append(scoreCSV1)
        
        weight_list.append(float(cross_val_to_number(scores)))
    return weight_list

def NB_k_fold_with_steps(jumpSteps, selection_parameter, dataset, class_var):
  with rpy2_context():
        try:
          df_cars = dataset
          r_from_pd_df = robjects.conversion.py2rpy(df_cars)
          robjects.globalenv['r_from_pd_df'] = r_from_pd_df
          robjects.r('r_from_pd_df <- as.data.frame(unclass(r_from_pd_df), stringsAsFactors = TRUE)')
          robjects.globalenv['class_variable'] = class_var

          tan = robjects.r("bnc('nb', class_variable, r_from_pd_df, smooth=0.01)")
          robjects.globalenv["tan"] = tan

          features = list(robjects.r('features(tan)'))
          g = nx.DiGraph()
          g.add_node(class_var)
          g.add_nodes_from(features)

          familia = robjects.r('families(tan)')
          edges_class_var = []
          rest_edges = []
          for x in familia:
              for i in range(1, len(x)):
                  g.add_edge(x[i], x[0])
                  if x[i] == class_var:
                      edges_class_var.append((x[i], x[0]))
                  else:
                      rest_edges.append((x[i], x[0]))

          distance = 0
          fixed_distance = 0.25
          fixed_distance_y = 0.5
          features_pos = {}
          for f in features:
              features_pos[f] = (distance,0.25)
              distance += fixed_distance

          df = df_cars
          df2 = pd.DataFrame()
          g.clear()

          bn = gum.BayesNet()
          df2[class_var] = df[class_var]
          bn.add(class_var)
          g.add_node(class_var)
          nodes_added = []

          distance = 0
          steps_aux = jumpSteps
          previous_score = 0.0

          if len(df[class_var].unique()) == 2:
              y = df[class_var].map({df[class_var].unique()[1]: True, df[class_var].unique()[0]: False})
          else:
              y = df[class_var]

          if selection_parameter == "Score":
              weight_list = obtain_weight_features_score(bn, df, df2, y, features, class_var)
          else:
              weight_list = obtain_weight_features_mi(features, class_var)

          weight_list_aux = weight_list
          features_list_aux = features
          figures_list = []

          while weight_list_aux:
              index_max = np.argmax(weight_list_aux)
              feature = features_list_aux.pop(index_max)
              weight_list_aux.pop(index_max)
              
              bn.add(feature)
              df2[feature] = df[feature]
              bn.addArc(class_var, feature)
              g.add_edge(class_var, feature)
              nodes_added.append(feature)

              features_pos[feature] = (distance,0.25)
              distance += fixed_distance

              if steps_aux == jumpSteps or not weight_list_aux:
                  steps_aux = 0
                  kf = KFold(n_splits=4, shuffle=True)
                  scores = []

                  for k in kf.split(df2):
                      df_train = df2.iloc[k[0]]
                      learner = gum.BNLearner(df_train)
                      learner.useSmoothingPrior()
                      bn2 = learner.learnParameters(bn.dag())
                      
                      bnc = skbn.BNClassifier()
                      bnc.fromTrainedModel(bn2, targetAttribute=class_var)
                      yTest = y.iloc[k[1]]

                      scoreCSV1 = bnc.score(df.iloc[k[1]], y=yTest)
                      scores.append(scoreCSV1)

                  score = float(cross_val_to_number(scores))
                  g[class_var][feature]['weight'] = float("{:.4f}".format(score - previous_score))
                  previous_score = score

                  fig = plt.figure()
                  nx.draw_networkx_nodes(g, pos=features_pos, nodelist=nodes_added, node_size=1500, margins=0.2)
                  nx.draw_networkx_nodes(g, pos={class_var: ((distance-fixed_distance)/2,fixed_distance_y)},
                                        nodelist=[class_var],node_color='#009900',node_size=1500)
                  features_pos[class_var] = ((distance-fixed_distance)/2,fixed_distance_y)
                  nx.draw_networkx_labels(g, pos=features_pos, font_weight='bold', font_size=5.5)
                  nx.draw_networkx_edges(g, features_pos, arrows=True, edgelist=[(class_var,node) for node in nodes_added],node_size=1500)
                  labels = nx.get_edge_attributes(g,'weight')
                  nx.draw_networkx_edge_labels(g, features_pos, edge_labels=labels, font_size=7.5)

                  figures_list.append((fig, scores, bn2))
              else:
                  steps_aux += 1

              if selection_parameter == "Score":
                  weight_list_aux = obtain_weight_features_score(bn, df, df2, y, features_list_aux, class_var)

          return figures_list
        except Exception as e:
                print(f"Error in R NB: {e}")
                traceback = robjects.r('geterrmessage()')
                #print(f"R traceback: {traceback[0]}")
                return None