import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pyAgrum as gum
import pyAgrum.skbn as skbn
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('SVG')
from rpy2.robjects import pandas2ri

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

utils = importr('utils')
bnclassify = importr('bnclassify')

def NB_TAN_k_fold_with_steps(jumpSteps, selection_parameter, dataset, class_var):
    with rpy2_context():
        try:
            df_cars = dataset
            r_from_pd_df = pandas2ri.py2rpy(df_cars)
            robjects.globalenv['r_from_pd_df'] = r_from_pd_df
            robjects.r('r_from_pd_df <- as.data.frame(unclass(r_from_pd_df), stringsAsFactors = TRUE)')
            robjects.globalenv['class_variable'] = class_var

            tan = robjects.r("bnc('nb', class_variable, r_from_pd_df, smooth=0.01)")
            robjects.globalenv["tan"] = tan

            modelo = robjects.r('modelstring(tan)')[0]
            class_var = robjects.r('class_var(tan)')[0]
            features = list(robjects.r('features(tan)'))

            g = nx.DiGraph()
            g.add_node(class_var)
            g.add_nodes_from(features)
            familia = robjects.r('families(tan)')

            edges_class_var = []
            rest_edges = []
            for x in familia:
                for i in range(1,len(x)):
                    g.add_edge(x[i], x[0])
                    if x[i] == class_var:
                        edges_class_var.append((x[i],x[0]))
                    else:
                        rest_edges.append((x[i],x[0]))

            distance = 0
            fixed_distance = 0.25
            fixed_distance_y = 0.5
            features_pos = {}
            for f in features:
                features_pos[f] = (distance,0.25)
                distance += fixed_distance

            df = pd.read_csv(dataset)
            df2 = pd.DataFrame()

            # Remove 'a' line
            # a

            g.clear()

            def obtain_weight_features(features_list):
                weight_list = []
                for f in features_list:
                    order = "cmi('"+f+"','"+class_var+"', r_from_pd_df)"
                    weight = float(robjects.r(order)[0])
                    weight_list.append(weight)
                return weight_list

            weight_list = obtain_weight_features(features)
            weight_list_aux = weight_list
            features_list_aux = features

            bn = gum.BayesNet()
            df2[class_var] = df[class_var]
            bn.add(class_var)
            g.add_node(class_var)

            steps_aux = jumpSteps
            i=0

            if len(df[class_var].unique()) == 2:
                y = df[class_var].map({df[class_var].unique()[1]: True, df[class_var].unique()[0]: False})
            else:
                y = df[class_var]

            figures_list = []
            while weight_list_aux:
                index_max = np.argmax(weight_list_aux)
                feature = features_list_aux.pop(index_max)
                weight_list_aux.pop(index_max)
                
                bn.add(feature)
                df2[feature] = df[feature]
                bn.addArc(class_var,feature)
                g.add_edge(class_var, feature)

                features_pos[feature] = (distance,0.25)
                distance += fixed_distance
                i+=1

                if steps_aux == jumpSteps or not weight_list_aux:
                    steps_aux = 0
                    fig = plt.figure()
                    nx.draw_networkx_nodes(g,pos=features_pos,nodelist=[feature], node_size=1500, margins=0.2)
                    nx.draw_networkx_nodes(g,pos={class_var: ((distance-fixed_distance)/2,fixed_distance_y)},
                                    nodelist=[class_var],node_color='#009900',node_size = 1500)
                    features_pos[class_var] = ((distance-fixed_distance)/2,fixed_distance_y)

                    nx.draw_networkx_labels(g, pos=features_pos, font_weight='bold', font_size=5.5)
                    nx.draw_networkx_edges(g, features_pos, arrows=True, edgelist=[(class_var, feature)],node_size=1500)

                    kf = KFold(n_splits=4, shuffle=True)
                    scores = []
                    for k in kf.split(df2):
                        df_train = df2.iloc[k[0]]
                        learner = gum.BNLearner(df_train)
                        learner.useSmoothingPrior()
                        bn2 = learner.learnParameters(bn.dag())

                        bnc = skbn.BNClassifier()
                        bnc.fromTrainedModel(bn2,targetAttribute=class_var)
                        yTest = y.iloc[k[1]]
                        scoreCSV1 = bnc.score(df.iloc[k[1]], y=yTest)
                        scores.append(scoreCSV1)

                    figures_list.append((fig, scores))
                else:
                    steps_aux+=1

            # Convert df2 again:
            r_from_pd_df = pandas2ri.py2rpy(df2)
            robjects.globalenv['r_from_pd_df'] = r_from_pd_df
            robjects.r('r_from_pd_df <- as.data.frame(unclass(r_from_pd_df), stringsAsFactors = TRUE)')

            tan = robjects.r("bnc('tan_cl', class_variable, r_from_pd_df, smooth=1, dag_args=list(score='aic'))")
            robjects.globalenv["tan"] = tan
            class_var = robjects.r('class_var(tan)')[0]
            features = list(robjects.r('features(tan)'))

            g = nx.DiGraph()
            g.add_node(class_var)
            g.add_nodes_from(features)
            familia = robjects.r('families(tan)')

            edges_class_var = []
            rest_edges = []
            for x in familia:
                for i in range(1,len(x)):
                    g.add_edge(x[i],x[0])
                    if(x[i] == class_var):
                        edges_class_var.append((x[i],x[0]))
                    else:
                        rest_edges.append((x[i],x[0]))

            def obtain_weight_edges(edges_list):
                weight_list = []
                for e in edges_list:
                    order = "cmi('"+e[0]+"','"+e[1]+"', z = '"+class_var+"', r_from_pd_df)"
                    weight = float(robjects.r(order)[0])
                    weight_list.append(weight)
                return weight_list

            edges = [k for k in g.edges if k[0]!=class_var]
            edges_weights = obtain_weight_edges(edges)
            index_max = np.argmax(edges_weights)
            edges_weights_aux = edges_weights

            g.clear_edges()
            bn = gum.BayesNet()
            for i in df2.columns:
                bn.add(i)

            for i in df2.columns:
                if(i != class_var):
                    g.add_edge(class_var,i)
                    bn.addArc(class_var,i)

              

            edges_added = []

            steps_aux = jumpSteps

            if len(df[class_var].unique()) == 2:
              y = df[class_var].map({df[class_var].unique()[1]: True, df[class_var].unique()[0]: False})
            else:
              y = df[class_var]

            while edges_weights_aux:
                index_max = np.argmax(edges_weights_aux)
                edge = edges.pop(index_max)
                edges_weights_aux.pop(index_max)
                bn.addArc(*edge)
                g.add_edge(*edge)
                edges_added.append(edge)
                #Draw net

                if steps_aux == jumpSteps or not edges_weights_aux:

                  steps_aux = 0

                  fig = plt.figure()

                  nx.draw_networkx_nodes(g,pos=features_pos,nodelist = features, node_size = 1500, margins= 0.2)

                  nx.draw_networkx_nodes(g,pos={class_var: ((distance-fixed_distance)/2,fixed_distance_y)},
                                    nodelist=[class_var],node_color='#009900',node_size = 1500)
                  
                  nx.draw_networkx_labels(g, pos = features_pos, font_weight='bold', font_size = 5.5)

                  nx.draw_networkx_edges(g,features_pos,arrows = True, edgelist=edges_class_var,node_size = 1500)

                  nx.draw_networkx_edges(g,features_pos,arrows = True, edgelist= edges_added, connectionstyle='arc3,rad=0.4',node_size = 1500)


                  #plt.show()

                  kf = KFold(n_splits= 4,shuffle=True)

                  scores = []
                  
                  for k in kf.split(df2):

                    df_train = df2.iloc[k[0]]

                    learner=gum.BNLearner(df_train) 
                    learner.useSmoothingPrior()
                    bn2 = learner.learnParameters(bn.dag())

                    bnc=skbn.BNClassifier()
                    bnc.fromTrainedModel(bn2,targetAttribute=class_var)
                    yTest = y.iloc[k[1]]

                    scoreCSV1 = bnc.score(df.iloc[k[1]], y =yTest)
                    #print("{0:.2f}% good predictions".format(100*scoreCSV1))
                    scores.append(scoreCSV1)

                  figures_list.append((fig, scores, bn2))
                  #bnc.showROC_PR('datos/cars_discrete.csv')
                
                else:
                  steps_aux+=1

            return figures_list
        except Exception as e:
                print(f"Error in R TAN : {e}")
                traceback = robjects.r('geterrmessage()')
                #print(f"R traceback: {traceback[0]}")
                return None