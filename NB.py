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

from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from contextlib import contextmanager

robjects.r('.libPaths(c("/home/ubuntu/miniconda3/lib/R/library", .libPaths()))')
pandas2ri.activate()

@contextmanager
def rpy2_context():
    pandas2ri.activate()
    robjects.conversion.set_conversion(default_converter + pandas2ri.converter)
    with localconverter(default_converter + pandas2ri.converter):
        yield

utils = importr('utils')
bnclassify = importr('bnclassify')

def cross_val_to_number(cross_list):
    return "{:.4f}".format(sum(cross_list) / len(cross_list) * 100)

def obtain_weight_features_mi(features_list, class_var):
    weight_list = []
    for f in features_list:
        #order = f"cmi('{f}','{class_var}', r_from_pd_df)"
        #weight = float(robjects.r(order)[0])
        order = f"as.numeric(cmi('{f}','{class_var}', r_from_pd_df))"
        res = robjects.r(order)
        weight = float(res[0])
        
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

        for train_idx, test_idx in kf.split(df3):
            df_train = df3.iloc[train_idx]
            learner = gum.BNLearner(df_train)
            learner.useSmoothingPrior()
            bn2 = learner.learnParameters(bn_aux.dag())
            
            bnc = skbn.BNClassifier()
            bnc.fromTrainedModel(bn2, targetAttribute=class_var)
            yTest = y.iloc[test_idx]
            scoreCSV1 = bnc.score(df.iloc[test_idx], y=yTest)
            scores.append(scoreCSV1)
        
        weight_list.append(float(cross_val_to_number(scores)))
    return weight_list

def NB_k_fold_with_steps(jumpSteps, selection_parameter, dataset, class_var):
    """
    Naive Bayes with step-wise feature selection. 
    'dataset' must be a Pandas DataFrame, NOT a path.
    """
    with rpy2_context():
        try:
            # -- Debug prints to see if there's something unusual --
            print("\n[NB_k_fold_with_steps] Before fixing columns:")
            print("DataFrame columns:", dataset.columns.tolist())
            print("DataFrame dtypes:\n", dataset.dtypes)
            print("DataFrame head:\n", dataset.head(5), "\n")

            # -- Reset index in case there's a MultiIndex or something unusual
            df_cars = dataset.reset_index(drop=True).copy(deep=True)

            # -- Force every column to string, then to category. 
            #    This avoids OrdDict or other object types that rpy2 can't convert.
            df_cars = df_cars.apply(lambda col: col.astype(str))
            df_cars = df_cars.apply(lambda col: col.astype('category'))

            print("[NB_k_fold_with_steps] After forcing to string->category:")
            print("DataFrame dtypes:\n", df_cars.dtypes)
            print("Head:\n", df_cars.head(5), "\n")

            # -- Convert to R data.frame
            r_from_pd_df = robjects.conversion.py2rpy(df_cars)
            robjects.globalenv['r_from_pd_df'] = r_from_pd_df
            robjects.r('r_from_pd_df <- as.data.frame(unclass(r_from_pd_df), stringsAsFactors = TRUE)')

            robjects.globalenv['class_variable'] = class_var
            nb_model = robjects.r("bnc('nb', class_variable, r_from_pd_df, smooth=0.01)")
            robjects.globalenv["tan"] = nb_model

            features = list(robjects.r('features(tan)'))
            g = nx.DiGraph()
            g.add_node(class_var)
            g.add_nodes_from(features)

            
            edges_class_var = []
            rest_edges = []
            familia_r = robjects.r('families(tan)')
            # Convert 'familia_r' into a pure Python list of lists
            familia_py = []
            for item in familia_r:
                # Each 'item' is an rpy2 object. Convert to a Python list of strings
                item_list = list(item)  # or [str(x) for x in item] if needed
                familia_py.append(item_list)

            # Now iterate 'familia_py' normally
            for x in familia_py:
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
                features_pos[f] = (distance, 0.25)
                distance += fixed_distance

            df = df_cars  # our working DataFrame
            df2 = pd.DataFrame()
            g.clear()

            bn = gum.BayesNet()
            df2[class_var] = df[class_var]
            bn.add(class_var)
            g.add_node(class_var)

            distance = 0
            steps_aux = jumpSteps
            previous_score = 0.0

            # If binary class
            unique_vals = df[class_var].unique().tolist()
            if len(unique_vals) == 2:
                # Map to True/False
                y = df[class_var].map({unique_vals[1]: True, unique_vals[0]: False})
            else:
                y = df[class_var]

            # Initial weighting
            if selection_parameter == "Score":
                weight_list = obtain_weight_features_score(bn, df, df2, y, features, class_var)
            else:
                weight_list = obtain_weight_features_mi(features, class_var)

            weight_list_aux = weight_list[:]
            features_list_aux = features[:]
            figures_list = []

            while weight_list_aux:
                index_max = np.argmax(weight_list_aux)
                feature = features_list_aux.pop(index_max)
                weight_list_aux.pop(index_max)
                
                bn.add(feature)
                df2[feature] = df[feature]
                bn.addArc(class_var, feature)
                g.add_edge(class_var, feature)

                features_pos[feature] = (distance, 0.25)
                distance += fixed_distance

                # Recalculate every jumpSteps steps or if out of features
                if steps_aux == jumpSteps or not weight_list_aux:
                    steps_aux = 0
                    kf = KFold(n_splits=4, shuffle=True)
                    scores = []

                    for train_idx, test_idx in kf.split(df2):
                        df_train = df2.iloc[train_idx]
                        learner = gum.BNLearner(df_train)
                        learner.useSmoothingPrior()
                        bn2 = learner.learnParameters(bn.dag())
                        
                        bnc = skbn.BNClassifier()
                        bnc.fromTrainedModel(bn2, targetAttribute=class_var)
                        yTest = y.iloc[test_idx]
                        scoreCSV1 = bnc.score(df.iloc[test_idx], y=yTest)
                        scores.append(scoreCSV1)

                    score = float(cross_val_to_number(scores))
                    g[class_var][feature]['weight'] = float("{:.4f}".format(score - previous_score))
                    previous_score = score

                    fig = plt.figure()
                    nx.draw_networkx_nodes(g, pos=features_pos, nodelist=[feature],
                                           node_size=1500, margins=0.2)
                    nx.draw_networkx_nodes(g, pos={class_var: ((distance - fixed_distance)/2, fixed_distance_y)},
                                           nodelist=[class_var], node_color='#009900', node_size=1500)
                    features_pos[class_var] = ((distance - fixed_distance)/2, fixed_distance_y)
                    nx.draw_networkx_labels(g, pos=features_pos, font_weight='bold', font_size=5.5)
                    nx.draw_networkx_edges(g, features_pos, arrows=True,
                                           edgelist=[(class_var, feature)], node_size=1500)
                    labels = nx.get_edge_attributes(g, 'weight')
                    nx.draw_networkx_edge_labels(g, features_pos, edge_labels=labels, font_size=7.5)

                    figures_list.append((fig, scores, bn2))
                else:
                    steps_aux += 1

                if selection_parameter == "Score":
                    weight_list_aux = obtain_weight_features_score(bn, df, df2, y, features_list_aux, class_var)

            return figures_list

        except Exception as e:
            print(f"[NB_k_fold_with_steps] Error in R NB: {e}")
            traceback = robjects.r('geterrmessage()')
            return None