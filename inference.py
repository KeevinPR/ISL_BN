import pyAgrum as gum
import io
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyAgrum.lib.bn2graph import BNinference2dot
import pybnesian
from functools import reduce
from pybnesian import DiscreteBN
from sklearn.model_selection import KFold
from pybnesianCPT_to_df import from_CPT_to_df
import pandas as pd
import copy
import random

def inference_getPosterior(bn, target, es):
    # Create a dictionary of evidence values
    evs_dict = {e[0]: e[1].get() for e in es if e[1].get() != ""}
    # Perform inference and compute the posterior distribution
    return gum.getPosterior(bn, target=target, evs=evs_dict)

def get_inference_graph(bn, es):
    # Configure gum library to use 'png' format for graph rendering
    gum.config["notebook", "graph_format"] = "png"
    # Create a dictionary of evidence values
    evs_dict = {e[0]: e[1] for e in es if e[1] != ""}
    #evs_dict = {e[0]: e[1].get() for e in es if e[1].get() != ""}
    size = 8
    # Adjust the size based on the number of variables in the Bayesian network
    if len(bn.names()) > 4:
        size = int(len(bn.names()) / 4) * 8
    # Create a graph representation of the Bayesian network with evidence
    graph = BNinference2dot(bn, size=size, evs=evs_dict)
    # Render the `pydot` graph as a PNG image
    png_str = graph.create_png(prog='dot')
    # Treat the PNG output as an image file
    sio = io.BytesIO(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    # Adjust the figure size based on the number of variables in the Bayesian network
    fig = plt.figure(figsize=(1.5 * 6.4, 1.5 * 4.8) if len(bn.names()) > 4 else None)
    # Plot the image
    imgplot = plt.imshow(img, aspect='equal')
    plt.axis('off')
    return fig

def likellihood_weighting(df_train, df_test, bn, class_variable):
    """
    Computes the accuracy of a Bayesian network classifier using the Likelihood Weighting algorithm.

    Parameters:
    - df_train (pandas.DataFrame): The training dataset.
    - df_test (pandas.DataFrame): The testing dataset.
    - bn (pyAgrum.BayesNet): The Bayesian network model.
    - class_variable (str): The name of the class variable.

    Returns:
    - accuracy (float): The accuracy of the classifier as a percentage.
    """
   
    order_list = bn.graph().topological_sort()
    
    possible_values = list(df_train[class_variable].unique())
    
    features = copy.deepcopy(order_list)
    features.remove(class_variable)

    number_of_success = 0

    for index, row in df_test.iterrows():
        w = [0 for _ in possible_values]
        w_total = 0

        df_aux = df_train.copy()
        for f in features:
            df_aux = df_aux[df_aux[f] == row[f]]
        factor = bn.cpd(class_variable)
        if factor.evidence():
            df_evidence = pd.DataFrame(df_aux[factor.evidence()])
            if df_evidence.empty:
                cpt = from_CPT_to_df(str(bn.cpd(class_variable)))
                for e in bn.cpd(class_variable).evidence():
                    cpt = cpt[cpt[e] == row[e]]

                random_number = random.uniform(0, 1)

                if random_number < float(cpt.iloc[0][possible_values[0]]):
                    samples = pd.DataFrame({0: [possible_values[0]]})
                else:
                    samples = pd.DataFrame({0: [possible_values[1]]})
                
            else:
                samples = pd.DataFrame(factor.sample(len(df_evidence), df_evidence).to_pandas())
        else:
            samples = pd.DataFrame(factor.sample(10).to_pandas())

        wi = []
        for node in features:
            cpt = from_CPT_to_df(str(bn.cpd(node)))
            if bn.cpd(node).evidence():
                for e in bn.cpd(node).evidence():
                    cpt = cpt[cpt[e] == row[e]]
                wi.append(cpt.iloc[0][row[node]])
            else:
                wi.append(cpt.iloc[0][row[node]])
        
        for index, sample in samples.iterrows():

            w[possible_values.index(sample[0])] = w[possible_values.index(sample[0])] + reduce(lambda x, y: float(x) * float(y), wi)
            w_total += reduce(lambda x, y: float(x) * float(y), wi)
        w = [x/w_total for x in w]

        if possible_values[w.index(max(w))] == row[class_variable]:
            number_of_success += 1

    return number_of_success/len(df_test.index) * 100
        
    
if __name__ == "__main__":
    df = pd.read_csv("datos/cars_discrete4.csv")

    for column in df.columns:
        df[column] = df[column].astype('category')
    
    kf = KFold(n_splits= 4,shuffle=True)
    k = next(kf.split(df))
    df_train = df.iloc[k[0]]

    df_test = df.iloc[k[1]]

    bn = DiscreteBN(df.columns)
    bn.add_arc("transmission","manufacturer_name")
    bn.fit(df)
    class_variable = "transmission"
    for i in range(1,100):
        likellihood_weighting(df_train, df_test, bn, class_variable)








