import numpy as np
import pybnesian
import random
import pandas as pd
import pyAgrum as gum
import pyAgrum.skbn as skbn
from sklearn.model_selection import KFold
from NB import cross_val_to_number
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import networkx as nx
import pprint

from inference import likellihood_weighting
from pybnesian import DiscreteBN

import time


class UMDA:

    class Solution:
        #Class for save solutions
        def __init__(self, chain, fitness, bn) -> None:
            self.chain = chain
            self.fitness = fitness
            self.bn = bn


    def __init__(self, selected_candidates, num_individuals, n_generations, dataset, class_variable, fitness_metric) -> None:
        #Initiate all parameters based on the parameters introduced by user
        self.selected_candidates = selected_candidates
        self.num_individuals = num_individuals
        self.n_generations = n_generations
        self.dataset = pd.read_csv(dataset)
        self.class_variable = class_variable
        self.fitness_metric = fitness_metric
        columns = list(self.dataset.columns)
        columns.remove(self.class_variable)
        columns.append(self.class_variable)
        self.nodes_list = columns
        self.nodes_number = len(columns)-1
        self.edges_number = int(self.nodes_number * (self.nodes_number-1) / 2)
        self.edges_dictionary = dict()
        aux = self.nodes_number
        for i in range(self.nodes_number):
            for j in range(i+1,self.nodes_number):
                self.edges_dictionary[aux] = (self.nodes_list[i],self.nodes_list[j])
                aux +=1
        self.fitness_dictionary = {}
        self.df2 = pd.DataFrame()

        for column in self.dataset.columns:
            self.df2 = pd.concat([self.df2, self.dataset.drop_duplicates(subset=[column])])
        
        self.df2 = self.df2.drop_duplicates()
        
    def generate_single_solution(self, variable_probability_vector):       
        individual = []
        
        for i in range(self.nodes_number):
            random_n = np.random.rand()
            i1 = variable_probability_vector[i][0]
            i2 = variable_probability_vector[i][1]
            i3 = variable_probability_vector[i][2]
            if random_n <= i1:
                individual.append(1)
            elif random_n <= i1+i2:
                individual.append(2)
            elif random_n <= i1+i2+i3:
                individual.append(3)
            else:
                individual.append(0)

        for i in range(self.nodes_number,self.nodes_number+self.edges_number):

            random_n = np.random.rand()
            i1 = variable_probability_vector[i][0]
            i2 = variable_probability_vector[i][1]
            if random_n <= i1:
                individual.append(1)
            elif random_n <= i1+i2:
                individual.append(2)
            else:
                individual.append(0)

        validation = self.validate_chain(individual)
        return self.Solution(validation[0], *self.obtain_fitness(*validation))

    def validate_chain(self, chain):
        """
        Validates the given chain by creating a pybnesian.Dag object and adding nodes and arcs according to the chain checking if they can be added updating the chain.

        Parameters:
        - chain (list): The chain representing the Bayesian network structure.

        Returns:
        - validated_chain (tuple): A tuple containing the validated chain, the list of nodes, and the list of arcs in the Bayesian network.
        """
        dag = pybnesian.Dag()
        dag.add_node(self.class_variable)
        nodes_list = list(self.nodes_list)
        nodes_list.remove(self.class_variable)
        random_number = random.uniform(0, self.nodes_number-1)

        for pos in range(self.nodes_number):

            aux_pos = int((pos + random_number) % self.nodes_number)
            node = nodes_list[aux_pos]
            if not dag.contains_node(node):
                dag.add_node(node)
            if chain[aux_pos] == 1 and dag.can_add_arc(node, self.class_variable):
                dag.add_arc(node, self.class_variable)
            elif chain[aux_pos] == 2 and dag.can_add_arc(self.class_variable, node):
                dag.add_arc(self.class_variable, node)
            elif chain[aux_pos] == 3:
                pass
            else:
                dag.remove_node(node)
                chain[aux_pos] = 0

        random_number = random.uniform(0, self.edges_number)

        for pos in range(self.nodes_number, self.nodes_number + self.edges_number):

            aux_pos = int(((pos+ random_number) % self.edges_number) + self.nodes_number)
            node1 = self.edges_dictionary[aux_pos][0]
            node2 = self.edges_dictionary[aux_pos][1]
            if not dag.contains_node(node1) or not dag.contains_node(node2):
                chain[aux_pos] = 0
            elif chain[aux_pos] == 1 and dag.can_add_arc(node1, node2):
                dag.add_arc(node1, node2)
            elif chain[aux_pos] == 2 and dag.can_add_arc(node2, node1):
                dag.add_arc(node2, node1)
            else:
                random_number = random.uniform(0, 1)
                if random_number < 0.5 and dag.can_add_arc(node1, node2):
                    dag.add_arc(node1, node2)
                    chain[aux_pos] = 1
                elif random_number < 0.5 and dag.can_add_arc(node2, node1):
                    dag.can_add_arc(node2, node1)
                    chain[aux_pos] = 2
                else:
                    chain[aux_pos] = 0

        #Chech spouses
        
        random_number = random.uniform(0, self.nodes_number-1)
        for pos in range(self.nodes_number):
            aux_pos = int((pos + random_number) % self.nodes_number)
            if chain[aux_pos] == 3:
                if not self.auxiliar_fun(chain, nodes_list[aux_pos]):
                    chain[aux_pos] = 0
                    dag.remove_node(nodes_list[aux_pos])
                    for i in self.list_arcs_contains_node(nodes_list[aux_pos]):
                        chain[i] = 0
        
        return (chain, dag.nodes(), dag.arcs())
    
    def list_arcs_contains_node(self, node):
        """
        Returns a list of positions corresponding to the arcs that contain the given node.

        Parameters:
        - node (str): The node name.

        Returns:
        - pos_list (list): A list of positions corresponding to the arcs that contain the given node.
        """
        pos_list = []
        for pos in range(self.nodes_number, self.nodes_number + self.edges_number):
            if self.edges_dictionary[pos][0] == node or self.edges_dictionary[pos][1] == node:
                pos_list.append(pos)
        return pos_list

    def auxiliar_fun(self, chain, node):
        """
        Calculates the count of edges connected to a given node in a chain.

        Parameters:
        - chain (list): The chain configuration.
        - node (str): The node name.

        Returns:
        - res (int): The count of edges connected to the given node.
        """
        res = 0

        for pos in range(self.nodes_number, self.nodes_number + self.edges_number):
            if ((chain[pos] == 2 and self.edges_dictionary[pos][0] == node and chain[self.nodes_list.index(self.edges_dictionary[pos][1])] == 2) 
                or (chain[pos] == 1 and self.edges_dictionary[pos][1] == node and chain[self.nodes_list.index(self.edges_dictionary[pos][0])] == 2)):
                res +=1
        return res

    
    def obtain_fitness(self, chain, nodes, arcs):
        """
        Obtain the fitness score of a Bayesian network chain using the pyAgrum library.

        Args:
            chain (list): The chain representing the Bayesian network structure.
            nodes (list): The list of nodes in the network.
            arcs (list): The list of arcs (edges) in the network.

        Returns:
            float: The fitness score of the Bayesian network.

        """
        if tuple(chain) in self.fitness_dictionary:
            return self.fitness_dictionary[tuple(chain)]
        else:
            bn = gum.BayesNet()
            for node in nodes:
                bn.add(node)
            
            for arc in arcs:
                bn.addArc(arc[0], arc[1])
            
            kf = KFold(n_splits= 4,shuffle=True)
            df_aux = self.dataset[nodes]

            if(len(df_aux[self.class_variable].unique()) == 2):
                y = df_aux[self.class_variable].map({df_aux[self.class_variable].unique()[1]: True, df_aux[self.class_variable].unique()[0]: False})
            else:
                y = df_aux[self.class_variable]

            scores = []

            for k in kf.split(self.dataset):
            
                df_train = df_aux.iloc[k[0]]
                learner=gum.BNLearner(df_train)
                learner.useSmoothingPrior()
                bn2 = learner.learnParameters(bn.dag())        
                bnc=skbn.BNClassifier()
                bnc.fromTrainedModel(bn2,targetAttribute=self.class_variable)          
                yTest = y.iloc[k[1]]
                scoreCSV1 = bnc.score(df_aux.iloc[k[1]], y = yTest)
                scores.append(scoreCSV1)
            self.fitness_dictionary[tuple(chain)] = ( float(cross_val_to_number(scores)), bn2)

            return (float(cross_val_to_number(scores)), bn2)
    
    def obtain_fitness_pybnessian(self, chain, nodes, arcs):
        """
        Obtain the fitness score of a Bayesian network chain using the pybnessian library.

        Args:
            chain (list): The chain representing the Bayesian network structure.
            nodes (list): The list of nodes in the network.
            arcs (list): The list of arcs (edges) in the network.

        Returns:
            float: The fitness score of the Bayesian network.

        """
        start_time_learning = time.time()
        bn = DiscreteBN(nodes)
        
        for arc in arcs:
            bn.add_arc(*arc)
        
        df = pd.read_csv(self.dataset_name)

        for column in df.columns:
            df[column] = df[column].astype('category')
        
        kf = KFold(n_splits= 4,shuffle=True)
        df_aux = df[nodes]

        bn.fit(df_aux)

        k = next(kf.split(df))
        df_train = df_aux.iloc[k[0]]
        df_test = df_aux.iloc[k[1]]
        score = likellihood_weighting(df_train, df_test, bn, self.class_variable)
        print("FOR LEARNING:--- %s seconds ---" % (time.time() - start_time_learning))
        return float("{:.4f}".format(score))

    def best_candidates_selection(self, population):
        """
        Select the best candidates from the population based on their fitness scores (taking the best ones).

        Args:
            population (list): The list of individuals in the population.

        Returns:
            list: The selected best candidates from the population.

        """
        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[:self.selected_candidates]
    
    def get_new_distribution(self, selected_candidates):
        """
        Calculate the new distribution based on the selected candidates.

        Args:
            selected_candidates (list): The list of selected candidates (individuals).

        Returns:
            list: The new distribution matrix.

        """
        dis1 = [[0,0,0] for _ in range(self.nodes_number)]
        dis2 = [[0,0] for _ in range(self.edges_number)]
        distribution = dis1+dis2
        for candidate in selected_candidates:
            for index1 in range(len(candidate.chain)):
                item1 = candidate.chain[index1]
                if item1:
                    distribution[index1][item1-1]  += 1
                    

        for index in range(len(distribution)):
            for i in range(len(distribution[index])):
                distribution[index][i] = distribution[index][i]/ self.selected_candidates
        
        return distribution
    
       
    def get_graph(self, chain):
        """
        Create a directed graph based on the given chain representation.

        Args:
            chain (list): The chain representation of the Bayesian network.

        Returns:
            nx.DiGraph: The created directed graph representing the Bayesian network.

        """
        g = nx.DiGraph()
        g.add_node(self.class_variable)
        for x in range(self.nodes_number):
            if chain[x] == 1:
                g.add_node(self.nodes_list[x])
                g.add_edge(self.nodes_list[x], self.class_variable)
            elif chain[x] == 2:
                g.add_node(self.nodes_list[x])
                g.add_edge(self.class_variable, self.nodes_list[x])
            elif chain[x] == 3:
                g.add_node(self.nodes_list[x])

        for y in range(self.nodes_number,self.nodes_number+self.edges_number):
            if chain[y] == 1:
                g.add_edge(*self.edges_dictionary[y])
            elif chain[y] == 2:
                g.add_edge(self.edges_dictionary[y][1], self.edges_dictionary[y][0])
        return g
    
    def from_chain_to_graph(self, chain):
        """
            Create a matplotlib graph of the Bayesian network representation from a given chain of the individual.

            Args:
                chain (list): The chain representing the Bayesian network.

            Returns:
                matplotlib.figure.Figure: The generated matplotlib Figure object.

        """
        
        fig = plt.figure()
        g = self.get_graph(chain)

        if len(list(g.nodes())) == 3:
            g.add_node("aux")
            pos = nx.circular_layout(g)
            g.remove_node("aux")
        else:
            pos = nx.circular_layout(g)

        #First the nodes
        nx.draw_networkx_nodes(g, pos= pos, node_size= 1500, margins= 0.2)

        #Second the class var
        nx.draw_networkx_nodes(g, pos= pos, node_color= "#CCCCFF", nodelist= [self.class_variable], node_size= 1500, margins= 0.2)

        #Draw edges
        nx.draw_networkx_edges(g, pos= pos, arrows = True, node_size= 1500)

        nx.draw_networkx_labels(g, pos= pos, font_size= 4.5)
        plt.show()
        return fig  
    
    def graph_between_chains(self, chain1, chain2):
        """
        Create a graph displaying the differences between two chains.

        Args:
            chain1 (list): The first chain representing a Bayesian network.
            chain2 (list): The second chain representing a Bayesian network.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib figure displaying the graph. 
            Red nodes/edges are missing edges in chain2 that are in chain1
            Green nodes/edges are new chains added in chain2
            Black nodes/edges are in both gens

        """

        fig = plt.figure()
        
        g1 = self.get_graph(chain1)
        g2 = self.get_graph(chain2)

        g1_nodes = set(g1.nodes())
        g1_edges = set(g1.edges())
        g2_nodes = set(g2.nodes())
        g2_edges = set(g2.edges())

        black_nodes = g1_nodes.intersection(g2_nodes)
        red_nodes = g1_nodes - g2_nodes
        green_nodes = g2_nodes - g1_nodes

        black_edges = g1_edges.intersection(g2_edges)
        red_edges = g1_edges - g2_edges
        green_edges = g2_edges - g1_edges

        g = nx.DiGraph()
        for i in g1_edges:
            g.add_edge(*i)
        for j in g2_edges:
            g.add_edge(*j)

        if len(list(g.nodes())) == 3:
            g.add_node("aux")
            pos = nx.circular_layout(g)
            g.remove_node("aux")
        else:
            pos = nx.circular_layout(g)

        nx.draw_networkx_nodes(g, pos= pos, nodelist= black_nodes, 
                               node_size= 1500, margins= 0.2)

        #Green and red nodes
        nx.draw_networkx_nodes(g, pos= pos, nodelist= green_nodes, node_color= "#7AD930",
                               node_size= 1500, margins= 0.2)
        nx.draw_networkx_nodes(g, pos= pos, nodelist= red_nodes, node_color= "#D82820",
                               node_size=1500, margins= 0.2)
        #Second the class var
        nx.draw_networkx_nodes(g, pos= pos, node_color= "#CCCCFF", nodelist= [self.class_variable], node_size= 1500, margins= 0.2)

        #Draw edges
        #print(red_edges_in_g)
        nx.draw_networkx_edges(g, pos = pos, edgelist= black_edges, arrows = True, node_size= 1500, connectionstyle='arc3,rad=0.05')       
        nx.draw_networkx_edges(g, pos = pos, edge_color= "#7AD930", edgelist= green_edges, arrows = True, node_size= 1500, connectionstyle='arc3,rad=0.05')
        nx.draw_networkx_edges(g, pos = pos, edge_color= "#D82820", edgelist= red_edges, arrows = True, node_size= 1500, connectionstyle='arc3,rad=0.05')

        nx.draw_networkx_labels(g, pos=pos, font_size = 4.5)
        plt.show()  
        return fig

    def from_generation_to_graph(self, generation_dict):
        """
        Create a matplotlib graph of the network from a dictionary representing the number of appearances of edges and nodes in a generation.

        Args:
            generation_dict (dict): Dictionary containing the count of edges and nodes in the generation.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib Figure object.

        """
        g = nx.DiGraph()
        g.add_node(self.class_variable)
        fig = plt.figure()
        features_in_g = []
        edges_in_g = []
    
        for x in generation_dict:
            if generation_dict[x]:
                if type(x) is tuple:
                    g.add_edge(*x)
                    edges_in_g.append(x)  
                else:
                    g.add_node(x)
                    features_in_g.append(x)

        if len(list(g.nodes())) == 3:
            g.add_node("aux")
            pos = nx.circular_layout(g)
            g.remove_node("aux")
        else:
            pos = nx.circular_layout(g)

        #First the nodes
        node_size= [(generation_dict[x]/self.num_individuals) * 1500 for x in features_in_g]
        nx.draw_networkx_nodes(g, pos= pos, nodelist= features_in_g, node_size= node_size, margins= 0.2)

        #Second the class var
        nx.draw_networkx_nodes(g, pos= pos, node_color= "#CCCCFF", nodelist= [self.class_variable], node_size= 1500, margins= 0.2)

        #Draw edges
        nx.draw_networkx_edges(g, pos= pos, edgelist= edges_in_g, width= [(generation_dict[x]/self.num_individuals) * 1.0 for x in edges_in_g], arrows = True, 
                            node_size= [1500]+node_size, connectionstyle='arc3,rad=0.05')

        nx.draw_networkx_labels(g, pos= pos, font_size= 4.5) 
        return fig

    
    def old_graph_between_generations(self, prev_generation_dict, actual_gen_dict):
        
        g = nx.DiGraph()
        g.add_node(self.class_variable)
        fig = plt.figure()
        green_features_in_g = []
        black_features_in_g = []
        red_features_in_g = []
        green_edges_in_g = []
        black_edges_in_g = []
        red_edges_in_g = []
        diff_gen_info = {}

        for x in actual_gen_dict:
            diff_gen_info[x] = (actual_gen_dict[x]- prev_generation_dict[x])

        for x in diff_gen_info:
            if diff_gen_info[x] > 0 and prev_generation_dict[x] == 0:
                if type(x) is tuple:
                    g.add_edge(*x)
                    green_edges_in_g.append(x)
                else:
                    g.add_node(x)
                    green_features_in_g.append(x)
            elif diff_gen_info[x] < 0 and actual_gen_dict[x] == 0:
                if type(x) is tuple:
                    g.add_edge(*x)
                    red_edges_in_g.append(x)
                else:
                    g.add_node(x)
                    red_features_in_g.append(x)
            elif prev_generation_dict[x]:
                if type(x) is tuple:
                    g.add_edge(*x)
                    black_edges_in_g.append(x)
                else:
                    g.add_node(x)
                    black_features_in_g.append(x)

        if len(list(g.nodes())) == 3:
            g.add_node("aux")
            pos = nx.circular_layout(g)
            g.remove_node("aux")
        else:
            pos = nx.circular_layout(g)
        #First the black nodes
        node_size= [(actual_gen_dict[x]/self.num_individuals) * 1500 for x in g.nodes() if x is not self.class_variable]

        nx.draw_networkx_nodes(g, pos= pos, nodelist= black_features_in_g, 
                               node_size= [(actual_gen_dict[x]/self.num_individuals) * 1500 for x in black_features_in_g], margins= 0.2)

        #Green and red nodes
        nx.draw_networkx_nodes(g, pos= pos, nodelist= green_features_in_g, node_color= "#7AD930",
                               node_size= [(actual_gen_dict[x]/self.num_individuals) * 1500 for x in green_features_in_g], margins= 0.2)
        nx.draw_networkx_nodes(g, pos= pos, nodelist= red_features_in_g, node_color= "#D82820",
                               node_size= [(prev_generation_dict[x]/self.num_individuals) * 1500 for x in red_features_in_g], margins= 0.2)
        #Second the class var
        nx.draw_networkx_nodes(g, pos= pos, node_color= "#CCCCFF", nodelist= [self.class_variable], node_size= 1500, margins= 0.2)

        #Draw edges
        nx.draw_networkx_edges(g, pos = pos, edgelist= black_edges_in_g, width= [(actual_gen_dict[x]/self.num_individuals) * 1.5 for x in black_edges_in_g], arrows = True, 
                            node_size= [1500]+node_size, connectionstyle='arc3,rad=0.05')
        
        nx.draw_networkx_edges(g, pos = pos, edge_color= "#7AD930", edgelist= green_edges_in_g, width= [(actual_gen_dict[x]/self.num_individuals) * 1.5 for x in green_edges_in_g], arrows = True, 
                            node_size= [1500]+node_size, connectionstyle='arc3,rad=0.05')
        
        nx.draw_networkx_edges(g, pos = pos, edge_color= "#D82820", edgelist= red_edges_in_g, width= [(prev_generation_dict[x]/self.num_individuals) * 1.5 for x in red_edges_in_g], arrows = True, 
                            node_size= [1500]+node_size, connectionstyle='arc3,rad=0.05')

        nx.draw_networkx_labels(g, pos=pos, font_size = 4.5)
        return fig
      
    def count_edges(self, group):
        """
        Count the occurrences of edges in a given group of individuals.

        Args:
            group (list): The group of individuals.

        Returns:
            list: The distribution of edge occurrences. The distribution is a list of lists, where each inner list represents the counts for a node or an edge.
            The first  lists correspond to node counts, and the remaining lists correspond to edge counts.

        """

        dis1 = [[0,0,0] for _ in range(self.nodes_number)]
        dis2 = [[0,0] for _ in range(self.edges_number)]
        distribution = dis1+dis2
        for individual in group:
            for index1 in range(len(individual.chain)):
                item1 = individual.chain[index1]
                if item1:
                    distribution[index1][item1-1]  += 1
        return distribution
    
    def print_data_generation(self, generation):

        """
            Save the count of individuals in a generation in a dictionary, categorized by nodes and edges.

            Args:
                generation (list): The generation of individuals.

            Returns:
                dict: A dictionary containing the count of individuals for each node and edge in the generation.
                    The dictionary keys are node_names representing the nodes or tuples representing edges, and the values are the respective counts.

        """

        count_individuals = self.count_edges(generation)
        nodes = count_individuals[:self.nodes_number]
        edges = count_individuals[self.nodes_number:]        
        count_dictionary = {}

        for i in range(self.nodes_number):
            count = sum(nodes[i])
            count_dictionary[self.nodes_list[i]] = count
            count_dictionary[(self.nodes_list[i], self.class_variable)] = nodes[i][0]
            count_dictionary[(self.class_variable, self.nodes_list[i])] = nodes[i][1]

        for i in range(self.nodes_number, self.nodes_number+self.edges_number):
            count_dictionary[self.edges_dictionary[i]] = edges[i-self.nodes_number][0]
            count_dictionary[(self.edges_dictionary[i][1],self.edges_dictionary[i][0])] = edges[i-self.nodes_number][1]

        return count_dictionary
    
    def execute_umda(self):
        probabilities = [[0.25, 0.25, 0.25] for _ in range(self.nodes_number)] + [[0.4, 0.4] for _ in range(self.edges_number)]
        population = []

        for i in range(1,self.num_individuals):
            population.append(self.generate_single_solution(probabilities))
        
        best_results = []
        generation_information = []

        for _ in range(self.n_generations):
            selected_candidates = self.best_candidates_selection(population)
            distribution = self.get_new_distribution(selected_candidates)
            population = [self.generate_single_solution(distribution) for _ in range(self.num_individuals)]
            population.sort(key=lambda x: x.fitness, reverse=True)
            population = population[:self.num_individuals]
            best_results.append(population[0])
            generation_information.append(self.print_data_generation(population))

        return (best_results, generation_information)
    
if __name__ == "__main__":
    nodes_number = 22
    probabilities = [[0.25, 0.25, 0.25] for _ in range(nodes_number)] + [[0.4, 0.4] for _ in range(int(nodes_number * (nodes_number-1) / 2))]
    num_selected_candidates = 5
    n_individuals = 10
    n_generations = 3
    umda = UMDA(num_selected_candidates , n_individuals, n_generations, "./datos/cars_discrete3.csv", "transmission", "")
    population = []
    

    for i in range(1,n_individuals):
        population.append(umda.generate_single_solution(probabilities))

    best_results = []
    generation_information = []

    for generation in range(n_generations):
        selected_candidates = umda.best_candidates_selection(population)
        distribution = umda.get_new_distribution(selected_candidates)
        population = [umda.generate_single_solution(distribution) for _ in range(n_individuals)]
        population.sort(key=lambda x: x.fitness, reverse=True)
        population = population[:n_individuals]
        best_results.append(population[0])
        generation_information.append(umda.print_data_generation(population))
        print(umda.from_chain_to_graph(best_results[0].chain))        

    print('-----')

    print([(x.chain, x.fitness) for x in best_results])

    diff_generation_information = []

    for i in range(len(generation_information)-1):
        diff_gen_info = {}
        actual_gen = generation_information[i]
        next_gen = generation_information[i+1]
        for x in actual_gen:
            diff_gen_info[x] = (next_gen[x]- actual_gen[x])/n_individuals
        diff_generation_information.append(diff_gen_info)
        umda.graph_between_generations(actual_gen, next_gen)