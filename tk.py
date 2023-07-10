import tkinter as tk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from NB import NB_k_fold_with_steps
from TAN import NB_TAN_k_fold_with_steps
from NB import cross_val_to_number 
from inference import inference_getPosterior
from inference import get_inference_graph
from MarkovBlanketEDAs import UMDA
from operator import attrgetter
import pandas as pd



#TO DO
#Añadir la variable clase como un parámetro
#causal feature selection markov blanket
#JMLR Senen Barro
#Poner que lo de comparar entre generaciones sea solo para comparar entre los mejores

steps = 0
figures_list = []

class StepsWindow(tk.Toplevel):
    """A class representing a window for displaying steps of performing the selected model construction.

    This window displays a specific step using a figure from a list of figures.
    It allows navigation between steps and provides a score label.

    Args:
        parent (Tk): The parent Tkinter window.
        figures_list (list): A list of figures representing the steps of the process.
        steps (int): The current step to display.

    Attributes:
        parent (Tk): The parent Tkinter window.
        figures_list (list): A list of figures representing the steps of the process.
        steps (int): The current step being displayed.

    """
    def __init__(self, parent, figures_list, steps):
        """Initialize the StepsWindow.

        Args:
            parent (Tk): The parent Tkinter window.
            figures_list (list): A list of figures representing the steps of the process.
            steps (int): The current step to display.

        """

        super().__init__(parent)
        self.geometry("780x550")
        self.title('steps')

        self.parent = parent

        self.figures_list = figures_list
        self.steps = steps

        fig_score = self.figures_list[self.steps]

        canvas = FigureCanvasTkAgg(fig_score[0], self)
            
        canvas.get_tk_widget().grid(row = 1, column = 1, columnspan=10, rowspan=10)
    
        # Revisar en un futuro si es integrable el toolbar con grid 
        # https://stackoverflow.com/questions/12913854/displaying-matplotlib-navigation-toolbar-in-tkinter-via-grid
        #toolbar = NavigationToolbar2Tk(canvas,my_w_child)
        #canvas._tkcanvas.pack(side= TOP, fill = BOTH , expand = True)
        #toolbar.pack()

        b2=tk.Button(self,text='Prev', command=lambda:self.change_page(-1))
        b2.grid(row=11,column=5)

        b2=tk.Button(self,text='Next', command=lambda:self.change_page(1))
        b2.grid(row=11,column=6)

        score_var = tk.StringVar()
        score_var.set("Score: " + cross_val_to_number(fig_score[1]))

        score_label = tk.Label(self,textvariable = score_var)
        score_label.grid(row=6, column = 12)

    def change_page(self, diff):
        """
        Change the displayed step in the window.

        Args:
            diff (int): The difference to apply to the current step index.

        """
        if self.steps+diff < len(self.figures_list):
            self.steps = self.steps + diff
            window = StepsWindow(self.parent, self.figures_list, self.steps)
            window.grab_set()
            self.destroy()

        elif self.steps+diff <0:
            pass
        else:
            window = OverviewWindow(self.parent, self.figures_list)
            window.grab_set()
            self.destroy()

class OverviewWindow(tk.Toplevel):
    """
    A class representing the overview window.

    This window provides an overview of the process with a selection menu, a score label,
    and the ability to choose a specific model for further inference.

    Args:
        parent (Tk): The parent Tkinter window.
        figures_list (list): A list of figures representing the steps of the process.

    Attributes:
        figures_list (list): A list of figures representing the steps of the process.
        my_var (IntVar): A Tkinter IntVar tracking the selected model index.
        score_var (StringVar): A Tkinter StringVar storing the current score.
        canvas (FigureCanvasTkAgg): A Tkinter canvas for displaying the figure.

    """
    def __init__(self, parent, figures_list):
        """
        Initialize the OverviewWindow.

        Args:
            parent (Tk): The parent Tkinter window.
            figures_list (list): A list of figures representing the steps of the process.

        """
        super().__init__(parent)
        
        self.figures_list = figures_list

        self.geometry("780x500")
        self.title('Overview')

        self.my_var = tk.IntVar()
        self.my_var.set(len(self.figures_list))
    
        self.score_var = tk.StringVar()
        self.score_var.set("Score: " + cross_val_to_number(self.figures_list[-1][1]))

        self.my_var.trace_add('write', self.callback)
        fig_score = self.figures_list[-1]

        self.canvas = FigureCanvasTkAgg(fig_score[0], self)
            
        self.canvas.get_tk_widget().grid(row = 1, column = 1, columnspan=10, rowspan=10)
    
        tk.OptionMenu(self, self.my_var, *range(1,len(self.figures_list)+1)).grid(row = 5, column = 12)

        score_label = tk.Label(self,textvariable = self.score_var)
        score_label.grid(row=6, column = 12)

        b2=tk.Button(self,text='Choose this model', command= lambda:self.open_window(parent, self.figures_list[self.my_var.get()-1][2]))
        b2.grid(row=7,column=12)

    # defining the callback function (observer) subfunction
    def callback(self, var, index, mode):
        """
        Callback function for updating the displayed model.

        Args:
            var: The variable that triggered the callback.
        """
        
        fig_score_aux = self.figures_list[self.my_var.get()-1]
        self.canvas = FigureCanvasTkAgg(fig_score_aux[0], self)
        self.canvas.get_tk_widget().grid(row = 1, column = 1, columnspan=10, rowspan=10)
        self.score_var.set("Score: " + cross_val_to_number(fig_score_aux[1]))

    def open_window(self, parent, bn):
        """
        Open the inference window with the chosen model.

        Args:
            parent (Tk): The parent Tkinter window.
            bn: The chosen Bayesian Network model.

        """
        window = InferenceWindow(parent, bn)
        window.grab_set()
        self.destroy()


class InferenceWindow(tk.Toplevel):
    """
    A class representing the inference window.

    This window allows the user to choose which variables wants to fix as evidence displaying the inference graphic 
    while letting the user change it.

    Args:
        parent (Tk): The parent Tkinter window.
        bn: The Bayesian Network model.

    Attributes:
        bn: The Bayesian Network model.
        tuple_list (list): A list of tuples containing the evidence variables and associated evidenced value chosen.
        canvas (FigureCanvasTkAgg): A Tkinter canvas for displaying the inference graph.

    """
    def __init__(self, parent, bn):
        """
        Initialize the InferenceWindow.

        Args:
            parent (Tk): The parent Tkinter window.
            bn: The Bayesian Network model.

        """
        super().__init__(parent)      
        self.bn = bn
        self.geometry("1000x500")
        self.title('Inference')
        tk.Label(self,text = "Choose the evidences: ").grid(row = 3, column = 2)
        #tk.OptionMenu(self, self.target, *bn.names()).grid(row = 3, column = 3) 
        count = 4
        self.tuple_list = []      
        for e  in self.bn.names():
            aux = tk.StringVar()
            aux.set("")
            aux.trace_add('write', self.callback)
            tk.Label(self,text = e).grid(row = count, column = 4)
            tk.OptionMenu(self, aux,"", *self.bn.variable(e).labels()).grid(row = count, column = 5)
            self.tuple_list.append((e,aux))
            count += 1
        
        figure = get_inference_graph(self.bn, self.tuple_list)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().grid(row = 3, column = 6, columnspan=10, rowspan=10)
        toolbar_frame = tk.Frame(self) 
        toolbar_frame.grid(row=13,column=6,columnspan=10) 
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame )

    def callback(self, var, index, mode):
        """
        Callback function for updating the inference graph.

        Args:
            var: The variable that triggered the callback. 

        """
        figure = get_inference_graph(self.bn, self.tuple_list)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().grid(row = 1, column = 6, columnspan=10, rowspan=10)



class EDAsWindow(tk.Toplevel):
    """
    A class representing the EDAsWindow.

    This window displays the best solution obtained by the algorithm and provides options to choose the model
    or display the changes between generations.

    Args:
        parent (Tk): The parent Tkinter window.
        umda: The UMDA algorithm object.

    Attributes:
        umda: The UMDA algorithm object.

    """
    def __init__(self, parent, umda):
        """
        Initialize the EDAsWindow.

        Args:
            parent (Tk): The parent Tkinter window.
            umda: The UMDA algorithm object.

        """
        super().__init__(parent)
        self.umda = umda

        self.title('Best solution')
        tk.Label(self,text="Best solution obtained by the algorithm: ").grid(row=1,column=1)

        self.parent = parent
        best_results, generation_information = self.umda.execute_umda()
        best_res = max(best_results, key=attrgetter('fitness'))

        canvas = FigureCanvasTkAgg(umda.from_chain_to_graph(best_res.chain), self)
        canvas.get_tk_widget().grid(row = 2, column = 1, columnspan=10, rowspan=10)

        b1=tk.Button(self,text='Choose this model', command= lambda:self.open_inference_window(parent, best_res.bn))
        b1.grid(row=12,column=5)

        b2=tk.Button(self,text='Show generations', command=lambda:self.show_generations_window(umda, best_results, generation_information))
        b2.grid(row=12,column=6)

        score_var = tk.StringVar()
        score_var.set("Score: " +  "{:.4f}".format(best_res.fitness))

        score_label = tk.Label(self,textvariable = score_var)
        score_label.grid(row=7, column = 12)


    def open_inference_window(self, parent, bn):
        """
        Open the inference window.

        Args:
            parent (Tk): The parent Tkinter window.
            bn: The Bayesian Network model.

        """
        window = InferenceWindow(parent, bn)
        window.grab_set()
        self.destroy()

    def show_generations_window(self, umda, best_results, generation_information):
        """
        Show the generations window.

        Args:
            umda: The UMDA algorithm object.
            best_results: The list of best results obtained during the generations.
            generation_information: The information about each generation.

        """
        window = EDAsGenerationWindow(self, umda, best_results, generation_information, 0)
        window.grab_set()

class EDAsGenerationWindow(tk.Toplevel):
    """
    A class representing the window for displaying the generation information in the EDAs application.

    Parameters:
    - parent: The parent tkinter window.
    - umda: An instance of the UMDA class.
    - best_results: The best results from each generation.
    - generation_information: The information of each generation.
    - steps: The current step/generation.

    Methods:
    - __init__(self, parent, umda, best_results, generation_information, steps): Initializes the EDAsGenerationWindow.
    - change_page(self, diff): Changes the page/generation.

    Attributes:
    - parent: The parent tkinter window.
    - umda: An instance of the UMDA class.
    - best_results: The best results from each generation.
    - generation_information: The information of each generation.
    - steps: The current step/generation.
    """
    def __init__(self, parent, umda, best_results, generation_information, steps):

        super().__init__(parent)
        self.geometry("1560x550")
        self.title('Generation Window')

        self.parent = parent
        self.umda = umda
        self.best_results = best_results
        self.generation_information = generation_information
        self.steps = steps

        #generation_fig = umda.from_generation_to_graph(self.generation_information[self.steps])
        generation_fig = umda.from_chain_to_graph(self.best_results[self.steps].chain)

        canvas = FigureCanvasTkAgg(generation_fig, self)
            
        canvas.get_tk_widget().grid(row = 1, column = 1, columnspan=10, rowspan=10)

        if steps > 0:
            #generation_diff_fig = umda.graph_between_generations(umda.print_data_generation([self.best_results[self.steps-1]]), umda.print_data_generation([self.best_results[self.steps]]))
            generation_diff_fig = umda.graph_between_chains(self.best_results[self.steps-1].chain, self.best_results[self.steps].chain)
            canvas2 = FigureCanvasTkAgg(generation_diff_fig, self)
            canvas2.get_tk_widget().grid(row = 1, column = 11, columnspan=10, rowspan=10)
    
        b2=tk.Button(self,text='Prev', command=lambda:self.change_page(-1))
        b2.grid(row=11,column=5)

        b2=tk.Button(self,text='Next', command=lambda:self.change_page(1))
        b2.grid(row=11,column=6)

    def change_page(self, diff):

        if self.steps+diff < len(self.generation_information):
            self.steps = self.steps + diff
            window = EDAsGenerationWindow(self.parent, self.umda, self.best_results, self.generation_information, self.steps)
            window.grab_set()
            self.destroy()

        elif self.steps+diff <0:
            pass
        else:
            #window = OverviewWindow(self.parent, self.figures_list)
            #window.grab_set()
            self.destroy()

class MainWindow(tk.Tk):
    """
    A class representing the main window of the application.

    Methods:
    - __init__(self): Initializes the MainWindow.
    - open_window(self, model, jumpSteps, no_steps, selection_parameter, dataset, class_variable): Opens a new window based on the selected model.
    - open_windowEDAs(self, n_generations, n_individuals, n_candidates, dataset, class_variable, fitness_metric): Opens a new window for the EDAs model.
    - open_file(self): Opens a file dialog to choose a dataset.
    - callbackInteger(self, P): Callback function for validating integer inputs.
    - callback(self, var, index, mode): Callback function for handling changes in the dropdown selection.

    Attributes:
    - dataset_name: The name of the chosen dataset.
    - model: The selected model.
    - frame: The tkinter frame.
    """
    def __init__(self):
        super().__init__()

        self.geometry('500x400')
        self.title('Main Window')

        l1_str=tk.StringVar()
        font1=('Times',18,'bold')	
        l1=tk.Label(self,textvariable=l1_str,font=font1)
        l1.grid(row=1,column=1)
        l1_str.set('Parameters')

        self.dataset_name = tk.StringVar()

        tk.Button(self,
                text='Choose dataset',
                command= self.open_file).grid(row = 2, column = 1)
        
        score_label = tk.Label(self,textvariable = self.dataset_name)
        score_label.grid(row=2, column = 2, columnspan= 2)

        self.model = tk.StringVar(self)
        self.model.set("-")

        self.model.trace_add('write', self.callback)

        l_model=tk.Label(self,text="Model: ")
        l_model.grid(row=3,column=1)

        dropdown = tk.OptionMenu(self, self.model, "Naive Bayes", "TAN", "Markov Blanket selection by EDAs")
        dropdown.grid(row=3, column = 2, sticky= tk.W)



    def open_window(self, model, jumpSteps, no_steps, selection_parameter, dataset, class_variable):

        self.steps = 0
   
        if model == "Naive Bayes":
            self.figures_list = NB_k_fold_with_steps(jumpSteps, selection_parameter, dataset, class_variable)
        elif model == "TAN":
            self.figures_list = NB_TAN_k_fold_with_steps(jumpSteps, selection_parameter, dataset, class_variable)
        
        if no_steps:
            window = OverviewWindow(self, self.figures_list)
            window.grab_set()
        else:
            window = StepsWindow(self, self.figures_list, self.steps)
            window.grab_set()
    
    def open_windowEDAs(self, n_generations, n_individuals, n_candidates, dataset, class_variable, fitness_metric):
        umda = UMDA(n_candidates, n_individuals, n_generations, dataset, class_variable, fitness_metric)
        window = EDAsWindow(self, umda)
        window.grab_set()
        return
 
    def open_file(self):
        self.file = tk.filedialog.askopenfile(
            title= "Choose dataset", filetypes = [("CSV files", ".csv")])
        self.dataset_name.set(self.file.name)
    
    def callbackInteger(self, P):
        if str.isdigit(P) or P == "":
            return True
        else:
            return False
        
    def callback(self, var, index, mode):
        
        if type(self.frame) is tk.Frame:
            self.frame.grid_forget()
        
        self.frame = tk.Frame(self)    
        self.frame.grid(row = 4, column = 0, columnspan=3, rowspan=4)

        if self.model.get() == "Naive Bayes" or self.model.get() == "TAN":

            l_jumpSteps=tk.Label(self.frame,text="Iterations between steps")
            l_jumpSteps.grid(row=4,column=1)

            vcmd = (self.register(self.callbackInteger))

            jumpSteps = tk.IntVar(self)
            jumpSteps.set(0)

            jumpStepsEntry = tk.Entry(self.frame,textvariable = jumpSteps, validate= 'all', validatecommand=(vcmd, '%P'))

            jumpStepsEntry.grid(row=4,column=2, sticky= tk.W)

            no_steps = tk.IntVar()

            c1 = tk.Checkbutton(self.frame, text='Skip all steps',variable=no_steps, onvalue=1, offvalue=0)
            c1.grid(row = 5, column=1)

            #Button to chose the selection parameter
            selection_parameter = tk.StringVar(self)
            selection_parameter.set("Mutual information")

            l_selection_parameter=tk.Label(self.frame,text="Selection parameter: ")
            l_selection_parameter.grid(row=6,column=1)

            selection_parameter_dropdown = tk.OptionMenu(self.frame, selection_parameter, "Mutual Information", "Score")
            selection_parameter_dropdown.grid(row=6, column = 2, sticky= tk.W)

            df = pd.read_csv(self.dataset_name.get())

            class_variable = tk.StringVar(self)
            class_variable.set("-")

            l_class_variable=tk.Label(self.frame,text="Class variable: ")
            l_class_variable.grid(row=7,column=1)

            class_variable_dropdown = tk.OptionMenu(self.frame, class_variable, *list(df.keys()))
            class_variable_dropdown.grid(row=7, column = 2, sticky= tk.W)



            b1=tk.Button(self.frame, text= 'Enter', command= lambda:self.open_window(self.model.get(), jumpSteps.get(), no_steps.get(), 
                                                                            selection_parameter.get(), self.dataset_name.get(), class_variable.get()))
            
            b1.grid(row=8,column=1)

        elif self.model.get() == "Markov Blanket selection by EDAs":

            vcmd = (self.register(self.callbackInteger))

            l_n_generations=tk.Label(self.frame,text="Number of generations")
            l_n_generations.grid(row=4,column=1)

            n_generations = tk.IntVar(self)
            n_generations.set(1)

            n_generationsEntry = tk.Entry(self.frame,textvariable = n_generations, validate= 'all', validatecommand=(vcmd, '%P'))
            n_generationsEntry.grid(row=4,column=2, sticky= tk.W)

            l_n_individuals=tk.Label(self.frame,text="Number of individuals per generation")
            l_n_individuals.grid(row=5,column=1)

            n_individuals = tk.IntVar(self)
            n_individuals.set(10)

            n_individualsEntry = tk.Entry(self.frame,textvariable = n_individuals, validate= 'all', validatecommand=(vcmd, '%P'))
            n_individualsEntry.grid(row=5,column=2, sticky= tk.W)

            l_n_candidates=tk.Label(self.frame,text="Number of selected candidates per generation")
            l_n_candidates.grid(row=6,column=1)

            n_candidates = tk.IntVar(self)
            n_candidates.set(5)
            
            n_candidatesEntry = tk.Entry(self.frame,textvariable = n_candidates, validate= 'all', validatecommand=(vcmd, '%P'))
            n_candidatesEntry.grid(row=6,column=2, sticky= tk.W)

            df = pd.read_csv(self.dataset_name.get())

            class_variable = tk.StringVar(self)
            class_variable.set("-")

            l_class_variable=tk.Label(self.frame,text="Class variable: ")
            l_class_variable.grid(row=7,column=1)

            class_variable_dropdown = tk.OptionMenu(self.frame, class_variable, *list(df.keys()))
            class_variable_dropdown.grid(row=7, column = 2, sticky= tk.W)

            fitness_metric = tk.StringVar(self)
            fitness_metric.set("Acurracy")

            l_fitness_metric=tk.Label(self.frame,text="Fitness metric: ")
            l_fitness_metric.grid(row=8,column=1)

            fitness_metric_dropdown = tk.OptionMenu(self.frame, fitness_metric, "Acurracy", "BIC")
            fitness_metric_dropdown.grid(row=8, column = 2, sticky= tk.W)

            b1=tk.Button(self.frame, text= 'Enter', command= lambda:self.open_windowEDAs(n_generations.get(), n_individuals.get(), 
                                                                            n_candidates.get(), self.dataset_name.get(), class_variable.get(), fitness_metric.get()))
            
            b1.grid(row=9,column=1)
        
if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
