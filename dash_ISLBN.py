import dash 
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import base64
import io
from operator import attrgetter
import matplotlib.pyplot as plt
import pyAgrum as gum

import ast


# Import custom modules
from NB import NB_k_fold_with_steps, cross_val_to_number
from TAN import NB_TAN_k_fold_with_steps
from inference import get_inference_graph
from MarkovBlanketEDAs import UMDA


# Initialize the Dash app
app = dash.Dash(
    __name__,
    requests_pathname_prefix='/Model/LearningFromData/ISLBNDash/',
    suppress_callback_exceptions=True
)

# Application Layout
app.layout = html.Div([
    html.H1("Bayesian Models Application", style={'textAlign': 'center'}),

    # Dataset upload
    html.H3("1. Load Dataset", style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and drop or ', html.A('select a CSV file')]),
        style={
            'width': '50%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '0 auto'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload', style={'textAlign': 'center'}),

    html.Hr(),

    # Model selection
    html.H3("2. Select Model", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
            {'label': 'TAN', 'value': 'TAN'},
            {'label': 'Markov Blanket selection by EDAs', 'value': 'EDAs'}
        ],
        placeholder='Select a model',
        style={'width': '50%', 'margin': '0 auto'}
    ),
    html.Div(id='model-parameters'),

    # Run model button
    html.Button('Run Model', id='run-button', n_clicks=0, style={'display': 'block', 'margin': '10px auto'}),

    # Output display
    html.Div(id='model-output'),

    # Hidden inputs to prevent callback errors
    html.Div([
        dcc.Input(id='jump-steps', type='number', style={'display': 'none'}),
        dcc.Checklist(id='no-steps', options=[{'label': 'Yes', 'value': 'yes'}], value=[], style={'display': 'none'}),
        dcc.Dropdown(id='selection-parameter', style={'display': 'none'}),
        dcc.Dropdown(id='class-variable', style={'display': 'none'}),
        dcc.Input(id='n-generations', type='number', style={'display': 'none'}),
        dcc.Input(id='n-individuals', type='number', style={'display': 'none'}),
        dcc.Input(id='n-candidates', type='number', style={'display': 'none'}),
        dcc.Dropdown(id='fitness-metric', style={'display': 'none'}),
    ]),

    # Hidden buttons
    html.Div([
        html.Button('Previous', id='prev-step-button', n_clicks=0, style={'display': 'none'}),
        html.Button('Next', id='next-step-button', n_clicks=0, style={'display': 'none'}),
        html.Button('Choose this model', id='choose-model-button', n_clicks=0, style={'display': 'none'}),
        html.Button('Previous Generation', id='prev-generation-button', n_clicks=0, style={'display': 'none'}),
        html.Button('Next Generation', id='next-generation-button', n_clicks=0, style={'display': 'none'}),
        html.Button('Choose this model (EDAs)', id='choose-model-button-edas', n_clicks=0, style={'display': 'none'}),
    ]),

    # Hidden stores to keep the state
    dcc.Store(id='uploaded-data-store'),
    dcc.Store(id='model-results-store'),
    dcc.Store(id='current-step-store'),
    dcc.Store(id='edas-results-store'),
    dcc.Store(id='current-generation-store'),
    dcc.Store(id='bn-model-store'),
    dcc.Store(id='inference-results'),
])

# Callback for uploading data
@app.callback(
    Output('output-data-upload', 'children'),
    Output('uploaded-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return html.Div([html.H5(filename), html.P('File uploaded successfully.')]), df.to_json(date_format='iso', orient='split')
        except Exception as e:
            return html.Div(['There was an error processing the file.']), None
    else:
        return '', None

# Callback for updating model parameters
@app.callback(
    Output('model-parameters', 'children'),
    Input('model-dropdown', 'value'),
    State('uploaded-data-store', 'data')
)
def update_parameters(model, data_json):
    if data_json is None:
        return html.Div('Please upload a dataset first.', style={'color': 'red', 'textAlign': 'center', 'margin-top':'5px'})
    df = pd.read_json(io.StringIO(data_json), orient='split')
    if model in ['Naive Bayes', 'TAN']:
        return html.Div([
            html.H3("Model Parameters", style={'textAlign': 'center'}),
            html.Div([
                html.Label('Iterations between steps:'),
                dcc.Input(id='jump-steps', type='number', value=0, min=0, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Skip all steps:'),
                dcc.Checklist(
                    id='no-steps',
                    options=[{'label': 'Yes', 'value': 'yes'}],
                    value=[]
                ),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Selection parameter:'),
                dcc.Dropdown(
                    id='selection-parameter',
                    options=[
                        {'label': 'Mutual Information', 'value': 'Mutual Information'},
                        {'label': 'Score', 'value': 'Score'}
                    ],
                    value='Mutual Information',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Class variable:'),
                dcc.Dropdown(
                    id='class-variable',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    placeholder='Select the class variable',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center'}),
        ])
    elif model == 'EDAs':
        return html.Div([
            html.H3("EDAs Model Parameters", style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of generations:'),
                dcc.Input(id='n-generations', type='number', value=1, min=1, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of individuals per generation:'),
                dcc.Input(id='n-individuals', type='number', value=10, min=1, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of selected candidates per generation:'),
                dcc.Input(id='n-candidates', type='number', value=5, min=1, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Class variable:'),
                dcc.Dropdown(
                    id='class-variable',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    placeholder='Select the class variable',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Fitness metric:'),
                dcc.Dropdown(
                    id='fitness-metric',
                    options=[
                        {'label': 'Accuracy', 'value': 'Accuracy'},
                        {'label': 'BIC', 'value': 'BIC'}
                    ],
                    value='Accuracy',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center'}),
        ])
    else:
        return ''

# Combined Callback to Handle Model Running and Navigation
@app.callback(
    Output('model-results-store', 'data'),
    Output('current-step-store', 'data'),
    Output('edas-results-store', 'data'),
    Output('current-generation-store', 'data'),
    Output('bn-model-store', 'data'),
    Input('run-button', 'n_clicks'),
    Input('prev-step-button', 'n_clicks'),
    Input('next-step-button', 'n_clicks'),
    Input('choose-model-button', 'n_clicks'),
    Input('prev-generation-button', 'n_clicks'),
    Input('next-generation-button', 'n_clicks'),
    Input('choose-model-button-edas', 'n_clicks'),
    State('model-results-store', 'data'),
    State('current-step-store', 'data'),
    State('edas-results-store', 'data'),
    State('current-generation-store', 'data'),
    State('bn-model-store', 'data'),
    State('model-dropdown', 'value'),
    State('uploaded-data-store', 'data'),
    State('jump-steps', 'value'),
    State('no-steps', 'value'),
    State('selection-parameter', 'value'),
    State('class-variable', 'value'),
    State('n-generations', 'value'),
    State('n-individuals', 'value'),
    State('n-candidates', 'value'),
    State('fitness-metric', 'value'),
    prevent_initial_call=True
)
def handle_model_run_and_navigation(
    run_clicks, prev_step_clicks, next_step_clicks, choose_model_clicks,
    prev_gen_clicks, next_gen_clicks, choose_model_edas_clicks,
    model_results_data, current_step,
    edas_results_data, current_generation,
    bn_model_data,
    model, data_json,
    jump_steps, no_steps, selection_parameter, class_variable,
    n_generations, n_individuals, n_candidates, fitness_metric
):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize return values
    model_results_data_out = model_results_data
    current_step_out = current_step
    edas_results_data_out = edas_results_data
    current_generation_out = current_generation
    bn_model_data_out = bn_model_data  # Corrected initialization

    print(f"Button pressed: {button_id}")

    if button_id == 'run-button':
        if data_json is None:
            print("No data uploaded.")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        df = pd.read_json(io.StringIO(data_json), orient='split')
        if model in ['Naive Bayes', 'TAN']:
            if class_variable is None:
                print("Class variable not selected.")
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            if model == 'Naive Bayes':
                # Ensure parameters have default values if None
                jump_steps = jump_steps or 0
                selection_parameter = selection_parameter or 'Mutual Information'
                no_steps = no_steps or []
                figures_list = NB_k_fold_with_steps(jump_steps, selection_parameter, df, class_variable)
            elif model == 'TAN':
                jump_steps = jump_steps or 0
                selection_parameter = selection_parameter or 'Mutual Information'
                no_steps = no_steps or []
                figures_list = NB_TAN_k_fold_with_steps(jump_steps, selection_parameter, df, class_variable)
            model_results_data_out = {
                'figures_list': serialize_figures_list(figures_list),
                'no_steps': 'yes' in no_steps
            }
            current_step_out = 0
            print("Naive Bayes/TAN model executed successfully.")
        elif model == 'EDAs':
            if class_variable is None:
                print("Class variable not selected.")
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            # Ensure parameters have default values if None
            n_generations = n_generations or 1
            n_individuals = n_individuals or 10
            n_candidates = n_candidates or 5
            fitness_metric = fitness_metric or 'Accuracy'
            # Convert DataFrame to CSV string and wrap in StringIO
            csv_string = df.to_csv(index=False)
            csv_buffer = io.StringIO(csv_string)
            umda = UMDA(n_candidates, n_individuals, n_generations, csv_buffer, class_variable, fitness_metric)
            best_results, generation_information = umda.execute_umda()
            # Serialize dataset
            dataset_json = data_json  # Already in JSON format
            edas_results_data_out = {
                'umda': serialize_umda(umda, dataset_json),
                'best_results': [serialize_solution(sol) for sol in best_results],
                'generation_information': serialize_generation_information(generation_information)
            }
            current_generation_out = 0
            print("EDAs model executed successfully.")
    elif button_id in ['prev-step-button', 'next-step-button']:
        if model_results_data is None or current_step is None:
            print("Model results or current step not available.")
            raise dash.exceptions.PreventUpdate
        figures_list = model_results_data['figures_list']
        if button_id == 'prev-step-button' and current_step > 0:
            current_step_out = current_step - 1
            print("Moved to previous step.")
        elif button_id == 'next-step-button' and current_step < len(figures_list) - 1:
            current_step_out = current_step + 1
            print("Moved to next step.")
    elif button_id == 'choose-model-button':
        if model_results_data is None or current_step is None:
            print("Model results or current step not available.")
            raise dash.exceptions.PreventUpdate
        figures_list = model_results_data['figures_list']
        bn_str = figures_list[current_step]['bn']
        bn = gum.BayesNet()
        try:
            bn.loadBN(bn_str)
            print("Bayesian Network loaded from string.")
        except Exception as e:
            print(f"Error loading BN from string: {e}")
            raise
        bn_model_data_out = serialize_bayesnet(bn)
        print("Bayesian Network serialized successfully.")

    elif button_id in ['prev-generation-button', 'next-generation-button']:
        if edas_results_data is None or current_generation is None:
            print("EDAs results or current generation not available.")
            raise dash.exceptions.PreventUpdate
        if button_id == 'prev-generation-button' and current_generation > 0:
            current_generation_out = current_generation - 1
            print("Moved to previous generation.")
        elif button_id == 'next-generation-button' and current_generation < len(edas_results_data['generation_information']) - 1:
            current_generation_out = current_generation + 1
            print("Moved to next generation.")
    elif button_id == 'choose-model-button-edas':
        if edas_results_data is None:
            print("EDAs results not available.")
            raise dash.exceptions.PreventUpdate
        best_results_data = edas_results_data['best_results']
        best_results = [deserialize_solution(sol_data) for sol_data in best_results_data]
        best_res = max(best_results, key=attrgetter('fitness'))
        bn = best_res.bn
        bn_model_data_out = serialize_bayesnet(bn)
        print("Best EDAs model serialized successfully.")

    return model_results_data_out, current_step_out, edas_results_data_out, current_generation_out, bn_model_data_out


# Callback for Inference
@app.callback(
    Output('inference-results', 'data'),
    Input('calculate-inference-button', 'n_clicks'),
    State({'type': 'evidence-dropdown', 'index': dash.ALL}, 'value'),
    State({'type': 'evidence-dropdown', 'index': dash.ALL}, 'id'),
    State('bn-model-store', 'data'),
    prevent_initial_call=True
)
def perform_inference(n_clicks, evidence_values, evidence_ids, bn_model_data):
    if bn_model_data is None:
        raise dash.exceptions.PreventUpdate
    bn = deserialize_bayesnet(bn_model_data)
    evidence = {}
    for value, id_dict in zip(evidence_values, evidence_ids):
        if value != '':
            var = id_dict['index']
            evidence[var] = value
    # Use the deserialized Bayesian Network for inference
    tuple_list = [(var, evidence.get(var, '')) for var in bn.names()]
    figure = get_inference_graph(bn, tuple_list)
    img = fig_to_base64_image(figure)
    content = html.Div([
        html.H4('Inference Results', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
    ])
    return content


# Callback to update model output
@app.callback(
    Output('model-output', 'children'),
    Input('model-results-store', 'data'),
    Input('current-step-store', 'data'),
    Input('edas-results-store', 'data'),
    Input('current-generation-store', 'data'),
    Input('bn-model-store', 'data'),
    Input('inference-results', 'data'),
    prevent_initial_call=True
)
def update_model_output(model_results_data, current_step, edas_results_data, current_generation, bn_model_data, inference_results):
    if inference_results is not None:
        return inference_results
    elif bn_model_data is not None:
        return display_inference_window(bn_model_data)
    elif edas_results_data is not None and current_generation is not None:
        return display_edas_generations(edas_results_data, current_generation)
    elif model_results_data is not None and current_step is not None:
        figures_list = model_results_data['figures_list']
        return display_step(figures_list, current_step)
    else:
        return html.Div('No model output to display.')

# Helper Functions for Display
def fig_to_base64_image(fig):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def display_step(figures_list, step_index):
    data = figures_list[step_index]
    img_data = data['fig']
    score = cross_val_to_number(data['scores'])
    total_steps = len(figures_list)
    return html.Div([
        html.H3(f'Step {step_index + 1} of {total_steps}', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img_data), style={'display': 'block', 'margin': '0 auto'}),
        html.P(f'Score: {score}', style={'textAlign': 'center'}),
        html.Div([
            html.Button('Previous', id='prev-step-button', n_clicks=0),
            html.Button('Next', id='next-step-button', n_clicks=0),
            html.Button('Choose this model', id='choose-model-button', n_clicks=0),
        ], style={'textAlign': 'center'}),
    ])


def display_edas_generations(edas_results_data, generation_index):
    umda_data = edas_results_data['umda']
    umda = deserialize_umda(umda_data)
    best_results_data = edas_results_data['best_results']
    best_results = [deserialize_solution(sol_data) for sol_data in best_results_data]
    total_generations = len(edas_results_data['generation_information'])
    figure = umda.from_chain_to_graph(best_results[generation_index].chain)
    img = fig_to_base64_image(figure)
    content = [
        html.H3(f'Generation {generation_index + 1} of {total_generations}', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
    ]
    if generation_index > 0:
        diff_figure = umda.graph_between_chains(best_results[generation_index-1].chain, best_results[generation_index].chain)
        diff_img = fig_to_base64_image(diff_figure)
        content.append(html.H4('Differences with the previous generation', style={'textAlign': 'center'}))
        content.append(html.Img(src='data:image/png;base64,{}'.format(diff_img), style={'display': 'block', 'margin': '0 auto'}))
    content.append(html.Div([
        html.Button('Previous', id='prev-generation-button', n_clicks=0),
        html.Button('Next', id='next-generation-button', n_clicks=0),
        html.Button('Choose this model', id='choose-model-button-edas', n_clicks=0),
    ], style={'textAlign': 'center'}))
    return html.Div(content)

def serialize_figures_list(figures_list):
    serialized_list = []
    for fig, scores, bn2 in figures_list:
        img_data = fig_to_base64_image(fig)
        bn_serialized = serialize_bayesnet(bn2)
        serialized_list.append({
            'fig': img_data,
            'scores': scores,
            'bn': bn_serialized
        })
    return serialized_list



def display_inference_window(bn_model_data):
    bn = deserialize_bayesnet(bn_model_data)
    variables = bn.names()
    evidence_selection = []
    for var in variables:
        var_labels = bn.variable(var).labels()
        # Convert var_labels (a tuple) to a list for concatenation
        var_labels_list = list(var_labels)
        
        evidence_selection.append(
            html.Div([
                html.Label(f'{var}:'),
                dcc.Dropdown(
                    id={'type': 'evidence-dropdown', 'index': var},
                    options=[{'label': label, 'value': label} for label in [''] + var_labels_list],
                    value='',
                    clearable=True,
                    style={'width': '200px'}
                )
            ], style={'marginBottom': '10px', 'display': 'inline-block', 'marginRight': '20px'})
        )
    return html.Div([
        html.H3('Inference', style={'textAlign': 'center'}),
        html.Div(evidence_selection, style={'columnCount': 2}),
        html.Button('Calculate Inference', id='calculate-inference-button', n_clicks=0),
    ])

# Serialization functions
def serialize_solution(solution):
    bn_serialized = serialize_bayesnet(solution.bn)
    return {
        'chain': solution.chain,
        'fitness': solution.fitness,
        'bn': bn_serialized
    }


def deserialize_solution(data):
    bn = deserialize_bayesnet(data['bn'])
    solution = UMDA.Solution(chain=data['chain'], fitness=data['fitness'], bn=bn)
    return solution

def serialize_generation_information(generation_info):
    # Convert list of dictionaries with tuple keys to string keys
    serialized_info = []
    for gen_dict in generation_info:
        serialized_dict = {str(k): v for k, v in gen_dict.items()}
        serialized_info.append(serialized_dict)
    return serialized_info

def serialize_umda(umda, dataset_json):
    # Convert tuple keys to strings
    edges_dict_serialized = {str(k): v for k, v in umda.edges_dictionary.items()}
    return {
        'class_variable': umda.class_variable,
        'nodes_list': umda.nodes_list,
        'edges_dictionary': edges_dict_serialized,
        'dataset': dataset_json  # Include dataset
    }

    
def deserialize_generation_information(serialized_info):
    deserialized_info = []
    for serialized_dict in serialized_info:
        deserialized_dict = {ast.literal_eval(k): v for k, v in serialized_dict.items()}
        deserialized_info.append(deserialized_dict)
    return deserialized_info


def deserialize_umda(data):
    # Convert string keys back to tuple keys
    edges_dict = {ast.literal_eval(k): v for k, v in data['edges_dictionary'].items()}
    dataset_json = data['dataset']
    df = pd.read_json(io.StringIO(dataset_json), orient='split')
    csv_buffer = io.StringIO(df.to_csv(index=False))
    umda = UMDA(
        selected_candidates=None,
        num_individuals=None,
        n_generations=None,
        dataset=csv_buffer,  # Pass dataset buffer
        class_variable=data['class_variable'],
        fitness_metric=None
    )
    umda.nodes_list = data['nodes_list']
    umda.edges_dictionary = edges_dict
    return umda

def serialize_bayesnet(bn):
    # Extract nodes and their variable types
    nodes = {}
    for node in bn.nodes():
        variable = bn.variable(node)
        labels = variable.labels()
        # Verify uniqueness of labels
        if len(labels) != len(set(labels)):
            print(f"Warning: Duplicate labels found in variable '{variable.name()}': {labels}")
        nodes[str(node)] = {
            'name': variable.name(),
            'description': variable.description(),
            'labels': labels
        }

    # Extract edges
    edges = [[str(parent), str(child)] for parent, child in bn.arcs()]

    # Extract CPTs
    cpts = {}
    for node in bn.nodes():
        cpt = bn.cpt(node)
        flat_cpt = cpt.toarray().flatten().tolist()  # Corrected method
        cpts[str(node)] = flat_cpt  # Store flat list of CPT values

        # Debug print
        print(f"CPT for node '{bn.variable(node).name()}' has size {cpt.toarray().size}, flat_cpt length: {len(flat_cpt)}")

    # Combine all information
    serialized_bn = {
        'nodes': nodes,
        'edges': edges,
        'cpts': cpts
    }
    return serialized_bn

def deserialize_bayesnet(serialized_bn):
    bn = gum.BayesNet()
    node_id_map = {}

    # Reconstruct nodes and variables
    for node_id_str, node_info in serialized_bn['nodes'].items():
        labels = node_info['labels']
        print(f"Deserializing node '{node_info['name']}' with labels: {labels}")

        # Check for duplicate labels
        if len(labels) != len(set(labels)):
            print(f"Duplicate labels found for variable '{node_info['name']}': {labels}")
            # Handle duplicates by appending a unique suffix
            seen_labels = {}
            unique_labels = []
            for label in labels:
                if label in seen_labels:
                    seen_labels[label] += 1
                    new_label = f"{label}_{seen_labels[label]}"
                    unique_labels.append(new_label)
                    print(f"Renamed duplicate label '{label}' to '{new_label}'")
                else:
                    seen_labels[label] = 0
                    unique_labels.append(label)
            print(f"Labels after ensuring uniqueness: {unique_labels}")
        else:
            unique_labels = labels.copy()
            print(f"No duplicate labels for variable '{node_info['name']}'. Using labels as is.")

        # Ensure unique_labels are indeed unique
        if len(unique_labels) != len(set(unique_labels)):
            raise ValueError(f"Labels are not unique after processing for variable '{node_info['name']}': {unique_labels}")

        num_labels = len(unique_labels)

        # Create variable with num_labels and temporary unique labels
        variable = gum.LabelizedVariable(node_info['name'], node_info['description'], num_labels)
        # Assign temporary labels that do not conflict
        for i in range(num_labels):
            temp_label = f"__temp_label_{i}__"
            try:
                variable.changeLabel(i, temp_label)
            except gum.DuplicateElement:
                print(f"Duplicate temporary label '{temp_label}' detected for variable '{node_info['name']}'")
                raise

        # Now change labels to unique_labels
        for i, label in enumerate(unique_labels):
            try:
                variable.changeLabel(i, label)
                print(f"Assigned label '{label}' to position {i} of variable '{node_info['name']}'")
            except gum.DuplicateElement:
                print(f"Duplicate label '{label}' detected when assigning to variable '{node_info['name']}'")
                raise

        # Verify that labels are correctly assigned
        print(f"Variable '{variable.name()}': labels = {variable.labels()}, domain size = {variable.domainSize()}")

        # Add variable to the Bayesian Network
        node_id = bn.add(variable)
        node_id_map[node_id_str] = node_id

    # Reconstruct edges
    for parent_id_str, child_id_str in serialized_bn['edges']:
        parent_id = node_id_map[parent_id_str]
        child_id = node_id_map[child_id_str]
        bn.addArc(parent_id, child_id)

    # Reconstruct CPTs
    for node_id_str, flat_cpt_list in serialized_bn['cpts'].items():
        node_id = node_id_map[node_id_str]
        # Ensure the CPT size matches
        expected_size = bn.cpt(node_id).toarray().size
        print(f"CPT for node '{bn.variable(node_id).name()}' expected size: {expected_size}, flat_cpt_list length: {len(flat_cpt_list)}")
        if len(flat_cpt_list) != expected_size:
            raise ValueError(
                f"Expected CPT size {expected_size}, but got {len(flat_cpt_list)} for node '{bn.variable(node_id).name()}'"
            )
        # Convert values to float if necessary
        flat_cpt_list = [float(value) for value in flat_cpt_list]
        # Fill CPT with the flattened list
        bn.cpt(node_id).fillWith(flat_cpt_list)

    return bn


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8053)