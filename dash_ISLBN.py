import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import base64
import io
from operator import attrgetter
import matplotlib.pyplot as plt
import pyAgrum as gum
import tempfile
import ast

from NB import NB_k_fold_with_steps, cross_val_to_number
from TAN import NB_TAN_k_fold_with_steps
from inference import get_inference_graph
from MarkovBlanketEDAs import UMDA

########################################################################
# 1) Create the Dash app, allowing callback exceptions,
#    and define a single layout that has *all* possible components/IDs.
########################################################################

app = dash.Dash(
    __name__,
    requests_pathname_prefix='/Model/LearningFromData/ISLBNDash/',
    suppress_callback_exceptions=True
)
server = app.server

# The main layout that includes everything
app.layout = dcc.Loading(
    id="global-spinner",
    overlay_style={"visibility": "visible", "filter": "blur(1px)"},
    type="circle",
    fullscreen=False,
    children=html.Div([
        html.H1("Interactive Structured Learning for Discrete BN", style={'textAlign': 'center'}),

        ########################################################
        # (A) Data upload
        ########################################################
        html.Div(className="card", children=[
                # Title or subtitle for this section
                html.H3("1. Upload Dataset", className="section-title"),
                
                # Container "card"
                html.Div([
                    # Top part with icon and text
                    html.Div([
                        html.Img(
                            src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",  # Cloud icon (or whichever you prefer)
                            className="upload-icon"
                        ),
                        html.Div("Drag and drop or select a CSV file", className="upload-text")
                    ]),
                    
                    # Upload component
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([], style={'display': 'none'}),
                        className="upload-dropzone",
                        multiple=False
                    ),
                ], className="upload-card"),
                # Checkbox for using a default dataset
                html.Div([
                    dcc.Checklist(
                        id='use-default-dataset',
                        options=[{'label': 'Use the default dataset', 'value': 'default'}],
                        value=[],
                        style={'textAlign': 'center', 'marginTop': '10px'}
                    )
                ], style={'textAlign': 'center'}),
                # Feedback message (uploaded file name or error)
                html.Div(id='output-data-upload'),
        ]),

        ########################################################
        # (B) Model selection
        ########################################################
        html.Div(className="card", children=[
            html.H3("2. Select Model", className="section-title", style={'textAlign': 'center'}),
        
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
                    {'label': 'TAN', 'value': 'TAN'},
                    {'label': 'Markov Blanket selection by EDAs', 'value': 'EDAs'}
                ],
                placeholder='Select a model',
                style={'width': '70%', 'margin': '0 auto'}
            ),
        ]),
        html.Div(id='model-parameters'),
        
        ########################################################
        # (C) Run model button and the main output area
        ########################################################
        html.Button('Run Model', id='run-button', n_clicks=0,
                    style={'display': 'block', 'margin': '10px auto'}),

        html.Div(id='model-output'),

        ########################################################
        # (D) Hidden inputs for parameters (NB/TAN or EDAs)
        ########################################################
        html.Div([
            dcc.Input(id='jump-steps', type='number', style={'display': 'none'}),
            dcc.Checklist(id='no-steps', 
                          options=[{'label': 'Yes', 'value': 'yes'}], 
                          value=[], 
                          style={'display': 'none'}),
            dcc.Dropdown(id='selection-parameter', style={'display': 'none'}),
            dcc.Dropdown(id='class-variable', style={'display': 'none'}),

            dcc.Input(id='n-generations', type='number', style={'display': 'none'}),
            dcc.Input(id='n-individuals', type='number', style={'display': 'none'}),
            dcc.Input(id='n-candidates', type='number', style={'display': 'none'}),
            dcc.Dropdown(id='fitness-metric', style={'display': 'none'}),
        ]),

        ########################################################
        # (E) Hidden buttons for step/gen navigation or model choices
        ########################################################
        html.Div([
            html.Button('Previous', id='prev-step-button', n_clicks=0, style={'display': 'none'}),
            html.Button('Next', id='next-step-button', n_clicks=0, style={'display': 'none'}),
            html.Button('Choose this model', id='choose-model-button', n_clicks=0, style={'display': 'none'}),

            html.Button('Previous Generation', id='prev-generation-button', n_clicks=0, style={'display': 'none'}),
            html.Button('Next Generation', id='next-generation-button', n_clicks=0, style={'display': 'none'}),
            html.Button('Choose this model (EDAs)', id='choose-model-button-edas', n_clicks=0, style={'display': 'none'}),
            html.Button('Show generations', id='show-generations-button-edas', n_clicks=0, style={'display': 'none'}),
        ]),

        ########################################################
        # (F) dcc.Store to keep data across callbacks
        ########################################################
        dcc.Store(id='uploaded-data-store'),
        dcc.Store(id='model-results-store'),
        dcc.Store(id='current-step-store'),
        dcc.Store(id='edas-results-store'),
        dcc.Store(id='current-generation-store'),
        dcc.Store(id='bn-model-store'),
        dcc.Store(id='inference-results'),
    ])
)

# Critical: Also set the validation_layout to the exact same layout
# so Dash never complains about missing IDs.
app.validation_layout = app.layout


########################################################################
# 2) Callbacks
########################################################################

# ----------------------------------------------------------------------------
# (A) Callback for uploading data
# ----------------------------------------------------------------------------
@app.callback(
    Output('output-data-upload', 'children'),
    Output('uploaded-data-store', 'data'),
    Input('upload-data', 'contents'),
    Input('use-default-dataset', 'value'),
    State('upload-data', 'filename')
)
def update_output(contents, use_default_value, filename):
    # Modify the path if needed
    default_path = '/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/ISLBN/cars_example.data'
    if 'default' in use_default_value:
        try:
            df = pd.read_csv(default_path)
            return (
                html.Div([html.P('Using default dataset: cars_example.data',
                                 style={'color': 'green', 'fontWeight': 'bold', 'margin': '10px 0'})]),
                df.to_json(date_format='iso', orient='split')
            )
        except Exception as e:
            return (html.Div([f'Error reading default dataset: {e}']), None)

    # Else if user uploaded a file
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return (
                html.Div([html.H5(filename), html.P('File uploaded successfully.')]),
                df.to_json(date_format='iso', orient='split')
            )
        except Exception:
            return (html.Div(['There was an error processing the file.']), None)
    else:
        return '', None


# ----------------------------------------------------------------------------
# (B) Update model parameters depending on user choice
# ----------------------------------------------------------------------------
@app.callback(
    Output('model-parameters', 'children'),
    Input('model-dropdown', 'value'),
    State('uploaded-data-store', 'data')
)
def update_parameters(model, data_json):
    if data_json is None:
        return html.Div('Please upload a dataset first.', 
                        style={'color': 'red','textAlign':'center','marginTop':'5px'})
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    if model in ['Naive Bayes', 'TAN']:
        return html.Div(className="card", children=[
            html.H3("3. Model Parameters", className="section-title", style={'textAlign': 'center'}),
            html.Div([
                html.Label('Iterations between steps:'),
                dcc.Input(id='jump-steps-nb', type='number', value=0, min=0, step=1, style={'width': '60px'}),
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
        return html.Div(className="card", children=[
            html.H3("EDAs Model Parameters", className="section-title", style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of generations:'),
                dcc.Input(id='n-generations', type='number', value=1, min=1, step=1, 
                          style={'width': '60px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of individuals per generation:'),
                dcc.Input(id='n-individuals', type='number', value=10, min=1, step=1,
                          style={'width': '60px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of selected candidates per generation:'),
                dcc.Input(id='n-candidates', type='number', value=5, min=1, step=1,
                          style={'width': '60px'}),
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


# ----------------------------------------------------------------------------
# (C) Main callback to handle "Run Model" and navigation
# ----------------------------------------------------------------------------
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
    Input('show-generations-button-edas', 'n_clicks'),

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
    run_clicks,
    prev_step_clicks,
    next_step_clicks,
    choose_model_clicks,
    prev_gen_clicks,
    next_gen_clicks,
    choose_model_edas_clicks,
    show_generations_edas_clicks,
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
    print(f"Button pressed: {button_id}")

    # Default outputs
    model_results_data_out = model_results_data
    current_step_out = current_step
    edas_results_data_out = edas_results_data
    current_generation_out = current_generation
    bn_model_data_out = bn_model_data

    # RUN MODEL
    if button_id == 'run-button':
        if data_json is None:
            print("No data uploaded.")
            return (dash.no_update,)*5

        df = pd.read_json(io.StringIO(data_json), orient='split')
        
        # NB or TAN
        if model in ['Naive Bayes', 'TAN']:
            if not class_variable:
                print("Class variable not selected.")
                return (dash.no_update,)*5
            if isinstance(class_variable, dict):
                class_variable = class_variable.get("value", None)

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                csv_path = tmp_file.name

            jump_steps = jump_steps or 0
            selection_parameter = selection_parameter or 'Mutual Information'
            no_steps = no_steps or []

            if model == 'Naive Bayes':
                figures_list = NB_k_fold_with_steps(jump_steps, selection_parameter, csv_path, class_variable)
            else:
                figures_list = NB_TAN_k_fold_with_steps(jump_steps, selection_parameter, csv_path, class_variable)

            skip_all = ('yes' in no_steps)
            model_results_data_out = {
                'figures_list': serialize_figures_list(figures_list),
                'no_steps': skip_all
            }
            if skip_all:
                current_step_out = len(figures_list) - 1
            else:
                current_step_out = 0

        # EDAs
        elif model == 'EDAs':
            if class_variable is None:
                print("Class variable not selected.")
                return (dash.no_update,)*5

            n_generations = n_generations or 1
            n_individuals = n_individuals or 10
            n_candidates = n_candidates or 5
            fitness_metric = fitness_metric or 'Accuracy'

            csv_string = df.to_csv(index=False)
            csv_buffer = io.StringIO(csv_string)
            umda = UMDA(n_candidates, n_individuals, n_generations, 
                        csv_buffer, class_variable, fitness_metric)
            best_results, generation_information = umda.execute_umda()

            edas_results_data_out = {
                'umda': serialize_umda(umda, data_json),
                'best_results': [serialize_solution(sol) for sol in best_results],
                'generation_information': serialize_generation_information(generation_information)
            }
            current_generation_out = None  # show best solution first

    # NB/TAN STEP NAV
    elif button_id in ['prev-step-button', 'next-step-button']:
        if not model_results_data or current_step is None:
            raise dash.exceptions.PreventUpdate
        figures_list = model_results_data['figures_list']
        if button_id == 'prev-step-button' and current_step > 0:
            current_step_out = current_step - 1
        elif button_id == 'next-step-button' and current_step < len(figures_list) - 1:
            current_step_out = current_step + 1

    # NB/TAN CHOOSE MODEL
    elif button_id == 'choose-model-button':
        if not model_results_data or current_step is None:
            raise dash.exceptions.PreventUpdate
        figures_list = model_results_data['figures_list']
        bn_serialized = figures_list[current_step]['bn']
        bn = deserialize_bayesnet(bn_serialized)
        bn_model_data_out = serialize_bayesnet(bn)

    # EDAs GENERATION NAV
    elif button_id in ['prev-generation-button', 'next-generation-button']:
        if not edas_results_data or current_generation is None:
            raise dash.exceptions.PreventUpdate
        total_gens = len(edas_results_data['generation_information'])
        if button_id == 'prev-generation-button' and current_generation > 0:
            current_generation_out = current_generation - 1
        elif button_id == 'next-generation-button' and current_generation < total_gens - 1:
            current_generation_out = current_generation + 1

    elif button_id == 'show-generations-button-edas':
        current_generation_out = 0

    # EDAs CHOOSE MODEL
    elif button_id == 'choose-model-button-edas':
        if not edas_results_data:
            raise dash.exceptions.PreventUpdate
        best_results_data = edas_results_data['best_results']
        best_results = [deserialize_solution(sol_data) for sol_data in best_results_data]
        best_res = max(best_results, key=attrgetter('fitness'))
        bn = best_res.bn
        bn_model_data_out = serialize_bayesnet(bn)

    return (
        model_results_data_out,
        current_step_out,
        edas_results_data_out,
        current_generation_out,
        bn_model_data_out
    )


# ----------------------------------------------------------------------------
# (D) Inference callback
# ----------------------------------------------------------------------------
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
    for val, id_dict in zip(evidence_values, evidence_ids):
        if val != '':
            var = id_dict['index']
            evidence[var] = val

    tuple_list = [(var, evidence.get(var, '')) for var in bn.names()]
    figure = get_inference_graph(bn, tuple_list)
    img = fig_to_base64_image(figure)

    return html.Div([
        html.H3('Inference Results', className="section-title", style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), 
                 className="zoomable", 
                 style={'display': 'block', 'margin': '0 auto'}),
    ])


# ----------------------------------------------------------------------------
# (E) Main output display logic
# ----------------------------------------------------------------------------
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
def update_model_output(model_results_data, current_step,
                        edas_results_data, current_generation,
                        bn_model_data, inference_results):

    # 1) If BN chosen => show inference
    if bn_model_data is not None:
        return display_inference_window(bn_model_data, inference_results)

    # 2) If EDAs => show best or generations
    elif edas_results_data is not None:
        if current_generation is None:
            return display_edas_best_solution(edas_results_data)
        else:
            return display_edas_generations(edas_results_data, current_generation)

    # 3) If NB/TAN => show steps
    elif model_results_data is not None and current_step is not None:
        figures_list = model_results_data['figures_list']
        return display_step(figures_list, current_step)

    # 4) Otherwise
    else:
        return html.Div('No model output to display.')


########################################################################
# 3) Helper functions exactly as in your code
########################################################################

def display_edas_best_solution(edas_results_data):
    umda_data = edas_results_data['umda']
    umda = deserialize_umda(umda_data)
    best_results_data = edas_results_data['best_results']
    best_results = [deserialize_solution(x) for x in best_results_data]
    best_res = max(best_results, key=attrgetter('fitness'))
    fig = umda.from_chain_to_graph(best_res.chain)
    img_base64 = fig_to_base64_image(fig)

    return html.Div(className="card", children=[
        html.H3('Best Markov Blanket structure obtained by the algorithm:', className="section-title", 
                style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img_base64), 
                 className="zoomable", 
                 style={'display': 'block', 'margin': '0 auto'}),
        html.P(f"Score (fitness): {best_res.fitness:.4f}", 
               style={'textAlign': 'center', 'fontWeight': 'bold'}),

        html.Div([
            html.Button('Choose this model (EDAs)', 
                        id='choose-model-button-edas', n_clicks=0),
            html.Button('Show generations', 
                        id='show-generations-button-edas', n_clicks=0),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
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
        html.H3(
            f'Generation {generation_index + 1} of {total_generations}',
            className="section-title",
            style={'textAlign': 'center'}
        ),
        html.Img(
            src='data:image/png;base64,{}'.format(img),
            className="zoomable",
            style={'display': 'block', 'margin': '0 auto'}
        ),
    ]

    if generation_index > 0:
        diff_figure = umda.graph_between_chains(
            best_results[generation_index-1].chain,
            best_results[generation_index].chain
        )
        diff_img = fig_to_base64_image(diff_figure)
        content.append(html.H4(
            'Differences with the previous generation',
            style={'textAlign': 'center'}
        ))
        content.append(html.Img(
            src='data:image/png;base64,{}'.format(diff_img),
            className="zoomable",
            style={'display': 'block', 'margin': '0 auto'}
        ))

    content.append(html.Div([
        html.Button('Previous', id='prev-generation-button', n_clicks=0),
        html.Button('Next', id='next-generation-button', n_clicks=0),
        html.Button('Choose this model (EDAs)', id='choose-model-button-edas', n_clicks=0),
        html.Button('Show generations', id='show-generations-button-edas', n_clicks=0,
                    style={'display': 'none'}),
    ], style={'textAlign': 'center'}))

    return html.Div(content, className="card")

def display_step(figures_list, step_index):
    data = figures_list[step_index]
    img_data = data['fig']
    score = cross_val_to_number(data['scores'])
    total_steps = len(figures_list)

    return html.Div(className="card", children=[
        html.H3(f'Step {step_index + 1} of {total_steps}', className="section-title", style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img_data), 
                 className="zoomable", 
                 style={'display': 'block', 'margin': '0 auto'}),
        html.P(f'Score: {score}', style={'textAlign': 'center'}),
        html.Div([
            html.Button('Previous', id='prev-step-button', n_clicks=0),
            html.Button('Next', id='next-step-button', n_clicks=0),
            html.Button('Choose this model', id='choose-model-button', n_clicks=0),
        ], style={'textAlign': 'center'}),
    ])

def display_inference_window(bn_model_data, inference_results=None):
    bn = deserialize_bayesnet(bn_model_data)
    variables = bn.names()
    evidence_selection = []

    for var in variables:
        labels = bn.variable(var).labels()
        evidence_selection.append(
            html.Div(children=[
                html.Label(f'{var}:'),
                dcc.Dropdown(
                    id={'type': 'evidence-dropdown', 'index': var},
                    options=[{'label': lb, 'value': lb} for lb in [''] + list(labels)],
                    value='',
                    clearable=True,
                    style={'width': '150px'}
                )
            ],
            style={'display': 'inline-block', 'verticalAlign': 'top', 'margin': '5px 15px'})
        )

    results_section = html.Div()
    if inference_results is not None:
        results_section = html.Div(className="card", children=[
            inference_results
        ], style={'marginTop': '20px'})

    return html.Div(children=[
            html.Div(className="card", children=[
            html.H3('Inference', className="section-title", style={'textAlign': 'center'}),
            html.Div(children=evidence_selection,
                    style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
            html.Div(children=[
                html.Button('Calculate Inference', id='calculate-inference-button', n_clicks=0)
            ], style={'textAlign': 'center', 'marginTop': '20px'}),
        ]),results_section,
    ])

def fig_to_base64_image(fig):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def serialize_figures_list(figures_list):
    serialized_list = []
    for step_tuple in figures_list:
        if len(step_tuple) == 2:
            fig, scores = step_tuple
            bn2 = None
        else:
            fig, scores, bn2 = step_tuple

        img_data = fig_to_base64_image(fig)
        bn_serialized = serialize_bayesnet(bn2) if bn2 else None

        serialized_list.append({
            'fig': img_data,
            'scores': scores,
            'bn': bn_serialized
        })
    return serialized_list

def serialize_solution(solution):
    bn_serialized = serialize_bayesnet(solution.bn)
    return {
        'chain': solution.chain,
        'fitness': solution.fitness,
        'bn': bn_serialized
    }

def deserialize_solution(data):
    bn = deserialize_bayesnet(data['bn'])
    sol = UMDA.Solution(chain=data['chain'], fitness=data['fitness'], bn=bn)
    return sol

def serialize_generation_information(generation_info):
    serialized_info = []
    for gen_dict in generation_info:
        serialized_dict = {str(k): v for k, v in gen_dict.items()}
        serialized_info.append(serialized_dict)
    return serialized_info

def serialize_umda(umda, dataset_json):
    edges_dict_serialized = {str(k): v for k, v in umda.edges_dictionary.items()}
    return {
        'class_variable': umda.class_variable,
        'nodes_list': umda.nodes_list,
        'edges_dictionary': edges_dict_serialized,
        'dataset': dataset_json
    }

def deserialize_umda(data):
    edges_dict = {ast.literal_eval(k): v for k, v in data['edges_dictionary'].items()}
    dataset_json = data['dataset']
    df = pd.read_json(io.StringIO(dataset_json), orient='split')
    csv_buffer = io.StringIO(df.to_csv(index=False))

    umda = UMDA(
        selected_candidates=None,
        num_individuals=None,
        n_generations=None,
        dataset=csv_buffer,
        class_variable=data['class_variable'],
        fitness_metric=None
    )
    umda.nodes_list = data['nodes_list']
    umda.edges_dictionary = edges_dict
    return umda

def serialize_bayesnet(bn):
    nodes = {}
    for node in bn.nodes():
        variable = bn.variable(node)
        labels = variable.labels()
        if len(labels) != len(set(labels)):
            print(f"Warning: Duplicate labels found in variable '{variable.name()}': {labels}")
        nodes[str(node)] = {
            'name': variable.name(),
            'description': variable.description(),
            'labels': labels
        }
    edges = [[str(p), str(c)] for p, c in bn.arcs()]
    cpts = {}
    for node in bn.nodes():
        cpt = bn.cpt(node)
        flat_cpt = cpt.toarray().flatten().tolist()
        cpts[str(node)] = flat_cpt

    return {
        'nodes': nodes,
        'edges': edges,
        'cpts': cpts
    }

def deserialize_bayesnet(serialized_bn):
    bn = gum.BayesNet()
    node_id_map = {}

    for node_id_str, node_info in serialized_bn['nodes'].items():
        labels = node_info['labels']
        # If any duplicates, rename them
        if len(labels) != len(set(labels)):
            seen = {}
            unique_labels = []
            for lb in labels:
                if lb in seen:
                    seen[lb] += 1
                    new_lb = f"{lb}_{seen[lb]}"
                    unique_labels.append(new_lb)
                else:
                    seen[lb] = 0
                    unique_labels.append(lb)
            labels = unique_labels

        var = gum.LabelizedVariable(node_info['name'], node_info['description'], len(labels))
        # Temporary labels
        for i in range(len(labels)):
            var.changeLabel(i, f"__temp__{i}")
        for i, lb in enumerate(labels):
            var.changeLabel(i, lb)

        new_id = bn.add(var)
        node_id_map[node_id_str] = new_id

    for edge in serialized_bn['edges']:
        p_id_str, c_id_str = edge
        bn.addArc(node_id_map[p_id_str], node_id_map[c_id_str])

    for node_id_str, flat_cpt_list in serialized_bn['cpts'].items():
        node_id = node_id_map[node_id_str]
        cpt_size = bn.cpt(node_id).toarray().size
        if len(flat_cpt_list) != cpt_size:
            raise ValueError("CPT size mismatch")
        bn.cpt(node_id).fillWith([float(x) for x in flat_cpt_list])

    return bn

########################################################################
# 4) Run the server
########################################################################
if __name__ == '__main__':
    app.run_server(debug=False, threaded=False, host='0.0.0.0', port=8053)