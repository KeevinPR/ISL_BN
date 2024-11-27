import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
from operator import attrgetter
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Imports
from NB import NB_k_fold_with_steps, cross_val_to_number
from TAN import NB_TAN_k_fold_with_steps
from inference import get_inference_graph
from MarkovBlanketEDAs import UMDA


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix='/Model/LearningFromData/ISLBNDash',
    suppress_callback_exceptions=True
)

def fig_to_base64_image(fig):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def display_step(figures_list, step_index):
    figure = figures_list[step_index][0]
    score = cross_val_to_number(figures_list[step_index][1])
    img = fig_to_base64_image(figure)
    total_steps = len(figures_list)
    return html.Div([
        html.H3(f'Paso {step_index + 1} de {total_steps}', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
        html.P(f'Score: {score}', style={'textAlign': 'center'}),
        html.Div([
            html.Button('Anterior', id='prev-step-button', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Siguiente', id='next-step-button', n_clicks=0),
        ], style={'textAlign': 'center', 'marginTop': '10px'}),
        html.Button('Elegir este modelo', id='choose-model-button', n_clicks=0, style={'display': 'block', 'margin': '10px auto'}),
    ])

def display_overview(figures_list):
    options = [{'label': f'Paso {i+1}', 'value': i} for i in range(len(figures_list))]
    last_figure = figures_list[-1][0]
    score = cross_val_to_number(figures_list[-1][1])
    img = fig_to_base64_image(last_figure)
    return html.Div([
        html.H3('Visión General', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='overview-dropdown',
            options=options,
            value=len(figures_list) - 1,
            clearable=False,
            style={'width': '200px', 'margin': '0 auto'}
        ),
        html.Div(id='overview-figure'),
        html.P(id='overview-score', style={'textAlign': 'center'}),
        html.Button('Elegir este modelo', id='choose-model-button-overview', n_clicks=0, style={'display': 'block', 'margin': '10px auto'}),
    ])

def display_edas_results(best_res, umda):
    figure = umda.from_chain_to_graph(best_res.chain)
    img = fig_to_base64_image(figure)
    return html.Div([
        html.H3('Mejor Solución Obtenida por el Algoritmo', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
        html.P(f'Score: {best_res.fitness:.4f}', style={'textAlign': 'center'}),
        html.Div([
            html.Button('Mostrar Generaciones', id='show-generations-button', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Elegir este modelo', id='choose-model-button-edas', n_clicks=0),
        ], style={'textAlign': 'center', 'marginTop': '10px'}),
    ])
    
app.layout = html.Div([
    dcc.Store(id='uploaded-data-store'),
    dcc.Store(id='model-results-store'),
    dcc.Store(id='current-step-store'),
    dcc.Store(id='current-model-store'),
    dcc.Store(id='edas-results-store'),
    dcc.Store(id='edas-best-results-store'),
    dcc.Store(id='bn-model-store'),
    html.H1("Aplicación de Modelos Bayesianos", style={'textAlign': 'center'}),
    html.Div(id='main-content'),
    html.Div(id='hidden-div', style={'display':'none'})  # Div oculto para evitar errores
])

@app.callback(
    Output('main-content', 'children'),
    Input('current-step-store', 'data'),
    Input('current-model-store', 'data'),
    Input('edas-results-store', 'data'),
    Input('bn-model-store', 'data'),
    State('model-results-store', 'data'),
    State('edas-best-results-store', 'data')
)
def display_main_content(current_step, current_model, edas_results, bn_model_data, model_results, edas_best_results):
    if current_step is None and current_model is None and edas_results is None and bn_model_data is None:
        return display_main_window()
    elif current_model == 'EDAs' and edas_results is not None:
        return display_edas_results(edas_best_results, edas_results['umda'])
    elif current_model in ['Naive Bayes', 'TAN'] and model_results is not None:
        if 'no_steps' in model_results and model_results['no_steps']:
            return display_overview(model_results['figures_list'])
        else:
            return display_step(model_results['figures_list'], current_step)
    elif bn_model_data is not None:
        return display_inference_window(bn_model_data)
    else:
        return html.Div('Ocurrió un error inesperado.')

def display_main_window():
    return html.Div([
        html.H3("1. Cargar Dataset", style={'textAlign': 'center'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Arrastra y suelta o ',
                html.A('selecciona un archivo CSV')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '0 auto'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload', style={'textAlign': 'center'}),
        html.Hr(),
        html.H3("2. Seleccionar Modelo", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
                {'label': 'TAN', 'value': 'TAN'},
                {'label': 'Markov Blanket selection by EDAs', 'value': 'EDAs'}
            ],
            placeholder='Selecciona un modelo',
            style={'width': '50%', 'margin': '0 auto'}
        ),
        html.Div(id='model-parameters'),
        html.Div([
            html.Button('Ejecutar Modelo', id='run-button', n_clicks=0)
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    ])

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
            return html.Div([
                html.H5(filename),
                html.P('Archivo cargado exitosamente.')
            ]), df.to_json(date_format='iso', orient='split')
        except Exception as e:
            return html.Div([
                'Hubo un error al procesar el archivo.'
            ]), None
    else:
        return '', None

@app.callback(
    Output('model-parameters', 'children'),
    Input('model-dropdown', 'value'),
    State('uploaded-data-store', 'data')
)
def update_parameters(model, data_json):
    if data_json is None:
        return html.Div('Por favor, carga un dataset primero.', style={'color': 'red', 'textAlign': 'center'})
    df = pd.read_json(data_json, orient='split')
    if model in ['Naive Bayes', 'TAN']:
        return html.Div([
            html.H3("Parámetros del Modelo", style={'textAlign': 'center'}),
            html.Div([
                html.Label('Iterations between steps:', style={'marginRight': '10px'}),
                dcc.Input(id='jump-steps', type='number', value=0, min=0, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Skip all steps:', style={'marginRight': '10px'}),
                dcc.Checklist(
                    id='no-steps',
                    options=[{'label': 'Sí', 'value': 'yes'}],
                    value=[]
                ),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Selection parameter:', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='selection-parameter',
                    options=[
                        {'label': 'Mutual Information', 'value': 'Mutual Information'},
                        {'label': 'Score', 'value': 'Score'}
                    ],
                    value='Mutual Information',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Class variable:', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='class-variable',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    placeholder='Selecciona la variable de clase',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center'}),
        ])
    elif model == 'EDAs':
        return html.Div([
            html.H3("Parámetros del Modelo EDAs", style={'textAlign': 'center'}),
            html.Div([
                html.Label('Number of generations:', style={'marginRight': '10px'}),
                dcc.Input(id='n-generations', type='number', value=1, min=1, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Number of individuals per generation:', style={'marginRight': '10px'}),
                dcc.Input(id='n-individuals', type='number', value=10, min=1, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Number of selected candidates per generation:', style={'marginRight': '10px'}),
                dcc.Input(id='n-candidates', type='number', value=5, min=1, step=1, style={'width': '60px'}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Class variable:', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='class-variable',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    placeholder='Selecciona la variable de clase',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label('Fitness metric:', style={'marginRight': '10px'}),
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


@app.callback(
    Output('model-results-store', 'data'),
    Output('current-step-store', 'data'),
    Output('current-model-store', 'data'),
    Output('edas-results-store', 'data'),
    Output('edas-best-results-store', 'data'),
    Output('main-content', 'children'),
    Input('run-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('jump-steps', 'value'),
    State('no-steps', 'value'),
    State('selection-parameter', 'value'),
    State('class-variable', 'value'),
    State('n-generations', 'value'),
    State('n-individuals', 'value'),
    State('n-candidates', 'value'),
    State('fitness-metric', 'value'),
    State('uploaded-data-store', 'data')
)
def run_model(n_clicks, model, jump_steps, no_steps, selection_parameter, class_variable,
              n_generations, n_individuals, n_candidates, fitness_metric, data_json):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    if data_json is None:
        return None, None, None, None, None, html.Div('Por favor, carga un dataset primero.', style={'color': 'red', 'textAlign': 'center'})
    df = pd.read_json(data_json, orient='split')
    if model in ['Naive Bayes', 'TAN']:
        if class_variable is None:
            return None, None, None, None, None, html.Div('Por favor, selecciona la variable de clase.', style={'color': 'red', 'textAlign': 'center'})
        # Ejecutar el modelo correspondiente
        if model == 'Naive Bayes':
            figures_list = NB_k_fold_with_steps(jump_steps, selection_parameter, df, class_variable)
        elif model == 'TAN':
            figures_list = NB_TAN_k_fold_with_steps(jump_steps, selection_parameter, df, class_variable)
        model_results_data = {
            'figures_list': figures_list,
            'no_steps': 'yes' in no_steps
        }
        current_step = 0
        return model_results_data, current_step, model, None, None, display_step(figures_list, current_step)
    elif model == 'EDAs':
        if class_variable is None:
            return None, None, None, None, None, html.Div('Por favor, selecciona la variable de clase.', style={'color': 'red', 'textAlign': 'center'})
        # Ejecutar el algoritmo UMDA
        umda = UMDA(n_candidates, n_individuals, n_generations, df, class_variable, fitness_metric)
        best_results, generation_information = umda.execute_umda()
        best_res = max(best_results, key=attrgetter('fitness'))
        edas_results_data = {
            'umda': umda,
            'best_results': best_results,
            'generation_information': generation_information
        }
        edas_best_results = best_res
        return None, None, model, edas_results_data, edas_best_results, display_edas_results(best_res, umda)
    else:
        return None, None, None, None, None, ''


@app.callback(
    Output('current-step-store', 'data'),
    Output('main-content', 'children'),
    Input('prev-step-button', 'n_clicks'),
    Input('next-step-button', 'n_clicks'),
    State('current-step-store', 'data'),
    State('model-results-store', 'data'),
    prevent_initial_call=True
)
def navigate_steps(prev_clicks, next_clicks, current_step, model_results):
    ctx = callback_context
    if not ctx.triggered or model_results is None:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'prev-step-button' and current_step > 0:
        current_step -= 1
    elif button_id == 'next-step-button' and current_step < len(model_results['figures_list']) - 1:
        current_step += 1
    return current_step, display_step(model_results['figures_list'], current_step)


@app.callback(
    Output('overview-figure', 'children'),
    Output('overview-score', 'children'),
    Input('overview-dropdown', 'value'),
    State('model-results-store', 'data')
)
def update_overview_figure(selected_index, model_results):
    if model_results is None:
        raise dash.exceptions.PreventUpdate
    figure = model_results['figures_list'][selected_index][0]
    score = cross_val_to_number(model_results['figures_list'][selected_index][1])
    img = fig_to_base64_image(figure)
    return html.Div([
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
    ]), f'Score: {score}'
    

@app.callback(
    Output('bn-model-store', 'data'),
    Output('main-content', 'children'),
    Input('choose-model-button', 'n_clicks'),
    Input('choose-model-button-overview', 'n_clicks'),
    Input('choose-model-button-edas', 'n_clicks'),
    State('current-step-store', 'data'),
    State('model-results-store', 'data'),
    State('overview-dropdown', 'value'),
    State('edas-best-results-store', 'data'),
    prevent_initial_call=True
)
def choose_model_for_inference(n_clicks_step, n_clicks_overview, n_clicks_edas, current_step, model_results, overview_index, edas_best_results):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'choose-model-button':
        bn = model_results['figures_list'][current_step][2]
    elif button_id == 'choose-model-button-overview':
        bn = model_results['figures_list'][overview_index][2]
    elif button_id == 'choose-model-button-edas':
        bn = edas_best_results.bn
    else:
        raise dash.exceptions.PreventUpdate
    # Convertir el modelo BN a una representación serializable si es necesario
    bn_data = bn  # Suponiendo que 'bn' es serializable
    return bn_data, display_inference_window(bn_data)

def display_inference_window(bn_model_data):
    bn = bn_model_data  # Recuperar el modelo BN
    variables = bn.names()
    evidence_selection = []
    for var in variables:
        var_labels = bn.variable(var).labels()
        evidence_selection.append(
            html.Div([
                html.Label(f'{var}:'),
                dcc.Dropdown(
                    id={'type': 'evidence-dropdown', 'index': var},
                    options=[{'label': label, 'value': label} for label in [''] + var_labels],
                    value='',
                    clearable=True,
                    style={'width': '200px'}
                )
            ], style={'marginBottom': '10px', 'display': 'inline-block', 'marginRight': '20px'})
        )
    return html.Div([
        html.H3('Inferencia', style={'textAlign': 'center'}),
        html.Div(evidence_selection, style={'columnCount': 2}),
        html.Button('Calcular Inferencia', id='calculate-inference-button', n_clicks=0, style={'display': 'block', 'margin': '10px auto'}),
        html.Div(id='inference-results')
    ])
    
@app.callback(
    Output('inference-results', 'children'),
    Input('calculate-inference-button', 'n_clicks'),
    State({'type': 'evidence-dropdown', 'index': ALL}, 'value'),
    State({'type': 'evidence-dropdown', 'index': ALL}, 'id'),
    State('bn-model-store', 'data'),
    prevent_initial_call=True
)
def perform_inference(n_clicks, evidence_values, evidence_ids, bn_model_data):
    if bn_model_data is None:
        raise dash.exceptions.PreventUpdate
    # Construir el diccionario de evidencias
    evidence = {}
    for value, id_dict in zip(evidence_values, evidence_ids):
        if value != '':
            var = id_dict['index']
            evidence[var] = value
    bn = bn_model_data  # Recuperar el modelo BN
    # Convertir evidence a la estructura esperada
    tuple_list = [(var, evidence.get(var, '')) for var in bn.names()]
    # Realizar la inferencia
    figure = get_inference_graph(bn, tuple_list)
    img = fig_to_base64_image(figure)
    return html.Div([
        html.H4('Resultados de la Inferencia', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
    ])
    
    
@app.callback(
    Output('main-content', 'children'),
    Input('show-generations-button', 'n_clicks'),
    State('edas-results-store', 'data'),
    prevent_initial_call=True
)
def show_generations(n_clicks, edas_results):
    if n_clicks == 0 or edas_results is None:
        raise dash.exceptions.PreventUpdate
    return display_edas_generations(edas_results, 0)


def display_edas_generations(edas_results, step_index):
    umda = edas_results['umda']
    best_results = edas_results['best_results']
    generation_information = edas_results['generation_information']
    total_steps = len(generation_information)
    figure = umda.from_chain_to_graph(best_results[step_index].chain)
    img = fig_to_base64_image(figure)
    content = [
        html.H3(f'Generación {step_index + 1} de {total_steps}', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), style={'display': 'block', 'margin': '0 auto'}),
    ]
    if step_index > 0:
        # Mostrar diferencias con la generación anterior
        diff_figure = umda.graph_between_chains(best_results[step_index-1].chain, best_results[step_index].chain)
        diff_img = fig_to_base64_image(diff_figure)
        content.append(html.H4('Diferencias con la generación anterior', style={'textAlign': 'center'}))
        content.append(html.Img(src='data:image/png;base64,{}'.format(diff_img), style={'display': 'block', 'margin': '0 auto'}))
    content.append(html.Div([
        html.Button('Anterior', id='prev-generation-button', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Siguiente', id='next-generation-button', n_clicks=0),
    ], style={'textAlign': 'center', 'marginTop': '10px'}))
    return html.Div(content)

@app.callback(
    Output('main-content', 'children'),
    Input('prev-generation-button', 'n_clicks'),
    Input('next-generation-button', 'n_clicks'),
    State('edas-results-store', 'data'),
    State('main-content', 'children'),
    prevent_initial_call=True
)
def navigate_generations(prev_clicks, next_clicks, edas_results, current_content):
    ctx = callback_context
    if not ctx.triggered or edas_results is None:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Encontrar el índice actual de la generación a partir del contenido actual
    for child in current_content:
        if isinstance(child, dict) and 'props' in child and 'children' in child['props']:
            if 'Generación' in child['props']['children']:
                current_step_text = child['props']['children']
                break
    else:
        current_step_text = 'Generación 1 de 1'
    current_step = int(current_step_text.split(' ')[1]) - 1
    total_steps = len(edas_results['generation_information'])
    if button_id == 'prev-generation-button' and current_step > 0:
        current_step -= 1
    elif button_id == 'next-generation-button' and current_step < total_steps - 1:
        current_step += 1
    return display_edas_generations(edas_results, current_step)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8053)
