import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
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
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',  # FontAwesome
        'https://bayes-interpret.com/Model/LearningFromData/ISLBNDash/assets/liquid-glass.css'  # Apple Liquid Glass CSS
    ],
    requests_pathname_prefix='/Model/LearningFromData/ISLBNDash/',
    suppress_callback_exceptions=True
)
server = app.server

# Safari Compatibility CSS Fix for Liquid Glass Effects
SAFARI_FIX_CSS = """
<style>
/* === SAFARI LIQUID GLASS COMPATIBILITY FIXES === */
/* Fixes for Safari 18 backdrop-filter + background-color bug */

/* Safari detection using CSS only */
@media not all and (min-resolution:.001dpcm) {
    @supports (-webkit-appearance:none) {
        
        /* Fix for main cards - separate background and blur */
        .card {
            background: transparent !important;
        }
        
        .card::before {
            background: rgba(255, 255, 255, 0.12) !important;
            -webkit-backdrop-filter: blur(15px) saturate(180%) !important;
            backdrop-filter: blur(15px) saturate(180%) !important;
        }
        
        /* Fix for buttons - use webkit prefix and avoid background conflicts */
        .btn {
            background: transparent !important;
            -webkit-backdrop-filter: blur(15px) !important;
            backdrop-filter: blur(15px) !important;
        }
        
        .btn::before {
            background: rgba(255, 255, 255, 0.12) !important;
        }
        
        /* Fix for form controls */
        .form-control {
            background: rgba(255, 255, 255, 0.15) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Fix for upload card */
        .upload-card {
            background: rgba(255, 255, 255, 0.05) !important;
        }
    }
}

/* Fallback for very old Safari versions */
@supports not (backdrop-filter: blur(1px)) {
    .card {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    .btn {
        background: rgba(255, 255, 255, 0.2) !important;
    }
}

/* === ENHANCED BUTTON STYLES === */
/* Hover effects for all styled buttons */
.btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
}

/* Specific hover effects by color */
.btn[style*="background-color: rgb(0, 123, 255)"]:hover,
.btn[style*="backgroundColor: #007bff"]:hover {
    background-color: #0056b3 !important;
    border-color: #0056b3 !important;
    box-shadow: 0 4px 12px rgba(0,123,255,0.5) !important;
}

.btn[style*="background-color: rgb(40, 167, 69)"]:hover,
.btn[style*="backgroundColor: #28a745"]:hover {
    background-color: #1e7e34 !important;
    border-color: #1e7e34 !important;
    box-shadow: 0 4px 12px rgba(40,167,69,0.4) !important;
}

.btn[style*="background-color: rgb(108, 117, 125)"]:hover,
.btn[style*="backgroundColor: #6c757d"]:hover {
    background-color: #545b62 !important;
    border-color: #545b62 !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
}

.btn[style*="background-color: rgb(23, 162, 184)"]:hover,
.btn[style*="backgroundColor: #17a2b8"]:hover {
    background-color: #117a8b !important;
    border-color: #117a8b !important;
    box-shadow: 0 4px 12px rgba(23,162,184,0.4) !important;
}

.btn[style*="background-color: rgb(0, 162, 225)"]:hover,
.btn[style*="backgroundColor: #00A2E1"]:hover {
    background-color: #0085b8 !important;
    border-color: #0085b8 !important;
    box-shadow: 0 6px 16px rgba(0,162,225,0.6) !important;
    transform: translateY(-3px) !important;
}

/* Active/pressed state */
.btn:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
}

/* Focus state for accessibility */
.btn:focus {
    outline: 3px solid rgba(0,123,255,0.3) !important;
    outline-offset: 2px !important;
}

/* === DROPDOWN Z-INDEX FIXES === */
/* Fix for dropdown menus appearing behind other elements */
.Select-menu-outer, .Select-menu {
    z-index: 9999 !important;
}

/* Dash dropdown menu container */
.dash-dropdown .Select-menu-outer {
    z-index: 9999 !important;
}

/* General dropdown fixes */
.dropdown-menu {
    z-index: 9999 !important;
}

/* Dash Core Components dropdown */
.dash-table-container .dash-dropdown .Select-menu-outer,
.dash-dropdown .Select-menu-outer {
    z-index: 9999 !important;
}

/* Additional dropdown container fixes */
.VirtualizedSelectOption, .VirtualizedSelectFocusedOption {
    z-index: 9999 !important;
}

/* Ensure cards don't interfere with dropdowns */
.card {
    position: relative;
    z-index: 1;
}

/* Ensure buttons don't interfere with dropdowns */
.btn {
    position: relative;
    z-index: 10;
}

/* But dropdown menus should always be on top */
div[data-dash-is-loading="true"] {
    z-index: 1 !important;
}
</style>
"""

# The main layout that includes everything
app.layout = html.Div([
    # Safari Compatibility Fix - inject CSS
    html.Div([
        dcc.Markdown(SAFARI_FIX_CSS, dangerously_allow_html=True)
    ], style={'display': 'none'}),
    
    dcc.Loading(
        id="global-spinner",
        overlay_style={"visibility": "visible", "filter": "blur(1px)"},
        type="circle",
        fullscreen=False,
        children=html.Div([
            html.H1("Interactive Structural Learning for Discrete BN", style={'textAlign': 'center'}),
            html.Div(
                className="link-bar",
                style={"textAlign": "center", "marginBottom": "20px"},
                children=[
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Original GitHub"
                        ],
                        href="https://github.com/IvanTelloLopez/ISL_BN",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Paper PDF"
                        ],
                        href="https://cig.fi.upm.es/wp-content/uploads/2024/01/Tello-I.-Interactive-Structure-Learning-for-Discrete-Bayesian-Network-Classifiers.pdf",
                        target="_blank",
                        className="btn btn-outline-primary me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Dash Adapted GitHub"
                        ],
                        href="https://github.com/KeevinPR/ISL_BN",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                ]
            ),
            ########################################################
            # Short explanatory text 
            ########################################################
            html.Div(
                [
                    html.P(
                        "Welcome to the Interactive Structural Learning tool. "
                        "Below, you can upload your dataset, select a model, "
                        "and explore the steps/generations of the Bayesian Network structure learning process.",
                        style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto"}
                    )
                ],
                style={"marginBottom": "20px"}
            ),
            

            
            ########################################################
            # (A) Data upload
            ########################################################
            html.Div(className="card", children=[
                    # Title or subtitle for this section
                    html.Div([
                        html.H3("1. Upload Dataset", style={"display": "inline-block", "marginRight": "10px", "textAlign": "center"}),
                        # Add help button right after the title
                        dbc.Button(
                            html.I(className="fa fa-question-circle"),
                            id="help-button-dataset-requirements",
                            color="link",
                            style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                        ),
                    ], style={"textAlign": "center", "position": "relative"}),
                    html.Div([
                        html.Div([
                            html.Img(
                                src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",
                                className="upload-icon"
                            ),
                            html.Div("Drag and drop or select a CSV file", className="upload-text")
                        ]),
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
                            style={'display': 'inline-block','textAlign': 'center', 'marginTop': '10px'}
                        ),
                        dbc.Button(
                            html.I(className="fa fa-question-circle"),
                            id="help-button-default-dataset",
                            color="link",
                            style={"display": "inline-block", "marginLeft": "8px"}
                        ),
                    ], style={'display': 'inline-block','textAlign': 'center'}),
                    # Feedback message (uploaded file name or error)
                    html.Div(id='output-data-upload'),
            ]),

            ########################################################
            # (B) Model selection
            ########################################################
            html.Div(className="card", children=[
                html.H3("2. Select Model", style={'textAlign': 'center'}),
            
                dbc.Select(
                    id='model-dropdown',
                    options=[
                        {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
                        {'label': 'TAN', 'value': 'TAN'},
                        {'label': 'Markov Blanket selection by EDAs', 'value': 'EDAs'}
                    ],
                    placeholder='Select a model',
                    style={
                        'width': '70%', 
                        'margin': '0 auto',
                        'border': '1px solid #d0d7de',
                        'borderRadius': '6px',
                        'padding': '8px 12px',
                        'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                        'backdropFilter': 'blur(10px)',
                        'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                        'transition': 'all 0.2s ease',
                        'fontSize': '14px'
                    }
                ),
            ]),
            
            ########################################################
            # (C) Model parameters - ALL POSSIBLE PARAMETERS ALWAYS PRESENT
            ########################################################
            # NB/TAN Parameters
            html.Div(id='nb-tan-parameters', className="card", children=[
                html.H3("3. Model Parameters", style={'textAlign': 'center'}),
                html.Div([
                    html.Label('Iterations between steps:'),
                    dcc.Input(id='jump-steps', type='number', value=0, min=0, step=1, style={'width': '60px'}),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Skip all steps:'),
                    dcc.Checklist(
                        id='no-steps',
                        options=[{'label': 'Yes', 'value': 'yes'}],
                        value=[]
                    ),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Selection parameter:'),
                    dbc.Select(
                        id='selection-parameter',
                        options=[
                            {'label': 'Mutual Information', 'value': 'Mutual Information'},
                            {'label': 'Score', 'value': 'Score'}
                        ],
                        value='Mutual Information',
                        style={
                            'width': '200px', 
                            'display': 'inline-block',
                            'border': '1px solid #d0d7de',
                            'borderRadius': '6px',
                            'padding': '8px 12px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'backdropFilter': 'blur(10px)',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'all 0.2s ease',
                            'fontSize': '14px'
                        }
                    ),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Class variable:'),
                    dbc.Select(
                        id='class-variable-nb-tan',
                        options=[],  # Will be populated dynamically
                        placeholder='Select the class variable',
                        style={
                            'width': '200px', 
                            'display': 'inline-block',
                            'border': '1px solid #d0d7de',
                            'borderRadius': '6px',
                            'padding': '8px 12px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'backdropFilter': 'blur(10px)',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'all 0.2s ease',
                            'fontSize': '14px'
                        }
                    ),
                ], style={'textAlign': 'center'}),
            ], style={'display': 'none'}),
            
            # EDAs Parameters
            html.Div(id='edas-parameters', className="card", children=[
                html.H3("3. EDAs Model Parameters", style={'textAlign': 'center'}),
                html.Div([
                    html.Label('Number of generations:'),
                    dcc.Input(id='n-generations', type='number', value=1, min=1, step=1, 
                              style={'width': '60px'}),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Number of individuals per generation:'),
                    dcc.Input(id='n-individuals', type='number', value=10, min=1, step=1,
                              style={'width': '60px'}),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Number of selected candidates per generation:'),
                    dcc.Input(id='n-candidates', type='number', value=5, min=1, step=1,
                              style={'width': '60px'}),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Class variable:'),
                    dbc.Select(
                        id='class-variable-edas',
                        options=[],  # Will be populated dynamically
                        placeholder='Select the class variable',
                        style={
                            'width': '200px', 
                            'display': 'inline-block',
                            'border': '1px solid #d0d7de',
                            'borderRadius': '6px',
                            'padding': '8px 12px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'backdropFilter': 'blur(10px)',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'all 0.2s ease',
                            'fontSize': '14px'
                        }
                    ),
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Label('Fitness metric:'),
                    dbc.Select(
                        id='fitness-metric',
                        options=[
                            {'label': 'Accuracy', 'value': 'Accuracy'},
                            {'label': 'BIC', 'value': 'BIC'}
                        ],
                        value='Accuracy',
                        style={
                            'width': '200px', 
                            'display': 'inline-block',
                            'border': '1px solid #d0d7de',
                            'borderRadius': '6px',
                            'padding': '8px 12px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'backdropFilter': 'blur(10px)',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'all 0.2s ease',
                            'fontSize': '14px'
                        }
                    ),
                ], style={'textAlign': 'center'}),
            ], style={'display': 'none'}),
            
            # (C) "Run" button + progress messages
            html.Div([
                html.Div([
                    dbc.Button(
                        [
                            html.I(className="fas fa-play-circle me-2"),
                            "Run Model"
                        ],
                        id='run-button',
                        n_clicks=0,
                        className="btn-lg",
                        style={
                            'fontSize': '1.2rem',
                            'padding': '0.875rem 2.5rem',
                            'borderRadius': '10px',
                            'boxShadow': '0 4px 8px rgba(0,162,225,0.4)',
                            'transition': 'all 0.3s ease',
                            'backgroundColor': '#00A2E1',
                            'borderColor': '#00A2E1',
                            'border': '2px solid #00A2E1',
                            'margin': '1rem 0',
                            'color': 'white',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        }
                    )
                ], style={'textAlign': 'center'}),
            ], style={'textAlign': 'center'}),

            # Main output area
            html.Div(id='model-output'),
            
            ########################################################
            # ALL NAVIGATION BUTTONS - ALWAYS PRESENT, CONTROLLED BY VISIBILITY
            ########################################################
            # Step navigation for NB/TAN
            html.Div(id='step-navigation', className="card", children=[
                html.Div([
                    dbc.Button('Previous', id='prev-step-button', n_clicks=0, 
                              className="me-2",
                              style={
                                  'backgroundColor': '#6c757d',
                                  'borderColor': '#6c757d',
                                  'color': 'white',
                                  'fontWeight': '500',
                                  'border': '2px solid #6c757d',
                                  'borderRadius': '8px',
                                  'padding': '8px 16px',
                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                                  'transition': 'all 0.3s ease'
                              }),
                    dbc.Button('Next', id='next-step-button', n_clicks=0, 
                              className="me-2",
                              style={
                                  'backgroundColor': '#6c757d',
                                  'borderColor': '#6c757d',
                                  'color': 'white',
                                  'fontWeight': '500',
                                  'border': '2px solid #6c757d',
                                  'borderRadius': '8px',
                                  'padding': '8px 16px',
                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                                  'transition': 'all 0.3s ease'
                              }),
                    dbc.Button('Choose this model', id='choose-model-button', n_clicks=0,
                              style={
                                  'backgroundColor': '#28a745',
                                  'borderColor': '#28a745',
                                  'color': 'white',
                                  'fontWeight': '600',
                                  'border': '2px solid #28a745',
                                  'borderRadius': '8px',
                                  'padding': '8px 20px',
                                  'boxShadow': '0 3px 6px rgba(40,167,69,0.3)',
                                  'transition': 'all 0.3s ease'
                              }),
                ], style={'textAlign': 'center'}),
            ], style={'display': 'none', 'marginTop': '20px'}),
            
            # Generation navigation for EDAs
            html.Div(id='generation-navigation', className="card", children=[
                html.Div([
                    dbc.Button('Previous Generation', id='prev-generation-button', n_clicks=0, 
                              className="me-2",
                              style={
                                  'backgroundColor': '#6c757d',
                                  'borderColor': '#6c757d',
                                  'color': 'white',
                                  'fontWeight': '500',
                                  'border': '2px solid #6c757d',
                                  'borderRadius': '8px',
                                  'padding': '8px 16px',
                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                                  'transition': 'all 0.3s ease'
                              }),
                    dbc.Button('Next Generation', id='next-generation-button', n_clicks=0, 
                              className="me-2",
                              style={
                                  'backgroundColor': '#6c757d',
                                  'borderColor': '#6c757d',
                                  'color': 'white',
                                  'fontWeight': '500',
                                  'border': '2px solid #6c757d',
                                  'borderRadius': '8px',
                                  'padding': '8px 16px',
                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                                  'transition': 'all 0.3s ease'
                              }),
                    dbc.Button('Choose this model (EDAs)', id='choose-model-button-edas', n_clicks=0, 
                              className="me-2",
                              style={
                                  'backgroundColor': '#28a745',
                                  'borderColor': '#28a745',
                                  'color': 'white',
                                  'fontWeight': '600',
                                  'border': '2px solid #28a745',
                                  'borderRadius': '8px',
                                  'padding': '8px 20px',
                                  'boxShadow': '0 3px 6px rgba(40,167,69,0.3)',
                                  'transition': 'all 0.3s ease'
                              }),
                    dbc.Button('Show generations', id='show-generations-button-edas', n_clicks=0,
                              style={
                                  'backgroundColor': '#17a2b8',
                                  'borderColor': '#17a2b8',
                                  'color': 'white',
                                  'fontWeight': '500',
                                  'border': '2px solid #17a2b8',
                                  'borderRadius': '8px',
                                  'padding': '8px 16px',
                                  'boxShadow': '0 2px 4px rgba(23,162,184,0.3)',
                                  'transition': 'all 0.3s ease'
                              }),
                ], style={'textAlign': 'center'}),
            ], style={'display': 'none', 'marginTop': '20px'}),
            
            # Inference interface
            html.Div(id='inference-interface', className="card", children=[
                html.H3('Inference', style={'textAlign': 'center'}),
                html.Div(id='evidence-controls', 
                        style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
                html.Div([
                    dbc.Button('Calculate Inference', id='calculate-inference-button', n_clicks=0, 
                              className="me-2",
                              style={
                                  'backgroundColor': '#007bff',
                                  'borderColor': '#007bff',
                                  'color': 'white',
                                  'fontWeight': '600',
                                  'border': '2px solid #007bff',
                                  'borderRadius': '8px',
                                  'padding': '10px 24px',
                                  'fontSize': '16px',
                                  'boxShadow': '0 3px 6px rgba(0,123,255,0.4)',
                                  'transition': 'all 0.3s ease'
                              }),
                    dbc.Button('Back to Results', id='back-to-results-button', n_clicks=0,
                              style={
                                  'backgroundColor': '#6c757d',
                                  'borderColor': '#6c757d',
                                  'color': 'white',
                                  'fontWeight': '500',
                                  'border': '2px solid #6c757d',
                                  'borderRadius': '8px',
                                  'padding': '10px 20px',
                                  'fontSize': '16px',
                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                                  'transition': 'all 0.3s ease'
                              })
                ], style={'textAlign': 'center', 'marginTop': '20px'}),
                html.Div(id='inference-results-display')
            ], style={'display': 'none', 'marginTop': '20px'}),

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
            dcc.Store(id='app-state-store', data={'current_view': 'initial'}),  # Track app state
        ])
    ), #End of dcc.Loading
    #  popover **outside** the dcc.Loading
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Default Dataset",
                    html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                ],
                style={
                    "backgroundColor": "#f8f9fa",
                    "fontWeight": "bold"
                }
            ),
            dbc.PopoverBody(
                [
                    html.P(
                        [
                            "For details and content of the dataset, check out: ",
                            html.A(
                                "cars_example.data",
                                href="https://github.com/KeevinPR/ISL_BN/blob/main/cars_example.data",
                                target="_blank",
                                style={"textDecoration": "underline", "color": "#0d6efd"}
                            ),
                        ]
                    ),
                    html.Hr(),
                    html.P("Feel free to upload your own dataset at any time.")
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "0 0 0.25rem 0.25rem",
                    "maxWidth": "300px"
                }
            ),
        ],
        id="help-popover-default-dataset",
        target="help-button-default-dataset",
        placement="right",
        is_open=False,
        trigger="hover",
        style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
    ),
    # New Popover for Dataset Upload Requirements
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Dataset Requirements",
                    html.I(className="fa fa-check-circle ms-2", style={"color": "#198754"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.Ul([
                        html.Li(
                            children=[
                                html.Strong("Format: "),
                                "CSV, .data, .dat, or plain text with headers. Auto-detects delimiter."
                            ]
                        ),
                        html.Li(
                            children=[
                                html.Strong("Data: "),
                                "Discrete/categorical variables preferred. Numerical values will be converted."
                            ]
                        ),
                        html.Li(
                            children=[
                                html.Strong("Missing Values: "),
                                "Use '?' symbol. Columns with >30% missing data are removed."
                            ]
                        ),
                        html.Li(
                            children=[
                                html.Strong("Cleaning: "),
                                "Constant columns and perfectly correlated features are automatically removed."
                            ]
                        ),
                    ]),
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "300px"}
            ),
        ],
        id="help-popover-dataset-requirements",
        target="help-button-dataset-requirements",
        placement="right",
        is_open=False,
        trigger="hover",
        style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
    ),
    #future popovers here
])
# Critical: Also set the validation_layout to the exact same layout
# so Dash never complains about missing IDs.
app.validation_layout = app.layout


########################################################################
# 2) Callbacks
########################################################################
import re
import numpy as np

def linear_dependent_features(df, threshold=1.0):
    """
    Very simple example:
    Returns a list of columns that are 100% correlated with another one (Pearson=1 or -1).
    Adjust this function to your own definition of linear dependence.
    """
    to_remove = set()
    corr_matrix = df.corr(numeric_only=True).abs()  # correlations in absolute value

    # To avoid redundant checks, iterate only the upper triangle
    # and check if correlation >= threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                colname_j = corr_matrix.columns[j]
                # Mark the j-th column as redundant (or i, depending on your preference)
                to_remove.add(colname_j)
    
    return list(to_remove)

def parse_content(content, filename, missing_threshold=0.3, sample_size=None):
    """
    Reads base64 content, applies multiple transformations:
      - Removes columns with > (missing_threshold * 100)% of NaNs
      - Removes rows with NaNs in the remaining columns
      - Drops constant columns
      - Renames columns to remove problematic characters (e.g., hyphens -> '_')
      - Converts object -> category, numbers -> float64
      - Resets the index
      - Removes linearly dependent features
      - (Optional) Samples the dataset if it's too large
    Returns a cleaned DataFrame or None if something fails.
    """
    if not content:
        return None

    try:
        # Separate the base64 content string and decode
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        # Read as CSV/text file
        filename_lower = filename.lower()
        # Extract the actual filename from a potential path
        base_filename = filename_lower.split('/')[-1]

        is_csv = base_filename.endswith('.csv')
        is_data = base_filename.endswith('.data')
        is_dat = base_filename.endswith('.dat')
        has_no_extension = '.' not in base_filename

        if is_csv or is_data or is_dat or has_no_extension:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),
                sep=None, # Auto-detect separator
                engine='python',
                na_values='?' # Common missing value indicator
            )
        else:
            # If the file has an extension and it's not one of the recognized ones
            print(f"Unsupported file type: {filename}. Attempting to parse will likely fail or lead to an error message.")
            # The function will return None, and the calling callback will display a generic error.
            # For a more specific message here, this function would need to return an error string/object.
            return None

        # 1) Remove columns that have more than X% of NaNs
        #    threshold=int(0.3*len(df)) => 30%: if a column has <70% non-null, drop it.
        df = df.dropna(axis=1, thresh=int((1 - missing_threshold) * len(df)))

        # 2) Remove rows with any NaNs in the remaining columns
        df = df.dropna(axis=0)

        # 3) Remove constant columns
        index_constant = np.where(df.nunique() == 1)[0]
        constant_columns = [df.columns[i] for i in index_constant]
        df.drop(columns=constant_columns, inplace=True)

        # 4) (Optional) Sampling if the dataset is too large
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        # 5) Rename columns with problematic characters (e.g., hyphens) => underscores
        #    You can adjust the regular expression to fit your needs:
        df.columns = [
            re.sub(r'[^a-zA-Z0-9]+', '_', str(col))  # anything non-alphanumeric => '_'
            for col in df.columns
        ]

        # 6) Convert types: object -> category, number -> float64
        cat_data = df.select_dtypes(include=['object']).astype('category')
        for c in cat_data.columns:
            df[c] = cat_data[c]

        float_data = df.select_dtypes(include=['int', 'float', 'number']).astype('float64')
        for c in float_data.columns:
            df[c] = float_data[c]

        # 7) Reset the index
        df.reset_index(drop=True, inplace=True)

        # 8) Remove linearly dependent features
        #    (Adjust 'threshold' in linear_dependent_features according to your use case)
        to_remove_features = linear_dependent_features(df, threshold=1.0)
        if to_remove_features:
            df.drop(columns=to_remove_features, inplace=True, errors='ignore')

        return df

    except Exception as e:
        print(f"Error parsing file: {e}")
        return None
# ----------------------------------------------------------------------------
# Callback to toggle the popover for the upload help button
# ----------------------------------------------------------------------------
@app.callback(
    Output("help-popover-upload", "is_open"),
    Input("help-button-upload", "n_clicks"),
    State("help-popover-upload", "is_open")
)
def toggle_popover_upload(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback for dataset requirements popover
@app.callback(
    Output("help-popover-dataset-requirements", "is_open"),
    Input("help-button-dataset-requirements", "n_clicks"),
    State("help-popover-dataset-requirements", "is_open")
)
def toggle_dataset_requirements_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# ----------------------------------------------------------------------------
# (A) Callback for uploading data
# ----------------------------------------------------------------------------
@app.callback(
    Output('output-data-upload', 'children'),
    Output('uploaded-data-store', 'data'),
    Output('use-default-dataset', 'value', allow_duplicate=True),
    Input('upload-data', 'contents'),
    Input('use-default-dataset', 'value'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_output(contents, use_default_value, filename):
    default_path = '/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/ISLBN/cars_example.data'
    
    # ---------------
    # A) If user has uploaded a file manually (PRIORITY OVER DEFAULT)
    # ---------------
    if contents is not None and filename is not None:
        df = parse_content(
            contents,
            filename,
            missing_threshold=0.3,
            sample_size=None  # or pick some limit if you want sampling
        )
        
        if df is not None:
            # Return success message + the cleaned df + CLEAR default checkbox
            return (
                html.Div([
                    html.H5(filename),
                    html.P('File uploaded and processed successfully.')
                ]),
                df.to_json(date_format='iso', orient='split'),
                []  # Clear the default dataset checkbox
            )
        else:
            return (
                html.Div(['There was an error processing the file with parse_content.']),
                None,
                []  # Clear the default dataset checkbox even on error
            )
    
    # ---------------
    # B) If user checks "Use default dataset" (and no file uploaded)
    # ---------------
    elif 'default' in use_default_value:
        try:
            # 1) Read the default file contents
            with open(default_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # 2) Convert it to base64 so we can use parse_content the same way
            content_b64 = "data:text/plain;base64," + base64.b64encode(raw_text.encode()).decode()
            
            # 3) Call parse_content. Let's pass a missing_threshold=0.3, sample_size=None, but you can tweak
            df = parse_content(content_b64, "cars_example.data", missing_threshold=0.3, sample_size=None)
            
            if df is None:
                return (
                    html.Div(["Error processing the default dataset with parse_content."]), 
                    None,
                    use_default_value  # Keep current checkbox state
                )
            
            # If everything is good, we notify the user + store the cleaned df
            return (
                html.Div([
                    html.P('Using default dataset: cars_example.data',
                           style={'color': 'green', 'fontWeight': 'bold', 'margin': '10px 0'})
                ]),
                df.to_json(date_format='iso', orient='split'),
                use_default_value  # Keep default checkbox checked
            )
        except Exception as e:
            return (
                html.Div([f'Error reading default dataset: {e}']), 
                None,
                use_default_value  # Keep current checkbox state
            )

    # ---------------
    # C) Otherwise (no upload, no default)
    # ---------------
    return '', None, use_default_value  # Keep current checkbox state

# ----------------------------------------------------------------------------
# (B) Update model parameters depending on user choice
# ----------------------------------------------------------------------------
@app.callback(
    Output('nb-tan-parameters', 'style'),
    Output('edas-parameters', 'style'),
    Output('class-variable-nb-tan', 'options'),
    Output('class-variable-edas', 'options'),
    Input('model-dropdown', 'value'),
    State('uploaded-data-store', 'data')
)
def update_parameters(model, data_json):
    if data_json is None:
        return {'display': 'none'}, {'display': 'none'}, [], []
    
    df = pd.read_json(io.StringIO(data_json), orient='split')
    class_options = [{'label': col, 'value': col} for col in df.columns]
    
    if model in ['Naive Bayes', 'TAN']:
        return {'display': 'block'}, {'display': 'none'}, class_options, []
    elif model == 'EDAs':
        return {'display': 'none'}, {'display': 'block'}, [], class_options
    else:
        return {'display': 'none'}, {'display': 'none'}, [], []




# ----------------------------------------------------------------------------
# (C) Main callback to handle "Run Model" and navigation
# ----------------------------------------------------------------------------
@app.callback(
    Output('model-results-store', 'data'),
    Output('current-step-store', 'data'),
    Output('edas-results-store', 'data'),
    Output('current-generation-store', 'data'),
    Output('bn-model-store', 'data'),
    Output('inference-results-display', 'children'),

    Input('run-button', 'n_clicks'),
    Input('prev-step-button', 'n_clicks'),
    Input('next-step-button', 'n_clicks'),
    Input('choose-model-button', 'n_clicks'),
    Input('prev-generation-button', 'n_clicks'),
    Input('next-generation-button', 'n_clicks'),
    Input('choose-model-button-edas', 'n_clicks'),
    Input('show-generations-button-edas', 'n_clicks'),
    Input('back-to-results-button', 'n_clicks'),

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
    State('class-variable-nb-tan', 'value'),
    State('class-variable-edas', 'value'),
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
    back_to_results_clicks,
    model_results_data, current_step,
    edas_results_data, current_generation,
    bn_model_data,
    model, data_json,
    jump_steps, no_steps, selection_parameter, class_variable_nb_tan, class_variable_edas,
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
    inference_results_out = dash.no_update  # Don't change inference results by default

    # RUN MODEL
    if button_id == 'run-button':
        if data_json is None:
            print("No data uploaded.")
            return (dash.no_update,)*5

        df = pd.read_json(io.StringIO(data_json), orient='split')
        
        # CLEAR PREVIOUS INFERENCE STATE when running new model
        bn_model_data_out = None
        
        # NB or TAN
        if model in ['Naive Bayes', 'TAN']:
            class_variable = class_variable_nb_tan
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
            
            # Clear EDAs results when running NB/TAN
            edas_results_data_out = None
            current_generation_out = None

        # EDAs
        elif model == 'EDAs':
            class_variable = class_variable_edas
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
            
            # Clear NB/TAN results when running EDAs
            model_results_data_out = None
            current_step_out = None

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
        # Clear model results to show inference interface
        model_results_data_out = None
        current_step_out = None
        # Clear previous inference results
        inference_results_out = html.Div()

    # EDAS GENERATION NAV
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

    # BACK TO RESULTS (clear inference state)
    elif button_id == 'back-to-results-button':
        bn_model_data_out = None

    # EDAS CHOOSE MODEL
    elif button_id == 'choose-model-button-edas':
        if not edas_results_data:
            raise dash.exceptions.PreventUpdate
        best_results_data = edas_results_data['best_results']
        best_results = [deserialize_solution(sol_data) for sol_data in best_results_data]
        best_res = max(best_results, key=attrgetter('fitness'))
        bn = best_res.bn
        bn_model_data_out = serialize_bayesnet(bn)
        # Clear EDAs results to show inference interface
        edas_results_data_out = None
        current_generation_out = None
        # Clear previous inference results
        inference_results_out = html.Div()

    return (
        model_results_data_out,
        current_step_out,
        edas_results_data_out,
        current_generation_out,
        bn_model_data_out,
        inference_results_out
    )


# ----------------------------------------------------------------------------
# (D) Inference callback
# ----------------------------------------------------------------------------
@app.callback(
    Output('inference-results-display', 'children', allow_duplicate=True),
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
        html.H3('Inference Results', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img), 
                 className="zoomable", 
                 style={'display': 'block', 'margin': '0 auto'}),
    ])


# ----------------------------------------------------------------------------
# (E) Main output display logic - controls both output and navigation visibility
# ----------------------------------------------------------------------------
@app.callback(
    Output('model-output', 'children'),
    Output('step-navigation', 'style'),
    Output('generation-navigation', 'style'),
    Output('inference-interface', 'style'),
    Output('evidence-controls', 'children'),
    Input('model-results-store', 'data'),
    Input('current-step-store', 'data'),
    Input('edas-results-store', 'data'),
    Input('current-generation-store', 'data'),
    Input('bn-model-store', 'data'),
    prevent_initial_call=True
)
def update_model_output_and_controls(model_results_data, current_step,
                                    edas_results_data, current_generation,
                                    bn_model_data):

    # Default styles (hidden)
    step_nav_style = {'display': 'none', 'marginTop': '20px'}
    gen_nav_style = {'display': 'none', 'marginTop': '20px'}
    inference_style = {'display': 'none', 'marginTop': '20px'}
    evidence_controls = []

    # PRIORITY LOGIC: Show most recent activity first
    
    # 1) If NB/TAN results are active (most recent activity) => show steps
    if model_results_data is not None and current_step is not None:
        figures_list = model_results_data['figures_list']
        output_content = display_step_content(figures_list, current_step)
        step_nav_style = {'display': 'block', 'marginTop': '20px'}

    # 2) If EDAs results are active (most recent activity) => show best or generations
    elif edas_results_data is not None:
        if current_generation is None:
            output_content = display_edas_best_solution_content(edas_results_data)
            gen_nav_style = {'display': 'block', 'marginTop': '20px'}
        else:
            output_content = display_edas_generations_content(edas_results_data, current_generation)
            gen_nav_style = {'display': 'block', 'marginTop': '20px'}

    # 3) If BN chosen (user explicitly chose a model for inference) => show inference
    elif bn_model_data is not None:
        bn = deserialize_bayesnet(bn_model_data)
        variables = bn.names()
        evidence_controls = []

        for var in variables:
            labels = bn.variable(var).labels()
            evidence_controls.append(
                html.Div(children=[
                    html.Label(f'{var}:'),
                    dbc.Select(
                        id={'type': 'evidence-dropdown', 'index': var},
                        options=[{'label': lb, 'value': lb} for lb in [''] + list(labels)],
                        value='',
                        style={
                            'width': '150px',
                            'border': '1px solid #d0d7de',
                            'borderRadius': '6px',
                            'padding': '8px 12px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'backdropFilter': 'blur(10px)',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'all 0.2s ease',
                            'fontSize': '14px'
                        }
                    )
                ],
                style={'display': 'inline-block', 'verticalAlign': 'top', 'margin': '5px 15px'})
            )

        inference_style = {'display': 'block', 'marginTop': '20px'}
        output_content = html.Div([
            html.H4('Model Selected for Inference', style={'textAlign': 'center', 'color': 'green'}),
            html.P('Select evidence values below and click "Calculate Inference" to see results.', 
                   style={'textAlign': 'center'})
        ])

    # 4) Otherwise
    else:
        output_content = html.Div('No model output to display.')

    return (output_content, step_nav_style, gen_nav_style, 
            inference_style, evidence_controls)


########################################################################
# 3) Helper functions exactly as in your code
########################################################################

def display_edas_best_solution_content(edas_results_data):
    umda_data = edas_results_data['umda']
    umda = deserialize_umda(umda_data)
    best_results_data = edas_results_data['best_results']
    best_results = [deserialize_solution(sol_data) for sol_data in best_results_data]
    best_res = max(best_results, key=attrgetter('fitness'))
    fig = umda.from_chain_to_graph(best_res.chain)
    img_base64 = fig_to_base64_image(fig)

    return html.Div(className="card", children=[
        html.H3('Best Markov Blanket structure obtained by the algorithm:',
                style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img_base64), 
                 className="zoomable", 
                 style={'display': 'block', 'margin': '0 auto'}),
        html.P(f"Score (fitness): {best_res.fitness:.4f}", 
               style={'textAlign': 'center', 'fontWeight': 'bold'}),
    ])

def display_edas_best_solution(edas_results_data):
    umda_data = edas_results_data['umda']
    umda = deserialize_umda(umda_data)
    best_results_data = edas_results_data['best_results']
    best_results = [deserialize_solution(sol_data) for sol_data in best_results_data]
    best_res = max(best_results, key=attrgetter('fitness'))
    fig = umda.from_chain_to_graph(best_res.chain)
    img_base64 = fig_to_base64_image(fig)

    return html.Div(className="card", children=[
        html.H3('Best Markov Blanket structure obtained by the algorithm:',
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

def display_edas_generations_content(edas_results_data, generation_index):
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

    return html.Div(content, className="card")

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

def display_step_content(figures_list, step_index):
    data = figures_list[step_index]
    img_data = data['fig']
    score = cross_val_to_number(data['scores'])
    total_steps = len(figures_list)

    return html.Div(className="card", children=[
        html.H3(f'Step {step_index + 1} of {total_steps}', style={'textAlign': 'center'}),
        html.Img(src='data:image/png;base64,{}'.format(img_data), 
                 className="zoomable", 
                 style={'display': 'block', 'margin': '0 auto'}),
        html.P(f'Score: {score}', style={'textAlign': 'center'}),
    ])

def display_step(figures_list, step_index):
    data = figures_list[step_index]
    img_data = data['fig']
    score = cross_val_to_number(data['scores'])
    total_steps = len(figures_list)

    return html.Div(className="card", children=[
        html.H3(f'Step {step_index + 1} of {total_steps}', style={'textAlign': 'center'}),
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



def fig_to_base64_image(fig):
    img_bytes = io.BytesIO()
    #fig.set_size_inches(10, 6)  # ancho x alto en pulgadas
    fig.savefig(img_bytes, dpi=500,format='png', bbox_inches='tight')
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
    app.run(debug=False, threaded=False, host='0.0.0.0', port=8053)