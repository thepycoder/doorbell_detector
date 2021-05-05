import os
import shutil
import socket
import logging
import requests

import dash
import rq_dashboard
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ALL

# Create the dash app object
APP = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
SERVER = APP.server

CLIP_FOLDER = '/app/assets/unlabeled_data'
BELL_FOLDER = '/app/labeled_data/bell'
AMBIENT_FOLDER = '/app/labeled_data/ambient'


def generate_clips():
    clips = []
    for i, clip in enumerate(sorted(os.listdir(CLIP_FOLDER))):
        clips.append(
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col(html.Audio(src=f'/assets/unlabeled_data/{clip}', controls=True))
                    ]),
                    dbc.Row([
                        dbc.Col(html.P(clip)),
                        dbc.Col(dbc.RadioItems(
                            id={
                                'type': 'filter-dropdown',
                                'index': i
                            },
                            options=[
                                {"label": "Correct", "value": True},
                                {"label": "Wrong", "value": False},
                            ],
                            labelClassName="date-group-labels",
                            labelCheckedClassName="date-group-labels-checked",
                            className="date-group-items",
                            inline=True,
                        ), align='center')
                    ])
                ], color='light')
            ], width=12)
        )
    if not clips:
        clips = [dbc.Row(dbc.Col(html.P('Nothing to review!')))]
    return clips



# Create the HTML layout
# Use html for 1:1 html tags
# Use dcc for more interactive stuff like button and sliders
APP.layout = dbc.Container([
    dcc.Location(id='url'),
    dcc.Interval(id='poll-timer', interval=5000),
    dcc.Store(id='job_id'),
    dbc.Row([
        dbc.Col(html.H1('Review doorbells'), width={"size": 4, "offset": 4})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row(generate_clips(), id='clips'),
            dbc.Row([
                dbc.Col(dbc.Button('Trigger Retraining', id='training_btn', color="primary", className="mr-1")),
                dbc.Col(dbc.Button('Save labels', id='save_btn', color="primary", className="mr-1")),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P(id='test')
                ])
            ]),
            dbc.Row([
                dbc.Alert(
                    "Training Successfully started!",
                    id="alert-good",
                    is_open=False,
                    duration=4000,
                ),
                dbc.Alert(
                    "Something went wrong in retraining!",
                    id="alert-bad",
                    is_open=False,
                    duration=4000,
                ),
            ])
        ], width={"size": 6, "offset": 3}, sm={"size": 12, "offset": 0}),
    ])
], style={'text-align': 'center'})

# Callbacks!
# These are the real magic of plotly dash
# Use any attribute within the layout as input
# and ouput to any other attribute based on the
# return value of a function
@APP.callback(
    Output('clips', 'children'),
    [Input('save_btn', 'n_clicks')],
    [State({'type': 'filter-dropdown', 'index': ALL}, 'value')]
)
def update_output_div(_, values):
    files = sorted(os.listdir(CLIP_FOLDER))
    if files:
        for i, value in enumerate(values):
            filename = files[i]
            if value is None:
                continue
            elif value:
                shutil.move(os.path.join(CLIP_FOLDER, filename),
                            os.path.join(BELL_FOLDER, filename))
            else:
                shutil.move(os.path.join(CLIP_FOLDER, filename),
                            os.path.join(AMBIENT_FOLDER, filename))
            print(f'File {files[i]} is {value}')
    return generate_clips()


@APP.callback(
    [Output('alert-good', 'is_open'),
     Output('alert-bad', 'is_open'),
     Output('job_id', 'data')],
    [Input('training_btn', 'n_clicks')]
)
def trigger_training(n_clicks):
    if n_clicks:
        response = requests.post('http://retraining_api:8080/start_retraining')
        if response.status_code == 201:
            return True, False, response.json()['job_id']
        else:
            return False, True, None
    else:
        raise PreventUpdate


@APP.callback(
    [Output('training_btn', 'children'),
     Output('training_btn', 'disabled')],
    [Input('poll-timer', 'n_intervals')],
    State('job_id', 'data')
)
def status_update(n_clicks, job_id):
    if n_clicks and job_id:
        response = requests.get(f'http://retraining_api:8080/status_retraining/{job_id}')
        if response.status_code == 200:
            meta = response.json()
            if meta['progress'] != 'Done!':
                return [dbc.Spinner(size="sm"), ' ' + meta['progress']], True
            else:
                return 'Trigger Retraining', False
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

if __name__ == '__main__':
    APP.run_server(host='0.0.0.0', debug=False)
