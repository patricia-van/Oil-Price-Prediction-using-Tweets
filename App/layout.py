from dash import dcc, html

layout = html.Div([
    html.H1("Oil Price Prediction and Trading Strategy Dashboard"),
    
    dcc.Tabs([
        dcc.Tab(label='Sentiment Analysis', children=[
            dcc.Graph(id='sentiment-graph'),
            dcc.Dropdown(
                id='topic-dropdown',
                options=[
                    {'label': 'All Topics', 'value': 'all'},
                    {'label': 'OPEC Decisions', 'value': 'opec'},
                    {'label': 'Geopolitical Events', 'value': 'geo'}
                ],
                value='all',
                placeholder="Select topic"
            ),
        ]),
        dcc.Tab(label='Trading Strategy Backtest & Simulation', children=[
            # Dropdown to select mode
            html.Label("Select Model:"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[
                    {'label': 'Random_Forest', 'value': 'Random_Forest'},
                    {'label': 'Random_Forest_Tuned', 'value': 'Random_Forest_Tuned'},
                    {'label': 'ARIMA', 'value': 'ARIMA'},
                    {'label': 'ARIMAX', 'value': 'ARIMAX'},
                    {'label': 'ARIMAX_Tuned', 'value': 'ARIMAX_Tuned'},
                    {'label': 'LSTM_MSE', 'value': 'LSTM_MSE'}
                ],
                value='LSTM_MSE'
            ),

            # Slider for threshold
            html.Label("Select Threshold:"),
            dcc.Slider(id='threshold-slider', min=0, max=1, step=0.01, value=0.5, marks={0: '0', 1: '1'}),

            # Button to run backtest
            html.Button("Run Backtest", id="run-backtest", n_clicks=0),

            # Results
            dcc.Graph(id='cumulative-return-chart'),
            dcc.Graph(id='brent-price-chart'),
            # dcc.Graph(id='buy-sell-signals'),
            # dcc.Graph(id='performance-metrics')
        ]),
        dcc.Tab(label='Live', children=[
            dcc.Graph(id='live-brent-price-chart'),
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ]),


        dcc.Tab(label='Real-time Monitoring', children=[
            html.P("Placeholder for Prometheus/Grafana Integration")
        ])
    ])
])
