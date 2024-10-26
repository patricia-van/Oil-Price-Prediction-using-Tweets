from dash import dcc, html
from dash.dependencies import Input, Output
import dash
from layout import layout
from callbacks import register_callbacks

app = dash.Dash(__name__)

# Set layout
app.layout = layout

# Register callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
