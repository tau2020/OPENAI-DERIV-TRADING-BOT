

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta

class Dashboard:
    def __init__(self, trading_strategy, host='0.0.0.0', port=8050):
        """
        Initializes the Dash dashboard.

        Args:
            trading_strategy (TradingStrategy): Instance of your trading strategy.
            host (str): Host address for the Dash app.
            port (int): Port number for the Dash app.
        """
        self.trading_strategy = trading_strategy
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.host = host
        self.port = port

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """
        Defines the layout of the Dash dashboard.
        """
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Real-Time Trading Dashboard"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.H5(f"Symbol: {self.trading_strategy.current_symbol}"), className="mb-4")
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='price-chart'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pl-chart'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='balance-info'), width=12)
            ]),
            # Interval component for live updates
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            )
        ], fluid=True)

    def setup_callbacks(self):
        """
        Sets up the callbacks for updating charts and balance information.
        """

        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('pl-chart', 'figure'),
             Output('balance-info', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            """
            Updates the price and P/L charts and balance information.

            Args:
                n (int): Number of intervals passed.

            Returns:
                tuple: Updated figures and balance information.
            """
            # Define the sliding window duration (e.g., last 60 minutes)
            window_duration = timedelta(minutes=60)
            current_time = datetime.now()

            # Acquire the lock to safely access shared data
            with self.trading_strategy.lock:
                # Filter Price Data within the sliding window
                filtered_tick_times = [t for t in self.trading_strategy.tick_times if current_time - t <= window_duration]
                filtered_tick_prices = self.trading_strategy.tick_prices[-len(filtered_tick_times):]

                # Filter P/L Data within the sliding window
                filtered_pl_times = [t for t in self.trading_strategy.pl_times if current_time - t <= window_duration]
                filtered_pl_values = self.trading_strategy.pl_values[-len(filtered_pl_times):]

                # Prepare Price Chart
                price_trace = go.Scatter(
                    x=filtered_tick_times,
                    y=filtered_tick_prices,
                    mode='lines',
                    name='Price',
                    line=dict(color='blue')
                )
                price_layout = go.Layout(
                    title=f"{self.trading_strategy.current_symbol} Price Chart",
                    xaxis=dict(
                        title='Time',
                        range=[
                            min(filtered_tick_times) if filtered_tick_times else current_time,
                            max(filtered_tick_times) if filtered_tick_times else current_time
                        ],
                        type='date'
                    ),
                    yaxis=dict(title='Price'),
                    showlegend=True
                )
                price_fig = go.Figure(data=[price_trace], layout=price_layout)

                # Prepare P/L Chart
                pl_trace = go.Scatter(
                    x=filtered_pl_times,
                    y=filtered_pl_values,
                    mode='lines',
                    name='Net Cumulative P/L',
                    line=dict(color='red')
                )
                pl_layout = go.Layout(
                    title="Net Cumulative Profit/Loss",
                    xaxis=dict(
                        title='Time',
                        range=[
                            min(filtered_pl_times) if filtered_pl_times else current_time,
                            max(filtered_pl_times) if filtered_pl_times else current_time
                        ],
                        type='date'
                    ),
                    yaxis=dict(title='Net P/L'),
                    showlegend=True
                )
                pl_fig = go.Figure(data=[pl_trace], layout=pl_layout)

                # Prepare Balance Information
                balance_info = f"Latest Account Balance: {self.trading_strategy.latest_balance} {self.trading_strategy.currency}"

            return price_fig, pl_fig, balance_info

    def run(self):
        """
        Runs the Dash server.
        """
        self.app.run_server(host=self.host, port=self.port, debug=False, use_reloader=False)
