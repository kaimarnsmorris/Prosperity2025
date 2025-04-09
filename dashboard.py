import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import base64  # Add base64 import for decoding uploaded files
from datetime import datetime
import re
import traceback  # For better error reporting
from utils2 import DataHandler  # Assuming DataHandler is defined in utils2.py


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Default log file
default_log_file = DataHandler.get_most_recent_log()

# App Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Trading Dashboard", className="text-center my-4"),
        ], width=12)
    ]),
    
    # File Selection Row
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Log File:"),
                dbc.Input(id="log-file-input", value=default_log_file, placeholder="Select log file..."),
                dbc.Button("Browse", id="browse-button", color="primary"),
            ], className="mb-3"),
            dcc.Upload(
                id='upload-log',
                children=html.Div(['Drag and Drop or ', html.A('Select Log File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
        ], width=12)
    ]),
    
    # Product Selection Row
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Product:"),
            dcc.Dropdown(id="product-dropdown", placeholder="Select a product"),
        ], width=4),
        dbc.Col([
            dbc.Label("Select Columns to Display:"),
            dcc.Checklist(
                id="column-checklist",
                options=[
                    {"label": "Bid Price 1", "value": "bid_price_1"},
                    {"label": "Ask Price 1", "value": "ask_price_1"},
                    {"label": "Mid Price", "value": "mid_price"},
                    {"label": "Weighted Mid Price", "value": "weighted_mid_price"},
                    {"label": "Spread %", "value": "spread_pct"},
                    {"label": "Log Return", "value": "log_return"}
                ],
                value=["bid_price_1", "ask_price_1"],
                inline=True
            ),
        ], width=8)
    ], className="mb-4"),
    
    # Main Chart Row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="main-chart", style={"height": "500px"}),
        ], width=9),
        dbc.Col([
            html.H4("Order Book", className="text-center mb-2"),
            html.Div(id="timestamp-display", className="text-center mb-2"),
            html.Div(id="order-book-display", style={"height": "450px", "overflowY": "auto"})
        ], width=3)
    ], className="mb-4"),
    
    # Comparison Chart Controls
    dbc.Row([
        dbc.Col([
            html.H4("Comparison Chart", className="mb-3"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Label("Left Y-Axis Column:"),
            dcc.Dropdown(id="left-axis-dropdown", placeholder="Select column for left axis"),
        ], width=3),
        dbc.Col([
            dbc.Label("Left Y-Axis Products:"),
            dcc.Dropdown(id="left-products-dropdown", multi=True, placeholder="Select products"),
        ], width=3),
        dbc.Col([
            dbc.Label("Right Y-Axis Column:"),
            dcc.Dropdown(id="right-axis-dropdown", placeholder="Select column for right axis"),
        ], width=3),
        dbc.Col([
            dbc.Label("Right Y-Axis Products:"),
            dcc.Dropdown(id="right-products-dropdown", multi=True, placeholder="Select products"),
        ], width=3)
    ], className="mb-3"),
    
    # Comparison Chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="comparison-chart", style={"height": "400px"}),
        ], width=12)
    ]),
    
    # Store components for data
    dcc.Store(id="prices-data-store"),
    dcc.Store(id="trades-data-store"),
    dcc.Store(id="selected-timestamp-store")
], fluid=True)

# Define callbacks

@app.callback(
    [Output("prices-data-store", "data"),
     Output("trades-data-store", "data"),
     Output("product-dropdown", "options"),
     Output("left-axis-dropdown", "options"),
     Output("right-axis-dropdown", "options"),
     Output("product-dropdown", "value")],
    [Input("log-file-input", "value"),
     Input("upload-log", "contents")],
    [State("upload-log", "filename")]
)
def load_data(log_file_path, contents, filename):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # If triggered by upload
    if triggered_id == "upload-log" and contents:
        print(f"File upload triggered: {filename}")
        try:
            content_type, content_string = contents.split(',')
            print(f"Content type: {content_type}")
            print(f"Content string length: {len(content_string)}")
            
            # Decode the file contents
            decoded = base64.b64decode(content_string)
            print(f"Successfully decoded file, size: {len(decoded)} bytes")
            
            # Save temp file and load it
            temp_file = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
            print(f"Saving to temporary file: {temp_file}")
            
            with open(temp_file, 'wb') as f:
                f.write(decoded)
            
            print(f"Loading data from temporary file")
            sandbox_logs, prices_df, trades_df = DataHandler.load_log_file(temp_file)
            print(f"Loaded data: {len(prices_df)} price records, {len(trades_df)} trade records")
            
            os.remove(temp_file)  # Clean up
            print(f"Removed temporary file")
            
            log_file_path = filename  # Update the input value
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            print(traceback.format_exc())  # Print the full traceback for debugging
            raise PreventUpdate
    else:
        # If no log file specified or doesn't exist, use default
        if not log_file_path or not os.path.exists(log_file_path):
            log_file_path = DataHandler.get_most_recent_log()
            if not log_file_path:
                # No valid log file available
                return None, None, [], [], [], None
        
        # Load the log file
        sandbox_logs,prices_df, trades_df = DataHandler.load_log_file(log_file_path)
    
    if prices_df.empty:
        return None, None, [], [], [], None
    
    # Convert DataFrames to dictionaries for storage
    prices_dict = prices_df.to_dict('records')
    trades_dict = trades_df.to_dict('records') if not trades_df.empty else []
    
    # Get unique products for dropdown
    products = prices_df['product'].unique().tolist() if 'product' in prices_df.columns else []
    product_options = [{"label": product, "value": product} for product in products]
    
    # Get columns for axis dropdowns
    numeric_columns = prices_df.select_dtypes(include=['number']).columns.tolist()
    column_options = [{"label": col, "value": col} for col in numeric_columns]
    
    # Set default product value
    default_product = products[0] if products else None
    
    return prices_dict, trades_dict, product_options, column_options, column_options, default_product

@app.callback(
    Output("main-chart", "figure"),
    [Input("prices-data-store", "data"),
     Input("trades-data-store", "data"),
     Input("product-dropdown", "value"),
     Input("column-checklist", "value")]
)
def update_main_chart(prices_data, trades_data, selected_product, selected_columns):
    if not prices_data or not selected_product:
        return {
            "data": [],
            "layout": {
                "title": "No data available",
                "height": 500
            }
        }
    
    # Convert dict back to DataFrame
    prices_df = pd.DataFrame(prices_data)
    trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
    
    # Filter for selected product
    product_prices = prices_df[prices_df['product'] == selected_product].copy() if 'product' in prices_df.columns else prices_df
    product_trades = trades_df[trades_df['symbol'] == selected_product].copy() if not trades_df.empty and 'symbol' in trades_df.columns else pd.DataFrame()
    
    if product_prices.empty:
        return {
            "data": [],
            "layout": {
                "title": f"No data available for {selected_product}",
                "height": 500
            }
        }
    
    # Create figure
    fig = go.Figure()
    
    # Add traces based on selected columns
    colors = {
        "bid_price_1": "blue",
        "ask_price_1": "red",
        "mid_price": "black",
        "weighted_mid_price": "purple",
        "spread_pct": "orange",
        "log_return": "green"
    }
    
    for column in selected_columns:
        if column in product_prices.columns:
            fig.add_trace(go.Scatter(
                x=product_prices.index if 'timestamp' not in product_prices.columns else product_prices['timestamp'],
                y=product_prices[column],
                mode='lines',
                name=column,
                line=dict(color=colors.get(column, "gray"), width=2),
                hovertemplate=f'{column}: %{{y:.2f}}<br>Index: %{{x}}'
            ))
    
    # Add trades if available
    if not product_trades.empty:
        # Buy trades
        buy_trades = product_trades[product_trades['buyer'] == 'SUBMISSION'] if 'buyer' in product_trades.columns else pd.DataFrame()
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['timestamp'] if 'timestamp' in buy_trades.columns else buy_trades.index,
                y=buy_trades['price'] if 'price' in buy_trades.columns else buy_trades['mid_price'],
                mode='markers',
                name='Buy Trades',
                marker=dict(color='green', size=8, symbol='circle'),
                hovertemplate='Price: %{y:.2f}<br>Timestamp: %{x}'
            ))
        
        # Sell trades
        sell_trades = product_trades[product_trades['seller'] == 'SUBMISSION'] if 'seller' in product_trades.columns else pd.DataFrame()
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['timestamp'] if 'timestamp' in sell_trades.columns else sell_trades.index,
                y=sell_trades['price'] if 'price' in sell_trades.columns else sell_trades['mid_price'],
                mode='markers',
                name='Sell Trades',
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate='Price: %{y:.2f}<br>Timestamp: %{x}'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{selected_product} Market Data",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Enable clicking on the chart
    fig.update_layout(clickmode='event')
    
    return fig

@app.callback(
    [Output("selected-timestamp-store", "data"),
     Output("timestamp-display", "children")],
    [Input("main-chart", "clickData")],
    [State("prices-data-store", "data"),
     State("product-dropdown", "value")]
)
def update_selected_timestamp(click_data, prices_data, selected_product):
    if not click_data or not prices_data or not selected_product:
        return None, "No timestamp selected"
    
    # Get timestamp from click
    point_index = click_data['points'][0]['pointIndex']
    x_value = click_data['points'][0]['x']
    
    # Convert dict back to DataFrame
    prices_df = pd.DataFrame(prices_data)
    
    # Filter for selected product
    product_prices = prices_df[prices_df['product'] == selected_product] if 'product' in prices_df.columns else prices_df
    
    # Get timestamp value
    if 'timestamp' in product_prices.columns:
        timestamp = product_prices.iloc[point_index]['timestamp'] if point_index < len(product_prices) else x_value
    else:
        timestamp = x_value
    
    return timestamp, f"Selected Timestamp: {timestamp}"

@app.callback(
    Output("order-book-display", "children"),
    [Input("selected-timestamp-store", "data")],
    [State("prices-data-store", "data"),
     State("product-dropdown", "value")]
)
def update_order_book(selected_timestamp, prices_data, selected_product):
    if not selected_timestamp or not prices_data or not selected_product:
        return html.Div("No data selected")
    
    # Convert dict back to DataFrame
    prices_df = pd.DataFrame(prices_data)
    
    # Filter for selected product
    product_prices = prices_df[prices_df['product'] == selected_product] if 'product' in prices_df.columns else prices_df
    
    # Find closest row to selected timestamp
    if 'timestamp' in product_prices.columns:
        closest_row = product_prices.iloc[(product_prices['timestamp'] - selected_timestamp).abs().argsort()[0]]
    else:
        closest_idx = min(int(selected_timestamp), len(product_prices) - 1) if isinstance(selected_timestamp, (int, float)) else 0
        closest_row = product_prices.iloc[closest_idx]
    
    # Create order book display
    book_columns = [col for col in closest_row.index if re.match(r'(bid|ask)_(price|volume)_\d+', col)]
    
    if not book_columns:
        return html.Div("Order book data not available")
    
    # Extract bid and ask columns
    bid_price_cols = sorted([col for col in book_columns if col.startswith('bid_price')])
    bid_volume_cols = sorted([col for col in book_columns if col.startswith('bid_volume')])
    ask_price_cols = sorted([col for col in book_columns if col.startswith('ask_price')])
    ask_volume_cols = sorted([col for col in book_columns if col.startswith('ask_volume')])
    
    # Create table header
    header_row = html.Tr([
        html.Th("Bid Vol", style={"width": "25%", "text-align": "center"}),
        html.Th("Bid Price", style={"width": "25%", "text-align": "center"}),
        html.Th("Ask Price", style={"width": "25%", "text-align": "center"}),
        html.Th("Ask Vol", style={"width": "25%", "text-align": "center"})
    ])
    
    # Create table rows
    table_rows = [header_row]
    max_levels = max(len(bid_price_cols), len(ask_price_cols))
    
    for i in range(max_levels):
        bid_price = closest_row[bid_price_cols[i]] if i < len(bid_price_cols) else "-"
        bid_vol = closest_row[bid_volume_cols[i]] if i < len(bid_volume_cols) else "-"
        ask_price = closest_row[ask_price_cols[i]] if i < len(ask_price_cols) else "-"
        ask_vol = closest_row[ask_volume_cols[i]] if i < len(ask_volume_cols) else "-"
        
        # Format numbers
        bid_price = f"{bid_price:.2f}" if isinstance(bid_price, (int, float)) else bid_price
        bid_vol = f"{bid_vol:.0f}" if isinstance(bid_vol, (int, float)) else bid_vol
        ask_price = f"{ask_price:.2f}" if isinstance(ask_price, (int, float)) else ask_price
        ask_vol = f"{ask_vol:.0f}" if isinstance(ask_vol, (int, float)) else ask_vol
        
        row = html.Tr([
            html.Td(bid_vol, style={"text-align": "center", "background-color": "#e6f7ff"}),
            html.Td(bid_price, style={"text-align": "center", "background-color": "#e6f7ff"}),
            html.Td(ask_price, style={"text-align": "center", "background-color": "#ffebee"}),
            html.Td(ask_vol, style={"text-align": "center", "background-color": "#ffebee"})
        ])
        
        table_rows.append(row)
    
    # Create the table
    table = dbc.Table(table_rows, bordered=True, hover=True, responsive=True, striped=False)
    
    # Additional information
    info_items = []
    for key, value in closest_row.items():
        if key not in book_columns and key not in ['day', 'timestamp', 'product']:
            formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
            info_items.append(html.Div([
                html.Strong(f"{key}: "), 
                html.Span(formatted_value)
            ], className="mb-1"))
    
    return html.Div([
        html.H5("Order Book", className="text-center mb-3"),
        table,
        html.Hr(),
        html.H5("Additional Info", className="text-center mb-3"),
        html.Div(info_items)
    ])

@app.callback(
    Output("comparison-chart", "figure"),
    [Input("prices-data-store", "data"),
     Input("left-axis-dropdown", "value"),
     Input("right-axis-dropdown", "value"),
     Input("left-products-dropdown", "value"),
     Input("right-products-dropdown", "value")]
)
def update_comparison_chart(prices_data, left_column, right_column, left_products, right_products):
    if not prices_data:
        return {
            "data": [],
            "layout": {
                "title": "No data available",
                "height": 400
            }
        }
    
    # Convert dict back to DataFrame
    prices_df = pd.DataFrame(prices_data)
    
    # Initialize figure
    fig = go.Figure()
    
    # Add traces for left axis
    if left_column and left_products:
        for product in left_products:
            product_data = prices_df[prices_df['product'] == product] if 'product' in prices_df.columns else prices_df
            if not product_data.empty and left_column in product_data.columns:
                fig.add_trace(go.Scatter(
                    x=product_data.index if 'timestamp' not in product_data.columns else product_data['timestamp'],
                    y=product_data[left_column],
                    mode='lines',
                    name=f"{product} - {left_column}",
                    line=dict(width=2),
                    hovertemplate=f'{product} {left_column}: %{{y:.2f}}<br>Index: %{{x}}'
                ))
    
    # Add traces for right axis
    if right_column and right_products:
        for product in right_products:
            product_data = prices_df[prices_df['product'] == product] if 'product' in prices_df.columns else prices_df
            if not product_data.empty and right_column in product_data.columns:
                fig.add_trace(go.Scatter(
                    x=product_data.index if 'timestamp' not in product_data.columns else product_data['timestamp'],
                    y=product_data[right_column],
                    mode='lines',
                    name=f"{product} - {right_column}",
                    line=dict(width=2, dash='dash'),
                    yaxis="y2",
                    hovertemplate=f'{product} {right_column}: %{{y:.2f}}<br>Index: %{{x}}'
                ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title="Comparison Chart",
        xaxis_title="Timestamp",
        yaxis=dict(
            title=left_column if left_column else "",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title=right_column if right_column else "",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@app.callback(
    [Output("left-products-dropdown", "options"),
     Output("right-products-dropdown", "options")],
    [Input("product-dropdown", "options")]
)
def update_product_dropdowns(product_options):
    return product_options, product_options

# Run the app
if __name__ == '__main__':
    app.run(debug=True)