import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import base64
from datetime import datetime
import re
import traceback
from utils2 import DataHandler  # Using your DataHandler class

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
    
    # File Selection Row - Simplified to just input and browse button
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Log File:"),
                dbc.Input(id="log-file-input", value=default_log_file, placeholder="Select log file..."),
                dbc.Button("Browse", id="browse-button", color="primary"),
            ], className="mb-3"),
        ], width=12)
    ]),
    
    # Product Selection Row
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Product:"),
            dcc.Dropdown(id="product-dropdown", placeholder="Select a product"),
        ], width=4),
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
    [Input("log-file-input", "value")]
)
def load_data(log_file_path):
    # If no log file specified or doesn't exist, use default
    if not log_file_path or not os.path.exists(log_file_path):
        log_file_path = DataHandler.get_most_recent_log()
        if not log_file_path:
            # No valid log file available
            return None, None, [], [], [], None
    
    # Load the log file
    try:
        print(f"Loading log file: {log_file_path}")
        s, prices_df, trades_df = DataHandler.load_log_file(log_file_path)
        print(f"Loaded data: {len(prices_df)} price records, {len(trades_df)} trade records")
        
        # Calculate position for each product based on trades
        if not trades_df.empty:
            print("Calculating positions from trades...")
            
            # Ensure required columns exist
            if 'symbol' in trades_df.columns and 'quantity' in trades_df.columns and 'timestamp' in trades_df.columns:
                
                # Get all unique products
                products_list = prices_df['product'].unique().tolist() if 'product' in prices_df.columns else []
                
                # Initialize position column with zeros
                if 'product' in prices_df.columns:
                    prices_df['position'] = 0
                    
                    # OPTIMIZATION: Calculate positions at each timestamp in a separate dataframe
                    # Sort trades chronologically
                    sorted_trades = trades_df.sort_values('timestamp')
                    
                    # Create a dictionary to track running positions
                    positions = {product: 0 for product in products_list}
                    
                    # Create a dictionary to store position changes by product and timestamp
                    # Format: {product: {timestamp: position}}
                    position_timeline = {product: {} for product in products_list}
                    
                    # Process trades to build position timeline
                    for _, trade in sorted_trades.iterrows():
                        symbol = trade['symbol']
                        quantity = float(trade['quantity']) if pd.notna(trade['quantity']) else 0
                        timestamp = trade['timestamp']
                        
                        # Skip if product not in our list
                        if symbol not in positions:
                            continue
                        
                        # Update position based on trade
                        if 'buyer' in trade and trade['buyer'] == 'SUBMISSION':
                            positions[symbol] += quantity
                        
                        if 'seller' in trade and trade['seller'] == 'SUBMISSION':
                            positions[symbol] -= quantity
                        
                        # Record the updated position at this timestamp
                        position_timeline[symbol][timestamp] = positions[symbol]
                    
                    # Now efficiently apply these positions to the prices dataframe
                    if 'timestamp' in prices_df.columns:
                        # For each product, find all its prices rows and apply positions
                        for product in products_list:
                            # Skip if no position changes for this product
                            if not position_timeline[product]:
                                continue
                            
                            # Get dataframe slice for this product
                            product_mask = prices_df['product'] == product
                            product_df = prices_df.loc[product_mask]
                            
                            if product_df.empty:
                                continue
                            
                            # For each price row, find the most recent position
                            for idx, row in product_df.iterrows():
                                time = row['timestamp']
                                
                                # Find the most recent position update before or at this timestamp
                                # This is the position at this point in time
                                current_position = 0
                                for trade_time, position in sorted(position_timeline[product].items()):
                                    if trade_time <= time:
                                        current_position = position
                                    else:
                                        break
                                
                                # Apply the position to this row
                                prices_df.at[idx, 'position'] = current_position
                    
                    print("Position calculation complete")
                else:
                    print("Cannot add position column - no 'product' column in prices dataframe")
            else:
                print("Cannot calculate positions - missing required columns in trades dataframe")
            
    except Exception as e:
        print(f"Error loading log file: {e}")
        print(traceback.format_exc())
        return None, None, [], [], [], None
    
    if prices_df.empty:
        return None, None, [], [], [], None
    
    # Convert DataFrames to dictionaries for storage
    prices_dict = prices_df.to_dict('records')
    trades_dict = trades_df.to_dict('records') if not trades_df.empty else []
    
    print(f"Prices DataFrame head:\n{prices_df.head()}")
    
    # Get unique products for dropdown
    if 'product' in prices_df.columns:
        products = [p for p in prices_df['product'].unique().tolist() 
                    if p is not None and pd.notna(p) and str(p).strip() != '']
    else:
        products = []
    
    print(f"Found products: {products}")
    
    # Create valid options for dropdown - ensure no null values
    product_options = [{"label": str(product), "value": str(product)} for product in products if product is not None]
    
    # If no valid options found, provide a default placeholder option
    if not product_options:
        product_options = [{"label": "No products available", "value": ""}]
        print("No valid products found, using placeholder option")
    else:
        print(f"Created {len(product_options)} product options")
    
    # Get columns for axis dropdowns
    numeric_columns = prices_df.select_dtypes(include=['number']).columns.tolist()
    column_options = [{"label": col, "value": col} for col in numeric_columns]
    
    # Set default product value - ensure it's not None or empty
    if products and len(products) > 0:
        default_product = str(products[0])
        print(f"Setting default product to: {default_product}")
    else:
        default_product = None
        print("No valid default product available")
    
    return prices_dict, trades_dict, product_options, column_options, column_options, default_product

@app.callback(
    Output("main-chart", "figure"),
    [Input("prices-data-store", "data"),
     Input("trades-data-store", "data"),
     Input("product-dropdown", "value")]
)
def update_main_chart(prices_data, trades_data, selected_product):
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
    
    # Define colors and visibility for different column types
    column_settings = {
        "bid_price_1": {"color": "blue", "width": 2, "visible": True},
        "ask_price_1": {"color": "red", "width": 2, "visible": True},
        "mid_price": {"color": "black", "width": 2.5, "visible": True},
        "weighted_mid_price": {"color": "purple", "width": 1.5, "visible": "legendonly"},
        "spread_pct": {"color": "orange", "width": 1, "visible": "legendonly"},
        "log_return": {"color": "green", "width": 1, "visible": "legendonly"}
    }
    
    # Add price columns
    price_columns = []
    
    # Find all bid and ask price columns
    for col in product_prices.columns:
        if re.match(r'bid_price_\d+', col) or re.match(r'ask_price_\d+', col):
            price_columns.append(col)
    
    # Add other standard columns
    standard_columns = ["mid_price", "weighted_mid_price", "spread_pct", "log_return"]
    for col in standard_columns:
        if col in product_prices.columns:
            price_columns.append(col)
    
    # Add any remaining numeric columns
    numeric_cols = product_prices.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col not in price_columns and col != 'timestamp' and not col.startswith('bid_volume') and not col.startswith('ask_volume'):
            price_columns.append(col)
    
    # Add all price columns to the plot
    for column in price_columns:
        if column in product_prices.columns:
            # Determine settings for this column
            if column in column_settings:
                color = column_settings[column]["color"]
                width = column_settings[column]["width"]
                visible = column_settings[column]["visible"]
            else:
                # For bid prices (other than bid_price_1)
                if column.startswith('bid_price_'):
                    color = "lightblue"
                    width = 1
                    visible = "legendonly"
                # For ask prices (other than ask_price_1)
                elif column.startswith('ask_price_'):
                    color = "lightpink"
                    width = 1
                    visible = "legendonly"
                # For other columns
                else:
                    color = "gray"
                    width = 1
                    visible = "legendonly"
            
            fig.add_trace(go.Scatter(
                x=product_prices.index if 'timestamp' not in product_prices.columns else product_prices['timestamp'],
                y=product_prices[column],
                mode='lines',
                name=column,
                line=dict(color=color, width=width),
                visible=visible,
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
                hovertemplate='Price: %{y:.2f}<br>Timestamp: %{x}<br>Quantity: %{text}',
                text=buy_trades['quantity'] if 'quantity' in buy_trades.columns else ''
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
                hovertemplate='Price: %{y:.2f}<br>Timestamp: %{x}<br>Quantity: %{text}',
                text=sell_trades['quantity'] if 'quantity' in sell_trades.columns else ''
            ))
    
    # Update layout with more top margin to avoid overlapping with title
    fig.update_layout(
        title=f"{selected_product} Market Data",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=40, r=40, t=80, b=40)  # Increased top margin from 60 to 80
    )
    
    # Enable clicking on the chart
    fig.update_layout(clickmode='event')
    
    return fig

@app.callback(
    [Output("selected-timestamp-store", "data"),
     Output("timestamp-display", "children")],
    [Input("main-chart", "clickData")],
    [State("prices-data-store", "data"),
     State("trades-data-store", "data"),
     State("product-dropdown", "value")]
)
def update_selected_timestamp(click_data, prices_data, trades_data, selected_product):
    if not click_data or not prices_data or not selected_product:
        return None, "No timestamp selected"
    
    try:
        # Get the direct x value from click - this is the timestamp
        x_value = click_data['points'][0]['x']
        print(f"Clicked point x value: {x_value}")
        
        # That's it! For both price lines and trade markers, the x value 
        # is the timestamp we want to display
        return x_value, f"Selected Timestamp: {x_value}"
        
    except Exception as e:
        print(f"Error extracting timestamp: {e}")
        print(traceback.format_exc())
        # Fallback to whatever x value we can extract
        return click_data['points'][0]['x'], f"Selected Timestamp: {click_data['points'][0]['x']}"

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
    
    if product_prices.empty:
        return html.Div(f"No data available for product: {selected_product}")
    
    # Find closest row to selected timestamp
    closest_row = None
    
    if 'timestamp' in product_prices.columns:
        try:
            # Convert both to numeric to avoid type comparison issues
            timestamp_series = pd.to_numeric(product_prices['timestamp'], errors='coerce')
            selected_timestamp_val = pd.to_numeric(pd.Series([selected_timestamp]), errors='coerce')[0]
            
            if pd.notna(selected_timestamp_val) and not timestamp_series.isna().all():
                # Calculate absolute difference and find minimum
                abs_diff = (timestamp_series - selected_timestamp_val).abs()
                
                if not abs_diff.empty:
                    min_idx = abs_diff.idxmin()
                    closest_row = product_prices.loc[min_idx]
                    print(f"Found closest timestamp at index {min_idx}")
                else:
                    print("Timestamp difference calculation resulted in empty series")
            else:
                print(f"Invalid timestamp values: selected={selected_timestamp_val}, series has all NaN: {timestamp_series.isna().all()}")
        except Exception as e:
            print(f"Error finding timestamp: {e}")
            print(traceback.format_exc())
    
    # If we couldn't find by timestamp, try by index
    if closest_row is None:
        try:
            if isinstance(selected_timestamp, (int, float)):
                closest_idx = min(int(selected_timestamp), len(product_prices) - 1)
            else:
                closest_idx = 0
            closest_row = product_prices.iloc[closest_idx]
            print(f"Using index-based selection: {closest_idx}")
        except Exception as e:
            print(f"Error using index selection: {e}")
            print(traceback.format_exc())
            return html.Div("Error retrieving order book data")
    
    # If we still couldn't get a row, return an error message
    if closest_row is None:
        return html.Div("Could not find matching data for the selected timestamp")
    
    # Extract bid and ask data
    bid_price_cols = [col for col in closest_row.index if col.startswith('bid_price_')]
    bid_volume_cols = [col for col in closest_row.index if col.startswith('bid_volume_')]
    ask_price_cols = [col for col in closest_row.index if col.startswith('ask_price_')]
    ask_volume_cols = [col for col in closest_row.index if col.startswith('ask_volume_')]
    
    if not bid_price_cols or not ask_price_cols:
        return html.Div("Order book data not available")
    
    # Sort columns by level
    bid_price_cols.sort(key=lambda x: int(x.split('_')[-1]))
    bid_volume_cols.sort(key=lambda x: int(x.split('_')[-1]))
    ask_price_cols.sort(key=lambda x: int(x.split('_')[-1]))
    ask_volume_cols.sort(key=lambda x: int(x.split('_')[-1]))
    
    # Create bid and ask dictionaries
    bids = {}
    for price_col, vol_col in zip(bid_price_cols, bid_volume_cols):
        price = closest_row[price_col]
        volume = closest_row[vol_col]
        if pd.notna(price) and pd.notna(volume) and volume > 0:
            # Round to integer price
            int_price = int(price)
            if int_price in bids:
                bids[int_price] += volume
            else:
                bids[int_price] = volume
    
    asks = {}
    for price_col, vol_col in zip(ask_price_cols, ask_volume_cols):
        price = closest_row[price_col]
        volume = closest_row[vol_col]
        if pd.notna(price) and pd.notna(volume) and volume > 0:
            # Round to integer price
            int_price = int(price)
            if int_price in asks:
                asks[int_price] += volume
            else:
                asks[int_price] = volume
    
    # If no valid bids or asks, return message
    if not bids or not asks:
        return html.Div("No valid order book data available")
    
    # Find price range
    min_price = min(min(bids.keys()), min(asks.keys()))
    max_price = max(max(bids.keys()), max(asks.keys()))
    
    # Create price range with integer steps
    price_range = list(range(min_price, max_price + 1))
    
    # Find maximum volume for color scaling
    max_bid_vol = max(bids.values()) if bids else 1
    max_ask_vol = max(asks.values()) if asks else 1
    max_vol = max(max_bid_vol, max_ask_vol)
    
    # Create table rows for order book
    table_rows = []
    
    # Header row
    header_row = html.Tr([
        html.Th("Bid Volume", style={"width": "30%", "text-align": "center"}),
        html.Th("Price", style={"width": "40%", "text-align": "center"}),
        html.Th("Ask Volume", style={"width": "30%", "text-align": "center"})
    ])
    table_rows.append(header_row)
    
    # Data rows
    for price in sorted(price_range, reverse=True):
        bid_vol = bids.get(price, 0)
        ask_vol = asks.get(price, 0)
        
        # Calculate color intensity based on volume
        bid_color_intensity = min(0.9, (bid_vol / max_vol) * 0.9) if bid_vol > 0 else 0
        ask_color_intensity = min(0.9, (ask_vol / max_vol) * 0.9) if ask_vol > 0 else 0
        
        bid_color = f"rgba(0, 0, 255, {bid_color_intensity})" if bid_vol > 0 else "transparent"
        ask_color = f"rgba(255, 0, 0, {ask_color_intensity})" if ask_vol > 0 else "transparent"
        
        # Format volumes as strings, blank if zero
        bid_vol_str = f"{bid_vol:.0f}" if bid_vol > 0 else ""
        ask_vol_str = f"{ask_vol:.0f}" if ask_vol > 0 else ""
        
        # Highlight mid price
        price_style = {"text-align": "center"}
        if 'mid_price' in closest_row and abs(closest_row['mid_price'] - price) < 0.5:
            price_style.update({"font-weight": "bold", "background-color": "#f8f9fa"})
        
        row = html.Tr([
            html.Td(bid_vol_str, style={"text-align": "center", "background-color": bid_color}),
            html.Td(f"{price}", style=price_style),
            html.Td(ask_vol_str, style={"text-align": "center", "background-color": ask_color})
        ])
        
        table_rows.append(row)
    
    # Create the table
    table = dbc.Table(table_rows, bordered=True, hover=True, responsive=True, striped=False)
    
    # Additional information (mid price, spread, etc.)
    info_items = []
    important_keys = ['mid_price', 'weighted_mid_price', 'spread_pct', 'log_return']
    
    for key in important_keys:
        if key in closest_row:
            value = closest_row[key]
            if pd.notna(value):
                formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
                info_items.append(html.Div([
                    html.Strong(f"{key}: "), 
                    html.Span(formatted_value)
                ], className="mb-1"))
    
    # Add a spacer
    info_items.append(html.Hr(style={"margin": "10px 0"}))
    
    # Add other info
    for key, value in closest_row.items():
        if (key not in important_keys and 
            key not in bid_price_cols and key not in bid_volume_cols and 
            key not in ask_price_cols and key not in ask_volume_cols and
            key not in ['day', 'timestamp', 'product']):
            if pd.notna(value):
                formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
                info_items.append(html.Div([
                    html.Strong(f"{key}: "), 
                    html.Span(formatted_value)
                ], className="mb-1"))
    
    return html.Div([
        html.H5("Order Book", className="text-center mb-3"),
        table,
        html.Hr(),
        html.H5("Price Info", className="text-center mb-3"),
        html.Div(info_items)
    ])

@app.callback(
    [Output("comparison-chart", "figure"),
     Output("left-axis-dropdown", "value"),
     Output("right-axis-dropdown", "value"),
     Output("left-products-dropdown", "value"),
     Output("right-products-dropdown", "value")],
    [Input("prices-data-store", "data"),
     Input("left-axis-dropdown", "value"),
     Input("right-axis-dropdown", "value"),
     Input("left-products-dropdown", "value"),
     Input("right-products-dropdown", "value"),
     Input("product-dropdown", "options")]
)
def update_comparison_chart(prices_data, left_column, right_column, left_products, right_products, product_options):
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Set default values if this is the first load or if product options changed
    first_load = triggered in [None, "prices-data-store", "product-dropdown"]
    
    # Convert options to list of product values
    all_products = [opt["value"] for opt in product_options if isinstance(opt, dict) and "value" in opt and opt["value"]]
    
    # Set default values if first load
    if first_load:
        # Set default columns
        default_left = "profit_and_loss" if prices_data and pd.DataFrame(prices_data).get('profit_and_loss', pd.Series()).notna().any() else None
        default_right = "position" if prices_data and pd.DataFrame(prices_data).get('position', pd.Series()).notna().any() else None
        
        left_column = default_left if default_left is not None else left_column
        right_column = default_right if default_right is not None else right_column
        
        # Set all products by default
        left_products = all_products if all_products and default_left is not None else left_products
        right_products = all_products if all_products and default_right is not None else right_products
    
    if not prices_data:
        empty_fig = {
            "data": [],
            "layout": {
                "title": "No data available",
                "height": 400
            }
        }
        return empty_fig, left_column, right_column, left_products, right_products
    
    # Convert dict back to DataFrame
    prices_df = pd.DataFrame(prices_data)
    
    # Initialize figure
    fig = go.Figure()
    
    # Add traces for left axis
    if left_column and left_products:
        # Generate different colors for each product
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, product in enumerate(left_products):
            product_data = prices_df[prices_df['product'] == product] if 'product' in prices_df.columns else prices_df
            if not product_data.empty and left_column in product_data.columns:
                fig.add_trace(go.Scatter(
                    x=product_data.index if 'timestamp' not in product_data.columns else product_data['timestamp'],
                    y=product_data[left_column],
                    mode='lines',
                    name=f"{product} - {left_column}",
                    line=dict(width=2, color=colors[i % len(colors)]),
                    hovertemplate=f'{product} {left_column}: %{{y:.2f}}<br>Index: %{{x}}'
                ))
    
    # Add traces for right axis
    if right_column and right_products:
        # Generate different dash patterns for each product
        dash_patterns = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, product in enumerate(right_products):
            product_data = prices_df[prices_df['product'] == product] if 'product' in prices_df.columns else prices_df
            if not product_data.empty and right_column in product_data.columns:
                fig.add_trace(go.Scatter(
                    x=product_data.index if 'timestamp' not in product_data.columns else product_data['timestamp'],
                    y=product_data[right_column],
                    mode='lines',
                    name=f"{product} - {right_column}",
                    line=dict(width=2, dash=dash_patterns[i % len(dash_patterns)], color=colors[i % len(colors)]),
                    yaxis="y2",
                    hovertemplate=f'{product} {right_column}: %{{y:.2f}}<br>Index: %{{x}}'
                ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title="Performance Metrics",
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
    
    return fig, left_column, right_column, left_products, right_products

@app.callback(
    [Output("left-products-dropdown", "options"),
     Output("right-products-dropdown", "options")],
    [Input("product-dropdown", "options")]
)
def update_product_dropdowns(product_options):
    # Validate options to ensure no null values
    valid_options = []
    for option in product_options:
        if isinstance(option, dict) and 'label' in option and 'value' in option:
            if option['label'] is not None and option['value'] is not None:
                valid_options.append(option)
    
    # If no valid options, provide a default
    if not valid_options:
        valid_options = [{"label": "No products available", "value": ""}]
        
    print(f"Updating product dropdowns with {len(valid_options)} valid options")
    return valid_options, valid_options

# Run the app
if __name__ == '__main__':
    app.run(debug=True)