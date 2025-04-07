import pandas as pd
import numpy as np
import os
import glob
import json
import re
import plotly.graph_objects as go

# --- Log File Parsing Functions ---

def most_recent_log():
    """Returns the path to the most recently modified log file."""
    list_of_files = glob.glob('logs/*')
    return max(list_of_files, key=os.path.getctime) if list_of_files else None

def load_log_file(file_path):
    """Load and parse a log file into three dataframes."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the file into sections
    sections = re.split(r'\n\n+(?:Sandbox logs:|Activities log:|Trade History:)\n', content)
    assert len(sections) == 3, "Expected 3 sections in the log file"

    sandbox_logs = parse_sandbox_logs(sections[0])
    activities_log = parse_activities_log(sections[1])
    trade_history = parse_trade_history(sections[2])
    
    return sandbox_logs, activities_log, trade_history

def parse_sandbox_logs(logs_text):
    """Parse sandbox logs section into a DataFrame."""
    logs_text = logs_text.replace('Sandbox logs:\n','').strip()
    logs_text = re.sub(r'}\s*\n\s*{', '},{', logs_text)
    
    if not logs_text.startswith('['): logs_text = '[' + logs_text
    if not logs_text.endswith(']'): logs_text = logs_text + ']'
    
    try:
        log_entries = json.loads(logs_text)
        for entry in log_entries:
            if 'lambdaLog' in entry and entry['lambdaLog']:
                entry['parsed_lambda'] = entry['lambdaLog']
        return pd.DataFrame(log_entries)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return pd.DataFrame(columns=['sandboxLog', 'lambdaLog', 'timestamp'])

def parse_activities_log(activities_text):
    """Parse activities log into a DataFrame with proper types."""
    lines = activities_text.strip().split('\n')
    header = lines[0].split(';')
    
    data = []
    for line in lines[1:]:
        if line.strip():
            values = line.split(';')
            data.append(values)
    
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns - ensuring we handle empty strings
    numeric_columns = ['timestamp', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2',
                       'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1', 'ask_price_2', 
                       'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def parse_trade_history(trade_text):
    """Parse trade history JSON into a DataFrame."""
    trades = json.loads(trade_text)
    df = pd.DataFrame(trades)
    # Ensure numeric types
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    return df

# --- Market Data Functions ---

def read_market_data(file_path):
    """Read a single prices CSV file into a dataframe."""
    df = pd.read_csv(file_path, sep=';')
    # Ensure numeric conversion for critical columns
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    for col in df.columns:
        if col.startswith('bid_price') or col.startswith('ask_price') or col == 'mid_price':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def read_all_prices_data(round_num, base_dir="round-{}-island-data-bottle", include_logs=False):
    """
    Read all prices data for a given round and concatenate into a single dataframe.
    If include_logs is True, also append out-of-sample data from the most recent log file.
    """
    # Read CSV files
    round_dir = base_dir.format(round_num)
    pattern = os.path.join(round_dir, f"prices_round_{round_num}_day_*.csv")
    files = glob.glob(pattern)
    
    dfs = []
    for file in files:
        df = read_market_data(file)
        day = int(file.split("_day_")[1].split(".csv")[0])
        df['day'] = day
        dfs.append(df)
    
    result_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Append log data if requested
    if include_logs:
        log_file = most_recent_log()
        if log_file:
            print(f"Including log data from {log_file}")
            _, activities_log, _ = load_log_file(log_file)
            
            # Ensure day is set to a value larger than existing days
            if not result_df.empty:
                max_day = result_df['day'].max() if 'day' in result_df.columns else 0
                activities_log['day'] = max_day + 1
            else:
                activities_log['day'] = 0
            
            # Make sure we have all required columns
            for col in result_df.columns:
                if col not in activities_log.columns:
                    activities_log[col] = np.nan
            
            # Append to result
            result_df = pd.concat([result_df, activities_log], ignore_index=True)
    
    # Ensure day is numeric
    if 'day' in result_df.columns:
        result_df['day'] = pd.to_numeric(result_df['day'], errors='coerce')
        
    return result_df

def read_trade_data(file_path):
    """Read a single trades CSV file into a dataframe."""
    df = pd.read_csv(file_path, sep=';')
    if 'price' in df.columns: 
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'quantity' in df.columns: 
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    if 'timestamp' in df.columns: 
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    return df

def read_all_trades_data(round_num, base_dir="round-{}-island-data-bottle", include_logs=False):
    """
    Read all trades data for a given round and concatenate into a single dataframe.
    If include_logs is True, also append out-of-sample data from the most recent log file.
    """
    # Read CSV files
    round_dir = base_dir.format(round_num)
    pattern = os.path.join(round_dir, f"trades_round_{round_num}_day_*.csv")
    files = glob.glob(pattern)
    
    dfs = []
    for file in files:
        df = read_trade_data(file)
        day = int(file.split("_day_")[1].split(".csv")[0])
        df['day'] = day
        dfs.append(df)
    
    result_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Append log data if requested
    if include_logs:
        log_file = most_recent_log()
        if log_file:
            print(f"Including trade history from {log_file}")
            _, _, trade_history = load_log_file(log_file)
            
            # Ensure day is set to a value larger than existing days
            if not result_df.empty:
                max_day = result_df['day'].max() if 'day' in result_df.columns else 0
                trade_history['day'] = max_day + 1
            else:
                trade_history['day'] = 0
            
            # Rename columns to match if needed
            column_mapping = {
                'product': 'symbol',  # In case the log uses different column names
                'currency': 'currency'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in trade_history.columns and new_col not in trade_history.columns:
                    trade_history[new_col] = trade_history[old_col]
            
            # Ensure required columns exist
            for col in ['symbol', 'price', 'quantity', 'timestamp']:
                if col not in trade_history.columns:
                    trade_history[col] = np.nan
            
            # Append to result
            result_df = pd.concat([result_df, trade_history], ignore_index=True)
    
    # Ensure day is numeric
    if 'day' in result_df.columns:
        result_df['day'] = pd.to_numeric(result_df['day'], errors='coerce')
        
    return result_df

def partition_prices(prices_df, products):
    """Partition a price dataframe by the specified products."""
    partitioned_dfs = []
    for product in products:
        product_df = prices_df[prices_df['product'] == product].copy()
        partitioned_dfs.append(product_df)
    return tuple(partitioned_dfs)

def plot_market_data(prices_df, trades_df, plot_width=2000, plot_height=1200):
    """
    Create interactive Plotly visualizations of market data.
    
    Parameters:
    prices_df (DataFrame): Price data with columns for product, timestamp, bid/ask prices
    trades_df (DataFrame): Trade data with columns for symbol, timestamp, price, quantity
    plot_width (int): Width of the plot in pixels (default 2000)
    plot_height (int): Height of the plot in pixels (default 1200)
    """
    for product in prices_df['product'].unique():
        # Filter data for this product
        product_log = prices_df[prices_df['product'] == product].copy()
        product_trades = trades_df[trades_df['symbol'] == product].copy()
        
        if product_log.empty:
            print(f"No price data for {product}")
            continue
            
        if product_trades.empty:
            print(f"No trade data for {product}")
            
        # Reset index to create sequential index
        product_log = product_log.reset_index(drop=True)
        
        # Create a lookup dictionary to map (day, timestamp) to index in price_log
        if 'day' in product_log.columns and 'day' in product_trades.columns:
            # Create (day, timestamp) lookup
            price_lookup = {}
            for idx, row in product_log.iterrows():
                price_lookup[(row['day'], row['timestamp'])] = idx
                
            # Map trades to nearest price index
            trade_indices = []
            for _, trade in product_trades.iterrows():
                key = (trade['day'], trade['timestamp'])
                if key in price_lookup:
                    trade_indices.append(price_lookup[key])
                else:
                    # Find closest timestamp within same day
                    same_day = product_log[product_log['day'] == trade['day']]
                    if not same_day.empty:
                        same_day_ts = pd.to_numeric(same_day['timestamp'], errors='coerce')
                        trade_ts = pd.to_numeric(trade['timestamp'], errors='coerce')
                        if pd.notna(trade_ts) and not same_day_ts.isna().all():
                            closest_idx = (same_day_ts - trade_ts).abs().idxmin()
                            trade_indices.append(closest_idx)
                        else:
                            trade_indices.append(np.nan)
                    else:
                        trade_indices.append(np.nan)
                        
            product_trades['plot_idx'] = trade_indices
        else:
            # If we don't have day columns, match only on timestamp
            price_lookup = dict(zip(product_log['timestamp'], product_log.index))
            
            # Find closest timestamp match for each trade
            trade_indices = []
            for _, trade in product_trades.iterrows():
                if trade['timestamp'] in price_lookup:
                    trade_indices.append(price_lookup[trade['timestamp']])
                else:
                    # Find closest timestamp
                    product_log_ts = pd.to_numeric(product_log['timestamp'], errors='coerce')
                    trade_ts = pd.to_numeric(trade['timestamp'], errors='coerce')
                    if pd.notna(trade_ts) and not product_log_ts.isna().all():
                        closest_idx = (product_log_ts - trade_ts).abs().idxmin()
                        trade_indices.append(closest_idx)
                    else:
                        trade_indices.append(np.nan)
                    
            product_trades['plot_idx'] = trade_indices
        
        # Drop trades with no matching index
        product_trades = product_trades.dropna(subset=['plot_idx'])
        product_trades['plot_idx'] = product_trades['plot_idx'].astype(int)
        
        fig = go.Figure()
        
        # Add bid and ask prices
        fig.add_trace(go.Scatter(
            x=product_log.index, 
            y=product_log['bid_price_1'],
            mode='lines',
            name='Bid Price 1',
            line=dict(color='blue', width=1, dash='solid'),
            opacity=0.7,
            hovertemplate='Index: %{x}<br>Bid: %{y}<br>Day: %{customdata}',
            customdata=product_log['day']
        ))
        
        fig.add_trace(go.Scatter(
            x=product_log.index, 
            y=product_log['ask_price_1'],
            mode='lines',
            name='Ask Price 1',
            line=dict(color='red', width=1, dash='solid'),
            opacity=0.7,
            hovertemplate='Index: %{x}<br>Ask: %{y}<br>Day: %{customdata}',
            customdata=product_log['day']
        ))
        
        fig.add_trace(go.Scatter(
            x=product_log.index, 
            y=product_log['mid_price'],
            mode='lines',
            name='Mid Price',
            line=dict(color='black', width=2),
            hovertemplate='Index: %{x}<br>Mid: %{y}<br>Day: %{customdata}',
            customdata=product_log['day']
        ))
        
        # Add trades with aligned indices
        buy_trades = product_trades[product_trades['buyer'] == 'SUBMISSION']
        sell_trades = product_trades[product_trades['seller'] == 'SUBMISSION']
        other_trades = product_trades[(product_trades['buyer'] != 'SUBMISSION') & 
                                     (product_trades['seller'] != 'SUBMISSION')]
        
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['plot_idx'],
                y=buy_trades['price'],
                mode='markers',
                name='Buy Trades',
                marker=dict(color='green', size=8, symbol='circle'),
                hovertemplate='Index: %{x}<br>Price: %{y}<br>Quantity: %{customdata}',
                customdata=buy_trades['quantity']
            ))
        
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['plot_idx'],
                y=sell_trades['price'],
                mode='markers',
                name='Sell Trades',
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate='Index: %{x}<br>Price: %{y}<br>Quantity: %{customdata}',
                customdata=sell_trades['quantity']
            ))
        
        if not other_trades.empty:
            fig.add_trace(go.Scatter(
                x=other_trades['plot_idx'],
                y=other_trades['price'],
                mode='markers',
                name='Other Trades',
                marker=dict(color='blue', size=6, symbol='circle'),
                opacity=0.5,
                hovertemplate='Index: %{x}<br>Price: %{y}<br>Quantity: %{customdata}',
                customdata=other_trades['quantity']
            ))
        
        # Add day boundaries if day column exists
        if 'day' in product_log.columns:
            day_changes = product_log[product_log['day'].diff() != 0]
            for idx, row in day_changes.iterrows():
                fig.add_vline(x=idx, line_dash="dash", line_color="gray", 
                             annotation_text=f"Day {row['day']}", annotation_position="top right")
        
        # Update layout with customizable size
        fig.update_layout(
            title=f'{product} Market Data',
            xaxis_title='Index',
            yaxis_title='Price',
            height=plot_height,
            width=plot_width,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='closest'
        )
        
        fig.show()

# Complete example of usage
def run_market_analysis(round_num=1, include_logs=True, plot_width=2000, plot_height=1200):
    """
    Run a complete market analysis including both historical data and log files.
    
    Parameters:
    round_num (int): Round number to analyze
    include_logs (bool): Whether to include log data
    plot_width (int): Width of plots in pixels
    plot_height (int): Height of plots in pixels
    
    Returns:
    tuple: (prices_df, trades_df, product_dfs)
    """
    print(f"Loading price data for round {round_num}...")
    prices_df = read_all_prices_data(round_num, include_logs=include_logs)
    
    print(f"Loading trade data for round {round_num}...")
    trades_df = read_all_trades_data(round_num, include_logs=include_logs)
    
    if prices_df.empty:
        print("No price data found!")
        return
        
    print(f"Found {len(prices_df)} price records and {len(trades_df)} trade records")
    
    # Display products in the data
    products = prices_df['product'].unique()
    print(f"Products found: {products}")
    
    # Create individual dataframes for each product
    print("Partitioning data by product...")
    product_dfs = partition_prices(prices_df, products)
    
    # Show some basic statistics
    print("\nBasic statistics:")
    for i, product in enumerate(products):
        product_df = product_dfs[i]
        print(f"\n{product}:")
        print(f"  Records: {len(product_df)}")
        print(f"  Days: {product_df['day'].nunique()}")
        print(f"  Price range: {product_df['mid_price'].min()} - {product_df['mid_price'].max()}")
        
        trades_count = len(trades_df[trades_df['symbol'] == product])
        print(f"  Trades: {trades_count}")
    
    # Plot the data with customizable size
    print("\nGenerating interactive visualizations...")
    plot_market_data(prices_df, trades_df, plot_width=plot_width, plot_height=plot_height)
    
    return prices_df, trades_df, product_dfs



if __name__ == "__main__":
    # Run the market analysis for round 1 and include logs
    prices_df, trades_df, product_dfs = run_market_analysis(round_num=1, include_logs=True, plot_width=1200, plot_height=800)