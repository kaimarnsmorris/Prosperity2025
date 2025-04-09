import pandas as pd
import numpy as np
import os
import glob
import json
import re
import plotly.graph_objects as go

# --- Log File Parsing Functions ---

def get_most_recent_log():
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
    activities_log = parse_order_books(sections[1])
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

def parse_order_books(activities_text):
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

def add_engineered_features(df):
    """
    Add engineered features to the market data:
    - weighted_mid_price: Mid price weighted by bid/ask volumes
    - spread_pct: Spread as percentage of bid price
    - log_return: Log return of mid price
    """
    # Calculate weighted mid price
    bid_vol_1 = df['bid_volume_1'].fillna(0)
    ask_vol_1 = df['ask_volume_1'].fillna(0)
    bid_vol_2 = df['bid_volume_2'].fillna(0)
    ask_vol_2 = df['ask_volume_2'].fillna(0)
    bid_vol_3 = df['bid_volume_3'].fillna(0)
    ask_vol_3 = df['ask_volume_3'].fillna(0)
    total_vol = bid_vol_1 + ask_vol_1 + bid_vol_2 + ask_vol_2 + bid_vol_3 + ask_vol_3
    
    # Avoid division by zero
    total_vol_safe = total_vol.copy()
    total_vol_safe[total_vol_safe == 0] = 1
    
    df['weighted_mid_price'] = (
        (df['bid_price_1'] * bid_vol_1 + df['ask_price_1'] * ask_vol_1 +
         df['bid_price_2'].fillna(0) * bid_vol_2 + df['ask_price_2'].fillna(0) * ask_vol_2 +
            df['bid_price_3'].fillna(0) * bid_vol_3 + df['ask_price_3'].fillna(0) * ask_vol_3) / total_vol_safe

    )
    
    # When only one side has volume, use that price
    bid_only = (bid_vol_1 > 0) & (ask_vol_1 == 0)
    ask_only = (ask_vol_1 > 0) & (bid_vol_1 == 0)
    df.loc[bid_only, 'weighted_mid_price'] = df.loc[bid_only, 'bid_price_1']
    df.loc[ask_only, 'weighted_mid_price'] = df.loc[ask_only, 'ask_price_1']
    
    # If neither side has volume, use simple mid price
    no_vol = (bid_vol_1 == 0) & (ask_vol_1 == 0)
    df.loc[no_vol, 'weighted_mid_price'] = df.loc[no_vol, 'mid_price']
    
    # Calculate spread as percentage
    with np.errstate(divide='ignore', invalid='ignore'):
        df['spread_pct'] = 100 * (df['ask_price_1'] - df['bid_price_1']) / df['bid_price_1']
    df['spread_pct'] = df['spread_pct'].replace([np.inf, -np.inf], np.nan)
    
    # Group by product and calculate log returns
    for product in df['product'].unique():
        product_mask = df['product'] == product
        
        # Sort chronologically within each product
        sorted_idx = df.loc[product_mask].sort_values(['day', 'timestamp']).index
        
        # Calculate log returns on mid price
        mid_price = df.loc[sorted_idx, 'mid_price']
        df.loc[sorted_idx, 'log_return'] = np.log(mid_price / mid_price.shift(1))
    

    # stddev of log returns
    df['volatility20'] = df['log_return'].rolling(window=20).std()
    df['return20'] = df['log_return'].rolling(window=20).mean()

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
        log_file = get_most_recent_log()
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
    
    # Sort the data chronologically by day and timestamp
    if not result_df.empty:
        result_df = result_df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    
    # Add engineered features
    if not result_df.empty:
        result_df = add_engineered_features(result_df)
        
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
        log_file = get_most_recent_log()
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
    
    # Sort the data chronologically by day and timestamp
    if not result_df.empty:
        result_df = result_df.sort_values(['day', 'timestamp']).reset_index(drop=True)
        
    return result_df

def partition_prices(prices_df, products):
    """Partition a price dataframe by the specified products."""
    partitioned_dfs = []
    for product in products:
        product_df = prices_df[prices_df['product'] == product].copy()
        partitioned_dfs.append(product_df)
    return tuple(partitioned_dfs)

def plot_market_data(prices_df, trades_df, plot_width=2000, plot_height=1200, features_to_plot=None):
    """
    Create interactive Plotly visualizations of market data.
    
    Parameters:
    prices_df (DataFrame): Price data with columns for product, timestamp, bid/ask prices
    trades_df (DataFrame): Trade data with columns for symbol, timestamp, price, quantity
    plot_width (int): Width of the plot in pixels (default 2000)
    plot_height (int): Height of the plot in pixels (default 1200)
    features_to_plot (list): List of additional feature columns to plot, defaults to 
                             ['weighted_mid_price', 'spread_pct', 'log_return']
    """
    # Default features to plot
    if features_to_plot is None:
        features_to_plot = ['weighted_mid_price', 'spread_pct', 'log_return']
    
    for product in prices_df['product'].unique():
        # Filter data for this product
        product_log = prices_df[prices_df['product'] == product].copy()
        product_trades = trades_df[trades_df['symbol'] == product].copy()
        
        if product_log.empty:
            print(f"No price data for {product}")
            continue
            
        if product_trades.empty:
            print(f"No trade data for {product}")
            
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
        if not product_trades.empty:  # Only convert to int if not empty
            product_trades['plot_idx'] = product_trades['plot_idx'].astype(int)
        
        # Create main price chart
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
        
        # Add engineered features if they exist
        for feature in features_to_plot:
            if feature in product_log.columns and feature != 'mid_price':
                if feature == 'weighted_mid_price':
                    color = 'purple'
                    width = 1.5
                elif feature == 'spread_pct':
                    color = 'orange'
                    width = 1
                elif feature == 'log_return':
                    color = 'green'
                    width = 1
                else:
                    color = 'teal'
                    width = 1
                
                fig.add_trace(go.Scatter(
                    x=product_log.index, 
                    y=product_log[feature],
                    mode='lines',
                    name=feature,
                    line=dict(color=color, width=width),
                    opacity=0.7,
                    hovertemplate=f'Index: %{{x}}<br>{feature}: %{{y}}<br>Day: %{{customdata}}',
                    customdata=product_log['day'],
                    visible='legendonly' if feature in ['spread_pct', 'log_return'] else True  # Only show by default if weighted_mid_price
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
            xaxis_title='Index (Chronological)',
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
        print(f"  Avg. spread %: {product_df['spread_pct'].mean():.2f}%")
        
        trades_count = len(trades_df[trades_df['symbol'] == product])
        print(f"  Trades: {trades_count}")
    
    # Plot the data with customizable size
    print("\nGenerating interactive visualizations...")
    plot_market_data(prices_df, trades_df, plot_width=plot_width, plot_height=plot_height)
    
    return prices_df, trades_df, product_dfs





def analyse_time_series(series, ts=None, non_negative=False, aut_corr_lags=np.arange(1, 20), windows=None):
    """
    Generate detailed plots for time series analysis.
    
    Parameters:
    -----------
    series : array-like
        The time series data to analyze
    ts : array-like, optional
        Time points corresponding to the series. If None, uses sequential indices.
    non_negative : bool, default=False
        If True, calculates log returns instead of differences for non-negative data
    aut_corr_lags : array-like, default=np.arange(1, 20)
        Lags to use for autocorrelation analysis
    windows : list, optional
        Window sizes for rolling statistics. If None, uses [len(series)//10, len(series)//25]
    
    Returns:
    --------
    figs : list
        List of matplotlib figure objects
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from scipy import signal
    from scipy.fft import fft, fftfreq
    
    if ts is None:
        ts = np.arange(len(series))
    if windows is None:
        windows = [len(series)//10, len(series)//25]
    
    # Convert to pandas Series for easier manipulation
    series_pd = pd.Series(series, index=ts)
    
    # Calculate differences or log returns
    if non_negative:
        # Calculate log returns for non-negative data
        transformed = np.log(series_pd / series_pd.shift(1)).dropna()
        transform_label = 'Log Returns'
    else:
        # Calculate differences
        transformed = series_pd.diff().dropna()
        transform_label = 'First Differences'
    
    figs = []  # List to store all figures
    
    # Helper function for autocorrelation with custom plotting
    def plot_autocorrelation(x, lags, ax, title):
        # Calculate autocorrelation
        acf = np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lags+1)])
        lags_array = np.arange(lags+1)
        
        # Plot as line with circles
        ax.plot(lags_array, acf, 'o-', markersize=5)
        ax.axhline(y=0, linestyle='--', color='gray', alpha=0.7)
        
        # Add confidence intervals (95%)
        conf_level = 1.96 / np.sqrt(len(x))
        ax.fill_between(lags_array, -conf_level, conf_level, alpha=0.2, color='blue')
        
        ax.set_title(title)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.grid(True, alpha=0.3)
        return acf
    
    # Figure 1: Original Series, Transformation, and Distributions
    fig1 = plt.figure(figsize=(15, 10))
    gs = fig1.add_gridspec(2, 3)
    
    # Original time series
    ax1 = fig1.add_subplot(gs[0, :2])
    ax1.plot(ts, series, linewidth=1.5)
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of original series
    ax2 = fig1.add_subplot(gs[0, 2])
    series_pd.hist(bins=30, density=True, alpha=0.6, ax=ax2)
    series_pd.plot.kde(ax=ax2, color='red', linewidth=2)
    ax2.set_title('Distribution of Original Series')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    
    # Transformed series
    ax3 = fig1.add_subplot(gs[1, :2])
    ax3.plot(transformed.index, transformed.values, linewidth=1.5, color='green')
    ax3.set_title(f'{transform_label}')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # Histogram of transformed series
    ax4 = fig1.add_subplot(gs[1, 2])
    transformed.hist(bins=30, density=True, alpha=0.6, ax=ax4)
    transformed.plot.kde(ax=ax4, color='red', linewidth=2)
    ax4.set_title(f'Distribution of {transform_label}')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    figs.append(fig1)
    
    # Figure 2: Autocorrelation Analysis
    fig2 = plt.figure(figsize=(15, 10))
    gs = fig2.add_gridspec(2, 2)
    
    # ACF of original series
    ax1 = fig2.add_subplot(gs[0, 0])
    plot_autocorrelation(series_pd.values, max(aut_corr_lags), ax1, 'Autocorrelation of Original Series')
    
    # ACF of transformed series
    ax2 = fig2.add_subplot(gs[0, 1])
    plot_autocorrelation(transformed.values, max(aut_corr_lags), ax2, f'Autocorrelation of {transform_label}')
    
    # Lag plot of original series
    ax3 = fig2.add_subplot(gs[1, 0])
    pd.plotting.lag_plot(series_pd, lag=1, ax=ax3)
    ax3.set_title('Lag 1 Plot of Original Series')
    ax3.grid(True, alpha=0.3)
    
    # Lag plot of transformed series
    ax4 = fig2.add_subplot(gs[1, 1])
    pd.plotting.lag_plot(transformed, lag=1, ax=ax4)
    ax4.set_title(f'Lag 1 Plot of {transform_label}')
    ax4.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    figs.append(fig2)
    
    # Figure 3: Fourier Analysis
    fig3 = plt.figure(figsize=(15, 10))
    gs = fig3.add_gridspec(2, 2)
    
    # FFT of original series
    ax1 = fig3.add_subplot(gs[0, :])
    # Compute FFT
    series_fft = fft(series_pd.values)
    n = len(series_pd.values)
    freq = fftfreq(n, d=(ts[1]-ts[0]) if len(ts) > 1 else 1)
    
    # Plot only the positive frequencies (up to Nyquist frequency)
    positive_freq_idx = np.arange(1, n // 2)
    ax1.plot(freq[positive_freq_idx], 2.0/n * np.abs(series_fft[positive_freq_idx]), '-o', markersize=3)
    ax1.set_title('Fourier Transform of Original Series')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Log scale for better visualization
    ax1.set_yscale('log')
    
    # FFT of transformed series
    ax2 = fig3.add_subplot(gs[1, :])
    # Compute FFT
    if len(transformed) > 1:  # Ensure we have enough data points
        trans_fft = fft(transformed.values)
        n_trans = len(transformed.values)
        freq_trans = fftfreq(n_trans, d=(transformed.index[1]-transformed.index[0]) if len(transformed.index) > 1 else 1)
        
        # Plot only the positive frequencies
        positive_freq_idx_trans = np.arange(1, n_trans // 2)
        ax2.plot(freq_trans[positive_freq_idx_trans], 
                 2.0/n_trans * np.abs(trans_fft[positive_freq_idx_trans]), 
                 '-o', markersize=3, color='green')
        ax2.set_title(f'Fourier Transform of {transform_label}')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Log scale for better visualization
        ax2.set_yscale('log')
    
    fig3.tight_layout()
    figs.append(fig3)
    
    # Figure 4: Rolling Statistics and Decomposition
    fig4 = plt.figure(figsize=(15, 12))
    
    # Determine grid layout based on whether we'll include decomposition
    include_decomp = len(series) >= 2 * max(windows)
    if include_decomp:
        gs = fig4.add_gridspec(3, 2)
    else:
        gs = fig4.add_gridspec(2, 2)
    
    # Rolling mean for different windows
    ax1 = fig4.add_subplot(gs[0, 0])
    ax1.plot(ts, series, label='Original', alpha=0.5)
    for window in windows:
        rolling_mean = series_pd.rolling(window=window).mean()
        ax1.plot(rolling_mean.index, rolling_mean.values, 
                 label=f'Window={window}', linewidth=2)
    ax1.set_title('Rolling Mean')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rolling standard deviation for different windows
    ax2 = fig4.add_subplot(gs[0, 1])
    for window in windows:
        rolling_std = series_pd.rolling(window=window).std()
        ax2.plot(rolling_std.index, rolling_std.values, 
                 label=f'Window={window}', linewidth=2)
    ax2.set_title('Rolling Standard Deviation')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Rolling autocorrelation (lag 1) for different windows
    ax3 = fig4.add_subplot(gs[1, 0])
    for window in windows:
        # Calculate rolling autocorrelation
        rolling_autocorr = series_pd.rolling(window=window).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan)
        ax3.plot(rolling_autocorr.index, rolling_autocorr.values, 
                 label=f'Window={window}', linewidth=2)
    ax3.set_title('Rolling Autocorrelation (Lag 1)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Autocorrelation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Rolling volatility (for financial time series)
    ax4 = fig4.add_subplot(gs[1, 1])
    if non_negative:
        for window in windows:
            # Calculate rolling volatility (standard deviation of log returns)
            log_returns = np.log(series_pd / series_pd.shift(1)).dropna()
            rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            ax4.plot(rolling_vol.index, rolling_vol.values, 
                     label=f'Window={window}', linewidth=2)
        ax4.set_title('Rolling Volatility (Annualized)')
    else:
        for window in windows:
            # For non-financial series, show rolling coefficient of variation
            rolling_mean = series_pd.rolling(window=window).mean()
            rolling_std = series_pd.rolling(window=window).std()
            rolling_cv = rolling_std / rolling_mean
            ax4.plot(rolling_cv.index, rolling_cv.values, 
                     label=f'Window={window}', linewidth=2)
        ax4.set_title('Rolling Coefficient of Variation')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add seasonal decomposition if enough data
    if include_decomp:
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series_pd, model='additive', period=max(windows))
            
            # Plot decomposition components
            ax5 = fig4.add_subplot(gs[2, :])
            ax5.plot(decomposition.trend.index, decomposition.trend.values, label='Trend', linewidth=2)
            ax5.plot(decomposition.seasonal.index, decomposition.seasonal.values, label='Seasonal', linewidth=1, alpha=0.7)
            ax5.plot(decomposition.resid.index, decomposition.resid.values, label='Residual', linewidth=1, alpha=0.5)
            ax5.set_title('Seasonal Decomposition')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Value')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        except:
            pass
    
    fig4.tight_layout()
    figs.append(fig4)
    
    # Figure 5: Advanced Lag Analysis
    fig5 = plt.figure(figsize=(15, 12))
    gs = fig5.add_gridspec(3, 3)
    
    # Multiple lag plots for original series
    for i, lag in enumerate(aut_corr_lags[:min(9, len(aut_corr_lags))]):
        ax = fig5.add_subplot(gs[i//3, i%3])
        pd.plotting.lag_plot(series_pd, lag=lag, ax=ax)
        ax.set_title(f'Lag {lag} Plot')
        ax.grid(True, alpha=0.3)
    
    fig5.tight_layout()
    figs.append(fig5)
    
    return figs

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_correlation(x, y):
    """
    Calculate Pearson correlation coefficient manually to avoid NumPy type issues.
    
    Parameters:
    -----------
    x : list or pandas Series
        First variable
    y : list or pandas Series
        Second variable
        
    Returns:
    --------
    float
        Pearson correlation coefficient
    """
    # Convert to Python floats
    x = [float(val) for val in x if not pd.isna(val)]
    y = [float(val) for val in y if not pd.isna(val)]
    
    # Check if we have enough data
    if len(x) != len(y) or len(x) < 2:
        return None
    
    # Calculate means
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    # Calculate correlation
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)
    
    # Avoid division by zero
    if denominator_x == 0 or denominator_y == 0:
        return None
    
    correlation = numerator / ((denominator_x ** 0.5) * (denominator_y ** 0.5))
    return correlation

def plot_lagged_correlation_matrices(df, products, columns, max_lag):
    """
    Generate correlation matrices for different lags between all product-attribute combinations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe containing market data
    products : list
        List of product names to analyze
    columns : list
        List of column names to include in the correlation analysis
    max_lag : int
        Maximum lag to consider
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for each lag (including lag=0)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert DataFrame columns to native Python types
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Dictionary to store figures for each lag
    figures = {}
    
    # Create product-column combinations
    product_columns = []
    for product in products:
        for column in columns:
            product_columns.append((product, column))
    
    # Iterate through each lag value (including lag=0)
    for lag in range(0, max_lag + 1):
        # Prepare data for correlation matrix
        corr_data = {}
        
        # Sort dataframe by timestamp to ensure proper lag calculation
        df_sorted = df.sort_values(by=['day', 'timestamp'])
        
        # For each product-column combination, create a time series
        for product, column in product_columns:
            # Filter data for current product
            product_df = df_sorted[df_sorted['product'] == product].copy()
            
            # Skip if not enough data
            if len(product_df) <= lag:
                continue
            
            # Add the current column values to the correlation data
            key = f"{product}_{column}"
            corr_data[key] = product_df[column].reset_index(drop=True)
            
            # Add the lagged column values to the correlation data
            key_lagged = f"{product}_{column}_lag{lag}"
            corr_data[key_lagged] = product_df[column].shift(lag).reset_index(drop=True)
        
        # Convert to dataframe and drop rows with NaN
        corr_df = pd.DataFrame(corr_data)
        corr_df = corr_df.dropna()
        
        # Skip this lag if not enough data
        if len(corr_df) < 2:
            figures[lag] = go.Figure().add_annotation(
                text=f"Not enough data for lag {lag} after creating lagged features",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            continue
        
        # Calculate correlation matrix
        current_columns = [f"{product}_{column}" for product, column in product_columns]
        lagged_columns = [f"{product}_{column}_lag{lag}" for product, column in product_columns]
        
        # Calculate correlations between current and lagged columns
        corr_matrix = []
        for current_col in current_columns:
            row_corrs = []
            for lagged_col in lagged_columns:
                corr_value = calculate_correlation(
                    corr_df[current_col].tolist(), 
                    corr_df[lagged_col].tolist()
                )
                row_corrs.append(corr_value if corr_value is not None else 0)
            corr_matrix.append(row_corrs)
        
        # Create readable labels for the heatmap
        x_labels = [f"{col.split('_lag')[0]} (t-{lag})" for col in lagged_columns]
        y_labels = [f"{col} (t)" for col in current_columns]
        
        # Prepare data for improved visualization with product grouping
        
        # Create matrix for product grouping borders
        product_borders = []
        product_positions = {}
        
        # Get positions for each product in the matrix
        current_position = 0
        for product in products:
            product_start = current_position
            product_columns_count = sum(1 for pc in product_columns if pc[0] == product)
            current_position += product_columns_count
            product_positions[product] = (product_start, current_position - 1)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=x_labels,
            y=y_labels,
            colorscale='RdBu_r',  # Red-Blue scale, reversed so blue is positive
            zmid=0,  # Center the color scale at 0
            colorbar=dict(
                title="Correlation",
                titleside="right",
                len=0.8
            ),
            text=[[round(val, 2) for val in row] for row in corr_matrix],
            texttemplate="%{text:.2f}",
            textfont={"size": 10, "family": "Arial, sans-serif", "color": "black"},
        ))
        
        # Add product group rectangles
        for product in products:
            y_start, y_end = product_positions[product]
            
            for other_product in products:
                x_start, x_end = product_positions[other_product]
                
                # Add rectangle to highlight product groups
                if product == other_product:
                    # Same product correlation areas get a more prominent border
                    fig.add_shape(
                        type="rect",
                        x0=x_start - 0.5,
                        y0=y_start - 0.5,
                        x1=x_end + 0.5,
                        y1=y_end + 0.5,
                        line=dict(
                            color="black",
                            width=3,
                        ),
                        fillcolor="rgba(0,0,0,0)",
                        layer="below"
                    )
                    
                    # Add product name annotation in the middle of the rectangle
                    mid_x = (x_start + x_end) / 2
                    mid_y = (y_start + y_end) / 2
                    
                    if lag == 0:  # Only add product labels for lag 0 to avoid cluttering
                        fig.add_annotation(
                            x=x_labels[int(mid_x)],
                            y=y_labels[int(mid_y)],
                            text=f"<b>{product}</b>",
                            showarrow=False,
                            font=dict(
                                size=14,
                                color="black"
                            ),
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            opacity=0.8
                        )
                else:
                    # Different product correlation areas get a lighter border
                    fig.add_shape(
                        type="rect",
                        x0=x_start - 0.5,
                        y0=y_start - 0.5,
                        x1=x_end + 0.5,
                        y1=y_end + 0.5,
                        line=dict(
                            color="gray",
                            width=1,
                            dash="dash"
                        ),
                        fillcolor="rgba(0,0,0,0)",
                        layer="below"
                    )
        
        # Special title for lag 0
        title_text = "Auto-Correlation Matrix" if lag == 0 else f"Correlation Matrix - Lag {lag}"
        
        # Update layout with improved aesthetics
        fig.update_layout(
            title_text=title_text,
            title_font=dict(size=24, family="Arial, sans-serif"),
            height=max(800, 200 + 24 * len(current_columns)),  # Dynamic height based on number of items
            width=max(1000, 200 + 24 * len(lagged_columns)),   # Dynamic width based on number of items
            xaxis=dict(
                tickangle=-45, 
                tickfont=dict(size=10, family="Arial, sans-serif"),
                title_font=dict(size=14),
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                autorange="reversed",  # To match standard correlation matrix orientation
                tickfont=dict(size=10, family="Arial, sans-serif"),
                title_font=dict(size=14),
                gridcolor='lightgray',
                showgrid=True
            ),
            margin=dict(l=150, r=50, t=100, b=150),  # Add margins for readability
            plot_bgcolor='rgba(240,240,240,0.3)',    # Light background
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans-serif"
            )
        )
        
        figures[lag] = fig
    
    return figures





if __name__ == "__main__":
    # Run the market analysis for round 1 and include logs
    prices_df, trades_df, product_dfs = run_market_analysis(round_num=1, include_logs=True, plot_width=1200, plot_height=800)

    kelp, resin, squid = partition_prices(prices_df, ["KELP", "RESIN", "SQUID"])

    kelp["theo"] = kelp["weighted_mid_price"].ewm(span=3).mean()

    plot_market_data(kelp, trades_df, plot_width=1200, plot_height=800, features_to_plot=["theo", "weighted_mid_price"])