import pandas as pd
import numpy as np
import os
import glob
import json
import re
import plotly.graph_objects as go
from io import StringIO
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
import io


class DataHandler:

    # -- Data Loading Functions --
    @staticmethod
    def load_log_file(file_path, convert_price_logs=False) -> tuple:
        """Load and parse a log file into three dataframes (sandbox_logs, activities_log, trade_history)."""
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Split the file into sections
        sections = re.split(r'\n\n+(?:Sandbox logs:|Activities log:|Trade History:)\n', content)
        assert len(sections) == 3, "Expected 3 sections in the log file"

        # Extract day as first column 3 lines into Activities log (before semicolon)
        day = int(sections[1].split('\n')[3].split(';')[0].split('_')[-1])

        # Parse the sections in parallel
        sandbox_logs = DataHandler.parse_sandbox_logs(sections[0])
        prices_log = DataHandler.parse_prices(sections[1])
        trade_history = DataHandler.parse_trade_history(sections[2], day=day, input_format='json')
        
        if convert_price_logs and not sandbox_logs.empty and 'lambdaLog' in sandbox_logs.columns:
            # Pre-process sandbox logs to extract product data - this is more efficient
            # than iterating through each row multiple times
            processed_data = []
            
            for _, log_entry in sandbox_logs.iterrows():
                if not log_entry.get('lambdaLog'):
                    continue
                    
                log_ts = log_entry.get('timestamp')
                if log_ts is None:
                    continue
                    
                # Extract product data between equals signs
                equals_matches = re.findall(r'=(.*?)=', log_entry['lambdaLog'], re.DOTALL)
                for equals_data in equals_matches:
                    try:
                        product_data = json.loads(equals_data)
                        if isinstance(product_data, dict):
                            processed_data.append((log_ts, product_data))
                    except (json.JSONDecodeError, Exception):
                        continue
            
            if processed_data and 'timestamp' in prices_log.columns:
                # Create lookup for prices_log timestamps to avoid repeated calculations
                prices_log_timestamps = prices_log['timestamp'].values
                product_set = set(prices_log['product'].values)
                
                # Create dictionary to store new column data
                new_columns = {}
                
                # Process all the extracted data in a single pass
                for log_ts, product_data in processed_data:
                    # Find closest timestamp index once
                    closest_idx = np.abs(prices_log_timestamps - log_ts).argmin()
                    
                    for product, data in product_data.items():
                        if product not in product_set:
                            continue
                            
                        if isinstance(data, dict):
                            for key, value in data.items():
                                col_name = key
                                if col_name not in new_columns:
                                    new_columns[col_name] = [None] * len(prices_log)
                                new_columns[col_name][closest_idx] = value
                        else:
                            col_name = f"{product.lower()}_value"
                            if col_name not in new_columns:
                                new_columns[col_name] = [None] * len(prices_log)
                            new_columns[col_name][closest_idx] = data
                
                # Add all new columns to prices_log at once
                for col_name, values in new_columns.items():
                    prices_log[col_name] = values

        return sandbox_logs, prices_log, trade_history


    @staticmethod
    def load_historical_round(round_num, base_dir="round-{}-island-data-bottle") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical data for a given round number and return two DataFrames: prices and trades."""
        round_dir = base_dir.format(round_num)

        # Use glob to find files matching the patterns
        price_files = glob.glob(os.path.join(round_dir, f"prices_round_{round_num}_day_*.csv"))
        trade_files = glob.glob(os.path.join(round_dir, f"trades_round_{round_num}_day_*.csv"))
        days = [int(file.split('_')[-1].split('.')[0]) for file in price_files]

        assert len(price_files) == len(trade_files), "Mismatch in number of price and trade files"

        prices_df = DataHandler.concat_ordered(
            [DataHandler.parse_prices(open(file).read()) for file in price_files],
        )
        
        trades_df = DataHandler.concat_ordered(
            [DataHandler.parse_trade_history(open(trade_files[i]).read(), day=days[i], input_format='csv') for i in range(len(trade_files))],
        )

        return prices_df, trades_df

        

    # -- Parsing Functions --
    @staticmethod
    def parse_sandbox_logs(logs_text) -> pd.DataFrame:
        """Parse sandbox logs section into a DataFrame."""
        logs_text = logs_text.replace('Sandbox logs:\n','').strip()
        
        # Add square brackets and comma separation to make it a valid JSON array
        logs_text = re.sub(r'}\s*{', '},{', logs_text.strip())
        if not logs_text.startswith('['):
            logs_text = '[' + logs_text
        if not logs_text.endswith(']'):
            logs_text = logs_text + ']'

        log_entries = json.loads(logs_text)

        # Convert to a pandas DataFrame
        return pd.DataFrame(log_entries)

    @staticmethod
    def parse_prices(activities_text) -> pd.DataFrame:
        """Parse activities log into a DataFrame with proper types."""
        df = pd.read_csv(StringIO(activities_text), sep=';')
        
        # Convert all columns except 'product' to numeric
        for col in df.columns:
            if col != 'product':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    @staticmethod
    def parse_trade_history(trade_text, day=None, input_format='json') -> pd.DataFrame:
        """Parse trade history JSON into a DataFrame."""
        df = pd.DataFrame()
        if input_format == 'json':
            trades = json.loads(trade_text)
            df = pd.DataFrame(trades)
        elif input_format == 'csv':
            # Handle CSV format if needed
            df = pd.read_csv(StringIO(trade_text), sep=';')

        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['day'] = day
        columns_order = ['day'] + [col for col in df.columns if col != 'day']
        
        return df[columns_order]



    # -- Other --

    @staticmethod
    def concat_ordered(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames into one sorted by day, thentimestamp."""
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(by=['day', 'timestamp'], ascending=[True, True])

        return combined_df

    @staticmethod
    def get_most_recent_log(folder='submissions') -> str:
        """Returns the path to the most recently modified log file."""
        list_of_files = glob.glob(f'{folder}/*')
        return max(list_of_files, key=os.path.getctime) if list_of_files else None

class TimeSeriesAnalysis:
    pass

if __name__ == "__main__":
    # Example usage
    log_file_path = DataHandler.get_most_recent_log()
    if log_file_path:
        sandbox_logs, prices_log, trade_history = DataHandler.load_log_file(log_file_path, convert_price_logs=True)
        print("Sandbox Logs:")
        print(sandbox_logs.head())
        print("\nPrices Log:")
        print(prices_log.head())
        print("\nTrade History:")
        print(trade_history.head())
    else:
        print("No log files found.")








