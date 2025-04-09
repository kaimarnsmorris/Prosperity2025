import pandas as pd
import numpy as np
import os
import glob
import json
import re
import plotly.graph_objects as go



class DataHandler:

    # -- Data Loading Functions --
    @staticmethod
    def load_log_file(file_path) -> pd.DataFrame:
        """Load and parse a log file into three dataframes (sandbox_logs, activities_log, trade_history)."""
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Split the file into sections
        sections = re.split(r'\n\n+(?:Sandbox logs:|Activities log:|Trade History:)\n', content)
        assert len(sections) == 3, "Expected 3 sections in the log file"

        sandbox_logs = DataHandler.parse_sandbox_logs(sections[0])
        prices_log = DataHandler.parse_prices(sections[1])
        trade_history = DataHandler.parse_trade_history(sections[2])

        return sandbox_logs, prices_log, trade_history

    @staticmethod
    def load_historical_round(round_num, base_dir="round-{}-island-data-bottle") -> pd.DataFrame:
        """Load historical data for a given round number. Will return two dataframes: prices and trades (concatenated from the three days)."""

        round_dir = base_dir.format(round_num)

        price_pattern = os.path.join(round_dir, f"prices_round_{round_num}_day_*.csv")
        trade_pattern = os.path.join(round_dir, f"trade_history_round_{round_num}_day_*.json")
        price_files = glob.glob(price_pattern)
        trade_files = glob.glob(trade_pattern)
        assert len(price_files) == len(trade_files), "Mismatch in number of price and trade files"

        prices_dfs = []
        for file in price_files:
            with open(file, 'r') as file:
                content = file.read()

            df = DataHandler.parse_prices(content)
            prices_dfs.append(df)

        prices_df = pd.concat(prices_dfs, ignore_index=True)
        
        trades_dfs = []
        for file in trade_files:
            with open(file, 'r') as file:
                content = file.read()

            df = DataHandler.parse_trade_history(content)
            trades_dfs.append(df)

        trades_df = pd.concat(trades_dfs, ignore_index=True)

        return prices_df, trades_df
        

    # -- Parsing Functions --
    @staticmethod
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

    @staticmethod
    def parse_prices(activities_text):
        """Parse activities log into a DataFrame with proper types."""
        lines = activities_text.strip().split('\n')
        header = lines[0].split(';')
        
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.split(';')
                data.append(values)
        
        df = pd.DataFrame(data, columns=header)
        
        for col in df.columns:
            if col != "product":
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    @staticmethod
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



    @staticmethod
    def get_most_recent_log():
        """Returns the path to the most recently modified log file."""
        list_of_files = glob.glob('logs/*')
        return max(list_of_files, key=os.path.getctime) if list_of_files else None




