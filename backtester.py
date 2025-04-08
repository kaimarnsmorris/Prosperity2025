import pandas as pd
import numpy as np
from datamodel import TradingState, OrderDepth, Listing, Observation, ConversionObservation, Trade
from trader import Trader
from typing import Dict, List, Tuple


class Backtester:
    """
    Backtester for trading strategies using historical price data.
    """
    
    def __init__(self, prices_df: pd.DataFrame, position_limits: Dict[str, int] = None):
        """
        Initialize the backtester with price data.
        
        Args:
            prices_df: DataFrame containing historical price data 
                       with columns including 'product', 'timestamp', 'bid_price_1', 'ask_price_1', etc.
            position_limits: Dictionary mapping products to their position limits
        """
        self.prices_df = prices_df.copy()
        self.trader = Trader()
        self.position_limits = position_limits or {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        
        # Initialize portfolio tracking
        self.positions = {product: 0 for product in self.position_limits.keys()}
        self.cash = 0
        self.trades_history = []
        self.pnl_history = []
        
        # Group data by product
        self.product_dfs = {product: group.reset_index(drop=True) 
                           for product, group in self.prices_df.groupby('product')}
        
    def create_order_depth(self, row: pd.Series) -> OrderDepth:
        """Create an OrderDepth object from a row of price data."""
        order_depth = OrderDepth()
        
        # Add buy orders
        if 'bid_price_1' in row and pd.notna(row['bid_price_1']) and 'bid_volume_1' in row and pd.notna(row['bid_volume_1']):
            order_depth.buy_orders[int(row['bid_price_1'])] = int(row['bid_volume_1'])
        if 'bid_price_2' in row and pd.notna(row['bid_price_2']) and 'bid_volume_2' in row and pd.notna(row['bid_volume_2']):
            order_depth.buy_orders[int(row['bid_price_2'])] = int(row['bid_volume_2'])
        if 'bid_price_3' in row and pd.notna(row['bid_price_3']) and 'bid_volume_3' in row and pd.notna(row['bid_volume_3']):
            order_depth.buy_orders[int(row['bid_price_3'])] = int(row['bid_volume_3'])
            
        # Add sell orders with negative volumes
        if 'ask_price_1' in row and pd.notna(row['ask_price_1']) and 'ask_volume_1' in row and pd.notna(row['ask_volume_1']):
            order_depth.sell_orders[int(row['ask_price_1'])] = -int(row['ask_volume_1'])
        if 'ask_price_2' in row and pd.notna(row['ask_price_2']) and 'ask_volume_2' in row and pd.notna(row['ask_volume_2']):
            order_depth.sell_orders[int(row['ask_price_2'])] = -int(row['ask_volume_2'])
        if 'ask_price_3' in row and pd.notna(row['ask_price_3']) and 'ask_volume_3' in row and pd.notna(row['ask_volume_3']):
            order_depth.sell_orders[int(row['ask_price_3'])] = -int(row['ask_volume_3'])
            
        return order_depth
    
    def create_trading_state(self, timestamp: int, current_data: Dict[str, pd.Series]) -> TradingState:
        """
        Create a TradingState object from the current price data.
        
        Args:
            timestamp: Current timestamp
            current_data: Dictionary mapping product names to their current price row
            
        Returns:
            TradingState object
        """
        # Create listings
        listings = {}
        order_depths = {}
        
        for product, row in current_data.items():
            # Create listing (symbol is the same as product for simplicity)
            listings[product] = Listing(symbol=product, product=product, denomination="USD")
            
            # Create order depth
            order_depths[product] = self.create_order_depth(row)
        
        # Create observations (simplistic - just using price data)
        # In a real scenario, you would have more complex observation data
        plain_value_observations = {}
        conversion_observations = {}
        
        for product, row in current_data.items():
            # For this example, we're creating a dummy ConversionObservation
            # In a real scenario, these values would come from market data
            conversion_observations[product] = ConversionObservation(
                bidPrice=float(row['bid_price_1']) if pd.notna(row['bid_price_1']) else 0.0,
                askPrice=float(row['ask_price_1']) if pd.notna(row['ask_price_1']) else 0.0,
                transportFees=1.0,
                exportTariff=0.5,
                importTariff=0.5,
                sugarPrice=5.0,
                sunlightIndex=0.8
            )
            
            # Plain value observations (simplified)
            plain_value_observations[product] = int(row['mid_price']) if pd.notna(row['mid_price']) else 0
        
        observations = Observation(plainValueObservations=plain_value_observations, 
                                   conversionObservations=conversion_observations)
        
        # Create TradingState
        state = TradingState(
            traderData="",  # Will be updated by trader's output
            timestamp=timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades={product: [] for product in current_data.keys()},
            market_trades={product: [] for product in current_data.keys()},
            position=self.positions.copy(),
            observations=observations
        )
        
        return state
    
    def execute_trades(self, orders: Dict[str, List], current_data: Dict[str, pd.Series]) -> List[Trade]:
        """
        Execute trades based on orders and current market prices.
        
        Args:
            orders: Dictionary mapping products to lists of Order objects
            current_data: Dictionary mapping products to their current price row
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for product, product_orders in orders.items():
            if product not in current_data:
                continue
                
            row = current_data[product]
            
            for order in product_orders:
                # Skip invalid orders
                if order.quantity == 0:
                    continue
                    
                # Buy order
                if order.quantity > 0:
                    # Check if there's a matching sell price in the market
                    if 'ask_price_1' in row and pd.notna(row['ask_price_1']) and order.price >= row['ask_price_1']:
                        # Execute at market ask price
                        trade_price = int(row['ask_price_1'])
                        available_volume = int(row['ask_volume_1']) if pd.notna(row['ask_volume_1']) else 0
                        trade_quantity = min(order.quantity, available_volume)
                        
                        if trade_quantity > 0:
                            # Create trade
                            trade = Trade(
                                symbol=product,
                                price=trade_price,
                                quantity=trade_quantity,
                                buyer="SUBMISSION",
                                seller="MARKET",
                                timestamp=row['timestamp'] if 'timestamp' in row else 0
                            )
                            
                            # Update position and cash
                            self.positions[product] += trade_quantity
                            self.cash -= trade_price * trade_quantity
                            
                            # Record trade
                            executed_trades.append(trade)
                            
                # Sell order
                elif order.quantity < 0:
                    # Check if there's a matching buy price in the market
                    if 'bid_price_1' in row and pd.notna(row['bid_price_1']) and order.price <= row['bid_price_1']:
                        # Execute at market bid price
                        trade_price = int(row['bid_price_1'])
                        available_volume = int(row['bid_volume_1']) if pd.notna(row['bid_volume_1']) else 0
                        trade_quantity = min(-order.quantity, available_volume)
                        
                        if trade_quantity > 0:
                            # Create trade
                            trade = Trade(
                                symbol=product,
                                price=trade_price,
                                quantity=trade_quantity,
                                buyer="MARKET",
                                seller="SUBMISSION",
                                timestamp=row['timestamp'] if 'timestamp' in row else 0
                            )
                            
                            # Update position and cash
                            self.positions[product] -= trade_quantity
                            self.cash += trade_price * trade_quantity
                            
                            # Record trade
                            executed_trades.append(trade)
                            
        return executed_trades
    
    def calculate_portfolio_value(self, current_data: Dict[str, pd.Series]) -> float:
        """Calculate the total portfolio value based on current positions and mid prices."""
        value = self.cash
        
        for product, position in self.positions.items():
            if product in current_data and 'mid_price' in current_data[product] and pd.notna(current_data[product]['mid_price']):
                value += position * current_data[product]['mid_price']
                
        return value
    
    def run(self, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
        """
        Run the backtest over the specified range of the price data.
        
        Args:
            start_idx: Starting index in the prices dataframe
            end_idx: Ending index in the prices dataframe (or None for all data)
            
        Returns:
            DataFrame containing backtest results
        """
        if end_idx is None:
            end_idx = len(self.prices_df)
            
        # Reset portfolio tracking
        self.positions = {product: 0 for product in self.position_limits.keys()}
        self.cash = 0
        self.trades_history = []
        self.pnl_history = []
        
        # Group price data by timestamp to process all products at each step
        timestamp_groups = self.prices_df.groupby(['day', 'timestamp'])
        
        trader_data = ""  # Initial trader state data
        
        for (day, timestamp), group in timestamp_groups:
            # Skip if outside specified range
            current_idx = group.index[0]
            if current_idx < start_idx or current_idx >= end_idx:
                continue
                
            # Get current price data for each product
            current_data = {}
            for _, row in group.iterrows():
                product = row['product']
                current_data[product] = row
                
            # Create trading state
            state = self.create_trading_state(int(timestamp), current_data)
            state.traderData = trader_data
            
            # Run trader's strategy
            orders_dict, conversions, trader_data = self.trader.run(state)
            
            # Execute trades
            executed_trades = self.execute_trades(orders_dict, current_data)
            self.trades_history.extend(executed_trades)
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(current_data)
            self.pnl_history.append({
                'day': day,
                'timestamp': timestamp,
                'cash': self.cash,
                'portfolio_value': portfolio_value,
                'trades': len(executed_trades)
            })
            
        # Create results dataframe
        results_df = pd.DataFrame(self.pnl_history)
        
        # Add position columns
        for product in self.position_limits.keys():
            results_df[f'{product}_position'] = [self.positions[product]] * len(results_df)
            
        return results_df
    
    def get_trades_summary(self) -> pd.DataFrame:
        """Create a summary of all executed trades."""
        if not self.trades_history:
            return pd.DataFrame()
            
        trades_data = []
        for trade in self.trades_history:
            trades_data.append({
                'symbol': trade.symbol,
                'price': trade.price,
                'quantity': trade.quantity,
                'timestamp': trade.timestamp,
                'is_buy': trade.buyer == "SUBMISSION",
                'is_sell': trade.seller == "SUBMISSION",
                'value': trade.price * trade.quantity
            })
            
        return pd.DataFrame(trades_data)
    
    def plot_results(self):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            
            # Create PnL chart
            results_df = pd.DataFrame(self.pnl_history)
            
            plt.figure(figsize=(12, 8))
            
            # Plot portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(results_df['portfolio_value'], label='Portfolio Value')
            plt.title('Backtest Performance')
            plt.xlabel('Time Step')
            plt.ylabel('Portfolio Value')
            plt.legend()
            
            # Plot position for each product
            plt.subplot(2, 1, 2)
            for product in self.positions.keys():
                product_trades = [t for t in self.trades_history if t.symbol == product]
                if not product_trades:
                    continue
                    
                positions = []
                position = 0
                for trade in product_trades:
                    if trade.buyer == "SUBMISSION":
                        position += trade.quantity
                    else:
                        position -= trade.quantity
                    positions.append(position)
                    
                plt.plot(positions, label=f'{product} Position')
                
            plt.xlabel('Trade #')
            plt.ylabel('Position')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting. Install with 'pip install matplotlib'")
            

# Example usage
def run_backtest(prices_df, days_to_backtest=None):
    """
    Run a backtest using the provided price data.
    
    Args:
        prices_df: DataFrame containing price data
        days_to_backtest: Number of days to backtest (None for all data)
        
    Returns:
        Tuple of (backtester, results_df)
    """
    # Filter data for specified days if needed
    if days_to_backtest is not None:
        unique_days = sorted(prices_df['day'].unique())
        if len(unique_days) > days_to_backtest:
            days_to_use = unique_days[:days_to_backtest]
            prices_df = prices_df[prices_df['day'].isin(days_to_use)]
    
    # Create and run backtester
    backtester = Backtester(prices_df)
    results_df = backtester.run()
    
    # Print summary statistics
    print(f"Backtest completed with {len(backtester.trades_history)} trades")
    print(f"Final portfolio value: {results_df['portfolio_value'].iloc[-1]:.2f}")
    print(f"Final positions: {backtester.positions}")
    
    # Return backtester and results
    return backtester, results_df


if __name__ == "__main__":
    # Example code for when run directly
    import os
    import sys
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import utility functions for loading data
    from utils import read_all_prices_data
    
    # Load data
    print("Loading price data...")
    prices_df = read_all_prices_data(round_num=1, include_logs=False)
    
    # Run backtest
    backtester, results = run_backtest(prices_df, days_to_backtest=2)
    
    # Plot results
    backtester.plot_results()