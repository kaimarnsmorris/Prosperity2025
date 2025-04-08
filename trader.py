from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle


# CONSTANTS
position_limits = {"RAINFOREST_RESIN" : 50, "KELP": 50, "SQUID_INK": 50}


def get_weighted_mid_price(order_depth: OrderDepth) -> float:
    total_weight = 0
    total_price = 0
    for price, amount in order_depth.buy_orders.items():
        total_weight += amount
        total_price += price * amount
    for price, amount in order_depth.sell_orders.items():
        total_weight -= amount # this value is negative
        total_price -= price * amount
    return total_price / total_weight if total_weight > 0 else 0



class Trader:
    def __init__(self):
        self.data = {}

    def trade_resin(self, state: TradingState, threshold=2, max_hedge_displacement=1.4) -> List[Order]:
        theo = 10000
        pos = self.get_position(state, "RAINFOREST_RESIN")
        pos_max = position_limits["RAINFOREST_RESIN"]

        orders: List[Order] = []

        # Hedge bias
        if abs(pos) > 4*pos_max / 5:
            theo = round(theo - max_hedge_displacement * (pos / pos_max))

        # Calculate available buying and selling capacity
        buy_capacity = pos_max - pos
        sell_capacity = pos_max + pos

        # Calculate order sizes for the two spreads (2/3 for first spread, 1/3 for second)
        first_buy_size = round(3 * buy_capacity / 4)
        second_buy_size = buy_capacity - first_buy_size
        
        first_sell_size = round(3 * sell_capacity / 4)
        second_sell_size = sell_capacity - first_sell_size

        # First spread (tighter, at theo±threshold)
        if first_buy_size > 0:
            orders.append(Order("RAINFOREST_RESIN", theo - threshold, first_buy_size))
        
        if first_sell_size > 0:
            orders.append(Order("RAINFOREST_RESIN", theo + threshold, -first_sell_size))
        
        # Second spread (wider, at theo±(threshold*2))
        wider_threshold = threshold * 2  # Make the second spread wider
        
        if second_buy_size > 0:
            orders.append(Order("RAINFOREST_RESIN", theo - wider_threshold, second_buy_size))
        
        if second_sell_size > 0:
            orders.append(Order("RAINFOREST_RESIN", theo + wider_threshold, -second_sell_size))

        return orders

    def trade_kelp(self, state: TradingState, threshold=1., max_hedge_displacement=0.7) -> List[Order]:
        order_depth = state.order_depths.get("KELP", OrderDepth())
        current_mid = get_weighted_mid_price(order_depth)
        
        # Historical prices
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
            historical_prices = data.get("kelp_prices", [])
        else:
            data = {}
            historical_prices = []
        
        historical_prices.append(current_mid)
        historical_prices = historical_prices[-2:]
        data["kelp_prices"] = historical_prices
        self.data = data
        
        # Calculate theo using EWM
        theo = 2000.0
        if len(historical_prices) >= 2:
            alpha = 2 / (3 + 1)
            theo = historical_prices[-1] * alpha + historical_prices[-2] * (1 - alpha)
        elif len(historical_prices) == 1:
            theo = historical_prices[0]
        
        pos = self.get_position(state, "KELP")
        pos_max = position_limits["KELP"]
        orders: List[Order] = []
        
        # Adjust theo based on current position
        theo -= max_hedge_displacement * (pos / pos_max)
        
        # Available capacity
        buy_capacity = pos_max - pos
        sell_capacity = pos_max + pos
        
        # Step 1: Take favorable prices that cross our theo (market taker)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            if best_ask < theo:
                # Take as much of the favorable ask as our capacity allows
                ask_volume = -order_depth.sell_orders[best_ask]  # Negative because sell order
                buy_volume = min(buy_capacity, ask_volume)
                if buy_volume > 0:
                    orders.append(Order("KELP", best_ask, buy_volume))
                    buy_capacity -= buy_volume
        
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            if best_bid > theo:
                # Take as much of the favorable bid as our capacity allows
                bid_volume = order_depth.buy_orders[best_bid]
                sell_volume = min(sell_capacity, bid_volume)
                if sell_volume > 0:
                    orders.append(Order("KELP", best_bid, -sell_volume))
                    sell_capacity -= sell_volume
        
        # Step 2: Use remaining capacity to place passive orders around theo (market maker)
        # First spread (tight)
        if buy_capacity > 0:
            orders.append(Order("KELP", round(theo - threshold), buy_capacity))
        
        if sell_capacity > 0:
            orders.append(Order("KELP", round(theo + threshold), -sell_capacity))
        
        # Second spread (wider)
        wider_threshold = threshold + 1
        second_buy_size = max(0, pos_max - pos - buy_capacity)
        second_sell_size = max(0, pos_max + pos - sell_capacity)
        
        if second_buy_size > 0:
            orders.append(Order("KELP", round(theo - wider_threshold), second_buy_size))
        
        if second_sell_size > 0:
            orders.append(Order("KELP", round(theo + wider_threshold), -second_sell_size))
        
        print("mid:", current_mid, "theo:", theo, "orders:", orders)
        return orders

    def trade_squid(self):
        return []

    def get_position(self, state: TradingState, product) -> dict:
        position = 0
        if product in state.position:
            position = state.position[product]
        return position

    
    def run(self, state: TradingState):

        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))

        # Deserialize traderData if it exists
        if state.traderData:
            self.data = jsonpickle.decode(state.traderData)
        else:
            self.data = {}

        result = {}

        result["RAINFOREST_RESIN"] = []# self.trade_resin(state)
        result["KELP"] = self.trade_kelp(state)  # Make sure to pass state
        result["SQUID_INK"] = self.trade_squid()
        
        # Serialize data to traderData
        traderData = jsonpickle.encode(self.data)
        
        conversions = 1
        return result, conversions, traderData