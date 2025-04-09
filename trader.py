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

    def trade_kelp(self, state: TradingState, threshold=0.5, max_hedge_displacement=0.7) -> List[Order]:
        # Get order depth for KELP
        order_depth = state.order_depths.get("KELP", OrderDepth())
        
        # Calculate current weighted mid price
        current_mid = get_weighted_mid_price(order_depth)
        
        # Initialize or retrieve historical prices from traderData
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
            if "kelp_prices" in data:
                historical_prices = data["kelp_prices"]
            else:
                historical_prices = []
        else:
            data = {}
            historical_prices = []
        
        historical_prices.append(current_mid)
        
        # Keep only last 2 prices
        historical_prices = historical_prices[-2:]
        
        # Store updated prices back to data dictionary
        data["kelp_prices"] = historical_prices
        
        # Store data back to traderData in the run method
        self.data = data
        
        # Calculate theo using EWM with span=3
        theo = 2000.0
        if len(historical_prices) >= 2:
            # Simple implementation of EWM with span=3
            alpha = 2/(3+1)  # alpha = 2/(span+1)
            theo = (historical_prices[-1] * alpha + historical_prices[-2] * (1-alpha))
        elif len(historical_prices) == 1:
            theo = (historical_prices[0])
        
        pos = self.get_position(state, "KELP")
        pos_max = position_limits["KELP"]

        orders: List[Order] = []

        # Hedge bias
        #if abs(pos) > 3*pos_max / 4:
        theo = (theo - max_hedge_displacement * (pos / pos_max))

        # Calculate available buying and selling capacity
        buy_capacity = pos_max - pos
        sell_capacity = pos_max + pos

        # Calculate order sizes for the two spreads (3/4 for first spread, 1/4 for second)
        first_buy_size = buy_capacity # round(4* buy_capacity / 4)
        second_buy_size = buy_capacity - first_buy_size
        
        first_sell_size = sell_capacity #round(4 * sell_capacity / 4)
        second_sell_size = sell_capacity - first_sell_size

        # First spread (tighter, at theo±threshold)
        if first_buy_size > 0:
            orders.append(Order("KELP", round(theo - threshold), first_buy_size))
        
        if first_sell_size > 0:
            orders.append(Order("KELP", round(theo + threshold), -first_sell_size))
        





        # OLD STRATEGY
        WINDOW_STARFRUIT = 8 # 6
        orders = []
        new_mean = get_weighted_mid_price(order_depth)

        if state.timestamp == 0:
            means_list = [0]*(WINDOW_STARFRUIT-1) + [new_mean]
            data["KELP"] = means_list
            return orders
        
        means_list = jsonpickle.decode(state.traderData)["KELP"]
        
        means_list.append(new_mean)
        data["KELP"] = means_list[1:]  # "{:.2f}".format(number)
        
        if state.timestamp < WINDOW_STARFRUIT*100:
            return orders
        
        def exp_mean(data):
            n = len(data)
            weights = [1]*int(n/3) + [2]*(n - int(n/3))
            return sum([data[i]*weights[i] for i in range(n)])/sum(weights)
        
        mean = exp_mean(means_list) # sum(means_list)/len(means_list)

        offset = float(pos)/50

        buy_band = mean - 0.5 - offset
        sell_band = mean + 0.5 - offset

        orders.append(Order("KELP", round(buy_band), pos_max-pos))
        orders.append(Order("KELP", round(sell_band), -pos_max-pos))
        print("position: " + str(pos) + " buy_band: " + str(buy_band) + " sell_band: " + str(sell_band))
        
        self.data = data

        return orders

    def trade_squid(self):
        return []

    def get_position(self, state: TradingState, product) -> dict:
        position = 0
        if product in state.position:
            position = state.position[product]
        return position

    
    def run(self, state: TradingState):

        print("traderData: " + state.traderData)
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