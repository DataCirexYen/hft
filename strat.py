import numpy as np
import polars as pl
from numba import njit
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY, SELL, GTX, LIMIT

@njit
def market_making_algo(hbt):
    asset_no = 0
    
    # Main simulation loop
    while True:
        # Elapse time, returns 0 if successful
        if hbt.elapse(10_000_000) != 0:
            break
        
        # Get market depth
        depth = hbt.depth(asset_no)
        
        # Check if market data is valid
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            continue
        
        # Calculate mid price
        mid_price = (depth.best_bid + depth.best_ask) / 2.0
        
        # Basic market-making parameters
        tick_size = depth.tick_size
        lot_size = depth.lot_size
        
        # Current position
        position = hbt.position(asset_no)
        
        # Define spread and order size
        half_spread = 0.0001  # 1 basis point spread
        order_qty = lot_size  # Use full lot size
        
        # Calculate bid and ask prices
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread
        
        # Convert prices to ticks
        bid_tick = int(bid_price / tick_size)
        ask_tick = int(ask_price / tick_size)
        
        # Clear existing orders
        hbt.clear_inactive_orders(asset_no)
        
        # Submit new orders
        # Buy order
        buy_order_id = bid_tick
        hbt.submit_buy_order(
            asset_no, 
            buy_order_id, 
            bid_tick * tick_size, 
            order_qty, 
            GTX,  # Good-til-cancel 
            LIMIT,  # Limit order
            False  # Not a post-only order
        )
        
        # Sell order
        sell_order_id = ask_tick
        hbt.submit_sell_order(
            asset_no, 
            sell_order_id, 
            ask_tick * tick_size, 
            order_qty, 
            GTX, 
            LIMIT, 
            False
        )
        
        # Wait for order responses
        timeout = 5_000_000_000  # 5 seconds in nanoseconds
        if not (hbt.wait_order_response(asset_no, buy_order_id, timeout) and 
                hbt.wait_order_response(asset_no, sell_order_id, timeout)):
            break
    
    return True

def run_backtest():
    # Backtest configuration
    symbol = "1000bonkusdt"
    date = "20240730"
    
    try:
        # Load market data
        market_data = np.load(f'usdm/{symbol}_{date}.npz')['data']
        
        # Convert to Polars DataFrame for initial inspection
        df = pl.DataFrame(market_data)
        print("Market Data Overview:")
        print(df.head())
        
        # Create BacktestAsset with detailed configuration
        asset = (
            BacktestAsset()
            .data([f'usdm/{symbol}_{date}.npz'])
            .initial_snapshot(f'usdm/{symbol}_{date}_eod.npz')
            .linear_asset(1.0)  # Linear asset type
            .intp_order_latency(
                np.load(f'usdm/feed_latency_{symbol}_{date}.npz')['data']
            )
            .power_prob_queue_model(2.0)  # Queue model 
            .no_partial_fill_exchange()  # Exchange fill model
            .trading_value_fee_model(-0.00005, 0.0007)  # Fee model
            .tick_size(0.00001)  # Minimum price movement
            .lot_size(1)  # Minimum trade quantity
        )
        
        # Create backtest environment
        hbt = HashMapMarketDepthBacktest([asset])
        
        # Run market-making algorithm
        result = market_making_algo(hbt)
        
        print(f"Backtest Completed. Result: {result}")
        
    except Exception as e:
        import traceback
        print(f"Backtest Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    run_backtest()