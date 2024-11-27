import numpy as np
import traceback

from numba import njit, uint64
from numba.typed import Dict

from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    GTX,
    LIMIT,
    BUY,
    SELL,
    BUY_EVENT,
    SELL_EVENT,
    Recorder
)
from hftbacktest.stats import LinearAssetRecord


def check_data(
    data_file='usdm/1000bonkusdt_20240730.npz',
    snapshot_file='usdm/1000bonkusdt_20240730_eod.npz', 
    latency_file='usdm/feed_latency_1000bonkusdt_20240730.npz'
):
    # Check the data file
    print(f"\nChecking {data_file}:")
    try:
        data = np.load(data_file)
        print("Arrays in the file:", data.files)
        for arr_name in data.files:
            print(f"{arr_name} shape:", data[arr_name].shape)
    except Exception as e:
        print(f"Error loading {data_file}:", e)

    # Check the EOD snapshot file
    print(f"\nChecking {snapshot_file}:")
    try:
        snapshot = np.load(snapshot_file)
        print("Arrays in the file:", snapshot.files)
        for arr_name in snapshot.files:
            print(f"{arr_name} shape:", snapshot[arr_name].shape)
    except Exception as e:
        print(f"Error loading {snapshot_file}:", e)

    # Check the latency file
    print(f"\nChecking {latency_file}:")
    try:
        latency = np.load(latency_file)
        print("Arrays in the file:", latency.files)
        for arr_name in latency.files:
            print(f"{arr_name} shape:", latency[arr_name].shape)
    except Exception as e:
        print(f"Error loading {latency_file}:", e)

@njit
def obi_mm(
    hbt,
    stat,
    half_spread,
    skew,
    c1,
    looking_depth,
    interval,
    window,
    order_qty_dollar,
    max_position_dollar,
    grid_num,
    grid_interval,
    roi_lb,
    roi_ub
):
    asset_no = 0
    imbalance_timeseries = np.full(1_000_000, np.nan, np.float64)  # Reduced size for testing

    tick_size = hbt.depth(0).tick_size
    lot_size = hbt.depth(0).lot_size

    t = 0
    roi_lb_tick = int(round(roi_lb / tick_size))
    roi_ub_tick = int(round(roi_ub / tick_size))

    try:
        while hbt.elapse(interval) == 0:
            print(f"Step {t}:")
            hbt.clear_inactive_orders(asset_no)
            depth = hbt.depth(asset_no)
            position = hbt.position(asset_no)
            orders = hbt.orders(asset_no)

            if depth is None or not hasattr(depth, 'best_bid') or not hasattr(depth, 'best_ask'):
                print("Invalid depth object.")
                continue

            best_bid = float(depth.best_bid)
            best_ask = float(depth.best_ask)
            mid_price = (best_bid + best_ask) / 2.0
            
            sum_ask_qty = 0.0
            from_tick = max(depth.best_ask_tick, roi_lb_tick)
            upto_tick = min(int(np.floor(mid_price * (1 + looking_depth) / tick_size)), roi_ub_tick)

            for price_tick in range(from_tick, upto_tick):
                sum_ask_qty += depth.ask_depth[price_tick - roi_lb_tick]
                    

            sum_bid_qty = 0.0
            from_tick = min(depth.best_bid_tick, roi_ub_tick)
            upto_tick = max(int(np.ceil(mid_price * (1 - looking_depth) / tick_size)), roi_lb_tick)

            for price_tick in range(from_tick, upto_tick, -1):
                sum_bid_qty += depth.bid_depth[price_tick - roi_lb_tick]

            imbalance_timeseries[t] = sum_bid_qty - sum_ask_qty
            
            # Standardizes the order book imbalance timeseries for a given window
            m = np.nanmean(imbalance_timeseries[max(0, t + 1 - window):t + 1])
            s = np.nanstd(imbalance_timeseries[max(0, t + 1 - window):t + 1]) 
            alpha = np.divide(imbalance_timeseries[t] - m, s)
        
            #--------------------------------------------------------
            # Computes bid price and ask price.
            
            order_qty = max(round((order_qty_dollar / mid_price) / lot_size) * lot_size, lot_size)
            fair_price = mid_price + c1 * alpha
            
            normalized_position = position / order_qty
            
            reservation_price = fair_price - skew * normalized_position

            bid_price = min(np.round(reservation_price - half_spread), best_bid)
            ask_price = max(np.round(reservation_price + half_spread), best_ask)
            
            bid_price = np.floor(bid_price / tick_size) * tick_size
            ask_price = np.ceil(ask_price / tick_size) * tick_size
            print(f"Step {t}: Position={position}, Best Bid={best_bid}, Best Ask={best_ask}, Mid Price={mid_price}")

            #--------------------------------------------------------
            # Updates quotes.
            
            # Creates a new grid for buy orders.
            new_bid_orders = Dict.empty(np.uint64, np.float64)
            if position * mid_price < max_position_dollar and np.isfinite(bid_price):
                for i in range(grid_num):
                    bid_price_tick = round(bid_price / tick_size)
                    
                    # order price in tick is used as order id.
                    new_bid_orders[uint64(bid_price_tick)] = bid_price
                    
                    bid_price -= grid_interval

            # Creates a new grid for sell orders.
            new_ask_orders = Dict.empty(np.uint64, np.float64)
            if position * mid_price > -max_position_dollar and np.isfinite(ask_price):
                for i in range(grid_num):
                    ask_price_tick = round(ask_price / tick_size)
                    
                    # order price in tick is used as order id.
                    new_ask_orders[uint64(ask_price_tick)] = ask_price

                    ask_price += grid_interval
                    
            order_values = orders.values();
            while order_values.has_next():
                order = order_values.get()
                # Cancels if a working order is not in the new grid.
                if order.cancellable:
                    if (
                        (order.side == BUY and order.order_id not in new_bid_orders)
                        or (order.side == SELL and order.order_id not in new_ask_orders)
                    ):
                        hbt.cancel(asset_no, order.order_id, False)
                        
            for order_id, order_price in new_bid_orders.items():
                # Posts a new buy order if there is no working order at the price on the new grid.
                if order_id not in orders:
                    hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)
                    
            for order_id, order_price in new_ask_orders.items():
                # Posts a new sell order if there is no working order at the price on the new grid.
                if order_id not in orders:
                    hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)
            
            t += 1
            
            if t >= len(imbalance_timeseries):
                print(f"Error: t={t} exceeds imbalance_timeseries size.")
                break

            
            # Records the current state for stat calculation.
            stat.record(hbt)
    except Exception:
        print(f"Error: {Exception}")


if __name__ == '__main__':
    # Strategy parameters
    check_data()
    print("General data health check completed.")
    print("Starting backtest...")
    symbol = "1000bonkusdt"
    date = "20240730"
    
    # Market making parameters
    half_spread = 10
    skew = 2
    c1 = 20
    depth = 0.001  # 0.1% from the mid price
    interval = 500_000_000  # 500ms
    window = 600_000_000_000 / interval  # 10min
    order_qty_dollar = 25_000
    max_position_dollar = order_qty_dollar * 20
    grid_num = 1
    roi_lb = 10000
    roi_ub = 50000
    
    # Load latency data
    latency_data = np.load(f'usdm/feed_latency_{symbol}_{date}.npz')['data']

    asset = (
        BacktestAsset()
            .data([f'usdm/{symbol}_{date}.npz'])
            .initial_snapshot(f'usdm/{symbol}_{date}_eod.npz')
            .linear_asset(1.0) 
            .intp_order_latency(latency_data)
            .power_prob_queue_model(2)
            .no_partial_fill_exchange()
            .trading_value_fee_model(-0.00005, 0.0007)
            .tick_size(0.1)
            .lot_size(0.001)
            .roi_lb(roi_lb)    
            .roi_ub(roi_ub)
    )

    hbt = ROIVectorMarketDepthBacktest([asset])
    recorder = Recorder(1, 30_000_000)
    grid_interval = hbt.depth(0).tick_size

    obi_mm(
        hbt,
        recorder.recorder,
        half_spread,
        skew,
        c1,
        depth,
        interval,
        window,
        order_qty_dollar,
        max_position_dollar,
        grid_num,
        grid_interval,
        roi_lb,
        roi_ub
    )

    # Save and analyze results
    output_file = f'stats/obi_{symbol}_{date}.npz'
    recorder.to_npz(output_file)

    data = np.load(output_file)['0']
    stats = (
        LinearAssetRecord(data)
            .resample('5m')
            .stats()
    )
    stats.summary()
    stats.plot()

