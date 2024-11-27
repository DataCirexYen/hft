# Import necessary libraries
import polars as pl
import gzip
from numba import njit
import numpy as np
import time
from hftbacktest.data.utils.snapshot import create_last_snapshot
from hftbacktest.data.utils import tardis
from hftbacktest.data.utils.binancefutures import convert as binance_convert



def analyze_and_recommend_params(input_files):
    """
    Analyze .npz input files to recommend tick_size and lot_size for market snapshot.

    Args:
        input_files (list): List of input .npz files.

    Returns:
        dict: Recommendations for `tick_size` and `lot_size`.
    """
    print("Analyzing input files for parameter recommendations...\n")

    all_prices = []
    all_quantities = []

    for file_path in input_files:
        print(f"Inspecting {file_path}:")
        data = np.load(file_path)['data']
        
        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")
        print("Sample rows:")
        print(data[:5])

        prices = data['px']
        quantities = data['qty']
        
        all_prices.extend(prices)
        all_quantities.extend(quantities)

        print(f"Price range in {file_path}: min={prices.min()}, max={prices.max()}, std={prices.std()}")
        print(f"Quantity range in {file_path}: min={quantities.min()}, max={quantities.max()}, std={quantities.std()}")
        print("-" * 50)

    all_prices = np.array(all_prices)
    all_quantities = np.array(all_quantities)

    print("\nAggregated Data Analysis Across All Files:")
    print(f"Overall price range: min={all_prices.min()}, max={all_prices.max()}, std={all_prices.std()}")
    print(f"Overall quantity range: min={all_quantities.min()}, max={all_quantities.max()}, std={all_quantities.std()}")

    tick_size = all_prices.std() / 10
    lot_size = all_quantities.std() / 10

    print("\nRecommended Parameters:")
    print(f"tick_size: {tick_size}")
    print(f"lot_size: {lot_size}")

    return {"tick_size": tick_size, "lot_size": lot_size}


def convert_binance_data(input_file, output_file, combined_stream=True):
    """
    Converts Binance futures data.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        combined_stream (bool): Whether to use combined streams.
    """
    print("Converting Binance Futures data...")
    binance_convert(
        input_file,
        output_filename=output_file,
        combined_stream=combined_stream,
    )
    print(f"Data saved to {output_file}")


def create_market_snapshot(input_files, tick_size, lot_size, output_snapshot_file=None, initial_snapshot_file=None):
    """
    Creates a market depth snapshot.

    Args:
        input_files (list): List of input files for snapshot creation.
        tick_size (float): Tick size for price.
        lot_size (float): Lot size for quantity.
        output_snapshot_file (str): Output file for the snapshot.
        initial_snapshot_file (str): Initial snapshot for reference (optional).
    """
    print("Creating market snapshot...")
    print("Input files:", input_files)
    
    for input_file in input_files:
        data = np.load(input_file)['data']
        print(f"{input_file}:")
        print(f"Shape: {data.shape}")
        print(f"dtype: {data.dtype}")
        print("Sample data:")
        print(data[:3])
        print("-" * 50)

    print("Tick size:", tick_size)
    print("Lot size:", lot_size)

    create_last_snapshot(
        input_files,
        tick_size=tick_size,
        lot_size=lot_size,
        output_snapshot_filename=output_snapshot_file,
        initial_snapshot=initial_snapshot_file,
    )
    
    if output_snapshot_file:
        snapshot_data = np.load(output_snapshot_file)['data']
        print(f"Snapshot saved to {output_snapshot_file}")
        print(f"Snapshot dtype: {snapshot_data.dtype}")
        print(f"Snapshot shape: {snapshot_data.shape}")
        print("Sample snapshot data:")
        print(snapshot_data[:3])


def convert_tardis_data(trade_file, book_file, output_file=None, buffer_size=200_000_000):
    """
    Converts data from Tardis.dev.

    Args:
        trade_file (str): Path to the trades file.
        book_file (str): Path to the book file.
        output_file (str): Path to save the converted data.
        buffer_size (int): Buffer size for large files.
    """
    print("Converting Tardis.dev data...")
    data = tardis.convert(
        [trade_file, book_file],
        output_filename=output_file,
        buffer_size=buffer_size,
    )
    if output_file:
        print(f"Converted data saved to {output_file}")
    return data



@njit
def generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp):
    """
    Generate order latency data using numba for performance.

    Args:
        data: Structured array with exchange and local timestamps.
        order_latency: Output array for storing latency data.
        mul_entry: Multiplier for entry latency.
        offset_entry: Offset for entry latency.
        mul_resp: Multiplier for response latency.
        offset_resp: Offset for response latency.
    """
    for i in range(len(data)):
        exch_ts = data[i].exch_ts
        local_ts = data[i].local_ts
        feed_latency = local_ts - exch_ts
        order_entry_latency = mul_entry * feed_latency + offset_entry
        order_resp_latency = mul_resp * feed_latency + offset_resp

        req_ts = local_ts
        order_exch_ts = req_ts + order_entry_latency
        resp_ts = order_exch_ts + order_resp_latency

        # Assign each field explicitly
        order_latency['req_ts'][i] = req_ts
        order_latency['exch_ts'][i] = order_exch_ts
        order_latency['resp_ts'][i] = resp_ts
def generate_order_latency(feed_file, output_file=None, mul_entry=1, offset_entry=0, mul_resp=1, offset_resp=0):
    print("Loading feed file:", feed_file)

    # Load the feed file and preserve the structured dtype
    data = np.load(feed_file)['data']
    
    # Filter rows based on specific `ev` values directly using numpy
    mask = np.isin(data['ev'], [3758096385, 3489660929])
    filtered_data = data[mask]
    print("Filtered Data Shape:", filtered_data.shape)

    if filtered_data.shape[0] == 0:
        print("No valid data found after filtering. Exiting latency generation.")
        return None

    # Group by timestamp (every second) using numpy operations
    timestamps = filtered_data['local_ts']
    bins = timestamps // 1_000_000_000  # Convert to seconds
    unique_bins = np.unique(bins)
    
    # Get last row for each second
    grouped_data = np.zeros(len(unique_bins), 
                          dtype=[('exch_ts', '<i8'), ('local_ts', '<i8')])
    
    for i, bin_val in enumerate(unique_bins):
        mask = bins == bin_val
        bin_data = filtered_data[mask]
        grouped_data[i]['exch_ts'] = bin_data['exch_ts'][-1]
        grouped_data[i]['local_ts'] = bin_data['local_ts'][-1]

    # Create output array
    order_latency = np.zeros(len(grouped_data), 
                           dtype=[('req_ts', 'i8'), ('exch_ts', 'i8'), ('resp_ts', 'i8')])

    # Generate latency data
    generate_order_latency_nb(grouped_data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp)

    if output_file is not None:
        np.savez_compressed(output_file, data=order_latency)
        print(f"Order latency data saved to {output_file}")

    return order_latency

def process_latency_data(feed_file, latency_output_file):
    """
    Wrapper function to process latency data.

    Args:
        feed_file: Path to the feed file to process.
        latency_output_file: Path to save the latency data file.
    """
    generate_order_latency(
        feed_file=feed_file,
        output_file=latency_output_file,
        mul_entry=4,
        mul_resp=3
    )


if __name__ == "__main__":
    # Convert Binance data
    name = "1000bonkusdt_20240730"
    convert_binance_data(
        input_file=f"usdm/{name}.gz",
        output_file=f"usdm/{name}.npz",
    )
    best_params = analyze_and_recommend_params([f"usdm/{name}.npz"])
    # Create market snapshot using the converted file
    create_market_snapshot(
        input_files=[f"usdm/{name}.npz"],
        tick_size=best_params["tick_size"],
        lot_size=best_params["lot_size"],
        output_snapshot_file=f"usdm/{name}_eod.npz",
    )

process_latency_data(
    feed_file=f"usdm/{name}.npz",
    latency_output_file=f"usdm/feed_latency_{name}.npz"
)