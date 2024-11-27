# Import necessary libraries
import polars as pl
import gzip
from numba import njit
import numpy as np
import time
from hftbacktest.data.utils.snapshot import create_last_snapshot
from hftbacktest.data.utils import tardis
from hftbacktest.data.utils.binancefutures import convert as binance_convert


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
    create_last_snapshot(
        input_files,
        tick_size=tick_size,
        lot_size=lot_size,
        output_snapshot_filename=output_snapshot_file,
        initial_snapshot=initial_snapshot_file,
    )
    if output_snapshot_file:
        print(f"Snapshot saved to {output_snapshot_file}")


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

    # Load the feed file
    data = np.load(feed_file)['data']
    df = pl.DataFrame(data)

    # Filter rows based on specific `ev` values (adjust as needed)
    valid_events = [3758096385, 3489660929]  # Add all relevant values here
    df = df.filter(pl.col('ev').is_in(valid_events))
    print("Filtered Data Shape:", df.shape)

    if df.shape[0] == 0:
        print("No valid data found after filtering. Exiting latency generation.")
        return None

    # Continue processing
    df = df.with_columns(
        pl.col('local_ts').alias('ts')
    ).group_by_dynamic(
        'ts', every='1000000000i'
    ).agg(
        pl.col('exch_ts').last(),
        pl.col('local_ts').last()
    ).drop('ts')

    # Convert to a structured array
    data = df.to_numpy(structured=True)
    order_latency = np.zeros(len(data), dtype=[('req_ts', 'i8'), ('exch_ts', 'i8'), ('resp_ts', 'i8')])

    # Generate latency data
    generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp)

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

    # Create market snapshot using the converted file
    create_market_snapshot(
        input_files=[f"usdm/{name}.npz"],
        tick_size=0.1,
        lot_size=0.001,
        output_snapshot_file=f"usdm/{name}_eod.npz",
    )

process_latency_data(
    feed_file=f"usdm/{name}.npz",
    latency_output_file=f"usdm/feed_latency_{name}.npz"
)