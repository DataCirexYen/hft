# Import necessary libraries
import polars as pl
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


if __name__ == "__main__":
    # Convert Binance data
    input_file = "usdm/1000bonkusdt_20240730.gz"
    converted_file = "usdm/1000bonkusdt_20240730.npz"
    
    convert_binance_data(
        input_file=input_file,
        output_file=converted_file,
    )

    # Create market snapshot using the converted file
    create_market_snapshot(
        input_files=[converted_file],  # Use the file we just converted
        tick_size=0.1,
        lot_size=0.001,
        output_snapshot_file=f"{converted_file}_eod.npz",
    )

