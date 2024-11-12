import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import re
import os
import time
from tqdm import tqdm  # for progress bars
import gzip
import pandas as pd
from typing import List, Optional
import json

class FileDownloader:
    def __init__(self, base_url, download_dir="downloads", delay_seconds=0):
        self.base_url = base_url.rstrip("/")
        self.download_dir = download_dir
        self.delay_seconds = delay_seconds
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Setup session with retries and headers
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Add headers to look more like a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_file_links(self):
        """Fetches .gz file links from the base URL."""
        try:
            print("Fetching file list...")
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            # Extract all .gz links
            file_links = re.findall(r'href="(.*?\.gz)"', response.text)
            links = [self.base_url + "/" + link for link in file_links if link.endswith(".gz")]
            print(f"Found {len(links)} total files")
            return links
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing URL {self.base_url}")
            print(f"Error details: {str(e)}")
            return []
    
    def filter_links_by_coin(self, coin_name):
        """Filters file links to only include those containing the specified coin name."""
        all_links = self.fetch_file_links()
        if not all_links:
            print("No links were fetched. Please check if the base URL is correct.")
            return []
        coin_links = [link for link in all_links if f"{coin_name.lower()}" in link.lower()]
        return coin_links
    
    def download_files(self, coin_name):
        """Downloads files for a specified coin."""
        coin_links = self.filter_links_by_coin(coin_name)
        
        if not coin_links:
            print(f"No files found for coin: {coin_name}")
            return
        
        print(f"Found {len(coin_links)} files for {coin_name}")
        
        for file_url in coin_links:
            filename = os.path.join(self.download_dir, file_url.split("/")[-1])
            
            # Skip if file already exists
            if os.path.exists(filename):
                print(f"Skipping {filename} (already exists)")
                continue
                
            try:
                # Get file size first
                response = self.session.get(file_url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Show progress bar while downloading
                print(f"\nDownloading: {filename}")
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
                
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = file.write(chunk)
                        progress_bar.update(size)
                progress_bar.close()
                
                if self.delay_seconds > 0:
                    print(f"Waiting {self.delay_seconds} seconds before next download...")
                    time.sleep(self.delay_seconds)
                    
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_url}")
                print(f"Error details: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error while downloading {file_url}")
                print(f"Error details: {str(e)}")
                continue
        
        print(f"\nDownload process completed for {coin_name.upper()}")

class DataProcessor:
    def __init__(self, data_dir: str = "downloads"):
        """
        Initialize the DataProcessor.
        
        Args:
            data_dir (str): Directory where the .gz files are stored
        """
        self.data_dir = data_dir
        
    def process_gz_files(self, coin_name: str) -> Optional[pd.DataFrame]:
        """
        Process all .gz files for a specific coin.
        First checks if a pickle file exists, if so reads that instead.
        Otherwise processes .gz files and saves as pickle.
        """
        # Check for existing pickle file
        pickle_dir = "pickles"
        if os.path.exists(pickle_dir):
            pickle_files = [f for f in os.listdir(pickle_dir) if f.startswith(coin_name) and f.endswith('.pkl')]
            if pickle_files:
                # Get most recent pickle file
                latest_pickle = max(pickle_files)
                pickle_path = os.path.join(pickle_dir, latest_pickle)
                try:
                    print(f"Found existing pickle file: {latest_pickle}")
                    df = pd.read_pickle(pickle_path)
                    print(f"Successfully loaded DataFrame from pickle with {len(df)} rows")
                    return df
                except Exception as e:
                    print(f"Error reading pickle file: {str(e)}")
                    print("Proceeding to process .gz files...")

        # If no pickle exists or failed to read, process .gz files
        # Get all .gz files for the specified coin
        gz_files = [f for f in os.listdir(self.data_dir) 
                    if f.endswith('.gz') and coin_name.lower() in f.lower()]
        
        if not gz_files:
            print(f"No .gz files found for {coin_name}")
            return None
        
        dataframes = []
        
        for gz_file in gz_files:
            gz_path = os.path.join(self.data_dir, gz_file)
            txt_path = gz_path[:-3]  # Remove .gz extension
            
            try:
                # Decompress the .gz file
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(txt_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Read the txt file
                df = self._read_trade_file(txt_path)
                if df is not None:
                    dataframes.append(df)
                
                # Clean up: remove both .gz and .txt files
                os.remove(gz_path)
                os.remove(txt_path)
                print(f"Processed and cleaned up: {gz_file}")
                
            except Exception as e:
                print(f"Error processing {gz_file}: {str(e)}")
                continue
        
        if dataframes:
            # Concatenate all DataFrames
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp')
            
            # Save as pickle
            self._save_as_pickle(combined_df, coin_name)
            
            return combined_df
        else:
            print("No data was processed")
            return None
    
    def _read_trade_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read a trade data file containing market depth data.
        Handles both depthUpdate and bookTicker messages.
        """
        try:
            df = pd.read_csv(file_path, 
                            sep=' ', 
                            header=None, 
                            names=['timestamp', 'json_data'],
                            dtype={'timestamp': str, 'json_data': str})
            
            rows = []
            for index, row in df.iterrows():
                try:
                    data = json.loads(row['json_data'])
                    market_data = data['data']
                    event_type = market_data['e']
                    
                    if event_type == 'depthUpdate':
                        # Handle depth update messages
                        # Process bids
                        for bid in market_data.get('b', []):
                            rows.append({
                                'timestamp': row['timestamp'],
                                'event_type': 'depthUpdate',
                                'event_time': market_data['E'],
                                'transaction_time': market_data['T'],
                                'symbol': market_data['s'],
                                'update_id': market_data['u'],
                                'type': 'bid',
                                'price': float(bid[0]),
                                'quantity': float(bid[1])
                            })
                        
                        # Process asks
                        for ask in market_data.get('a', []):
                            rows.append({
                                'timestamp': row['timestamp'],
                                'event_type': 'depthUpdate',
                                'event_time': market_data['E'],
                                'transaction_time': market_data['T'],
                                'symbol': market_data['s'],
                                'update_id': market_data['u'],
                                'type': 'ask',
                                'price': float(ask[0]),
                                'quantity': float(ask[1])
                            })
                    
                    elif event_type == 'bookTicker':
                        # Handle bookTicker messages
                        rows.append({
                            'timestamp': row['timestamp'],
                            'event_type': 'bookTicker',
                            'event_time': market_data['E'],
                            'transaction_time': market_data['T'],
                            'symbol': market_data['s'],
                            'update_id': market_data['u'],
                            'type': 'bid',
                            'price': float(market_data['b']),
                            'quantity': float(market_data['B'])
                        })
                        
                        rows.append({
                            'timestamp': row['timestamp'],
                            'event_type': 'bookTicker',
                            'event_time': market_data['E'],
                            'transaction_time': market_data['T'],
                            'symbol': market_data['s'],
                            'update_id': market_data['u'],
                            'type': 'ask',
                            'price': float(market_data['a']),
                            'quantity': float(market_data['A'])
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at row {index}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Error processing row {index}: {str(e)}")
                    continue
            
            if not rows:
                print(f"No valid data found in {file_path}")
                return None
            
            # Create final DataFrame
            result_df = pd.DataFrame(rows)
            
            # Set proper data types
            result_df = result_df.astype({
                'timestamp': 'int64',
                'event_type': 'str',
                'event_time': 'int64',
                'transaction_time': 'int64',
                'symbol': 'str',
                'update_id': 'int64',
                'type': 'str',
                'price': 'float64',
                'quantity': 'float64'
            })
            
            print(f"Successfully processed {len(result_df)} order book updates")
            return result_df
                
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _save_as_pickle(self, df: pd.DataFrame, coin_name: str, max_rows: int = 0) -> None:
        """
        Save DataFrame as a pickle file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            coin_name (str): Name of the coin for the filename
            max_rows (int): Maximum number of rows to save. If 0, save all rows.
        """
        try:
            # Create a directory for pickles if it doesn't exist
            pickle_dir = "pickles"
            os.makedirs(pickle_dir, exist_ok=True)
            
            # Cut dataframe if max_rows specified
            if max_rows > 0:
                df = df.head(max_rows)
            
            # Generate filename with timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{coin_name}_{timestamp}.pkl"
            filepath = os.path.join(pickle_dir, filename)
            
            # Save DataFrame as pickle
            df.to_pickle(filepath)
            print(f"Successfully saved DataFrame to {filepath}")
            
            # Optional: Print file size
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
            print(f"File size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"Error saving pickle file: {str(e)}")

def main():
    def download_data(coin_name: str):
        base_url = "https://reach.stratosphere.capital/data/usdm/"
        downloader = FileDownloader(base_url, delay_seconds=1)

        coin_name = "adausdt"
        downloader.download_files(coin_name)

    processor = DataProcessor()
    dataframes = processor.process_gz_files("adausdt")
    
    if dataframes:
        print(f"Successfully processed {len(dataframes)} files")
        # Do something with the dataframes
        for i, df in enumerate(dataframes):
            print(f"\nPreview of dataframe {i+1}:")
            print(df.head())
    else:
        print("No data was processed")

if __name__ == "__main__":
    main()