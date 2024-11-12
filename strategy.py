import pandas as pd

class ImbalanceAllocator:
    def __init__(self, window_size, historical_sd_multiplier=1.2):
        self.window_size = window_size
        self.historical_sd_multiplier = historical_sd_multiplier
        self.historical_sd = None  # To store the historical standard deviation

    def calculate_imbalance(self, df):
        # Calculate rolling imbalance using the specified window size
        df['bid_volume'] = df.apply(lambda x: x['quantity'] if x['type'] == 'bid' else 0, axis=1)
        df['ask_volume'] = df.apply(lambda x: x['quantity'] if x['type'] == 'ask' else 0, axis=1)
        
        # Calculate rolling sum of bid and ask volumes
        df['rolling_bid_volume'] = df['bid_volume'].rolling(self.window_size).sum()
        df['rolling_ask_volume'] = df['ask_volume'].rolling(self.window_size).sum()

        # Calculate rolling imbalance
        df['imbalance'] = (df['rolling_bid_volume'] - df['rolling_ask_volume']) / (
            df['rolling_bid_volume'] + df['rolling_ask_volume']
        )
        df['imbalance'].fillna(0, inplace=True)  # Replace NaN values with zero

    def rolling_std(self, df):
        # Calculate rolling standard deviation of the imbalance
        df['rolling_sd'] = df['imbalance'].rolling(self.window_size).std()
        
        # Set historical standard deviation if not set
        if self.historical_sd is None:
            self.historical_sd = df['imbalance'].std()

    def allocate_based_on_imbalance(self, df):
        allocations = []
        
        for _, row in df.iterrows():
            imbalance = row['imbalance']
            rolling_sd = row['rolling_sd']
            
            # Calculate allocation increment based on volatility
            if rolling_sd > self.historical_sd * self.historical_sd_multiplier:
                increment = 0.05  # 5% if volatility is high
            else:
                increment = 0.03  # 3% if volatility is normal

            # Calculate allocation percentage based on imbalance
            allocation_percentage = min(abs(imbalance) // increment * (1/20), 1)  # Cap allocation at 100%
            allocations.append(allocation_percentage)
        
        return allocations

    def determine_position(self, df):
        # Define position based on imbalance and allocation
        positions = []
        
        for _, row in df.iterrows():
            if row['allocation'] > 0.0:  # Arbitrary threshold to take a position
                if row['imbalance'] > 0:
                    positions.append('long')
                elif row['imbalance'] < 0:
                    positions.append('short')
                else:
                    positions.append('nothing')
            else:
                positions.append('nothing')
        
        return positions

    def run(self, data):
        # Calculate imbalance and rolling standard deviation
        self.calculate_imbalance(data)
        self.rolling_std(data)
        
        # Determine allocation percentages
        data['allocation'] = self.allocate_based_on_imbalance(data)
        
        # Determine position based on allocation and imbalance
        data['position'] = self.determine_position(data)
        
        # Drop intermediate columns for cleaner output
        data.drop(['bid_volume', 'ask_volume', 'rolling_bid_volume', 'rolling_ask_volume'], axis=1, inplace=True)
        return data[['timestamp', 'price', 'quantity', 'type', 'imbalance', 'rolling_sd', 'allocation', 'position']]

# Load the data
df = pd.read_pickle('pickles/adausdt_20241112_104855.pkl')
df = df.head(10000)
print(f"Successfully loaded DataFrame with {len(df)} rows")

# Instantiate and run the strategy
allocator = ImbalanceAllocator(window_size=500)  # Example window size
result_df = allocator.run(df)

# Print the result
print(result_df.head(10000))  # Display the first 10,000 rows of the resulting DataFrame
