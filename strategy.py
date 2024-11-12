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
            imbalance = abs(row['imbalance'])  # Use absolute value for allocation magnitude
            rolling_sd = row['rolling_sd']
            
            # Determine threshold based on volatility
            threshold = 0.05 if rolling_sd > self.historical_sd * self.historical_sd_multiplier else 0.03
            
            # Calculate allocation in fractions of 1/20
            allocation_percentage = min((imbalance // threshold) * (1/20), 1)  # Cap at 100%
            allocations.append(allocation_percentage)
        
        return allocations

    def determine_position(self, df):
        # Define position based on the threshold and imbalance
        positions = []
        
        for _, row in df.iterrows():
            imbalance = row['imbalance']
            rolling_sd = row['rolling_sd']
            
            # Determine the appropriate threshold
            threshold = 0.05 if rolling_sd > self.historical_sd * self.historical_sd_multiplier else 0.03
            
            # Determine position based on imbalance and threshold
            if imbalance > threshold:
                positions.append('long')
            elif imbalance < -threshold:
                positions.append('short')
            else:
                positions.append('nothing')
        
        return positions

    def get_trend_change_rows(self, df):
        # Identify rows where the position changes from the previous row
        trend_change_rows = df[df['position'] != df['position'].shift()]
        return trend_change_rows

    def run(self, data):
        # Calculate imbalance and rolling standard deviation
        self.calculate_imbalance(data)
        self.rolling_std(data)
        
        # Determine allocation percentages
        data['allocation'] = self.allocate_based_on_imbalance(data)
        
        # Determine position based on allocation and imbalance
        data['position'] = self.determine_position(data)
        
        # Get rows where a trend change occurs
        trend_change_rows = self.get_trend_change_rows(data)
        
        # Drop intermediate columns for cleaner output
        data.drop(['bid_volume', 'ask_volume', 'rolling_bid_volume', 'rolling_ask_volume'], axis=1, inplace=True)
        
        return data[['timestamp', 'price', 'quantity', 'type', 'imbalance', 'rolling_sd', 'allocation', 'position']], trend_change_rows

# Load the data
df = pd.read_pickle('pickles/adausdt_20241112_104855.pkl')
df = df.head(10000)
print(f"Successfully loaded DataFrame with {len(df)} rows")

# Instantiate and run the strategy
allocator = ImbalanceAllocator(window_size=500)  # Example window size
result_df, trend_change_rows = allocator.run(df)

# Print the last 100 rows of the result with trend changes
pd.set_option('display.max_rows', 100)  # Control maximum rows displayed in a single print

# Get mean imbalance
mean_imbalance = result_df['imbalance'].mean()
print(f"Mean imbalance: {mean_imbalance}")

# Show trend change rows
print("Trend change rows:")
print(trend_change_rows)
