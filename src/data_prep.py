import pandas as pd
import glob
import os


def process_milan_traffic(raw_dir: str, output_path: str):
    """Loads, optimizes, and aggregates daily Milan telecom files."""
    
    # The dataset lacks headers. These are the official columns.
    columns = ['square_id', 'time_interval', 'country_code', 
               'sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
    
    # We only care about these three columns for forecasting
    use_cols = ['square_id', 'time_interval', 'internet']
    
    # Downcasting types to save massive amounts of RAM
    # square_id max is 10,000 (fits in uint16)
    # time_interval is epoch milliseconds (needs uint64)
    # internet is a decimal (float32 is plenty of precision)
    dtypes = {
        'square_id': 'uint16',
        'time_interval': 'uint64',
        'internet': 'float32'
    }
    
    file_paths = sorted(glob.glob(f"{raw_dir}/*.txt"))
    daily_frames = []
    
    print(f"Found {len(file_paths)} daily files. Beginning processing...")
    
    for file in file_paths:
        # 1. Load only what we need
        df = pd.read_csv(file, sep='\t', header=None, names=columns, 
                         usecols=use_cols, dtype=dtypes)
        
        # 2. Fill missing internet traffic with 0
        df['internet'] = df['internet'].fillna(0)
        
        # 3. Aggregate: The raw data splits traffic by 'country_code'. 
        # We must sum it to get total traffic per square per 10-minute interval.
        daily_agg = df.groupby(['square_id', 'time_interval'], as_index=False)['internet'].sum()
        
        daily_frames.append(daily_agg)
        print(f"Processed: {os.path.basename(file)}")

    # 4. Combine all 62 days into one master DataFrame
    print("Concatenating all days...")
    master_df = pd.concat(daily_frames, ignore_index=True)
    
    # 5. Convert epoch timestamps to readable datetimes (Milan is UTC+1)
    print("Converting timestamps...")
    master_df['datetime'] = pd.to_datetime(master_df['time_interval'], unit='ms')
    master_df = master_df.drop(columns=['time_interval'])
    
    # 6. Save as Parquet (compresses 20GB down to a few hundred MBs)
    print(f"Saving optimized dataset to {output_path}...")
    master_df.to_parquet(output_path, index=False)
    print("Done!")

# Get the absolute paths to the root of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the paths to the raw and processed data
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cdrs')
OUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'milan_internet_traffic.parquet')

# Create the processed folder if it doesn't exist yet
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

process_milan_traffic(RAW_DIR, OUT_FILE)