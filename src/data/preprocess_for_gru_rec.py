import pandas as pd
import numpy as np
from datetime import datetime

# Load MovieLens 1M data
# Download from: https://grouplens.org/datasets/movielens/1m/
ratings = pd.read_csv('../../data/raw/ratings.dat', 
                      sep='::', 
                      engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      dtype={'user_id': np.int32, 'movie_id': np.int32, 
                             'rating': np.float32, 'timestamp': np.int32})

# Filter out low ratings (optional - keep only ratings >= 4 to indicate interest)
ratings = ratings[ratings['rating'] >= 4.0]

# Sort by user and timestamp
ratings = ratings.sort_values(['user_id', 'timestamp'])

# Rename columns to match GRU4Rec format
ratings = ratings.rename(columns={
    'user_id': 'SessionId',
    'movie_id': 'ItemId',
    'timestamp': 'Time'
})

# Keep only required columns
data = ratings[['SessionId', 'ItemId', 'Time']]

# Remove sessions with only 1 interaction
session_lengths = data.groupby('SessionId').size()
valid_sessions = session_lengths[session_lengths >= 2].index
data = data[data['SessionId'].isin(valid_sessions)]

# Split into train and test sets (last day of each session for test)
def train_test_split_by_time(data, test_days=1):
    # Get the last timestamp for each session
    session_max_time = data.groupby('SessionId')['Time'].max()
    
    train_list = []
    test_list = []
    
    for session_id in data['SessionId'].unique():
        session_data = data[data['SessionId'] == session_id].copy()
        max_time = session_max_time[session_id]
        
        # Use last interaction for test
        n_items = len(session_data)
        if n_items >= 2:
            train_data = session_data.iloc[:-1]
            test_data = session_data.iloc[-1:]
            
            if len(train_data) > 0:
                train_list.append(train_data)
                test_list.append(test_data)
    
    train = pd.concat(train_list)
    test = pd.concat(test_list)
    
    return train, test

# Create train_full and test split
train_full, test = train_test_split_by_time(data)

# Create train_tr and train_valid from train_full for hyperparameter tuning
train_tr, train_valid = train_test_split_by_time(train_full)

# Save to TSV files
train_full.to_csv('../../data/movielens1m_train_full.tsv', sep='\t', index=False)
test.to_csv('../../data/movielens1m_test.tsv', sep='\t', index=False)
train_tr.to_csv('../../data/movielens1m_train_tr.tsv', sep='\t', index=False)
train_valid.to_csv('../../data/movielens1m_train_valid.tsv', sep='\t', index=False)

print(f"Train full: {len(train_full)} interactions, {train_full['SessionId'].nunique()} sessions")
print(f"Test: {len(test)} interactions, {test['SessionId'].nunique()} sessions")
print(f"Train TR: {len(train_tr)} interactions, {train_tr['SessionId'].nunique()} sessions")
print(f"Train Valid: {len(train_valid)} interactions, {train_valid['SessionId'].nunique()} sessions")