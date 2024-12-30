import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('water_potability.csv')

# Split data
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define directory paths
data_path = os.path.join('data', 'raw')

# Create the directory if it does not exist
os.makedirs(data_path, exist_ok=True)

# Save train and test datasets
train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
