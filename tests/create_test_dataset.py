import pandas as pd
import os

# Set the paths
main_dataset_path = r'C:\Users\DanTh\Documents\Coding\Within Take Home\dataset\instagram_data.csv'
test_data_dir = 'tests'
test_dataset_path = os.path.join(test_data_dir, 'test_instagram_data.csv')

# Create test_data directory if it doesn't exist
os.makedirs(test_data_dir, exist_ok=True)

# Read the main dataset
df = pd.read_csv(main_dataset_path)

# Create a small subset (e.g., 1% of the data or 100 rows, whichever is smaller)
sample_size = min(100, int(len(df) * 0.01))
test_df = df.sample(n=sample_size, random_state=42)  # Using random_state for reproducibility

# Save the test dataset
test_df.to_csv(test_dataset_path, index=False)

print(f"Test dataset created at {test_dataset_path}")
print(f"Number of rows in test dataset: {len(test_df)}")