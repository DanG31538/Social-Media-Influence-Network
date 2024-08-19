import os

# Get the directory of the current file (tests directory)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the test dataset
TEST_DATASET_PATH = os.path.join(TESTS_DIR, 'test_instagram_data.csv')

# Path to the main dataset (for reference)
MAIN_DATASET_PATH = os.path.join(TESTS_DIR, '..', 'dataset', 'instagram_data.csv')

# You can add more configuration variables here as needed