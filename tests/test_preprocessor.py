import unittest
import pandas as pd
from data_prep.preprocessor import DataPreprocessor
from test_config import TEST_DATASET_PATH

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.test_df = pd.read_csv(TEST_DATASET_PATH)

    def test_preprocess_instagram_data(self):
        processed_df = self.preprocessor.preprocess_instagram_data(self.test_df)
        
        self.assertIsNotNone(processed_df)
        self.assertIsInstance(processed_df, pd.DataFrame)
        
        # Check if created_at is converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['created_at']))
        
        # Check if is_video is boolean
        self.assertTrue(pd.api.types.is_bool_dtype(processed_df['is_video']))
        
        # Check if comments and likes are integers
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['comments']))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['likes']))
        
        # Check if location is processed (assuming it's converted to a string)
        self.assertTrue(pd.api.types.is_string_dtype(processed_df['location']))
        
        # Print the first few rows of the processed dataset
        print("First few rows of the processed dataset:")
        print(processed_df.head().to_string())

        # Print column info
        print("\nColumn information after preprocessing:")
        print(processed_df.info())

if __name__ == '__main__':
    unittest.main()