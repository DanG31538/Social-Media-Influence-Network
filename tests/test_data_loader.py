import os
import unittest
import pandas as pd
import numpy as np
from data_prep.data_loader import DataLoader
from test_config import TEST_DATASET_PATH

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(os.path.dirname(TEST_DATASET_PATH))

    def test_load_instagram_data(self):
        df = self.data_loader.load_instagram_data('test_instagram_data.csv')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check if the loaded data has the same shape as the test dataset
        test_df = pd.read_csv(TEST_DATASET_PATH)
        self.assertEqual(df.shape, test_df.shape)
        
        # Check if essential columns are present
        essential_columns = ['owner_id', 'owner_username', 'shortcode', 'is_video', 'caption', 'comments', 'likes', 'created_at']
        for col in essential_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(df['owner_id']))
        self.assertTrue(pd.api.types.is_string_dtype(df['owner_username']))
        self.assertTrue(pd.api.types.is_string_dtype(df['shortcode']))
        self.assertTrue(pd.api.types.is_bool_dtype(df['is_video']))
        self.assertTrue(pd.api.types.is_string_dtype(df['caption']) or pd.api.types.is_object_dtype(df['caption']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['comments']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['likes']))
        self.assertTrue(pd.api.types.is_datetime64_dtype(df['created_at']))

        # Print the first few rows of the test dataset
        print("First few rows of the test dataset:")
        print(df.head().to_string())

        # Print column info
        print("\nColumn information:")
        print(df.info())

if __name__ == '__main__':
    unittest.main()