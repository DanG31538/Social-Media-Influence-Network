import unittest
import pandas as pd
import numpy as np
from data_prep.data_normalizer import DataNormalizer
from data_prep.preprocessor import DataPreprocessor
from test_config import TEST_DATASET_PATH

class TestDataNormalizer(unittest.TestCase):
     
    def setUp(self):
        self.normalizer = DataNormalizer()
        self.preprocessor = DataPreprocessor()
        self.test_df = pd.read_csv(TEST_DATASET_PATH)
        self.processed_df = self.preprocessor.preprocess_instagram_data(self.test_df)

    def test_normalize_data(self):
        normalized_df = self.normalizer.normalize_data(self.processed_df)
        
        self.assertIsNotNone(normalized_df)
        self.assertIsInstance(normalized_df, pd.DataFrame)
        
        # Check if numeric columns are normalized
        numeric_columns = ['comments', 'likes', 'followers', 'following']
        for col in numeric_columns:
            if col in normalized_df.columns:
                self.assertTrue(-10 < normalized_df[col].mean() < 10)  # Loose check for centering
                self.assertTrue(0 < normalized_df[col].std() < 10)  # Loose check for scaling
        
        # Check if boolean columns are converted to integers
        if 'is_video' in normalized_df.columns:
            self.assertTrue(set(normalized_df['is_video'].unique()).issubset({0, 1}))
        
        # Check if 'created_at' is normalized
        if 'created_at' in normalized_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(normalized_df['created_at']))
        
        # Check if 'caption' is excluded from normalization
        if 'caption' in normalized_df.columns:
            self.assertTrue(pd.api.types.is_string_dtype(normalized_df['caption']) or 
                            pd.api.types.is_object_dtype(normalized_df['caption']))

    def test_inverse_transform(self):
        normalized_df = self.normalizer.normalize_data(self.processed_df)
        
        # Test inverse transform for a numeric column
        original_likes = self.processed_df['likes'].values
        normalized_likes = normalized_df['likes'].values
        inverse_transformed_likes = self.normalizer.inverse_transform(normalized_likes.reshape(-1, 1), ['likes'])
        
        np.testing.assert_array_almost_equal(original_likes, inverse_transformed_likes.flatten(), decimal=0)
        
        print("\nInverse transform test results:")
        print(f"Original likes (first 5): {original_likes[:5]}")
        print(f"Inverse transformed likes (first 5): {inverse_transformed_likes.flatten()[:5]}")

    def test_get_feature_names(self):
        normalized_df = self.normalizer.normalize_data(self.processed_df)
        feature_names = self.normalizer.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertEqual(set(feature_names), set(normalized_df.columns))
        
        print("\nFeature names after normalization:")
        print(feature_names)

    def test_get_normalization_summary(self):
        self.normalizer.normalize_data(self.processed_df)
        summary = self.normalizer.get_normalization_summary()
        
        self.assertIsInstance(summary, dict)
        
        print("\nNormalization summary:")
        for feature, details in summary.items():
            print(f"{feature}: {details}")

if __name__ == '__main__':
    unittest.main()