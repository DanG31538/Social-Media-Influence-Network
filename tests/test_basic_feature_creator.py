import unittest
import pandas as pd
from features.basic_feature_creator import BasicFeatureCreator
from data_prep.preprocessor import DataPreprocessor
from test_config import TEST_DATASET_PATH

class TestBasicFeatureCreator(unittest.TestCase):
    def setUp(self):
        self.feature_creator = BasicFeatureCreator()
        self.preprocessor = DataPreprocessor()
        self.test_df = pd.read_csv(TEST_DATASET_PATH)
        self.processed_df = self.preprocessor.preprocess_instagram_data(self.test_df)

    def test_create_basic_features(self):
        featured_df = self.feature_creator.create_basic_features(self.processed_df)
        
        self.assertIsNotNone(featured_df)
        self.assertIsInstance(featured_df, pd.DataFrame)
        
        # Check if new features are created
        expected_new_features = ['post_age_days', 'total_engagement', 'caption_length', 'hashtag_count']
        for feature in expected_new_features:
            self.assertIn(feature, featured_df.columns)
        
        # Print the first few rows of the featured dataset
        print("First few rows of the featured dataset:")
        print(featured_df.head().to_string())

        # Print new column info
        print("\nNew columns information:")
        new_columns = set(featured_df.columns) - set(self.processed_df.columns)
        print(featured_df[list(new_columns)].info())

if __name__ == '__main__':
    unittest.main()