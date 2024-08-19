import unittest
import pandas as pd
import numpy as np
from PIL import Image
import os
import tempfile
from data_prep.image_feature_extractor import ImageFeatureExtractor

class TestImageFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ImageFeatureExtractor()
        self.test_dir = tempfile.mkdtemp()
        
        # Create a sample image for testing
        self.sample_image = Image.new('RGB', (100, 100), color='red')
        self.sample_image_path = os.path.join(self.test_dir, 'sample_image.jpg')
        self.sample_image.save(self.sample_image_path)
        
        # Create a sample DataFrame
        self.sample_df = pd.DataFrame({
            'imageUrl': [self.sample_image_path, self.sample_image_path],
            'other_column': ['data1', 'data2']
        })

    def tearDown(self):
        # Clean up the temporary directory
        os.remove(self.sample_image_path)
        os.rmdir(self.test_dir)

    def test_extract_image_features(self):
        result_df = self.extractor.extract_image_features(self.sample_df, use_local_files=True)
        
        self.assertIsNotNone(result_df)
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check if new features are created
        expected_new_features = ['img_width', 'img_height', 'img_aspect_ratio', 'img_phash', 'img_dominant_color_1', 'img_brightness_level', 'img_edge_level', 'img_texture_level']
        for feature in expected_new_features:
            self.assertIn(feature, result_df.columns, f"Feature {feature} not found in result DataFrame")

    def test_process_local_image(self):
        features = self.extractor._process_local_image(self.sample_image_path)
        self.assertIsNotNone(features)
        self.assertIn('width', features)
        self.assertIn('height', features)
        self.assertIn('aspect_ratio', features)
        self.assertIn('dominant_color_1', features)
        self.assertIn('brightness_level', features)
        self.assertIn('edge_level', features)
        self.assertIn('texture_level', features)

    def test_extract_color_features(self):
        img_array = np.array(self.sample_image)
        color_features = self.extractor._extract_color_features(img_array)
        self.assertIn('dominant_color_1', color_features)
        self.assertIn('dominant_color_2', color_features)
        self.assertIn('dominant_color_3', color_features)
        print("\nExtracted color features:", color_features)

    def test_extract_brightness_feature(self):
        img_array = np.array(self.sample_image)
        brightness_feature = self.extractor._extract_brightness_feature(img_array)
        self.assertIn('brightness_level', brightness_feature)
        self.assertIn(brightness_feature['brightness_level'], ['dark', 'medium', 'bright'])
        print("\nExtracted brightness feature:", brightness_feature)

    def test_extract_edge_feature(self):
        img_array = np.array(self.sample_image)
        edge_feature = self.extractor._extract_edge_feature(img_array)
        self.assertIn('edge_level', edge_feature)
        self.assertIn(edge_feature['edge_level'], ['low', 'medium', 'high'])
        print("\nExtracted edge feature:", edge_feature)

    def test_extract_texture_feature(self):
        img_array = np.array(self.sample_image)
        texture_feature = self.extractor._extract_texture_feature(img_array)
        self.assertIn('texture_level', texture_feature)
        self.assertIn(texture_feature['texture_level'], ['smooth', 'moderate', 'rough'])
        print("\nExtracted texture feature:", texture_feature)

    def test_get_feature_descriptions(self):
        descriptions = self.extractor.get_feature_descriptions()
        self.assertIsInstance(descriptions, dict)
        self.assertTrue(len(descriptions) > 0)
        print("\nFeature descriptions:")
        for feature, description in descriptions.items():
            print(f"{feature}: {description}")

if __name__ == '__main__':
    unittest.main()