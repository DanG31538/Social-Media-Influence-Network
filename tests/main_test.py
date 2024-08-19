import unittest
import os
import sys

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from test_data_loader import TestDataLoader
from test_preprocessor import TestDataPreprocessor
from test_basic_feature_creator import TestBasicFeatureCreator
from test_image_feature_extractor import TestImageFeatureExtractor
from test_data_normalizer import TestDataNormalizer

def run_all_tests():
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add all tests from each test class
    test_classes = [
        TestDataLoader,
        TestDataPreprocessor,
        TestBasicFeatureCreator,
        TestImageFeatureExtractor,
        TestDataNormalizer
    ]

    for test_class in test_classes:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == "__main__":
    run_all_tests()