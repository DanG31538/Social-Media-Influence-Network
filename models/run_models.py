import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_prep.data_loader import DataLoader
from data_prep.preprocessor import DataPreprocessor
from data_prep.data_normalizer import DataNormalizer
from data_prep.image_feature_extractor import ImageFeatureExtractor
from features.basic_feature_creator import BasicFeatureCreator
from features.feature_engineering import FeatureEngineer
from features.exploratory_data_analysis import ExploratoryDataAnalysis
from models.community_detection_model import CommunityDetectionModel
from models.influential_users_model import InfluentialUsersModel
from models.viral_content_model import ViralContentModel

def main():
    # Load and preprocess data
    data_dir = os.path.join(project_root, 'dataset')
    loader = DataLoader(data_dir)
    raw_df = loader.load_instagram_data('instagram_data.csv')
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_instagram_data(raw_df)

    # Extract image features
    image_extractor = ImageFeatureExtractor()
    df_with_image_features = image_extractor.extract_image_features(processed_df)

    # Create basic features
    basic_feature_creator = BasicFeatureCreator()
    df_with_basic_features = basic_feature_creator.create_basic_features(df_with_image_features)

    # Normalize data
    normalizer = DataNormalizer()
    normalized_df = normalizer.normalize_data(df_with_basic_features)

    # Feature engineering
    feature_engineer = FeatureEngineer(normalized_df)
    featured_df = feature_engineer.engineer_features()

    # Exploratory Data Analysis
    eda_vis_dir = os.path.join(project_root, 'visualization', 'exploratory_data_analysis')
    eda = ExploratoryDataAnalysis(eda_vis_dir)
    eda.run_eda(featured_df)

    # Run models
    models_vis_dir = os.path.join(project_root, 'visualization', 'models')

    community_model = CommunityDetectionModel()
    community_model.load_data()
    community_model.create_graph()
    community_model.detect_communities()
    community_model.analyze_communities()
    community_model.visualize_communities(output_dir=models_vis_dir)

    influential_model = InfluentialUsersModel()
    influential_model.load_data()
    influential_model.train_model()
    influential_model.analyze_feature_importance(output_dir=models_vis_dir)

    viral_model = ViralContentModel()
    viral_model.load_data()
    viral_model.engineer_features()
    viral_model.train_model()
    viral_model.analyze_feature_importance(output_dir=models_vis_dir)

    print("All processes and models have been run successfully.")

if __name__ == "__main__":
    main()