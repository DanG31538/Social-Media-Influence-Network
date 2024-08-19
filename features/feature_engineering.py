import pandas as pd
import numpy as np
from typing import List
import os
import sys

# Add the parent directory to the Python path to allow imports from data_prep
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data_prep.data_loader import DataLoader
from data_prep.preprocessor import DataPreprocessor

class FeatureEngineer:
    """
    Key Thoughts and Decisions:
    1. User-Level Focus: 
    I made the decision to aggregate features at the user level rather than keeping them at the post level. 
    This choice was driven by our ultimate goal of understanding and potentially predicting influencer performance. 
    User-level features allow us to capture overall patterns and tendencies of each influencer.

    2. Comprehensive Metrics: 
    I included a variety of metrics covering different aspects of an influencer's presence, including audience size (followers, following), engagement (likes, comments, engagement rate), content style (video ratio, caption length), and hashtag usage. 

    3. Ratio Features: 
    I incorporated several ratio features, such as follower-following ratio and video ratio. 
    These relative measures can often be more informative than absolute numbers, providing insights into an influencer's strategy and audience dynamics.

    4. Averages Over Time: 
    For many metrics, I chose to use averages (e.g., average likes, average caption length) rather than totals. 
    This approach normalizes for the varying number of posts across influencers, making comparisons more meaningful.

    5. Handling Edge Cases: 
    I implemented logic to handle potential edge cases, such as division by zero when calculating the follower-following ratio. 
    The solution of replacing infinity values with a large number preserves the essence of the ratio while avoiding technical issues in subsequent analyses.

    Challenges and Solutions:
    One significant challenge was deciding how to aggregate post-level data into user-level features without losing too much information.
    I addressed this primarily by using averages and ratios, which capture overall tendencies while being insensitive to the total number of posts.
    Another consideration was how to handle potential outliers or extreme values, particularly in ratio calculations. 
    The approach of replacing infinity values in the follower-following ratio is one example of managing this.

    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngineer with preprocessed data.

        Args:
            df (pd.DataFrame): Preprocessed Instagram data.
        """
        self.df = df

    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer user-level features from the preprocessed data.

        This method creates a set of features for each user, including:
        - User-level metrics (followers, following, engagement rate, etc.)
        - Content-based features (video ratio, caption length, etc.)
        - Simple network features (follower-following ratio)

        Returns:
            pd.DataFrame: A DataFrame containing user-level engineered features.
        """
        # Group by user to create user-level features
        user_features = self.df.groupby('owner_username').agg({
            # User-level features
            'followers': 'first',  # Followers count
            'following': 'first',  # Following count
            'engagement_rate': 'mean',  # Average engagement rate
            'likes': 'mean',  # Average likes per post
            'comments': 'mean',  # Average comments per post
            
            # Content-based features
            'is_video': 'mean',  # Ratio of video posts
            'multiple_images': 'mean',  # Ratio of multiple image posts
            'caption': lambda x: x.str.len().mean(),  # Average caption length
            'hashtags': lambda x: x.str.count(',').mean() + 1  # Average hashtag count
        }).reset_index()

        # Rename columns for clarity
        user_features.columns = [
            'username', 'followers', 'following', 'avg_engagement_rate',
            'avg_likes', 'avg_comments', 'video_ratio', 'multiple_images_ratio',
            'avg_caption_length', 'avg_hashtag_count'
        ]

        # Simple network feature
        # Calculate follower-following ratio
        user_features['follower_following_ratio'] = user_features['followers'] / user_features['following']

        # Replace infinity with NaN and then with a large number
        user_features['follower_following_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        user_features['follower_following_ratio'].fillna(1e6, inplace=True)

        return user_features

def main():
    """
    Main function to run the feature engineering process.

    This function loads the raw data, preprocesses it, engineers features,
    and saves the result to a CSV file.
    """
    # Construct the path to the main dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dataset_path = os.path.join(current_dir, '..', 'dataset', 'instagram_data.csv')
    
    # Load and preprocess the data
    loader = DataLoader(os.path.dirname(main_dataset_path))
    raw_df = loader.load_instagram_data(os.path.basename(main_dataset_path))
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_instagram_data(raw_df)
    
    # Engineer features
    feature_engineer = FeatureEngineer(processed_df)
    user_features = feature_engineer.engineer_features()
    
    # Print summary of engineered features
    print("Engineered Features:")
    print(user_features.head())
    print("\nFeature Descriptions:")
    print(user_features.describe())

    # Save the engineered features
    output_path = os.path.join(current_dir, 'engineered_features.csv')
    user_features.to_csv(output_path, index=False)
    print(f"\nEngineered features saved to: {output_path}")

if __name__ == "__main__":
    main()