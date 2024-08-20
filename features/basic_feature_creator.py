import pandas as pd
import numpy as np
from typing import Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BasicFeatureCreator:
    """
    The goal was to create a solid foundation of derived attributes that could provide immediate insights and serve as building blocks for more complex analyses down the line.

    Key Thoughts and Decisions:
    1. Modular Structure: 
    I organized the feature creation process into distinct methods (_create_time_based_features, _create_engagement_features, etc.). 
    This structure allows for easy extension or modification of specific feature groups without affecting others. 
    It also makes the code more readable and maintainable.

    2. Time-Based Features: 
    I put significant emphasis on extracting various time-related features. 
    This decision was driven by the understanding that timing can be crucial in social media engagement. 
    Features like post hour, day of the week, and seasonality can reveal important patterns in user behavior and post performance.

    3. Engagement Metrics: 
    I created several engagement-related features, including total engagement and engagement rate. 
    These metrics are fundamental in assessing the performance of posts and accounts. 
    I also included an 'engagement velocity' feature to capture the speed at which posts gather engagement.

    4. Content Analysis: 
    For content-based features, I focused on easily extractable attributes like caption length, word count, and hashtag usage. 
    These features can provide insights into content strategies without delving into complex natural language processing techniques.

    5. User-Based Features: 
    I included some basic user-level features like follower-to-following ratio and average engagement. 
    These help in profiling users and can be indicative of account authenticity and influence.

    Challenges and Solutions:
    One challenge was handling potential division by zero errors, particularly when calculating ratios involving follower counts. 
    I addressed this using the .replace(0, 1) method when dividing, ensuring we don't encounter runtime errors while still preserving the essence of the ratio.
    Another consideration was how to handle posts with multiple images (carousels) versus single image or video posts. 
    I created a 'media_type' feature using numpy's select function to categorize posts into 'image', 'video', or 'carousel'. 
    This provides a clean way to analyze engagement across different media types.

    In developing this feature creator, I aimed to strike a balance between comprehensiveness and simplicity. 
"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic features from the preprocessed Instagram data.

        Args:
            df (pd.DataFrame): Preprocessed Instagram data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added basic features.
        """
        self.df = df.copy()
        logging.info("Starting basic feature creation...")

        self._create_time_based_features()
        self._create_engagement_features()
        self._create_content_based_features()
        self._create_user_based_features()

        logging.info("Basic feature creation completed.")
        return self.df

    def _create_time_based_features(self):
        """Create features based on time and date information."""
        logging.info("Creating time-based features...")
        
        # Post age in days
        current_time = datetime.now()
        self.df['post_age_days'] = (current_time - self.df['created_at']).dt.total_seconds() / (24 * 3600)
        
        # Time of day features
        self.df['post_hour'] = self.df['created_at'].dt.hour
        self.df['post_minute'] = self.df['created_at'].dt.minute
        self.df['is_night_post'] = ((self.df['post_hour'] >= 22) | (self.df['post_hour'] < 6)).astype(int)
        
        # Day of week features
        self.df['post_day'] = self.df['created_at'].dt.day_name()
        self.df['is_weekend'] = self.df['created_at'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Month and season
        self.df['post_month'] = self.df['created_at'].dt.month
        self.df['post_season'] = pd.cut(self.df['created_at'].dt.month, 
                                        bins=[0, 3, 6, 9, 12], 
                                        labels=['Winter', 'Spring', 'Summer', 'Fall'],
                                        include_lowest=True)

    def _create_engagement_features(self):
        """Create features related to post engagement."""
        logging.info("Creating engagement-based features...")
        
        # Basic engagement metrics
        self.df['total_engagement'] = self.df['likes'] + self.df['comments']
        self.df['engagement_rate'] = self.df['total_engagement'] / self.df['followers'].replace(0, 1)  # Avoid division by zero
        
        # Likes and comments ratios
        self.df['likes_to_followers_ratio'] = self.df['likes'] / self.df['followers'].replace(0, 1)
        self.df['comments_to_followers_ratio'] = self.df['comments'] / self.df['followers'].replace(0, 1)
        
        # Engagement velocity (engagement per day since posting)
        self.df['engagement_velocity'] = self.df['total_engagement'] / self.df['post_age_days'].replace(0, 1)

    def _create_content_based_features(self):
        """Create features based on content information."""
        logging.info("Creating content-based features...")
        
        self.df['caption_length'] = self.df['caption'].fillna('').str.len()
        self.df['word_count'] = self.df['caption'].fillna('').str.split().str.len()
        self.df['avg_word_length'] = self.df['caption'].fillna('').str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
        
        self.df['hashtag_count'] = self.df['hashtags'].str.count(',') + 1
        self.df['mention_count'] = self.df['caption'].fillna('').str.count('@')
        self.df['has_hashtags'] = (self.df['hashtag_count'] > 0).astype(int)
        self.df['has_mentions'] = (self.df['mention_count'] > 0).astype(int)
        
        # Ensure 'is_video' and 'multiple_images' are boolean
        self.df['is_video'] = self.df['is_video'].astype(bool)
        self.df['multiple_images'] = self.df['multiple_images'].astype(bool)
        
        self.df['is_carousel'] = (self.df['multiple_images'] & ~self.df['is_video']).astype(int)
        self.df['media_type'] = np.select(
            [self.df['is_video'], self.df['multiple_images'] & ~self.df['is_video']],
            ['video', 'carousel'],
            default='image'
        )

    def _create_user_based_features(self):
        """Create features based on user information."""
        logging.info("Creating user-based features...")
        
        # Follower to following ratio
        self.df['followers_to_following_ratio'] = self.df['followers'] / self.df['following'].replace(0, 1)
        
        # Average engagement per post for each user
        user_avg_engagement = self.df.groupby('owner_username')['total_engagement'].transform('mean')
        self.df['user_avg_engagement'] = user_avg_engagement
        
        # Post frequency (average days between posts for each user)
        user_post_frequency = self.df.groupby('owner_username')['created_at'].transform(lambda x: (x.max() - x.min()).total_seconds() / (86400 * (len(x) - 1)) if len(x) > 1 else np.nan)
        self.df['user_post_frequency_days'] = user_post_frequency

    def get_feature_descriptions(self) -> dict:
        """
        Get descriptions of the created features.

        Returns:
            dict: A dictionary containing feature names and their descriptions.
        """
        return {
            "post_age_days": "Age of the post in days",
            "post_hour": "Hour of the day when the post was made (0-23)",
            "is_night_post": "Whether the post was made at night (22:00-06:00)",
            "post_day": "Day of the week when the post was made",
            "is_weekend": "Whether the post was made on a weekend",
            "post_season": "Season when the post was made",
            "total_engagement": "Sum of likes and comments",
            "engagement_rate": "Total engagement divided by number of followers",
            "engagement_velocity": "Engagement per day since posting",
            "caption_length": "Number of characters in the post caption",
            "word_count": "Number of words in the post caption",
            "hashtag_count": "Number of hashtags in the post caption",
            "mention_count": "Number of user mentions in the post caption",
            "is_carousel": "Whether the post is a carousel (multiple images)",
            "media_type": "Type of media (image, video, or carousel)",
            "followers_to_following_ratio": "Ratio of followers to accounts followed",
            "user_avg_engagement": "Average engagement per post for each user",
            "user_post_frequency_days": "Average number of days between posts for each user"
        }

if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor

    data_dir = "path/to/dataset"
    loader = DataLoader(data_dir)
    raw_df = loader.load_instagram_data()

    if raw_df is not None:
        # Preprocess the data
        preprocessor = DataPreprocessor()
        preprocessed_df = preprocessor.preprocess_instagram_data(raw_df)

        # Create basic features
        feature_creator = BasicFeatureCreator()
        featured_df = feature_creator.create_basic_features(preprocessed_df)

        print("Instagram data with basic features:")
        print(featured_df.head())
        print(f"\nDataset shape after feature creation: {featured_df.shape}")
        print(f"\nNew features added: {set(featured_df.columns) - set(preprocessed_df.columns)}")

        # Print feature descriptions
        print("\nFeature Descriptions:")
        for feature, description in feature_creator.get_feature_descriptions().items():
            print(f"{feature}: {description}")
    else:
        print("Failed to load Instagram data.")