import pandas as pd
import numpy as np
from typing import Optional
import logging
from datetime import datetime
import ast
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    I developed this preprocessor to serve as the central hub for cleaning, transforming, and enriching our raw Instagram data. 
    The goal was to create a robust pipeline that could handle the quirks and inconsistencies of social media data while preparing it for more advanced analysis and modeling stages.

    Key Thoughts and Decisions:
    1. Modular Design: 
    I structured the preprocessor with several private methods (_convert_data_types, _handle_missing_values, etc.). 
    This modular approach makes the code more manageable and easier to update or extend in the future. 
    It also allows us to easily reorder or optionally skip certain preprocessing steps if needed.

    2. Type Conversion: 
    I put a strong emphasis on converting data to appropriate types early in the process. 
    This decision was driven by past experiences where inconsistent data types caused subtle bugs later in the analysis pipeline. 
    By establishing correct types upfront, we reduce the risk of such issues.

    3. Missing Value Strategy: 
    For handling missing values, I chose a mixed approach. 
    Critical columns result in row deletion, while less critical missing data are filled with neutral values (like empty strings for text or 0 for numeric data). 
    This balance aims to preserve as much data as possible without compromising the integrity of our analysis.

    4. Feature Creation: 
    I implemented several basic feature creation steps, like extracting hashtags and calculating engagement rates. 
    The goal here was to surface potentially useful information that's implicit in the raw data. 
    These derived features often prove valuable in modeling stages.

    5. Text Cleaning: 
    I included a method for cleaning text data, focusing on lowercase conversion and removal of URLs and special characters. 
    This standardization is crucial for any subsequent text analysis or natural language processing tasks.

    Challenges and Solutions:
    One significant challenge was dealing with the 'location' field, which appeared to be stored as a string representation of a dictionary. 
    I implemented a parsing function to extract usable information from this field. 
    This solution allows us to utilize location data without being hampered by its original quirky format.
    Another challenge was deciding how to calculate engagement rate given that not all accounts might have follower counts.
    I addressed this by using a '.replace(0, 1)' operation when calculating engagement rate to avoid division by zero errors. 
    This is a simple solution that allows us to proceed with analysis, though it's worth noting as a potential area for refinement.

    Relation to Other Components:
    This preprocessor is designed to work closely with the DataLoader class. 
    It expects input in the format provided by the loader and produces output that's ready for feature extraction and modeling stages. 

    The decisions made here lay the groundwork for all subsequent analysis, so I focused on creating clean, consistent, and information-rich data.
"""

    def __init__(self):
        pass

    def preprocess_instagram_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the Instagram dataset."""
        logging.info("Preprocessing Instagram data...")
        
        # Convert data types
        df['owner_id'] = pd.to_numeric(df['owner_id'], errors='coerce')
        df['is_video'] = df['is_video'].astype(bool)
        df['comments'] = pd.to_numeric(df['comments'], errors='coerce')
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
        df['following'] = pd.to_numeric(df['following'], errors='coerce')

        # Handle location data
        df['location'] = df['location'].astype(str)

        # Extract hashtags from caption
        df['hashtags'] = df['caption'].str.findall(r'#(\w+)').apply(lambda x: ','.join(x) if isinstance(x, list) else '')

        # Calculate engagement rate
        df['engagement_rate'] = ((df['likes'] + df['comments']) / df['followers']) * 100

        # Handle imageUrl
        df['has_image'] = df['imageUrl'].notna()

        # Handle missing values
        df = df.dropna(subset=['owner_id', 'shortcode', 'created_at'])

        logging.info(f"Instagram data preprocessed. Shape: {df.shape}")
        return df

    def _parse_location(self, location_str: str) -> Optional[dict]:
        """Parse the location string into a dictionary."""
        if pd.isna(location_str):
            return None
        try:
            return json.loads(location_str)
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON in location field: {location_str}")
            return None
    def _convert_data_types(self):
        """Convert columns to appropriate data types."""
        logging.info("Converting data types...")
        
        # Boolean conversions
        bool_columns = ['is_video', 'multiple_images']
        for col in bool_columns:
            self.df[col] = self.df[col].astype(bool)
        
        # Datetime conversion
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], unit='s')
        
        # Numeric conversions
        numeric_columns = ['owner_id', 'comments', 'likes', 'followers', 'following']
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        logging.info("Handling missing values...")
        
        # Fill missing text data with empty strings
        text_columns = ['caption', 'location']
        for col in text_columns:
            self.df[col].fillna('', inplace=True)
        
        # Fill missing numeric data with 0
        numeric_columns = ['comments', 'likes', 'followers', 'following']
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)
        
        # Drop rows with missing critical data
        critical_columns = ['owner_id', 'shortcode', 'created_at']
        self.df.dropna(subset=critical_columns, inplace=True)

    def _process_location_data(self):
        """Process and extract information from the location column."""
        logging.info("Processing location data...")
        
        def extract_location_info(loc_str):
            if not loc_str:
                return ''
            try:
                loc_dict = ast.literal_eval(loc_str)
                return loc_dict.get('name', '')
            except:
                return ''

        self.df['location_name'] = self.df['location'].apply(extract_location_info)
        self.df['has_location'] = self.df['location_name'] != ''

    def _create_basic_features(self):
        """Create basic features that don't require complex processing."""
        logging.info("Creating basic features...")
        
        # Time-based features
        current_time = datetime.now()
        self.df['post_age_days'] = (current_time - self.df['created_at']).dt.total_seconds() / (24 * 3600)
        self.df['post_hour'] = self.df['created_at'].dt.hour
        self.df['post_day'] = self.df['created_at'].dt.day_name()
        self.df['is_weekend'] = self.df['created_at'].dt.dayofweek.isin([5, 6])
        
        # Content-based features
        self.df['caption_length'] = self.df['caption'].str.len()
        self.df['hashtag_count'] = self.df['caption'].str.count('#')
        self.df['mention_count'] = self.df['caption'].str.count('@')
        
        # Engagement features
        self.df['total_engagement'] = self.df['likes'] + self.df['comments']
        self.df['engagement_rate'] = self.df['total_engagement'] / self.df['followers'].replace(0, 1)  # Avoid division by zero

    def _clean_text_data(self):
        """Clean and preprocess text data in captions."""
        logging.info("Cleaning text data...")
        
        def clean_text(text):
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        self.df['cleaned_caption'] = self.df['caption'].apply(clean_text)

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get the processed Instagram data.

        Returns:
            Optional[pd.DataFrame]: The processed Instagram data DataFrame, or None if not processed.
        """
        return self.df

    def get_preprocessing_summary(self) -> dict:
        """
        Get a summary of the preprocessing steps and basic dataset statistics.

        Returns:
            dict: A dictionary containing preprocessing summary and statistics.
        """
        if self.df is None:
            return {"error": "Data has not been preprocessed yet."}

        return {
            "original_shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.to_dict(),
            "numeric_columns_stats": self.df.describe().to_dict(),
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'bool']).columns),
            "time_range": {
                "earliest_post": self.df['created_at'].min(),
                "latest_post": self.df['created_at'].max()
            },
            "total_posts": len(self.df),
            "unique_users": self.df['owner_username'].nunique()
        }

if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader

    data_dir = "path/to/dataset"
    loader = DataLoader(data_dir)
    raw_df = loader.load_instagram_data()

    if raw_df is not None:
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_instagram_data(raw_df)

        print("Preprocessed Instagram data sample:")
        print(processed_df.head())
        print(f"\nProcessed dataset shape: {processed_df.shape}")
        print(f"\nColumns after preprocessing: {processed_df.columns.tolist()}")

        # Print preprocessing summary
        summary = preprocessor.get_preprocessing_summary()
        print("\nPreprocessing Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("Failed to load Instagram data for preprocessing.")