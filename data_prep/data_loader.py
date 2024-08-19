import pandas as pd
import logging
from typing import Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Key Thoughts and Decisions:
    1. Error Handling: 
        I chose to use a try-except block in the load_instagram_data method. 
        This was crucial because data loading is often where things can go wrong (file not found, permission issues, etc.). 
        By catching exceptions here, we can provide meaningful error messages and avoid crashes later in the pipeline.

    2. Data Type Specification: 
        I explicitly set the 'caption' column to string type during loading. 
        This decision came from past experiences where mixed data types in text columns caused headaches down the line. 
        It's a small step that can save a lot of trouble later, especially when we're dealing with user-generated content like Instagram captions.

    3. Datetime Conversion: 
        Converting the 'created_at' column to datetime right away was a deliberate choice. 
        Time-based analysis is often crucial in social media data, so having this in a proper datetime format from the start makes subsequent operations much smoother.

    4. Validation Method: 
        I implemented a separate _validate_instagram_data method. 
        While it's simple now (just checking for expected columns), I structured it this way thinking about future extensions. 
        We might want to add more sophisticated validations later, and having a dedicated method makes that easier.

    5. Logging: 
        I incorporated logging throughout the class. 
        In a data pipeline, it's crucial to know what's happening at each stage, especially when things go wrong. 
        The logging here will be invaluable for debugging and monitoring as we scale up.

    Challenges and Solutions:
        One challenge I faced was deciding how to handle missing data at this stage. 
        I chose to fill NA values in the 'caption' column with empty strings. 
        This was a balance between data integrity (not losing posts just because they have no caption) and practicality (avoiding None/NaN issues in later text processing).

    It feeds directly into the preprocessing stage (preprocessor.py), so I made sure the output DataFrame has a consistent structure that the preprocessor can rely on.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.instagram_df: Optional[pd.DataFrame] = None

    def load_instagram_data(self, file_name: str = "instagram_data.csv") -> Optional[pd.DataFrame]:
        file_path = self.data_dir / file_name
        try:
            df = pd.read_csv(file_path, dtype={'caption': str}, encoding='utf-8')
            logging.info(f"Loaded Instagram data with shape: {df.shape}")
            df['caption'] = df['caption'].fillna('')
            # Convert 'created_at' to datetime
            df['created_at'] = pd.to_datetime(df['created_at'], unit='s', errors='coerce')
            self.instagram_df = df
            return df
        except Exception as e:
            logging.error(f"Error loading Instagram data: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame instead of None
    def _validate_instagram_data(self, df: pd.DataFrame):
        expected_columns = {
            'owner_id', 'owner_username', 'shortcode', 'is_video', 'caption',
            'comments', 'likes', 'created_at', 'location', 'imageUrl',
            'multiple_images', 'username', 'followers', 'following'
        }
        if not expected_columns.issubset(df.columns):
            missing_columns = expected_columns - set(df.columns)
            logging.warning(f"Instagram data is missing expected columns: {missing_columns}")
            raise ValueError(f"Instagram data is missing expected columns: {missing_columns}")

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded Instagram data.

        Returns:
            Optional[pd.DataFrame]: The loaded Instagram data DataFrame, or None if not loaded.
        """
        return self.instagram_df

if __name__ == "__main__":
    # Example usage
    data_dir = Path("path/to/dataset")
    loader = DataLoader(str(data_dir))
    instagram_df = loader.load_instagram_data()

    if instagram_df is not None:
        print("Instagram data sample:")
        print(instagram_df.head())
        print(f"\nDataset shape: {instagram_df.shape}")
        print(f"\nColumns: {instagram_df.columns.tolist()}")
    else:
        print("Failed to load Instagram data.")