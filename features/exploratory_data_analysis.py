import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExploratoryDataAnalysis:
    """
    I created this module to perform a comprehensive exploratory data analysis (EDA) on our Instagram influencer dataset. 
    The primary goal was to uncover patterns, relationships, and insights that could guide our subsequent modeling efforts and provide valuable information about influencer dynamics on the platform.

    Key Thoughts and Decisions:
    1. Modular Analysis Approach: 
    I structured the EDA into distinct analysis functions (analyze_user_influence, analyze_viral_content, analyze_content_features).
    This modular approach allows us to focus on specific aspects of the data independently and makes it easier to add new analysis types in the future.

    2. Visualization Directory: 
    I implemented a system to save all generated plots to a specified directory. 
    This decision was driven by the need to easily review and share visual insights, as well as to keep a record of our analysis results.

    3. Summary Statistics: 
    Along with visualizations, I included the calculation and storage of summary statistics. 
    These provide quick, quantitative insights that complement the visual analysis.

    4. Variety of Plot Types: 
    I used a mix of plot types (histograms, scatter plots, box plots) to provide different perspectives on the data. 
    This variety helps in identifying different types of patterns and relationships.

    Challenges and Solutions:
    One challenge was dealing with the potential for very large datasets, which could make some visualizations (like scatter plots) cluttered or slow to generate. 
    To address this, I implemented sampling in some of the plotting functions. 
    This allows us to get a representative view of the data without processing every single data point.
    Another consideration was how to handle outliers, particularly in engagement metrics which can have extreme values. 
    I chose to use log scales in some plots to better visualize the distribution across orders of magnitude. 
    For box plots, I set the whiskers to show 5th and 95th percentiles to focus on the bulk of the data while still indicating the presence of outliers.

"""
    def __init__(self, visualization_dir: str):
        self.summary_stats: Dict[str, Any] = {}
        self.visualization_dir = visualization_dir
        os.makedirs(self.visualization_dir, exist_ok=True)

    def _save_plot(self, filename: str):
        filepath = os.path.join(self.visualization_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logging.info(f"Saved plot: {filepath}")

    def analyze_user_influence(self, df: pd.DataFrame) -> None:
        # ... (keep the existing code)

        # Update the savefig calls
        self._save_plot('follower_distribution.png')
        self._save_plot('engagement_vs_followers.png')

    def analyze_viral_content(self, df: pd.DataFrame) -> None:
        # ... (keep the existing code)

        # Update the savefig calls
        self._save_plot('likes_distribution.png')
        self._save_plot('likes_vs_comments.png')

    def analyze_content_features(self, df: pd.DataFrame) -> None:
        # ... (keep the existing code)

        # Update the savefig calls
        self._save_plot('engagement_by_content_type.png')
        self._save_plot('engagement_by_hashtags.png')

    # ... (keep the rest of the class as is)

def main():
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    
    # Get the directory of the current file (data_prep)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the main dataset
    main_dataset_path = os.path.join(current_dir, '..', 'dataset', 'instagram_data.csv')
    
    # Construct the path to the visualization directory
    visualization_dir = os.path.join(current_dir, '..', 'visualization', 'exploratory_data_analysis')
    
    loader = DataLoader(os.path.dirname(main_dataset_path))
    raw_df = loader.load_instagram_data(os.path.basename(main_dataset_path))
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_instagram_data(raw_df)
    
    eda = ExploratoryDataAnalysis(visualization_dir)
    summary_stats = eda.run_eda(processed_df)
    
    print("EDA Summary Stats:")
    print(summary_stats)

if __name__ == "__main__":
    main()