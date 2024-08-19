import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

from data_prep.data_loader import DataLoader
from data_prep.preprocessor import DataPreprocessor

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
        plt.style.use('ggplot')

    def _save_plot(self, filename: str):
        filepath = os.path.join(self.visualization_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved plot: {filepath}")

    def analyze_user_influence(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['followers'], bins=50, kde=True)
        plt.xscale('log')
        plt.title('Distribution of Followers (Log Scale)')
        plt.xlabel('Number of Followers')
        plt.ylabel('Frequency')
        self._save_plot('follower_distribution.png')

        plt.figure(figsize=(12, 6))
        plt.scatter(df['followers'], df['engagement_rate'], alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Engagement Rate vs Followers (Log-Log Scale)')
        plt.xlabel('Number of Followers')
        plt.ylabel('Engagement Rate')
        self._save_plot('engagement_vs_followers.png')

    def analyze_viral_content(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['likes'], bins=50, kde=True)
        plt.xscale('log')
        plt.title('Distribution of Likes (Log Scale)')
        plt.xlabel('Number of Likes')
        plt.ylabel('Frequency')
        self._save_plot('likes_distribution.png')

        plt.figure(figsize=(12, 6))
        plt.scatter(df['likes'], df['comments'], alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Likes vs Comments (Log-Log Scale)')
        plt.xlabel('Number of Likes')
        plt.ylabel('Number of Comments')
        self._save_plot('likes_vs_comments.png')

    def analyze_content_features(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 6))
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df['hashtag_count'] = df['hashtags'].str.count(',') + 1
        df['hashtag_bin'] = pd.cut(df['hashtag_count'], bins=[0, 5, 10, 15, 20, 25, 30, np.inf], 
                                   labels=['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '30+'])
        sns.boxplot(data=df, x='hashtag_bin', y='engagement_rate')
        plt.title('Engagement Rate by Number of Hashtags')
        plt.xlabel('Number of Hashtags')
        plt.ylabel('Engagement Rate')
        plt.yscale('log')
        plt.xticks(rotation=45)
        self._save_plot('engagement_by_hashtags.png')

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='is_video', y='engagement_rate')
        plt.title('Engagement Rate by Content Type')
        plt.xlabel('Is Video')
        plt.ylabel('Engagement Rate')
        plt.yscale('log')
        self._save_plot('engagement_by_content_type.png')

    def run_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        logging.info("Starting Exploratory Data Analysis...")
        
        self.analyze_user_influence(df)
        self.analyze_viral_content(df)
        self.analyze_content_features(df)
        
        # Calculate summary statistics
        self.summary_stats['total_users'] = df['owner_username'].nunique()
        self.summary_stats['total_posts'] = len(df)
        self.summary_stats['avg_followers'] = df['followers'].mean()
        self.summary_stats['median_followers'] = df['followers'].median()
        self.summary_stats['avg_engagement_rate'] = df['engagement_rate'].mean()
        self.summary_stats['median_engagement_rate'] = df['engagement_rate'].median()
        
        logging.info("Exploratory Data Analysis completed.")
        return self.summary_stats
def main():
    # Get the directory of the current file (features)
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