# Instagram Influencer Analysis Project

## Project Overview

This project implements a machine learning pipeline for analyzing Instagram influencer data. It covers the main stages of a ML workflow, from data preparation to model deployment, within the constraints of limited time and resources.

## About Dataset

Data Type: Instagram Account Posts Data
Last Updated: February 2024

Fields:
1. owner_id: Integer (Unique identifier for the owner of the Instagram account)
2. owner_username: String (Username of the owner of the Instagram account)
3. shortcode: String (Unique identifier for the post)
4. is_video: Boolean (Indicator if the post is a video or not)
5. caption: String (Caption text accompanying the post)
6. comments: Integer (Number of comments on the post)
7. likes: Integer (Number of likes on the post)
8. created_at: Integer (Timestamp indicating the creation time of the post)
9. location: String (Location associated with the post, if any)
10. imageUrl: String (URL of the image associated with the post)
11. multiple_images: Boolean (Indicator if the post contains multiple images)
12. username: String (Username of the Instagram account)
13. followers: Integer (Number of followers of the Instagram account)
14. following: Integer (Number of accounts the Instagram account is following)

## Pipeline Stages

### 1. Data Preparation and Preprocessing

- **Data Loading**: The `DataLoader` class reads the CSV file and performs basic validations.
- **Data Preprocessing**: The `DataPreprocessor` class cleans and transforms the raw data, handling missing values and converting data types.
- **Data Normalization**: The `DataNormalizer` class standardizes numeric features and encodes categorical variables.
- **Image Feature Extraction**: The `ImageFeatureExtractor` class processes image URLs to extract basic visual features like dominant colors, brightness, and texture.

#### Unit Testing

I implemented unit tests for the data preparation stage to ensure the reliability and correctness of these crucial initial steps. The tests cover:

- Data loading
- Data preprocessing
- Basic feature creation
- Image feature extraction
- Data normalization

The test scripts are located in the `tests` directory and can be run using the `main_test.py` script.

### 2. Basic Feature Creation

The `BasicFeatureCreator` class adds derived features to the dataset, including:
- Time-based features (post age, hour of day, day of week)
- Engagement metrics (total engagement, engagement rate)
- Content-based features (caption length, hashtag count)

### 3. Exploratory Data Analysis (EDA)

The `ExploratoryDataAnalysis` class performs analysis of the data, including:
- User influence analysis
- Viral content analysis
- Content feature analysis

Visualizations are generated to show distributions and relationships within the data:

![Follower Distribution](visualization/exploratory_data_analysis/follower_distribution.png)
*Figure 1: Distribution of follower counts across users*

![Engagement vs Followers](visualization/exploratory_data_analysis/engagement_vs_followers.png)
*Figure 2: Relationship between engagement and follower count*

![Likes Distribution](visualization/exploratory_data_analysis/likes_distribution.png)
*Figure 3: Distribution of likes across posts*

![Likes vs Comments](visualization/exploratory_data_analysis/likes_vs_comments.png)
*Figure 4: Relationship between likes and comments*

### 4. Feature Engineering

The `FeatureEngineer` class creates user-level features including:
- Engagement metrics (average likes, comments, engagement rate)
- Content style features (video ratio, average caption length)
- Audience metrics (follower-following ratio)

### 5. Modeling

Three models were developed to analyze different aspects of influencer data:

1. **Community Detection Model:** 
   - Uses graph-based techniques to identify influencer communities based on shared hashtags.
   - Implements the Louvain method for community detection.
   - Provides visualization of the community structure.

   ![Community Structure](visualization/models/community_structure.png)
   *Figure 5: Visualization of detected influencer communities*

2. **Influential Users Model:** 
   - Uses a Random Forest Classifier to predict influential users.
   - Defines influential users as those in the top 25% of follower counts.
   - Includes feature importance analysis.

   ![Feature Importance for Influential Users](visualization/models/InfluentialUsersModel_feature_importance.png)
   *Figure 6: Feature importance for predicting influential users*

3. **Viral Content Model:** 
   - Predicts the likelihood of content going viral using a Random Forest Classifier.
   - Defines viral content as posts in the top 10% of likes.
   - Uses content-specific features and engagement metrics.

   ![Feature Importance for Viral Content](visualization/models/ViralContentModel_feature_importance.png)
   *Figure 7: Feature importance for predicting viral content*

### 6. Deployment (Docker)

The project is containerized using Docker:
- Dockerfile for building the application image
- `run_models.py` script as the entry point for the container
- Volume mounts for data and visualization outputs

## Setup and Usage

1. Clone the repository
2. Install Docker
3. Build the Docker image:
   ```
   docker build -t instagram-analysis .
   ```
4. Run the analysis:
   ```
   docker run -v /path/to/dataset:/app/dataset -v /path/to/visualization:/app/visualization instagram-analysis
   ```