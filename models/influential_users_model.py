import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the Python path to allow imports from other project directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from features.feature_engineering import FeatureEngineer

class InfluentialUsersModel:
    """

    I created this module to develop a model capable of identifying influential users within our Instagram dataset. 
    The goal was to move beyond simple follower counts and create a more nuanced understanding of what makes an Instagram user truly influential.

    Key Thoughts and Decisions:
    1. Random Forest Classifier: 
    I chose to use a Random Forest Classifier for this task. 
    This decision was based on Random Forest's ability to handle non-linear relationships, its robustness to outliers, and its built-in feature importance capabilities. 
    It's also less prone to overfitting compared to some other algorithms.

    2. Feature Selection: 
    I included a variety of features beyond just follower count, such as engagement rates, content type ratios, and caption characteristics. 
    This was to capture different aspects of a user's Instagram presence that might contribute to their influence.

    3. Influential User Definition: 
    I defined influential users as those in the top 25% of follower counts. 
    This threshold is somewhat arbitrary and could be adjusted based on specific needs or insights from domain experts.

    4. Feature Scaling: 
    I implemented StandardScaler to normalize our features. 
    This ensures that all features contribute equally to the model and prevents features with larger scales from dominating the others.

    5. Feature Importance Analysis: 
    I included a detailed analysis and visualization of feature importances. 
    This not only helps in understanding our model but also provides actionable insights about what factors contribute most to a user's influence.

    Challenges and Solutions:
    One challenge was deciding how to define an "influential" user. 
    The 75th percentile of follower counts was chosen as a starting point, but this is definitely an area that could benefit from further refinement, perhaps incorporating engagement rates or other metrics.
    Another consideration was the potential for multicollinearity among our features. 
    While Random Forests are generally robust to this, it's something to keep in mind for feature selection and interpretation.

"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def load_data(self):
        features_path = os.path.join(parent_dir, 'features', 'engineered_features.csv')
        self.data = pd.read_csv(features_path)
        
        # Check for and remove any rows with NaN values
        self.data = self.data.dropna()
        
        follower_threshold = self.data['followers'].quantile(0.75)
        self.data['is_influential'] = (self.data['followers'] > follower_threshold).astype(int)

    def preprocess_data(self):
        features = ['followers', 'following', 'avg_engagement_rate', 'avg_likes', 
                    'avg_comments', 'video_ratio', 'multiple_images_ratio', 
                    'avg_caption_length', 'avg_hashtag_count', 'follower_following_ratio']
        
        X = self.data[features]
        y = self.data['is_influential']
        
        # Print some information about the data
        print("Data shape:", X.shape)
        print("Features info:")
        print(X.info())
        print("\nTarget distribution:")
        print(y.value_counts(normalize=True))
        
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print feature importances
        features = ['followers', 'following', 'avg_engagement_rate', 'avg_likes', 
                    'avg_comments', 'video_ratio', 'multiple_images_ratio', 
                    'avg_caption_length', 'avg_hashtag_count', 'follower_following_ratio']
        importances = self.model.feature_importances_
        for f, imp in zip(features, importances):
            print(f"{f}: {imp}")

    def analyze_feature_importance(self):
        features = ['followers', 'following', 'engagement_rate', 'likes', 'comments',
                    'caption_length', 'hashtag_count', 'is_video', 'multiple_images']
        importances = self.model.feature_importances_
        
        print("\nFeature importances:")
        for feature, importance in zip(features, importances):
            print(f"{feature}: {importance}")
        
        # Check for NaN or Inf values
        valid_mask = ~np.isnan(importances) & ~np.isinf(importances)
        valid_importances = importances[valid_mask]
        valid_features = [f for f, v in zip(features, valid_mask) if v]
        
        if len(valid_importances) == 0:
            print("\nWarning: All feature importances are NaN or Inf. Unable to create visualization.")
            return
        
        # Normalize valid importances
        normalized_importances = valid_importances / np.sum(valid_importances)
        
        feature_imp = pd.DataFrame({'Feature': valid_features, 'Importance': normalized_importances})
        feature_imp = feature_imp.sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(feature_imp['Feature'], feature_imp['Importance'])
        plt.title('Normalized Feature Importance for Prediction')
        plt.xlabel('Normalized Importance')
        
        # Add value labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, 'visualization', f'{self.__class__.__name__}_feature_importance.png'))
        plt.close()
        
        print("\nNormalized feature importances:")
        for feature, importance in zip(valid_features, normalized_importances):
            print(f"{feature}: {importance:.4f}")

    def predict_influential_users(self, users_data):
        features = ['followers', 'following', 'avg_engagement_rate', 'avg_likes', 
                    'avg_comments', 'video_ratio', 'multiple_images_ratio', 
                    'avg_caption_length', 'avg_hashtag_count', 'follower_following_ratio']
        
        X = users_data[features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

def main():
    model = InfluentialUsersModel()
    model.load_data()
    model.train_model()
    model.analyze_feature_importance()
    
    sample_users = model.data.sample(5)
    predictions = model.predict_influential_users(sample_users)
    print("\nSample Predictions:")
    print(pd.DataFrame({
        'username': sample_users['username'],
        'followers': sample_users['followers'],
        'influential_probability': predictions
    }))

if __name__ == "__main__":
    main()