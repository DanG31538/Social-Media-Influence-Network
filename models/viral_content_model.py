import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the Python path to allow imports from other project directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data_prep.data_loader import DataLoader
from data_prep.preprocessor import DataPreprocessor

class ViralContentModel:
    """
I developed this module to create a model capable of predicting whether a piece of content is likely to go viral on Instagram. 
The goal was to understand the factors that contribute to content virality and provide a tool for predicting the potential reach of new content.

Key Thoughts and Decisions:
1. Random Forest Classifier: 
Similar to the influential users model, I chose a Random Forest Classifier for this task. 
The decision was based on its ability to handle complex, non-linear relationships and its built-in feature importance capabilities. 
It's also relatively robust against overfitting, which is crucial when dealing with the often noisy world of social media data.

2. Feature Engineering: 
I created several content-specific features, such as caption length and hashtag count, in addition to using engagement metrics. 
This was to capture both the characteristics of the content itself and its initial reception.

3. Viral Content Definition: 
I defined viral content as posts in the top 10% of likes. 
This threshold is adjustable and could be refined based on platform-specific insights or business goals.

4. Data Preprocessing: 
I implemented StandardScaler to normalize our features, ensuring that all features contribute equally to the model regardless of their original scale.

5. Performance Metrics: 
In addition to the classification report, I included the ROC AUC score as a metric. 
This provides a more nuanced view of the model's performance, especially important when dealing with imbalanced classes (as viral content is likely to be the minority class).

Challenges and Solutions:
One significant challenge was defining what constitutes "viral" content. 
The top 10% threshold was chosen as a starting point, but this definition could be refined. 
For instance, we might consider the rate of engagement growth rather than just the total number of likes.
Another challenge was feature selection. 
While I included a variety of features, there's always the question of whether we're capturing all relevant aspects of what makes content go viral. 
This is an area for ongoing refinement and possibly the incorporation of more advanced features (e.g., image analysis, sentiment analysis of captions).

"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.data = None

    def load_data(self):
        loader = DataLoader(os.path.join(parent_dir, 'dataset'))
        raw_df = loader.load_instagram_data('instagram_data.csv')
        
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess_instagram_data(raw_df)
        
        # Check for and remove any rows with NaN values
        self.data = self.data.dropna()

    def engineer_features(self):
        self.data['caption_length'] = self.data['caption'].str.len()
        self.data['hashtag_count'] = self.data['hashtags'].str.count(',') + 1
        
        engagement_threshold = self.data['likes'].quantile(0.9)
        self.data['is_viral'] = (self.data['likes'] > engagement_threshold).astype(int)

    def prepare_data(self):
        features = ['followers', 'following', 'engagement_rate', 'likes', 'comments',
                    'caption_length', 'hashtag_count', 'is_video', 'multiple_images']
        
        X = self.data[features]
        y = self.data['is_viral']
        
        # Print some information about the data
        print("Data shape:", X.shape)
        print("Features info:")
        print(X.info())
        print("\nTarget distribution:")
        print(y.value_counts(normalize=True))
        
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])}")
        
        # Print feature importances
        features = ['followers', 'following', 'engagement_rate', 'likes', 'comments',
                    'caption_length', 'hashtag_count', 'is_video', 'multiple_images']
        importances = self.model.feature_importances_
        for f, imp in zip(features, importances):
            print(f"{f}: {imp}")

    def analyze_feature_importance(self, output_dir=None):
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
        plt.title('Normalized Feature Importance for Viral Content Prediction')
        plt.xlabel('Normalized Importance')
        
        # Add value labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{self.__class__.__name__}_feature_importance.png'))
        else:
            plt.savefig(os.path.join(parent_dir, 'visualization', f'{self.__class__.__name__}_feature_importance.png'))
        plt.close()
        
        print("\nNormalized feature importances:")
        for feature, importance in zip(valid_features, normalized_importances):
            print(f"{feature}: {importance:.4f}")
    def predict_viral_content(self, new_posts):
        features = ['followers', 'following', 'engagement_rate', 'likes', 'comments',
                    'caption_length', 'hashtag_count', 'is_video', 'multiple_images']
        
        X = new_posts[features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

def main():
    model = ViralContentModel()
    model.load_data()
    model.engineer_features()
    model.train_model()
    model.analyze_feature_importance()
    
    sample_posts = model.data.sample(5)
    predictions = model.predict_viral_content(sample_posts)
    print("\nSample Predictions:")
    print(pd.DataFrame({
        'caption': sample_posts['caption'],
        'likes': sample_posts['likes'],
        'viral_probability': predictions
    }))

if __name__ == "__main__":
    main()