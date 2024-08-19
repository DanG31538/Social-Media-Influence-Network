import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataNormalizer:
    """ 
    The goal was to transform our raw data into a format that's more suitable for machine learning algorithms, while preserving as much information as possible.

    Key Thoughts and Decisions:
    1. Flexible Normalization: 
        I designed the DataNormalizer class to handle different types of data (numeric, categorical, datetime) dynamically. 
        This flexibility was important because Instagram data can be quite varied, and I wanted a solution that could adapt to changes in the dataset structure.

    2. Scalers and Encoders: 
        I chose to use StandardScaler for numeric data and OneHotEncoder for categorical data. 
        These are tried-and-true methods in data science, and I felt they struck a good balance between effectiveness and simplicity.
        I stored these in dictionaries (self.scalers and self.encoders) to keep track of how each column was processed, which will be crucial for inverse transformations later.

    3. Feature Names Tracking: 
        I implemented a system to track feature names throughout the normalization process. 
        This was born out of frustration from past projects where feature names got lost in the shuffle, making interpretation difficult later on.

    4. Special Handling for Certain Columns: 
        I made deliberate choices to skip normalization for 'caption' and 'location' columns.
        These text fields require more specialized processing that we'll handle in a separate text analysis component.

    Challenges and Solutions:
    A significant challenge was deciding how to handle datetime data. 
    I ended up converting it to Unix timestamp (seconds since epoch) and then normalizing. 
    This preserves the relative time differences while putting the data in a format our models can work with. 
    However, I'm aware this might not be ideal for all use cases, so it's something to keep an eye on.
    Another tricky aspect was handling potential errors in categorical encoding, especially for columns with many unique values. 
    I implemented a try-except block to skip problematic columns, logging a warning instead of crashing the entire process. 
    This trades off some data loss for increased robustness.

    This normalizer sits between the initial data loading/preprocessing (handled by data_loader.py and preprocessor.py) and our eventual model training. 
    It ensures that the data fed into our models is consistently scaled and encoded, which is crucial for many machine learning algorithms.
    In creating this normalizer, I tried to balance thoroughness with flexibility. 
    The Instagram influencer landscape can change rapidly, and I wanted the data processing to be adaptable to those changes while still providing consistent, well-structured data for our analysis pipeline.
"""
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.df = None
        self.feature_names = []

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df.copy()
        normalized_df = self.df.copy()
        self.feature_names = []

        for column in normalized_df.columns:
            if column in ['caption', 'location']:
                self.feature_names.append(column)
                continue  # Skip these columns
            elif pd.api.types.is_datetime64_any_dtype(normalized_df[column]):
                normalized_df[column] = normalized_df[column].astype('int64') // 10**9
                self.feature_names.append(column)
            elif pd.api.types.is_bool_dtype(normalized_df[column]):
                normalized_df[column] = normalized_df[column].astype(int)
                self.feature_names.append(column)
            elif pd.api.types.is_numeric_dtype(normalized_df[column]):
                normalized_df[column] = self._normalize_numeric(normalized_df[column], column)
                self.feature_names.append(column)
            elif pd.api.types.is_categorical_dtype(normalized_df[column]) or pd.api.types.is_object_dtype(normalized_df[column]):
                if column != 'caption':  # Ensure we're not encoding the caption
                    try:
                        encoded_df = self._encode_categorical(normalized_df[[column]], column)
                        normalized_df = pd.concat([normalized_df.drop(columns=[column]), encoded_df], axis=1)
                        self.feature_names.extend(encoded_df.columns)
                    except Exception as e:
                        print(f"Skipping encoding for column {column}: {str(e)}")
                        self.feature_names.append(column)

        return normalized_df

    def _normalize_numeric(self, series: pd.Series, column_name: str) -> pd.Series:
        if column_name not in self.scalers:
            self.scalers[column_name] = StandardScaler()
        return pd.Series(self.scalers[column_name].fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

    def _encode_categorical(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        if column_name not in self.encoders:
            self.encoders[column_name] = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        encoded_columns = self.encoders[column_name].fit_transform(df[[column_name]])
        feature_names = self.encoders[column_name].get_feature_names_out([column_name])
        return pd.DataFrame(encoded_columns, columns=feature_names, index=df.index)

    def get_feature_names(self):
        return self.feature_names

    def inverse_transform(self, data: np.ndarray, columns: list) -> np.ndarray:
        result = data.copy()
        for i, column in enumerate(columns):
            if column in self.scalers:
                result[:, i] = self.scalers[column].inverse_transform(result[:, i].reshape(-1, 1)).flatten()
        return result

    def get_normalization_summary(self) -> dict:
        summary = {}
        for column, scaler in self.scalers.items():
            summary[column] = {
                'scaler_type': type(scaler).__name__,
                'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
            }
        for column, encoder in self.encoders.items():
            summary[column] = {
                'encoder_type': type(encoder).__name__,
                'categories': encoder.categories_
            }
        return summary