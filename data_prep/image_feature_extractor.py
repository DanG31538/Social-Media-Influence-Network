import pandas as pd
import numpy as np
from typing import Optional, List
import logging
import requests
from io import BytesIO
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class ImageFeatureExtractor:
    """
    I created this file to tackle the challenge of extracting meaningful features from Instagram post images. 
    The goal was to transform visual data into a format that could be used alongside our textual and numerical data in our machine learning models.

    Key Thoughts and Decisions:
    1. Threading for Performance: 
    I implemented threading using ThreadPoolExecutor. 
    This decision came from the realization that image processing can be quite slow, especially when dealing with thousands of Instagram posts. 
    By processing images in parallel, we can significantly speed up this stage of our pipeline.

    2. Diverse Feature Set: 
    I chose to extract a variety of features including basic properties (width, height, aspect ratio), color information (dominant colors), and higher-level characteristics (brightness, edge levels, texture). 
    This diverse set aims to capture different aspects of image composition that might be relevant to an image's performance on Instagram.

    3. Perceptual Hashing: 
    I included image hashing (specifically, perceptual hashing or pHash) as a feature. 
    This was inspired by the idea that similar images might perform similarly, and pHash gives us a way to quantify image similarity efficiently.
    The original intention was to use pHash more extensively in other features and analyses, such as identifying trends in viral content or grouping influencers with similar visual styles. 
    However, due to time constraints and the complexity of implementing these advanced features before the deadline, the current version does not use this.

    4. Error Handling: 
    I implemented robust error handling throughout the class. When dealing with web resources (like Instagram image URLs), many things can go wrong. 
    By catching and logging errors, we ensure that a problem with one image doesn't crash the entire process.

    5. Local File Support: 
    I added an option to process local image files. 
    This decision was driven by the consideration that we might not always have direct URLs to images, especially when working with historical or archived data.

    Challenges and Solutions:
    A major challenge was balancing feature richness with processing speed. 
    Extracting complex features from high-resolution images could be very slow. 
    I addressed this by resizing images to a standard size (100x100) before feature extraction, which sacrifices some detail but greatly improves speed.
    Another challenge was dealing with the diversity of image types and qualities on Instagram. 
    To handle this, I made the feature extraction methods as robust as possible, using try-except blocks liberally to handle unexpected issues with specific images.

    I spent a considerable amount of time and effort trying to make this work but could not in time. Despite this , I kept this in the code base to further outline my thought process. 
"""
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    def extract_image_features(self, df: pd.DataFrame, sample_size: Optional[int] = None, use_local_files: bool = False) -> pd.DataFrame:
        self.df = df.copy()
        logging.info("Starting image feature extraction...")

        if sample_size:
            self.df = self.df.sample(n=min(sample_size, len(self.df)))

        with ThreadPoolExecutor() as executor:
            if use_local_files:
                features = list(executor.map(self._process_local_image, self.df['imageUrl']))
            else:
                features = list(executor.map(self._process_image, self.df['imageUrl']))

        self.df['image_features'] = features
        self._unpack_image_features()

        logging.info("Image feature extraction completed.")
        return self.df

    def get_feature_descriptions(self) -> dict:
        return {
            "img_width": "Width of the image in pixels",
            "img_height": "Height of the image in pixels",
            "img_aspect_ratio": "Aspect ratio of the image (width/height)",
            "img_phash": "Perceptual hash of the image, useful for finding similar images",
            "img_dominant_color_1": "The most dominant color in the image (RGB)",
            "img_dominant_color_2": "The second most dominant color in the image (RGB)",
            "img_dominant_color_3": "The third most dominant color in the image (RGB)",
            "img_brightness_level": "Overall brightness of the image (dark, medium, bright)",
            "img_edge_level": "Level of edge detail in the image (low, medium, high)",
            "img_texture_level": "Texture characteristic of the image (smooth, moderate, rough)"
        }

    def _process_image(self, url: str) -> Optional[dict]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return self._extract_features_from_image(img)
        except Exception as e:
            logging.error(f"Error processing image {url}: {str(e)}")
            return None

    def _process_local_image(self, file_path: str) -> Optional[dict]:
        try:
            img = Image.open(file_path)
            return self._extract_features_from_image(img)
        except Exception as e:
            logging.error(f"Error processing local image {file_path}: {str(e)}")
            return None

    def _extract_features_from_image(self, img: Image.Image) -> dict:
        width, height = img.size
        aspect_ratio = width / height
        
        img_array = np.array(img.resize((100, 100)))  # Smaller resize for faster processing
        
        color_features = self._extract_color_features(img_array)
        brightness_feature = self._extract_brightness_feature(img_array)
        edge_feature = self._extract_edge_feature(img_array)
        texture_feature = self._extract_texture_feature(img_array)
        phash = str(imagehash.phash(img))
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'phash': phash,
            **color_features,
            **brightness_feature,
            **edge_feature,
            **texture_feature
        }

    def _extract_color_features(self, img_array: np.ndarray) -> dict:
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        return {
            f'dominant_color_{i+1}': f'rgb{tuple(color)}' for i, color in enumerate(dominant_colors)
        }

    def _extract_brightness_feature(self, img_array: np.ndarray) -> dict:
        brightness = np.mean(img_array)
        if brightness < 64:
            brightness_level = "dark"
        elif brightness < 192:
            brightness_level = "medium"
        else:
            brightness_level = "bright"
        return {'brightness_level': brightness_level}

    def _extract_edge_feature(self, img_array: np.ndarray) -> dict:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        edges_vertical = np.abs(gray[:, 1:] - gray[:, :-1])
        edges_horizontal = np.abs(gray[1:, :] - gray[:-1, :])
        
        # Ensure the shapes match
        min_height = min(edges_vertical.shape[0], edges_horizontal.shape[0])
        min_width = min(edges_vertical.shape[1], edges_horizontal.shape[1])
        edges = np.maximum(edges_vertical[:min_height, :min_width], edges_horizontal[:min_height, :min_width])
        
        edge_density = np.mean(edges)
        if edge_density < 10:
            edge_level = "low"
        elif edge_density < 30:
            edge_level = "medium"
        else:
            edge_level = "high"
        return {'edge_level': edge_level}

    def _extract_texture_feature(self, img_array: np.ndarray) -> dict:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        texture = np.std(gray)
        if texture < 20:
            texture_level = "smooth"
        elif texture < 50:
            texture_level = "moderate"
        else:
            texture_level = "rough"
        return {'texture_level': texture_level}

    def _unpack_image_features(self):
        if 'image_features' not in self.df.columns:
            return

        non_none_features = [item for item in self.df['image_features'] if item is not None]
        if not non_none_features:
            return

        first_features = non_none_features[0]
        feature_keys = first_features.keys()

        for key in feature_keys:
            self.df[f'img_{key}'] = self.df['image_features'].apply(lambda x: x.get(key) if x else None)

        self.df = self.df.drop(columns=['image_features'])