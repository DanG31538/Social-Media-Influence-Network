import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the Python path to allow imports from other project directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data_prep.data_loader import DataLoader
from data_prep.preprocessor import DataPreprocessor

class CommunityDetectionModel:
    """
    I developed this module to uncover and analyze the community structure within our Instagram influencer dataset. 
    The goal was to identify clusters of influencers who are connected through shared interests or content themes, providing insights into the broader ecosystem of influencer relationships.

    Key Thoughts and Decisions:
    1. Graph Construction: 
    I chose to create edges between users based on shared hashtags. 
    This decision was driven by the idea that hashtags serve as a proxy for shared interests or content themes. 
    Users who frequently use the same hashtags are likely operating in similar niches or targeting similar audiences.

    2. Louvain Method: 
    For community detection, I implemented the Louvain method. 
    This choice was based on its effectiveness in handling large networks and its ability to automatically determine the number of communities. 
    It offers a good balance between performance and the quality of the detected communities.

    3. Weighted Edges: 
    I decided to use weighted edges, where the weight represents the number of shared hashtags between two users. 
    This approach allows us to capture the strength of connections between users, rather than just binary relationships.

    4. Visualization: 
    I included a visualization component to help understand the structure of the detected communities. 
    However, recognizing that our network might be very large, I implemented a node limit to ensure the visualization remains manageable and informative.

    5. Community Analysis: 
    Beyond just detecting communities, I added functionality to analyze the characteristics of these communities, particularly focusing on the largest one. 
    This provides insights into what distinguishes different influencer communities.

    Challenges and Solutions:
    One major challenge was dealing with the potential scale of the network. 
    With a large number of influencers and hashtags, the graph could become very dense and computationally expensive to process. 
    To address this, I implemented the option to visualize only a subset of nodes, allowing us to get a sense of the community structure without overwhelming computational resources.
    Another consideration was how to meaningfully interpret the detected communities. 
    The analysis of the largest community's characteristics is a start, but there's potential for much deeper analysis here.
"""
    def __init__(self):
        self.G = nx.Graph()
        self.communities = None

    def load_data(self):
        # Load preprocessed data
        loader = DataLoader(os.path.join(parent_dir, 'dataset'))
        raw_df = loader.load_instagram_data('instagram_data.csv')
        
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess_instagram_data(raw_df)

    def create_graph(self):
        # Create edges between users who use the same hashtags
        hashtag_users = self.data.groupby('hashtags')['owner_username'].apply(list)
        for users in hashtag_users:
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    if self.G.has_edge(users[i], users[j]):
                        self.G[users[i]][users[j]]['weight'] += 1
                    else:
                        self.G.add_edge(users[i], users[j], weight=1)

    def detect_communities(self):
        # Apply Louvain method for community detection
        self.communities = community_louvain.best_partition(self.G)

    def analyze_communities(self):
        community_sizes = pd.Series(self.communities).value_counts()
        print("Number of communities:", len(community_sizes))
        print("\nTop 10 largest communities:")
        print(community_sizes.head(10))

        # Analyze characteristics of the largest community
        largest_community = community_sizes.index[0]
        largest_community_users = [user for user, comm in self.communities.items() if comm == largest_community]
        
        community_data = self.data[self.data['owner_username'].isin(largest_community_users)]
        print("\nCharacteristics of the largest community:")
        print(community_data[['followers', 'following', 'engagement_rate']].describe())

    def visualize_communities(self, output_dir=None, max_nodes=1000):
        sub_graph = self.G.subgraph(list(self.G.nodes())[:max_nodes])
        pos = nx.spring_layout(sub_graph)
        
        plt.figure(figsize=(12, 8))
        nx.draw(sub_graph, pos, node_color=list(self.communities.values()), 
                cmap=plt.cm.rainbow, node_size=30, with_labels=False)
        plt.title(f"Community Structure Visualization\n{len(set(self.communities.values()))} communities, {len(sub_graph)} nodes")
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'community_structure.png'))
        else:
            plt.savefig('community_structure.png')
        plt.close()

def main():
    model = CommunityDetectionModel()
    model.load_data()
    model.create_graph()
    model.detect_communities()
    model.analyze_communities()
    model.visualize_communities()

if __name__ == "__main__":
    main()