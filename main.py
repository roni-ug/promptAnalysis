import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, models, util
from nlpUtils import *
from tagAnalysisUtils import *
import os
import time
from datasets import load_dataset
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import umap

import faiss
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy
import matplotlib.patches as mpatches
from wordcloud import STOPWORDS
from sklearn.cluster import MiniBatchKMeans

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

first_run = False



def similar_vectors(embeddings):
    threshold = 0.9 # Set the threshold for similarity.
    n_neighbors = 1000  # Set the number of neighbors to consider.

    # Perform batch processing because processing all data at once may cause resource shortage.
    batch_size = 1000  # Set the batch size (i.e., the number of data items to be processed at once).
    similar_vectors = []  # Create an empty list to store similar vectors.
    index = faiss.IndexFlatIP(384)
    # Normalize the input vector and add it to the IndexFlatIP
    index.add(F.normalize(embeddings).cpu().numpy())

    for i in tqdm(range(0, len(embeddings), batch_size)):
        # Get the target batch for processing.
        batch_data = embeddings.cpu().numpy()[i:i + batch_size]
        # Neighborhood search based on cosine similarity.
        similarities, indices = index.search(batch_data, n_neighbors)

        # Extract indexes and similarities of data to be deleted.
        for j in range(similarities.shape[0]):
            close_vectors = indices[j, similarities[j] >= threshold]
            index_base = i
            # Get only the similar vectors that exclude itself
            close_vectors = close_vectors[close_vectors != index_base + j]
            similar_vectors.append((index_base + j, close_vectors))
    return similar_vectors

def filter_similarities(prompts, embeddings, similar_vectors):
    # Get all indices that are in the similar vectors
    all_similar_indices = [indices for _, indices in similar_vectors]
    all_similar_indices = np.concatenate(all_similar_indices)
    #print(len(all_similar_indices))

    # Create a mask where only the first occurrence of each index is True
    mask = np.ones(len(embeddings), dtype=bool)
    mask[all_similar_indices] = False
    mask = mask.tolist()
    # Apply the mask to the embeddings to get only the unique embeddings
    unique_embeddings = embeddings[mask]
    unique_prompts = [prompt for prompt, m in zip(prompts, mask) if m]
    return unique_embeddings,unique_prompts

def remove_similarities_and_save(embeddings,prompts):
    sim = similar_vectors(embeddings)
    emb_filtered, prompts_filtered = filter_similarities(prompts, embeddings, sim)

    torch.save(emb_filtered, 'data/filtered_similarities_embeddings.pt')
    torch.save(prompts_filtered, 'data/filtered_similarities_prompts.pt')

def most_frequent_terms(texts, ngram_range=(1, 2), top_n=1):
    vec = CountVectorizer(ngram_range=ngram_range).fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return ' '.join([wf[0] for wf in words_freq[:top_n]])

def plot_clusters(df, word_df):
    # Plot scatter plots for each cluster
    fig, ax = plt.subplots(figsize=(20, 10))
    legend_patches = []

    for label in np.unique(df['label']):
        if label == -1:
            continue

        cluster_df = df[df['label'] == label].sample(
            min(1000, len(df[df['label'] == label])))  # take a subset of max 1000 points

        ax.scatter(cluster_df['x'], cluster_df['y'], color=cluster_df['color'].iloc[0],
                   alpha=0.5)  # use the first color of the cluster and set alpha for better visibility

        # add legend patch
        legend_patches.append(
            mpatches.Patch(color=cluster_df['color'].iloc[0], label=f"Cluster {label} : {word_df.loc[label, 'word']}"))

    ax.legend(handles=legend_patches, loc='upper right', title='Clusters and their Most Frequent Words')
    plt.show()

def effiecnt_prompt_cluster(emb_filtered,prompts_filtered):
    try:
        #Dimention reduction
        logging.info('Dimension reduction')
        umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(
            emb_filtered)

        # Clustering
        logging.info('Clustering')
        cluster = MiniBatchKMeans(n_clusters=20, batch_size=1000)
        umap_embeddings = np.array(umap_embeddings)
        cluster_labels = cluster.fit_predict(umap_embeddings)

        # Prepare data for visualization
        logging.info('Preparing data for visualization')
        df = pd.DataFrame()
        df['x'], df['y'] = umap_embeddings[:, 0], umap_embeddings[:, 1]
        df['label'] = cluster_labels
        df['prompts'] = prompts_filtered

        # Get most frequent words for each cluster
        logging.info('Getting most frequent words for each cluster')
        topics = {}
        for label in np.unique(cluster_labels):
            if label == -1:
                continue

            prompts_in_cluster = df[df['label'] == label]['prompts'].tolist()
            words_in_cluster = [word for sentence in prompts_in_cluster for word in sentence.lower().split() if
                                word not in STOPWORDS]

            most_common_words = ', '.join(
                [word for word, _ in Counter(words_in_cluster).most_common(3)])  # change number of words here
            topics[label] = most_common_words

        # Create a new dataframe for the words and their counts
        logging.info('Creating dataframe for words and their counts')
        word_df = pd.DataFrame.from_dict(topics, orient='index', columns=['word'])

        # Assign colors to data points
        color_palette = sns.color_palette('viridis', n_colors=len(topics))
        df['color'] = df['label'].apply(lambda label: color_palette[label])
    except Exception as e:
        logging.exception("Exception occurred")
    return df, word_df

if __name__ == '__main__':

    #load model
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    if(first_run == True):
        # deletes duplicates, embeds it and saves both the prompts and the corresponding embeddings in files for quick retrival
        # needs to be used only once for each new dataset
        preProcess_Data('data/metadata.parquet', st_model)

        # The following code is used to verify the embedding model on a test dataset of prompts +  embeddings provided by Kaggle
        print(test_embedding_model('data/test_prompts.csv', 'data/test_embeddings.csv', st_model))

        embeddings = torch.load('data/embedded_corpus.pt')
        prompts = torch.load('data/corpus_unique.pt')
        #Removes very similar prompts by cosine similarity with a given threshold, saves in file
        remove_similarities_and_save(embeddings,prompts)

    # Load Spacy English model
    nlp = spacy.load('en_core_web_sm')

    # Load embeddings and corresponding prompts
    emb_filtered = torch.load('data/filtered_similarities_embeddings.pt')
    prompts_filtered = torch.load('data/filtered_similarities_prompts.pt')

    df, word_df = effiecnt_prompt_cluster(emb_filtered,prompts_filtered)
    print("plotting")
    plot_clusters(df, word_df)


    #analyse_tags(prompts)



