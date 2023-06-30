import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, models
from fastparquet import ParquetFile

import pyarrow.parquet as pq


#used to varify embedding model is working correctly with the "test_prompts.csv" and it's embedding
# "test_embeddings.csv" test datasets
def test_embedding_model(prompt_path, embeddings_path,model):
    # Example of using this method:
    # if test_embedding_model('test_prompts.csv','test_embeddings.csv.csv',st_model):
    #         print ("embedding model working correctly")

    test_prompts = pd.read_csv('prompts.csv')
    test_weights = pd.read_csv('sample_submission.csv')
    prompt_embeddings = model.encode(test_prompts['prompt']).flatten()
    are_similiar = np.all(np.isclose(test_weights['val'].values, prompt_embeddings, atol=1e-06))
    return are_similiar

def embed_corpus(corpus_sentences, model):
    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    return corpus_embeddings


def preProcess_Data(path,model):
    table = pq.read_table(path)
    df = table.to_pandas()
    corpus_sentences = set()


    df_formatted_unique = df.drop_duplicates(subset=['prompt'])
    for prompt in df_formatted_unique['prompt']:
        corpus_sentences.add(prompt)

    corpus_sentences = list(corpus_sentences)
    embedded_df = embed_corpus(corpus_sentences, model)
    torch.save(embedded_df, 'data/embedded_corpus.pt')
    torch.save(corpus_sentences, 'data/corpus_unique.pt')
