import re, pickle, os, torch, csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bertopic import BERTopic
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from cuml.preprocessing import Normalizer
from sklearn.utils.extmath import safe_sparse_dot

import time
import resource

def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2)  # Convert to gigabytes



def gpu_cosine_similarity(matrix, matrix2):
  #  sparse = csr_matrix(matrix)
    norm = Normalizer()
    norm_matrix = norm.transform(matrix)
    norm_matrix2 = norm.transform(matrix2)
    K = safe_sparse_dot(norm_matrix, norm_matrix2.T, dense_output=True) 
    return K

def main():
    dir_0 = Path('/mnt/scratch/ande2472/sjrouts/0to264_topjournals')
    dir_12 = Path('/mnt/scratch/ande2472/sjrouts/1200to1526_topjournals')


    topic_model_0 = BERTopic.load(dir_0/'model_outliers_reduced')
    topic_model_12 = BERTopic.load(dir_12/'model_outliers_reduced')

    topic_freq_0 = pd.read_csv(dir_0/'topic_freq.csv').drop(columns='Unnamed: 0')
    topic_freq_12 = pd.read_csv(dir_12/'topic_freq.csv').drop(columns='Unnamed: 0')

    embeddings = topic_model_0.c_tf_idf_[topic_model_0._outliers:]
    embeddings2 = topic_model_12.c_tf_idf_[topic_model_12._outliers:]

    freq_df = topic_freq_0.loc[topic_freq_0.Topic != -1, :]
    freq_df2 = topic_freq_12.loc[topic_freq_12.Topic != -1, :]

    topics = sorted(freq_df.Topic.to_list())
    topics2 = sorted(freq_df2.Topic.to_list())

    start_time = time.time()

    distance_matrix = gpu_cosine_similarity(embeddings.T, embeddings2.T)

    
    with open(dir_0/'matrix.pickle', "wb") as f:
      pickle.dump(distance_matrix, f)
    print(f'documents Saved at: {dir_0/'matrix.pickle'}')
    
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    print(f"Memory usage: {memory_usage()} GB")
    


if __name__ == '__main__':
    main()
