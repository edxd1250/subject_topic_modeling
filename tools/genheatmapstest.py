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

import time
import resource

def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2)  # Convert to gigabytes

def converttopicdf(topics_list):
    topic_dict = {}
    for index, topic in enumerate(topics_list):
        if topic not in topic_dict:
            topic_dict[topic] = []
        topic_dict[topic].append(index)
    df = pd.DataFrame(topic_dict.items(), columns=['topic', 'document_indices'])
    df['topic_length'] = df['document_indices']
    for i in range(len(df)):
     df['topic_length'][i] = len(df['document_indices'][i])
    return df


def calculate_similarity_matrix(topic1, topic2):
    
 matrix = np.zeros((len(topic1),len(topic2)))

    # Calculate the Jaccard similarity between each pair of topics
 for i in range(len(topic1)):
        for j in range(len(topic2)):
            docs1 = set(topic1[topic1['topic'] == topic1['topic'][i]]['document_indices'].iloc[0])
            docs2 = set(topic2[topic2['topic'] == topic2['topic'][j]]['document_indices'].iloc[0])
            intersection = len(docs1.intersection(docs2))
            union = len(docs1.union(docs2))
            matrix[i, j] = intersection / union
    
 return matrix

def sort_singularity_mat(matrix):
    max_values = np.max(matrix, axis=1)
    sorted_indices = np.argsort(max_values)[::-1]
    sorted_similarity_matrix = matrix[sorted_indices]
    max_values_col = np.max(sorted_similarity_matrix, axis=0)
    sorted_indices_col = np.argsort(max_values_col)[::-1]
    sorted_similarity_matrix2 = sorted_similarity_matrix[:, sorted_indices_col]
    return sorted_similarity_matrix2


def main():
    dir_0full = Path('/mnt/scratch/ande2472/sjrouts/0to264_full/')
    dir_12full = Path('/mnt/scratch/ande2472/sjrouts/1200to1526_full/')
    doc_path0 = Path('/mnt/scratch/ande2472/data/0_topjournals')
    doc_path12 = Path('/mnt/scratch/ande2472/data/1200_topjournals')



    with open(doc_path12/'1200to1526_topjournals_cleans.pickle', "rb") as f:
        cleans_12 = pickle.load(f)

    with open(doc_path0/'0to264_topjournals_cleans.pickle', "rb") as f:
        cleans_0 = pickle.load(f)

    # with open(dir_0/'predictions.pickle', "rb") as f:
    #     pred_0 = pickle.load(f)

    # with open(dir_12/'new_docs.pickle', "rb") as f:
    #     documents_12 = pickle.load(f)

    topic_model_0 = BERTopic.load(dir_0full/'model_outliers_reduced')
    topic_model_12 = BERTopic.load(dir_12full/'model_outliers_reduced')


    start_time = time.time()
    pred_12, prob_12 = topic_model_0.transform(cleans_12)

    with open(dir_0full/'1200docs_on_0_predictions.pickle', "wb") as f:
           pickle.dump(pred_12, f)

    with open(dir_0full/'1200docs_on_0_prob.pickle', "wb") as f:
           pickle.dump(prob_12, f)
    
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    print(f"Memory usage: {memory_usage()} GB")
    
    start_time = time.time()
    pred_0, prob_0 = topic_model_12.transform(cleans_0)

    with open(dir_12full/'0docs_on_1200_predictions.pickle', "wb") as f:
           pickle.dump(pred_0, f)

    with open(dir_12full/'0docs_on_1200_prob.pickle', "wb") as f:
           pickle.dump(prob_0, f)
    
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    print(f"Memory usage: {memory_usage()} GB")

    ogpreds_0_path = Path('/mnt/scratch/ande2472/sjrouts/0to264_full/generated_topics/0to264_topjournals_embs_generated_topics.pickle')
    with open(ogpreds_0_path, "rb") as f:
        ogpred_0 = pickle.load(f)

    topic_vec_0 = converttopicdf(ogpred_0)
    topic_vec_12 = converttopicdf(pred_0)

    similarity_matrix_0_12 = calculate_similarity_matrix(topic_vec_0, topic_vec_12)
    sorted_similarity_matrix_0_12 = sort_singularity_mat(similarity_matrix_0_12)

    plt.figure(figsize=(8, 6))
    plt.axes().set_facecolor("blue")
    sns.heatmap(sorted_similarity_matrix_0_12, cmap='spring', )

    plt.xlabel('Topics in 1%')
    plt.ylabel('Topic in 5%')
    plt.title('Similarity between Topics')
    plt.show()
    plt.savefig('/mnt/home/ande2472/data/heatmap1.png')

if __name__ == '__main__':
    main()
