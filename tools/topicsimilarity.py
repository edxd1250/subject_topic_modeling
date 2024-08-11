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

import statsmodels.api as sm
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import interp1d
from sklearn.utils.extmath import safe_sparse_dot
import plotly.graph_objects as go
from cuml.preprocessing import Normalizer
import time
import resource

import plotly.figure_factory as ff

def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2)  # Convert to kilobytes

def gpu_cosine_similarity(matrix, matrix2=None):
    norm = Normalizer()
    norm_matrix = norm.transform(matrix)
    if matrix2 is None:
        K = safe_sparse_dot(norm_matrix, norm_matrix.T, dense_output=True) 
    else:
        norm_matrix2 = norm.transform(matrix2)
        K = safe_sparse_dot(norm_matrix, norm_matrix2.T)
    return K

def get_bins(array):
    bins = np.arange(0, 1, 0.025)
    bin_indices = np.digitize(array, bins, right=True)
    bin_counts = np.bincount(bin_indices, minlength=len(bins))
    bin_counts[-2] += bin_counts[-1]
    bin_counts = bin_counts[:-1]
    bin_count_df = pd.DataFrame({'Bin': bins, 'Count': bin_counts})
    return bin_count_df


def main():
    dir_0full = Path('/mnt/scratch/ande2472/sjrouts/0to264_full/')

    topic_model_0 = BERTopic.load(dir_0full/'model_outliers_reduced')
    with open(dir_0full/'new_docs.pickle', "rb") as f:
        docs_0 = pickle.load(f)


    topic_list = list(docs_0[docs_0['Topic'] == 1].index)
    topic_unlist = list(docs_0[docs_0['Topic'] != 1].index)
    file = '/mnt/scratch/ande2472/data/0_topjournals/0to264_topjournals_embs.pickle'
    with open(file, "rb") as f:
        emb = pickle.load(f)
    start_time = time.time()

    within_sim_mat = gpu_cosine_similarity(emb[topic_list])
    between_sim_mat = gpu_cosine_similarity(emb[topic_list],emb[topic_unlist])
    
    save_file1= dir_0full/'sim_matrix_test.pickle'
    save_file2= dir_0full/'sim_matrix_test2.pickle'

    with open(save_file1, "wb") as f:
      pickle.dump(within_sim_mat, f)
    with open(save_file2, "wb") as f:
      pickle.dump(between_sim_mat, f)

    upper_triangle_no_diag = within_sim_mat[np.triu_indices_from(within_sim_mat, k=1)]
   # fig = ff.create_distplot([upper_triangle_no_diag],['Topic 1'] ,bin_size=.025)
    
    within_bin = get_bins(upper_triangle_no_diag)
    between_bin = get_bins(between_sim_mat.flatten())

    fig = go.Figure(data=[
    go.Bar(
        x=within_bin['Bin'],
        y=within_bin['Count'],
        width=0.02,
        name='Within Bins'
    ),
    go.Bar(
        x=between_bin['Bin'],
        y=between_bin['Count'],
        width=0.02,
        name='Between Bins'
    )])

    fig.update_layout(
    title='Histogram of Bin Values and Counts',
    xaxis_title='Bin Value',
    yaxis_title='Count',
    bargap=0.2)  # Adjust gap between bars

    
    fig.write_html(dir_0full/"test_1.html")

    

    print(f'predictions Saved at: {dir_0full}')
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    print(f"Memory usage: {memory_usage()} GB")

if __name__ == '__main__':
    main()
