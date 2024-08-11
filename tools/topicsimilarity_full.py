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

from cuml.preprocessing import Normalizer
import time
import resource

import plotly.figure_factory as ff

def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2)  # Convert to kilobytes

def gpu_cosine_similarity(matrix):
    norm = Normalizer()
    norm_matrix = norm.transform(matrix)
    K = safe_sparse_dot(norm_matrix, norm_matrix.T, dense_output=True) 
    return K


def main():
    dir_0full = Path('/mnt/scratch/ande2472/sjrouts/0to264_full/')

    topic_model_0 = BERTopic.load(dir_0full/'model_outliers_reduced')
    with open(dir_0full/'new_docs.pickle', "rb") as f:
        docs_0 = pickle.load(f)


    topic_list = list(docs_0[docs_0['Topic'] == 1].index)
    file = '/mnt/scratch/ande2472/data/0_topjournals/0to264_topjournals_embs.pickle'
    with open(file, "rb") as f:
        emb = pickle.load(f)
    start_time = time.time()

    within_sim_mat = gpu_cosine_similarity(emb[topic_list])
    
    save_dir = dir_0full/'sim_matrix_test.pickle'
    with open(save_dir, "wb") as f:
      pickle.dump(within_sim_mat, f)
    print(f'predictions Saved at: {save_dir}')
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    print(f"Memory usage: {memory_usage()} GB")