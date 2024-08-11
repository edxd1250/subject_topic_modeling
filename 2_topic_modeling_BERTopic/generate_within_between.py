import argparse
import re, pickle, os, torch, csv
import numpy as np
import pandas as pd

import tracemalloc
import linecache
from pathlib import Path
from tqdm import tqdm

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from cuml.cluster import hdbscan
from datetime import datetime
import time
import resource
import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
import plotly.figure_factory as ff
from cuml.preprocessing import Normalizer
from sklearn.utils.extmath import safe_sparse_dot
import math

# Reproducibility
seed = 20220609

def generate_within_between(emb, docs, labels, save_dir, max_val=1000000, topics=None, ):
    if topics is None:
        for topic in tqdm(docs['Topic'].unique()):
            topic_list = list(docs[docs['Topic'] == topic].index)
            topic_unlist = list(docs[docs['Topic'] != topic].index)
            within_similarity = gpu_cosine_similarity(emb[topic_list],max=max_val)
            between_similarity = gpu_cosine_similarity(emb[topic_unlist],max=max_val)
            colors = ['rgb(0, 200, 200)','rgb(0, 0, 100)']
            fig = ff.create_distplot([within_similarity,between_similarity], ['Within Topic','Between Topic'],bin_size=.025, show_rug=False, colors=colors)
            label = labels[topic]
            fig.update_layout(title_text=f'Document Similarity Distribution for {label}')
            
            fig.update_xaxes(title_text='Cosine Similarity Score')
            fig.update_yaxes(showgrid=False, title_text='Relative Frequency')
            
            fig.write_image(save_dir/f"topic_{topic}_within_between_sim.pdf")
         

    else:
        for topic in tqdm(topics):
            topic_list = list(docs[docs['Topic'] == topic].index)
            topic_unlist = list(docs[docs['Topic'] != topic].index)
            within_similarity = gpu_cosine_similarity(emb[topic_list],max=max_val)
            between_similarity = gpu_cosine_similarity(emb[topic_unlist],max=max_val)
            colors = ['rgb(0, 200, 200)','rgb(0, 0, 100)']
            fig = ff.create_distplot([within_similarity,between_similarity], ['Within Topic','Between Topic'],bin_size=.025, show_rug=False, colors=colors)
            label = labels[topic]
            fig.update_layout(title_text=f'Document Similarity Distribution for {label}')
            
            fig.update_xaxes(title_text='Cosine Similarity Score')
            fig.update_yaxes(showgrid=False, title_text='Relative Frequency')
   
            fig.write_image(save_dir/f"topic_{topic}_within_between_sim.pdf")
           
# replication of scikit-learn's cosine similarity, with cuml's normalizer for speed
def gpu_cosine_similarity(matrix, matrix2=None, max=None):
    norm = Normalizer()
    norm_matrix = norm.transform(matrix)
    if max is None:
        if matrix2 is None:
            K = safe_sparse_dot(norm_matrix, norm_matrix.T, dense_output=True) 
        else:
            norm_matrix2 = norm.transform(matrix2)
            K = safe_sparse_dot(norm_matrix, norm_matrix2.T)
        return K
    else:
        similarity_values = []
        if matrix2 is None:
            num_samples = round(math.sqrt(max))
            idx1 = np.random.randint(0, matrix.shape[0], num_samples)
            idx2 = np.random.randint(0, matrix.shape[0], num_samples)
            for i in idx1:
                for j in idx2:
                    similarity_values += [np.dot(norm_matrix[i],norm_matrix[j])]
        else:
            norm_matrix2 = norm.transform(matrix2)
            num_samples = round(math.sqrt(max))
            idx1 = np.random.randint(0, matrix.shape[0], num_samples)
            idx2 = np.random.randint(0, matrix2.shape[0], num_samples)
            for i in idx1:
                for j in idx2:
                    similarity_values += [np.dot(norm_matrix[i],norm_matrix2[j])]
        return similarity_values
                    



def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2)  # Convert to kilobytes

def get_args():
  """Get command-line arguments"""
  parser = argparse.ArgumentParser(
    description='Generate topics from pre-generated Sci-BERT embeddings',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-c', '--config_file', 
                      type=str,
                      help='Config file path',
                      default='./config.yaml')

  args = parser.parse_args()
  
  return args

def main():
  
  
  parser = argparse.ArgumentParser(description= "Generate Within/Between Topic Similarity Graphs for a given model")

  #Add arguments for input files and output file
  parser.add_argument("-m", "--model_dir", help="Path to topic model")
  parser.add_argument("-e", "--emb_dir", help="Path to embedding matrix")
  parser.add_argument("-d", "docs_dir",help="Path to docs file")

  parser.add_argument("-o", "--output", help="Path to the output dir")


  args = parser.parse_args()
  modelpath = Path(args.model_dir)
  embpath = Path(args.emb_dir)
  docspath = Path(args.docs_dir)
  plots_dir = Path(args.output)
  #filename = embpath.stem


  #load requirements
  with open(embpath, "rb") as f:
    emb = pickle.load(f)
  
  with open(docspath, "rb") as f:
    docs = pickle.load(f)

  topic_model_0 = BERTopic.load(modelpath)

  #generate topic labels
  topic_labels = topic_model_0.generate_topic_labels(nr_words=3, word_length=20, aspect="KeyBERT", separator='|')
  generate_within_between(emb, docs, topic_labels, save_dir=plots_dir)


if __name__ == '__main__':
    main()