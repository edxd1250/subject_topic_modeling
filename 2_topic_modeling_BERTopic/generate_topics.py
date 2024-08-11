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

# Reproducibility
seed = 20220609

def expand_docs(documents, data, topic_model, save_dir=None):
    docs = documents
    docs['Year'] = pd.to_datetime(data['Date']).dt.year
    docs['Journal'] = data['Journal']
   # docs = docs.drop(columns='ID')

    docs['Date'] = pd.to_datetime(data['Date'])
# documents['Timestamp'] = documents['Date'].dt.timestamp()
    ts_for_bins = list(docs['Date'])
    ts_for_bins.sort()

    bin_num  = 20
    bin_size = int(len(ts_for_bins)/bin_num)
    bin_idxs = [idx for idx in range(0, len(ts_for_bins), bin_size)]

    bin_timestamps = [ts_for_bins[idx] for idx in bin_idxs]

    max_timestamp      = max(ts_for_bins) + pd.Timedelta(1, unit='D')

    bin_df         = pd.DataFrame(list(zip(bin_idxs, bin_timestamps)),
            columns=['bin_start_idx', 'bin_start_date'])

    bin_df['Count'] = bin_df['bin_start_idx'].diff().fillna(bin_df['bin_start_idx'].iloc[0]).astype(int)
    bin_df['bin_end_date'] = bin_df['bin_start_date'].shift(-1) - pd.Timedelta(days=1)
    bin_df['bin_end_date'][20] = max(docs['Date']) + pd.Timedelta(1, unit='D')

    bin_period = []
    docs['bin_period'] = 0

    for i in tqdm(range(len(docs))):
        period = 0
        while docs['Date'][i] > bin_df['bin_end_date'][period] and period < len(bin_df):
            period +=1
        
            # doc = documents['Date'][i]
            # bindate = bin_df['bin_end_date'][period]

            # print(f'Period: {period}.. {doc} < {bindate}')

    # print(f'Assigning Document: {i} bin: {period}')
        docs['bin_period'][i] = period

    if save_dir is not None:
        with open(save_dir/'new_docs.pickle', "wb") as f:
            pickle.dump(docs, f)
        print(f"Docs saved at: {save_dir}")
    
    return docs

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
  
  
  parser = argparse.ArgumentParser(description= "Generate Topics using input Embedding Matrix")

  #Add arguments for input files and output file
  parser.add_argument("emb", help="Path to embedding matrix")
  parser.add_argument("clean", help="Path to cleaned document")


  parser.add_argument("-o", "--output", help="Path to the output dir")
  parser.add_argument("-r", help="Run number (optional for naming purposes)", default=None)

  args = parser.parse_args()
  embpath = Path(args.emb)
  cleanpath = Path(args.clean)
  output_dir = Path(args.output)
  #filename = embpath.stem
  r = args.r 
 # if r is not None:
 #   filename = r + filename


  with open(embpath, "rb") as f:
    embeddings = pickle.load(f)
  
  with open(cleanpath, "rb") as f:
    docs_clean = pickle.load(f)

  # Setting working directory
 # work_dir   = Path("/mnt/home/shius/scratch/topic_modeling_example")

  #device = torch.device('cuda')
  # BERT model to use
  model_name     = "allenai/scibert_scivocab_uncased"
  model_name_mod = "-".join(model_name.split("/"))
  emb_model  = SentenceTransformer(model_name)
  # the target term


  ## outputs for topic modeling
  #Declaring output dirs
  # output_dir_generated_topics = output_dir / "generated_topics"
  # output_dir_generated_model = output_dir / "full_generated_model"
  # output_dir_generated_probabilities = output_dir / "generated_probabilities"
  
  # #creating paths if they dont exist
  # if not os.path.exists(output_dir_generated_model):
  #   os.makedirs(output_dir_generated_model)
  # if not os.path.exists(output_dir_generated_probabilities):
  #   os.makedirs(output_dir_generated_probabilities)
  #   if not os.path.exists(output_dir_generated_topics):
  #       os.makedirs(output_dir_generated_topics)

#
  # generated during cleaning and embedding
  topics_file = output_dir / f"generated_topics.pickle"
  topic_model_file= output_dir / f"generated_model"
  topic_model_file_step5= output_dir/ f"model_outliers_reduced"
  probs_file = output_dir/ f"generated_probabilities.pickle"

  torch.cuda.is_available(), torch.__version__



  # HDBSCAN clustering setting
  # min_cluster_size         = 100 # This is 500 for the full dataset run
  # metric                   = 'euclidean' 
  # cluster_selection_method ='eom' 
  # prediction_data          = True 
  # min_samples              = 5

  # BERTopic setting
  calculate_probabilities = True
  n_neighbors             = 15  
  #nr_topics               = 100
  n_gram_range            = (1,2)

  start_time = time.time()
  #init hbscan
  # hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
  #                       metric=metric, 
  #                       cluster_selection_method=cluster_selection_method, 
  #                       prediction_data=prediction_data, 
  #                       min_samples=min_samples)


  # KeyBERT
  keybert_model = KeyBERTInspired()

  # Part-of-Speech
  #pos_model = PartOfSpeech("en_core_web_sm")

  # MMR
  mmr_model = MaximalMarginalRelevance(diversity=0.3)

  # GPT-3.5
  #client = openai.OpenAI(api_key="sk-...")
  prompt = """
  I have a topic that contains the following documents: 
  [DOCUMENTS]
  The topic is described by the following keywords: [KEYWORDS]

  Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
  topic: <topic label>
  """
  aimodel = "gpt-3.5-turbo"
 # openai_model = OpenAI(client, exponential_backoff=True, chat=True, prompt=prompt)

  # All representation models
  representation_model = {
      "KeyBERT": keybert_model,
    #  "OpenAI": openai_model,  # Uncomment if you will use OpenAI
      "MMR": mmr_model,
     # "POS": pos_model
  }

  hdbscan_gpu = hdbscan.HDBSCAN(min_cluster_size=100,
                                    min_samples=5,
                                    
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True,
                                    output_type = 'numpy')
  
  #init/train topic model
  topic_model = BERTopic(hdbscan_model=hdbscan_gpu,
                      calculate_probabilities=calculate_probabilities,
                       n_gram_range=n_gram_range,
                       #nr_topics=nr_topics,
                       top_n_words=20,
                       representation_model= representation_model,
                       embedding_model = emb_model,

                       verbose=True)
 #
 #  topic_model.tocuda()
  topics, probabilities = topic_model.fit_transform(docs_clean,
                                          embeddings)

  # snapshot = tracemalloc.take_snapshot()
  # # display_top(snapshot)
  # size, peak = tracemalloc.get_traced_memory()
  # print(f"{size=}, {peak=}")

  

  topic_model.save(topic_model_file)
  print(f'Model Saved at: {topic_model_file}')

  with open(topics_file, "wb") as f:
      pickle.dump(topics, f)
  print(f'Topics Saved at: {topics_file}')

  with open(probs_file, "wb") as f:
    pickle.dump(probabilities, f)
  print(f'Probabilities Saved at {probs_file}')

  print("--- %s seconds ---" % round(time.time() - start_time, 2))
  print(f"Memory usage: {memory_usage()} GB")

  print("----step5...----")

  
  probability_threshold = np.percentile(probabilities, 95)
  new_topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 
                                                            for prob in probabilities]
  n_unassigned= pd.Series(new_topics).value_counts().loc[-1]
  n_unassigned/len(new_topics)  

  topic_model.update_topics(docs_clean, new_topics)

  new_documents = pd.DataFrame({"Document": docs_clean, "Topic": new_topics})
  topic_model._update_topic_size(new_documents)

  topic_model.save(topic_model_file_step5)

  topic_info = topic_model.get_topic_info()
  # topic_info.to_csv(outputfile1)
  
  # topic_freq = topic_model.get_topic_freq()
  # topic_freq.to_csv(outputfile2)
  
  # fig = topic_model.visualize_topics()
  # fig.write_html(outputfile3)

  topic_info_sorted = topic_info.sort_values(by=['Count'])
  topic_names  = list(topic_info_sorted.Name.values)
  topic_counts = list(topic_info_sorted.Count.values)
  y_pos        = len(topic_names)

  print("--- %s seconds ---" % round(time.time() - start_time, 2))
  print(f"Memory usage: {memory_usage()} GB")
  data = pd.read_csv(Path(args.csv_file), sep='\t')

  new_docs = expand_docs(new_documents, data, topic_model, output_dir)


if __name__ == '__main__':
    main()