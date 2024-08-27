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

  r = args.r 



  with open(embpath, "rb") as f:
    embeddings = pickle.load(f)
  
  with open(cleanpath, "rb") as f:
    docs_clean = pickle.load(f)

  # BERT model used for embedding generation
  model_name     = "allenai/scibert_scivocab_uncased"
  model_name_mod = "-".join(model_name.split("/"))
  emb_model  = SentenceTransformer(model_name)


  topics_file = output_dir / f"generated_topics.pickle"
  topic_model_file= output_dir / f"generated_model"
  topic_model_file_step5= output_dir/ f"model_outliers_reduced"
  probs_file = output_dir/ f"generated_probabilities.pickle"

  torch.cuda.is_available(), torch.__version__


  start_time = time.time()



#####################################
  # BERTopic setting
  calculate_probabilities = True
  n_neighbors             = 15  
  #nr_topics               = 100
  n_gram_range            = (1,2)



######################################################
  ## optional representation model add ons 

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
###################################

#optional add on for gpu accelerated hdbscan model

  hdbscan_gpu = hdbscan.HDBSCAN(min_cluster_size=100,
                                    min_samples=5,
                                    
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True,
                                    output_type = 'numpy')

#########################################

  #init/train topic model
  topic_model = BERTopic(hdbscan_model=hdbscan_gpu,
                      calculate_probabilities=calculate_probabilities,
                       n_gram_range=n_gram_range,
                       #nr_topics=nr_topics,
                       top_n_words=20,
                       representation_model= representation_model,
                       embedding_model = emb_model,

                       verbose=True)

  topics, probabilities = topic_model.fit_transform(docs_clean,
                                          embeddings)

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

################################################
  print("----step5...----")

  #reducing outliers and saving reduced outlier model

  probability_threshold = np.percentile(probabilities, 95)
  new_topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 
                                                            for prob in probabilities]
  n_unassigned= pd.Series(new_topics).value_counts().loc[-1]
  n_unassigned/len(new_topics)  

  topic_model.update_topics(docs_clean, new_topics)

  new_documents = pd.DataFrame({"Document": docs_clean, "Topic": new_topics})
  topic_model._update_topic_size(new_documents)

  topic_model.save(topic_model_file_step5)


  print("--- %s seconds ---" % round(time.time() - start_time, 2))
  print(f"Memory usage: {memory_usage()} GB")




if __name__ == '__main__':
    main()