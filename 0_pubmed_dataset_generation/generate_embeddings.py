import argparse
import re, pickle, os, torch, csv
import numpy as np
import pandas as pd

import yaml
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer

from dateutil.relativedelta import relativedelta


# def get_args():
#   """Get command-line arguments"""
#   parser = argparse.ArgumentParser(
#     description='Clean Target TSV and Generate Embeddings',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
#   parser.add_argument('-c', '--config_file', 
#                       type=str,
#                       help='Config file path',
#                       default='./config.yaml')


def main():
  
  # print("Get config")
  # args  = get_args()
  # config_file = Path(args.config_file)
  # print(f"  config_file: {config_file}\n")
  # with open(config_file, 'r') as f:
  #     config = yaml.safe_load(f)

  # xmls_path = Path(config['parse_xml']['xml_path'])
  # print(f"  xmls_path: {xmls_path}")
  # output_dir = Path(config['parse_xml']['output_dir'])

  seed = 20220609
  
  
  
  parser = argparse.ArgumentParser(description= "Clean Target TSV and Generate Embeddings")

  #Add arguments for input files and output file
  parser.add_argument("tsv", help="Path to TSV file")

  parser.add_argument("-o", "--output", help="Path to the output dir")

  args = parser.parse_args()
  tsvpath = Path(args.tsv)
  output_dir = Path(args.output)
  filename = tsvpath.stem



  # Setting working directory
 # work_dir   = Path("/mnt/home/shius/scratch/topic_modeling_example")


  # BERT model to use
  model_name     = "allenai/scibert_scivocab_uncased"
  model_name_mod = "-".join(model_name.split("/"))

  # the target term


  ## outputs for topic modeling
  #Declaring output dirs
  output_dir_cleanfiles = output_dir / "clean_files"
  output_dir_genembed = output_dir / "generated_embeddings"
  output_dir_savedcsv = output_dir / "saved_csv"
  
  #creating paths if they dont exist
  if not os.path.exists(output_dir_cleanfiles):
    os.makedirs(output_dir_cleanfiles)
  if not os.path.exists(output_dir_genembed):
    os.makedirs(output_dir_genembed)
  if not os.path.exists(output_dir_savedcsv):
    os.makedirs(output_dir_savedcsv)


  # generated during cleaning and embedding
  docs_clean_file  = output_dir / f"clean_files/{filename}_docs_clean.pickle"
  emb_file         = output_dir/ f"generated_embeddings/{filename}_embeddings_scibert.pickle"
  csvfile = output_dir/f"{filename}_cleaned.csv"

  torch.cuda.is_available(), torch.__version__



  def clean_text(x, stop_words_dict):
      x = str(x)
      x = x.lower()
      # Replace any non-alphanumric characters of any length
      # Q: Not sure what the # character do.
      x = re.sub(r'#[A-Za-z0-9]*', ' ', x)
      # tokenize and rid of any token matching stop words
      tokens = word_tokenize(x)
      x = ' '.join([w for w in tokens if not w in stop_words_dict])
      return x

  #text preprocessing 
  print("Pre-processing docs")
  if docs_clean_file.is_file():
    print(f"Preprocessed doc found for tsv {filename}")
    with open(docs_clean_file, "rb") as f:
      docs_clean = pickle.load(f)
      print(f"Loading preprocessed tsv from {docs_clean_file}")
  else:
    print("  read corpus and process docs")
    corpus_target_df = pd.read_csv(tsvpath, sep='\t', 
                                    )

    corpus_target_df = corpus_target_df[corpus_target_df.duplicated() == False]

    #for use if desired to remove articles if BOTH the abstract AND title are empty
    # corpus_target_df['txt'] = corpus_target_df['Title'].fillna('') + ". " + corpus_target_df['Abstract'].fillna('')
    # num = sum(corpus_target_df['txt'] == ". ")
    # tot = len(corpus_target_df['txt'])
    # print("Txt NAN:", sum(corpus_target_df['txt'] == ". "))
    # print(f'Dropping {num} records (txt contains NaN) out of {tot}')
    # corpus_target_df = corpus_target_df.drop(corpus_target_df[corpus_target_df['txt'] == ". "].index)
    # #fix indexing issue after removing Na's
    # corpus_target_df.reset_index(inplace=True)

    #removes articles if EITHER the abstract OR title are empty
    corpus_target_df['txt'] = corpus_target_df['Title']+ ". " + corpus_target_df['Abstract']
    num = sum(corpus_target_df['txt'].isna())
    tot = len(corpus_target_df['txt'])
    print("Txt NAN:", sum(corpus_target_df['txt'] == ". "))
    print(f'Dropping {num} records (txt contains NaN) out of {tot}')
    corpus_target_df = corpus_target_df.drop(corpus_target_df[corpus_target_df['txt'].isna()].index)
    #fix indexing issue after removing Na's
    corpus_target_df.reset_index(inplace=True)

    docs       = corpus_target_df['txt']
    stop_words = stopwords.words('english')
    stop_words_dict = {}
    for i in stop_words:
      stop_words_dict[i] = 1

    docs_clean = []
    for doc_idx in tqdm(range(len(docs))):
      doc = docs[doc_idx]
      docs_clean.append(clean_text(doc, stop_words_dict))
    with open(docs_clean_file, "wb") as f:
      pickle.dump(docs_clean, f)
      print(f"Cleaned doc stored at {docs_clean_file}")
    corpus_target_df.to_csv(csvfile, sep='\t', index=False)



  #Generate embeddings
  if emb_file.is_file():
    print(f"Embeddings for tsv {filename} exists")
    print(f"Embeddings found at {emb_file}")
  else:
    print(f"Generating embeddings for {filename}")
    
    emb_model  = SentenceTransformer(model_name)
    embeddings = emb_model.encode(docs_clean, batch_size=128, show_progress_bar=True)

    #Running embeddings in parallel
    #pool = emb_model.start_multi_process_pool()
    #embeddings = emb_model.encode_multi_process(docs_clean, pool= pool, batch_size=128)


    # Output embeddings
    with open(emb_file, "wb") as f:
        pickle.dump(embeddings, f)
        print(f"Embeddings stored at {emb_file}")

 # emb_model.stop_multi_process_pool(pool)

if __name__ == '__main__':
    main()