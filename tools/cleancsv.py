import argparse
import re, pickle, os, torch, csv
import numpy as np
import pandas as pd
from os import listdir
import yaml
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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



    parser = argparse.ArgumentParser(description= "Clean Target TSV")

    #Add arguments for input files and output file
    parser.add_argument("tsv", help="Path to TSV file")

    parser.add_argument("-o", "--output", help="Path to the output dir")

    args = parser.parse_args()
    tsvpath = Path(args.tsv)
    tsvfiles= listdir(tsvpath)
    tsvfiles.sort()
    #tsvfiles.sort(key=lambda x: int(x.split('_')[0]))
    output_dir = Path(args.output)
    



    # Setting working directory
    # work_dir   = Path("/mnt/home/shius/scratch/topic_modeling_example")




    ## outputs for topic modeling
    #Declaring output dirs
    output_dir_savedcsv = output_dir / "saved_csv"

    #creating paths if they dont exist
    if not os.path.exists(output_dir_savedcsv):
        os.makedirs(output_dir_savedcsv)


    # generated during cleaning and embedding
    totaldropped = 0
    totalrecords = 0
    totalkept = 0
    skippedlist =[]

    torch.cuda.is_available(), torch.__version__


    for file in tqdm(tsvfiles):
        
        
        filepath = tsvpath/file
        filename = filepath.stem
        csvfile = output_dir/f"saved_csv/{filename}_noNA.csv"
        
        #text preprocessing 
        print(f"Pre-processing {filename}")
        #   if docs_clean_file.is_file():
        #     print(f"Preprocessed doc found for tsv {filename}")
        #     with open(docs_clean_file, "rb") as f:
        #       docs_clean = pickle.load(f)
        #       print(f"Loading preprocessed tsv from {docs_clean_file}")
        try:
            corpus_target_df = pd.read_csv(filepath, sep='\t', 
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
        
            print(f'Dropping {num} records (txt contains NaN) out of {tot}')
            corpus_target_df = corpus_target_df.drop(corpus_target_df[corpus_target_df['txt'].isna()].index)
            #fix indexing issue after removing Na's
            corpus_target_df.reset_index(inplace=True)

            corpus_target_df.to_csv(csvfile, sep='\t', index=False)
            totaldropped += num
            totalrecords += tot
            totalkept += tot - num
        except:
            print(f'Error reading {filename}. Skipping...')
            skippedlist += [filename]
    print(f'Total records analyized: {totalrecords}')
    print(f'Total dropped: {totaldropped}')
    print(f'Total kept: {totalkept}')
    print(f'Total files skipped: {len(skippedlist)}')
    print(f'Files skipped: {skippedlist}')


if __name__ == '__main__':
    main()