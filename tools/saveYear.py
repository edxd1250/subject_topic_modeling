import re, pickle, os, torch, csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bertopic import BERTopic
import datetime
import math


def main():
    year = [2020]
    cleanfile = '/mnt/home/ande2472/data/bigdataoutput/complete_1147_clean_combined.pickle'
    embfile = '/mnt/home/ande2472/data/bigdataoutput/complete_1147_emb_combined.pickle'
    fulldatafile = '/mnt/scratch/ande2472/data/1147_latest.csv'
    sortedranksfile = '/mnt/home/ande2472/data/full_clean_data/sortedranks.csv'

    savenewcleans = '/mnt/home/ande2472/data/2020journals/2020journals_cleans.pickle'
    savenewembs = '/mnt/home/ande2472/data/2020journals/2020journals_embs.pickle'

    fulldata = pd.read_csv(fulldatafile, sep='\t')
    fulldata['DateTime'] = pd.to_datetime(fulldata['Date'])
    fulldata['Year'] = fulldata['DateTime'].dt.year

    with open(cleanfile, "rb") as f:
        bigcleans = pickle.load(f)
    with open(embfile, "rb") as f:
        bigembs = pickle.load(f)

    year2020articles =  fulldata[fulldata['Year'].isin(year)]
    indices_of_filtered_data = year2020articles.index.tolist()
   
    #
    # sortedranks['Journal'][:400].tolist()
    
    # sortedranks = pd.read_csv(sortedranksfile, sep='\t')
    # 
    # filtered_data = fulldata[fulldata['Journal'].isin(top400journals)]
    # indices_of_filtered_data = filtered_data.index.tolist()
    year2020_bigcleans = [bigcleans[i] for i in indices_of_filtered_data]
    year2020_bigembs = bigembs[indices_of_filtered_data]

    with open(savenewcleans, "wb") as f:
        pickle.dump(year2020_bigcleans, f)
    print(f'Cleans saved at {savenewcleans}')
    with open(savenewembs, "wb") as f:
        pickle.dump(year2020_bigembs, f)
    print(f'Embs saved at {savenewembs}')



if __name__ == '__main__':
    main()