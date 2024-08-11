import re, pickle, os, torch, csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bertopic import BERTopic
import datetime
import math
import argparse


def main():


    parser = argparse.ArgumentParser(description= "Partiton Big Dataset into N amount of journals by SJR Journal Rank")
    #parser.add_argument('-d', '--data_file', help='Path to full data csv')
    parser.add_argument('-r','--ranks', nargs='+', help='<Required> Ranks of desired subsets', required=True, type=int)
    parser.add_argument('-n','--num', help='Minimum amount of articles per subset', required=True, type=int)
    args = parser.parse_args()
    partitions = args.ranks
    num = args.num

    cleanfile = '/mnt/home/ande2472/data/bigdataoutput/complete_1147_clean_combined.pickle'
    embfile = '/mnt/home/ande2472/data/bigdataoutput/complete_1147_emb_combined.pickle'
    fulldatafile = '/mnt/scratch/ande2472/data/1147_latest.csv'
    sortedranksfile = '/mnt/home/ande2472/data/full_clean_data/sortedranks.csv'



    fulldata = pd.read_csv(fulldatafile, sep='\t')
    with open(cleanfile, "rb") as f:
       bigcleans = pickle.load(f)
    with open(embfile, "rb") as f:
       bigembs = pickle.load(f)
    sortedranks = pd.read_csv(sortedranksfile, sep='\t')

    def getpartitions(partitions, n=1000000, minjournals = 200):
        ends = []
        for start in partitions:
            end = start+minjournals
            numval = sortedranks['Cumulative Count'][end]-sortedranks['Cumulative Count'][start]
            while numval < n:
                end = end+1
                numval = sortedranks['Cumulative Count'][end]-sortedranks['Cumulative Count'][start]
                if numval >= n:
                    break
            maxval = sortedranks['sjr2020'][start]
            minval = sortedranks['sjr2020'][end]
            numval = sortedranks['Cumulative Count'][end]-sortedranks['Cumulative Count'][start]
            meanval = sortedranks['sjr2020'][start:end].mean()
            print(f'For partition from rank {start} - {end}:')
            print(f'    Number of journals: {end - start}')
            print(f'    Number of articles: {numval}')
            print(f'    Mean SJR: {meanval}')
            print(f'    Max SJR: {maxval}')
            print(f'    Min SJR: {minval}')
            ends += [end]
        return ends
    
    endparts = getpartitions(partitions, num)

    for i in range(len(partitions)):
        startnum = partitions[i]
        endnum = endparts[i]
        topNjournals = sortedranks['Journal'][startnum:endnum].tolist()
        filtered_data = fulldata[fulldata['Journal'].isin(topNjournals)]
        indices_of_filtered_data = filtered_data.index.tolist()
        topN_bigcleans = [bigcleans[x] for x in indices_of_filtered_data]
        topN_bigembs = bigembs[indices_of_filtered_data]
        
        Path(f'/mnt/scratch/ande2472/data/{startnum}_topjournals').mkdir(parents=True, exist_ok=True)


        savenewcleans = f'/mnt/scratch/ande2472/data/{startnum}_topjournals/{startnum}_topjournals_cleans.pickle'
        savenewembs = f'/mnt/scratch/ande2472/data/{startnum}_topjournals/{startnum}_topjournals_embs.pickle'
        savenewfile = f'/mnt/scratch/ande2472/data/{startnum}_topjournals/{startnum}_topjournals.csv'
        with open(savenewcleans, "wb") as f:
           pickle.dump(topN_bigcleans, f)
        print(f'Cleans for {startnum} to {endnum} saved at {savenewcleans}')
        with open(savenewembs, "wb") as f:
           pickle.dump(topN_bigembs, f)
        print(f'Embs for {startnum} to {endnum} saved at {savenewembs}')
        filtered_data.to_csv(savenewfile, sep='\t')
        



if __name__ == '__main__':
    main()