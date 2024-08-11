from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import os
import argparse

def main():
  

  
  
  
    parser = argparse.ArgumentParser(description= "total number of empty files removed")

    #Add arguments for input files and output file
    parser.add_argument("tsv", help="Path to TSV directory")
    parser.add_argument("-output", help="output_dir")
    args = parser.parse_args()
    tsvpath = Path(args.tsv)
    outpath = Path(args.output)
    sortedfile  = outpath / 'sorted_full.csv'
    journalsfile  = outpath / 'journal_groupby.csv'
    
    fulldataset = pd.read_csv(tsvpath, sep='\t')
    fulldataset['date_column'] = pd.to_datetime(fulldataset['Date'])
    sorteddataset = fulldataset.sort_values('date_column')

    sorteddataset.to_csv(sortedfile, sep='\t')
    print(f'Sorted dataset saved at: {sortedfile}')

    journals = sorteddataset.groupby('Journal').sum()
    journals.to_csv(journalsfile, sep='\t')
    print(f'Journal Groupby saved at: {sortedfile}')





    


if __name__ == '__main__':
    main()