import pandas as pd 
from pathlib import Path
import argparse
from tqdm import tqdm
from os import listdir
import time
import resource
import math

def main():
  

  
  
  
    parser = argparse.ArgumentParser(description= "Split a combined tsv file into n partitions")

    #Add arguments for input files and output file
    parser.add_argument("tsv", help="Path to TSV file")

    parser.add_argument("-n", help="Number of partitions")
    parser.add_argument("-output", help="output_dir")
    

    args = parser.parse_args()
    tsv = Path(args.tsv)
    n = int(args.n)
    outpath = Path(args.output)
    print('Loading Data...')
    data = pd.read_csv(tsv, sep='\t')
    print("Splitting TSVs...")
    splittsv(data, n, outpath)


    
def splittsv(data, n_parts, save_dir):
        dat_len = len(data)
        part_len = math.floor(dat_len/n_parts)
        remainder = len(data)%n_parts
        n = 0
        for i in tqdm(range(n_parts)):
            if i != n_parts - 1:
                newdat = data.iloc[n:n+part_len]
            
                savefile = save_dir/f'{i + 1:03d}_ordered.csv'
                newdat.to_csv(savefile, sep='\t', index=False)
                n += part_len
            if i == n_parts-1:
                newdat = data.iloc[n:]
                savefile = save_dir/f'{i + 1:03d}_ordered.csv'
                newdat.to_csv(savefile, sep='\t', index=False)
                n += part_len+remainder

def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2)  # Convert to kilobytes

if __name__ == '__main__':
    main()