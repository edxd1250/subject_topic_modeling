import pandas as pd 
import random
from pathlib import Path
import argparse
import pickle
import numpy as np
from os import listdir
from tqdm import tqdm

def main():
    #argument parser
    parser = argparse.ArgumentParser(description= "Combine multiple embedding matricies and clean docs into one for testing")

    #Add arguments for input files and output file
    parser.add_argument("embdir", help="Path to the folder where TSV files are located")

    parser.add_argument("cleandir", help="Path to the folder where clean files are located")

    parser.add_argument("-n", help="Number of files to be combined")

    parser.add_argument("-s", help="Start Index")

    parser.add_argument("-o", "--output", help="Path to the output location")
    
    args = parser.parse_args()
    embdir = Path(args.embdir)
    cleandir = Path(args.cleandir)
    num = int(args.n)
    value = int(args.s)
    output = Path(args.output)
    
    embout = output / f'complete_{num}_emb_combined.pickle'
    cleanout = output / f'complete_{num}_clean_combined.pickle'


    # key=lambda x: int(x.split('_')[0])
    embfiles = listdir(embdir)
    embfiles.sort()
    cleanfiles = listdir(cleandir)
    cleanfiles.sort()

    
    #read in files and combine
    

    
    combined_embeddings = []
    combined_cleanfiles = []
    #value =len(embfiles) - 1



    for i in tqdm(range(num)):
        #value = random.randint(0,len(embfiles))
        

        embfile = embdir / embfiles[value]
        cleanfile = cleandir / cleanfiles[value]
        #print(f'Adding {embfiles[value]} and {cleanfiles[value]} to combined file...')
        with open(cleanfile, "rb") as f:
            clean = pickle.load(f)
        with open(embfile, "rb") as f:
            embog = pickle.load(f)
       # print(f"emb shape: {embog.shape}")
        emb = embog.tolist()
        if combined_embeddings is None:
            combined_embeddings = emb
        else:
           # combined_embeddings = np.concatenate((combined_embeddings, emb))
           combined_embeddings += emb
         
        #value = value - 1
        value = value + 1
        
        combined_cleanfiles += clean
        
    print("creating matrix...")   
    matrix = np.array(combined_embeddings)
        

    
            
    
   

    
    print("saving matrix...")
    with open(embout, "wb") as f:
        pickle.dump(matrix, f)
    
    print("saving cleanfiles...")
    with open(cleanout, "wb") as f:
        pickle.dump(combined_cleanfiles, f)
 


    print(f"Combined data saved to {args.output}")
    print(f"Total records: {len(combined_cleanfiles)}")

if __name__ == '__main__':
    main()

    