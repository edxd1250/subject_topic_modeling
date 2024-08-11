## for calculating total amount of documents removed
## to be used after tsv files and clean files are generated

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

    parser.add_argument("-clean", help="Path to cleaned files")
    parser.add_argument("-output", help="output_dir")
    

    args = parser.parse_args()
    tsvpath = Path(args.tsv)
    cleanpath = Path(args.clean)
    outpath = Path(args.output)

    missingness_file  = outpath / "NA_removed_table.pickle"
    cleanfiles = os.listdir(cleanpath)
    ogfiles = os.listdir(tsvpath)

    # so that files are in order
    cleanfiles.sort()
    ogfiles.sort()

    num_files_removed = 0
    files_removed = np.zeros(len(cleanfiles))
    original_count = np.zeros(len(cleanfiles))
    cleaned_count = np.zeros(len(cleanfiles))
    removepct = np.zeros(len(cleanfiles))

    for i in range(len(cleanfiles)):
        path_to_cleanfile = cleanpath / cleanfiles[i]
        path_to_originfile = tsvpath / ogfiles[i]
        try:
            original = pd.read_csv(path_to_originfile, sep='\t')
            with open(path_to_cleanfile, "rb") as f:
                clean = pickle.load(f)
            
            removed = len(original) - len(clean)
            files_removed[i] = removed
            num_files_removed += removed
            original_count[i] = len(original)
            cleaned_count[i] = len(cleaned_count)
            removepct[i] = files_removed[i]/original_count[i]
        except:
            print(f"Problem with {ogfiles[i]}. Skipping...")
    data = np.array([files_removed, removepct, original_count,cleaned_count]).T
    NAtable = pd.dataFrame(data, columns= ['Files_Removed', 'Removed_Pct','Original_Count','Cleaned_Count'])

    print(f'Number of files removed for missingness: {num_files_removed}')
    with open(missingness_file, "wb") as f:
      pickle.dump(NAtable, f)

if __name__ == '__main__':
    main()