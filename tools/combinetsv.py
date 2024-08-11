import pandas as pd 
from pathlib import Path
import argparse
from tqdm import tqdm
from os import listdir

def main():
    #argument parser
    parser = argparse.ArgumentParser(description= "Combine multiple TSV files into one.")

    #Add arguments for input files and output file
    parser.add_argument("filedir", help="Path to the folder where TSV files are located")

    parser.add_argument("-o", "--output", help="Path to the output TSV file")

    parser.add_argument('-n', "--number", help="Number")
    
     
    args = parser.parse_args()
    xmls = Path(args.filedir)
   
    xmlfiles = listdir(xmls)
    xmlfiles.sort()
    #xmlfiles.sort(key=lambda x: int(x.split('_')[0]))
    #read in files and combine
    if args.number == 'full':
        num = len(xmlfiles)
    else:  
        num = int(args.number)
    n =0
    combined_df = []
    for i in tqdm(range(num)):

        if xmlfiles[n].endswith(".csv"):
            print(f'Combining: {xmlfiles[n]}')
            try:
                df = pd.read_csv(xmls/xmlfiles[n], delimiter='\t', on_bad_lines='warn')
                combined_df.append(df)
                
            except pd.errors.ParserError as e:
                print(f"Error reading {xmlfiles[n]}: {e}")
        n += 1
    combined_df = pd.concat(combined_df, axis=0, ignore_index=True)    
            
    
   

    output_dir = Path(args.output)
    output_file = output_dir/f'{num}_latest.csv'

    combined_df.to_csv(output_file, sep='\t', index=False)

    print(f"Combined data saved to {output_file}")

main()

    