''' 
Changelog:

06/08/2024: Added negative set and labels


'''

from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pd
import yaml
from tqdm import tqdm


def create_corpus(dataset, journals, cleans, n=None):
    '''
    Create subset of given dataset using list of given journal names

    Parameters:
    dataset (pandas df): Dataframe of pubmed articles. Dataframe must be in the format of previously parsed 
                        pubmed articles.
    journals (list of str): List of journals to be filtered. Names must specifically match Pubmed journal titles. 
                            For more information see classification.ipynb
    cleans (list of str): List of text preprocessed articles corresponding with given dataset.

    Returns:
    filtered_data (pandas df): Subset of given dataset containing only articles in given journal list
    indices_of_filtered_data (list of int): List of indicies of selected articles. Useful for matching with cleans or embeddings
    new_clean: Subset of cleans corresponding with filtered_data

    
    '''
    if n is None: 
        filtered_data = dataset[dataset['Journal'].isin(journals)]
        filtered_data['Label'] = 1
        indices_of_filtered_data = filtered_data.index.tolist()
        new_clean = [cleans[x] for x in tqdm(indices_of_filtered_data)]
        return filtered_data, indices_of_filtered_data, new_clean
    else: 
        filtered_data = dataset[dataset['Journal'].isin(journals)]
        sampled_data = filtered_data.sample(n=n)
        sampled_data['Label'] = 1
        indices_of_sampled_data = sampled_data.index.tolist()
        new_clean = [cleans[x] for x in tqdm(indices_of_sampled_data)]
        return sampled_data, indices_of_sampled_data, new_clean


def create_neg_set(dataset, journals, cleans, n):
    filtered_data = dataset[-dataset['Journal'].isin(journals)]
    sampled_data = filtered_data.sample(n=n)
    sampled_data['Label'] = 0
    indices_of_sampled_data = sampled_data.index.tolist()
    new_clean = [cleans[x] for x in tqdm(indices_of_sampled_data)]
    return sampled_data, indices_of_sampled_data, new_clean
    

def get_args():
  """Get command-line arguments"""
  parser = argparse.ArgumentParser(
    description='Generate training corpus for journal-type classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-c', '--config_file', 
                      type=str,
                      help='Config file path',
                      default='./config.yaml')

  args = parser.parse_args()
  
  return args
def main():
    print(f"Generating positive set...")
    if use_full_pos == True: 
        new_data, indicies, new_cleans = create_corpus(dataset, journalslist, cleans)
    else:

        new_data, indicies, new_cleans = create_corpus(dataset, journalslist, cleans, num_pos)
    pos_len = len(new_cleans)
    print(f"    positive set of {pos_len} samples generated.")
    total_number = pos_len/pos_rate
    number = int(total_number - pos_len)
    print(f"Generating negative set...")
    neg_data, neg_indicies, neg_cleans = create_neg_set(dataset, journalslist, cleans, number)
    neg_len = len(neg_cleans)
    print(f"    negative set of {neg_len} samples generated.")
    total_data = pd.concat([new_data,neg_data])
    total_indicies = indicies + neg_indicies
    total_cleans = new_cleans + neg_cleans
    tot_len = len(total_cleans)
    print(f"Full dataset length: {tot_len}")

    with open(indicies_output, "wb") as f:
        pickle.dump(total_indicies,f)
    print(f"Indicies saved at: {indicies_output}")

    with open(clean_output, "wb") as f:
        pickle.dump(total_cleans,f)
    print(f'Cleans saved at: {clean_output}')

    total_data.to_csv(filtered_output, sep='\t')
    print(f'Dataset saved at: {filtered_output}')


 

    

    
    

        
    

if __name__ == '__main__':

       #parser = argparse.ArgumentParser(description= "Generate TF-IDF matrix for given corpus and matrix")

    #Add arguments for input files and output file
    # parser.add_argument("-d", "--dataset", help="Path to dataset")
    # parser.add_argument("-c", "--corpus", help="Path to cleaned text corpus")
    # parser.add_argument("-o", "--output", help="Path to the output dir")
    # parser.add_argument("-n", "--name", help="<Optional> naming scheme", default='')
    #args = parser.parse_args()
    print("Getting config...")
    args        = get_args()
    config_file = Path(args.config_file)  # config file path
    print(f"  config_file: {config_file}\n")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    corpus_path = Path(config['env']['clean_path'])
    dataset_path = Path(config['env']['dataset_path'])
    output_dir = Path(config['env']['output_dir'])
    



    corpus_path = Path(config['env']['clean_path'])
    dataset_path = Path(config['env']['dataset_path'])
    output_dir = Path(config['env']['output_dir'])
    journals_dir = Path(config['env']['journals_dir'])
    
    corpus_output_path = Path(output_dir/config['env']['data_dir_name'])
    corpus_output_path.mkdir(parents=True, exist_ok=True)
    
    name = config['create_corpus']['run_name']
    issnlist = config['create_corpus']['issn_list']
    pos_rate = config['create_corpus']['pos_rate']
    use_full_pos =  config['create_corpus']['use_full_pos']
    num_pos = config['create_corpus']['num_pos']

    journals = pd.read_csv(journals_dir, sep='\t')
    journalslist = list(journals[journals['issn'].isin(issnlist)]['Journal'])
    print("Loading dataset...")
    dataset = pd.read_csv(dataset_path, sep='\t')

    print("Loading cleans...")
    with open(corpus_path, "rb") as f:
        cleans = pickle.load(f)


    if type(name) is list:
        for x in name:
            new_name = x
            filtered_output = corpus_output_path/f'{new_name}_filtered_data.tsv'
            indicies_output = corpus_output_path/f'{new_name}_filtered_indicies.pickle'
            clean_output = corpus_output_path/f'{new_name}_filtered_cleans.pickle'
            main()

    elif name is None:
        filtered_output = corpus_output_path/f'filtered_data.tsv'
        indicies_output = corpus_output_path/f'filtered_indicies.pickle'
        clean_output = corpus_output_path/f'filtered_cleans.pickle'
        main()
    else: 
        filtered_output = corpus_output_path/f'{name}_filtered_data.tsv'
        indicies_output = corpus_output_path/f'{name}_filtered_indicies.pickle'
        clean_output = corpus_output_path/f'{name}_filtered_cleans.pickle'
        main()
    
    