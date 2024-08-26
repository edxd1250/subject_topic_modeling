import pickle
import numpy as np
from tensorflow.keras import models, preprocessing
from pathlib import Path
import pandas as pd
import argparse
import gensim
import yaml

def load_w2v_model(model_path):
    '''Load a pre-trained Word2Vec model'''
    with open(model_path, "rb") as f:
        model_w2v = pickle.load(f)
    return model_w2v

def load_keras_model(model_path):
    '''Load the pre-trained Keras model'''
    model = models.load_model(model_path)
    return model

def load_tokenizer(tokenizer_path):
    '''Load the trained tokenizer'''
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def get_embeddings(texts, model_w2v, tokenizer, max_len=500):
    '''Generate embeddings for a list of texts'''
    lst_text2seq = tokenizer.texts_to_sequences(texts)
    X_w2v = preprocessing.sequence.pad_sequences(lst_text2seq, maxlen=max_len, padding="post", truncating="post")

    vocab_size = len(tokenizer.word_index) + 1
    embeddings = np.zeros((vocab_size, model_w2v.vector_size))
    for word, idx in tokenizer.word_index.items():
        try:
            embeddings[idx] = model_w2v.wv[word]
        except KeyError:
            pass

    return embeddings, X_w2v

def classify_abstracts(abstracts, w2v_model, keras_model, tokenizer):
    '''Classify a list of abstracts'''
    _, X_w2v = get_embeddings(abstracts, w2v_model, tokenizer)
    predictions = keras_model.predict(X_w2v)
    y_pred = np.argmax(predictions, axis=1)
    return y_pred

def save_classified_abstracts(abstracts, predictions, output_file):
    '''Save classified abstracts to a file'''
    df = pd.DataFrame({'Abstract': abstracts, 'Prediction': predictions})
    df.to_csv(output_file, sep = '\t')
    print(f"Classified abstracts saved to {output_file}")

def main(w2v_model_path, keras_model_path, tokenizer_path, input_file, output_file):
    '''Main function to load models, classify abstracts, and save results'''
    print("Loading models and tokenizer...")
    w2v_model = load_w2v_model(w2v_model_path)
    keras_model = load_keras_model(keras_model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    print("Loading abstracts...")

    with open(input_file, "rb") as f:
        abstracts = pickle.load(f)

    print("Classifying abstracts...")
    predictions = classify_abstracts(abstracts, w2v_model, keras_model, tokenizer)

    print("Saving classified abstracts...")
    save_classified_abstracts(abstracts, predictions, output_file)

def get_args():
  """Get command-line arguments"""
  parser = argparse.ArgumentParser(
    description='Classify abstracts using a pre-trained Word2Vec and Keras model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-c', '--config_file', 
                      type=str,
                      help='Config file path',
                      default='./config.yaml')

  args = parser.parse_args()
  
  return args   
    

if __name__ == '__main__':

    args        = get_args()
    config_file = Path(args.config_file)  # config file path
    print(f"  config_file: {config_file}\n")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
   

    main(config['w2v_model'], config['keras_model'], config['tokenizer'], config['input_file'], config['output_file'])