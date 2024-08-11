import pandas as pd
import pickle
import itertools
import sys
import yaml
import argparse
import gensim
from pathlib import Path
from sklearn import model_selection, metrics
import os
import numpy as np
#import tensorrt
from tensorflow.keras import models, layers, callbacks, preprocessing



def get_embeddings(corpus, model_w2v, tokenizer, dic_vocab_token):

  # Transforms each text in texts to a sequence of integers.
  lst_text2seq = tokenizer.texts_to_sequences(corpus)

  # pad or trucate sequence
  X_w2v = preprocessing.sequence.pad_sequences(
                    lst_text2seq,      # List of sequences, each a list of ints 
                    maxlen=500,        # maximum length of all sequences
                    padding="post",    # 'pre' or 'post' 
                    truncating="post") # remove values from sequences > maxlen

  ## start the matrix (length of vocabulary x vector size) with all 0s

  embeddings = np.zeros((len(dic_vocab_token)+1, 300))
  not_in_emb = {}
  for word, idx in dic_vocab_token.items():
      ## update the row with vector
      try:
          embeddings[idx] =  model_w2v.wv[word]
      ## if word not in model then skip and the row stays all 0s
      except KeyError:
          not_in_emb[word] = 1

  return embeddings, X_w2v

def train_tokenizer(corpus, param):
  '''Train a tokenizer
  Args:
    corpus (list): a nested list of word lists
    param (list): for tokenizer and vocab output file names
  Return:
    tokenizer (keras.preprocessing.text.Tokenizer): trained tokenizer
    dic_vocab_token (dict): token as key, index as value
  '''

  # intialize tokenizer
  # See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
  # This is replaced by tf.keras.layers.TextVectorization
  tokenizer = preprocessing.text.Tokenizer(lower=True, split=' ', 
                oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

  # tokenize corpus 
  tokenizer.fit_on_texts(corpus)

  # get token dictionary, with token as key, index number as value
  dic_vocab_token = tokenizer.word_index

  # Save tokenizer and vocab
  [min_count, window, ngram] = param
  tok_name   = work_dir / f"model_cln_w2v_token_{min_count}-{window}-{ngram}"
  vocab_name = work_dir / f"model_cln_w2v_vocab_{min_count}-{window}-{ngram}"

  if not tok_name.is_file():
    with open(tok_name, "wb") as f:
      pickle.dump(tokenizer, f)

  if not vocab_name.is_file():
    with open(vocab_name, "wb") as f:
      pickle.dump(dic_vocab_token, f)

  return tokenizer, dic_vocab_token   

def split_train_validate_test(labels, cleans, rand_state):
  '''Load data and split train, validation, test subsets for the cleaned texts
  Args:
    corpus_combo_file (str): path to the json data file
    rand_state (int): for reproducibility
  Return:
    train, valid, test (pandas dataframes): training, validation, testing sets
  '''
  clean_txt = pd.Series(cleans, name= 'Text')
  corpus = pd.DataFrame([labels, clean_txt])
  corpus = corpus.T
  # Split train test
  train, test = model_selection.train_test_split(corpus, 
      test_size=0.2, stratify=corpus['Label'], random_state=rand_state)

  # Split train validate
  train, valid = model_selection.train_test_split(train, 
      test_size=0.25, stratify=train['Label'], random_state=rand_state)

  X_train = train['Text']
  X_valid = valid['Text']
  X_test  = test['Text']
  y_train = train['Label'].astype(np.int32)
  y_valid = valid['Label'].astype(np.int32)
  y_test  = test['Label'].astype(np.int32)

  print(f"    size: train={X_train.shape}, valid={X_valid.shape}," +\
        f" test={X_test.shape}")

  return [X_train, X_valid, X_test, y_train, y_valid, y_test]

def get_hyperparameters(w2v_param):
  ''' Return a list with hyperparameters based on the passed dictionary
  Adopted from:
    https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
  Args:
    param (dict): a dictionary specified in the config.txt file.
  Return:
    param_list (list): a nested list of hyperparameters 
  '''
  print(w2v_param)
  keys, values = zip(*w2v_param.items())
  param_list = [v for v in itertools.product(*values)]
  
  return keys, param_list

def get_unigram(corpus):
  unigram = []
  for txt in corpus:
    lst_words = txt.split()
    unigram.append(lst_words)

  return unigram

def get_ngram(X_corpus, ngram, min_count, subset, work_dir):
  '''Check if ngrams files exisit, if not get ngrams based on passed parameters
  Args:
    X_corpus (pandas series): texts to get ngrams from
    ngram (int): uni (1), bi (2), or tri (3) grams
    min_count (int): minmumal number of term occurence in corpus
    subset (str): train, valid, or test; for file name
    work_dir (Path): does not really need this for call within this script, but
      if called as module, this needs to be passed. So make this required.
  Output:
    ngram_file (pickle): model_cln_ngrams_{subset}_{min_count}-{ngram}
  Return:
    unigrams, bigrams, or trigrams
  '''

  # Check if ngram file exist
  ngram_file = work_dir / f"model_cln_ngrams_{subset}_{min_count}-{ngram}"
  # if ngram_file.is_file():
  #   print("    load ngrams")
  #   with open(ngram_file, "rb") as f:
  #       ngrams = pickle.load(f)
  #   return ngrams

  #else:
    # ngrams file does not exist, generate it
  print("    generate ngrams")


  unigrams = get_unigram(X_corpus)
  ngrams = unigrams
  if ngram == 1:
    print('ngram=1')
    ngrams = unigrams
  # ngram >1
  else:
    # Get bigrams
    bigrams_detector  = gensim.models.phrases.Phrases(
                    unigrams, delimiter=" ", min_count=min_count, threshold=10)
    bigrams_detector  = gensim.models.phrases.Phraser(bigrams_detector)
    bigrams = list(bigrams_detector[unigrams])

    # Return bigrams
    if ngram == 2:
      ngrams = bigrams
    # Get trigrams and return them
    elif ngram == 3:
      trigrams_detector = gensim.models.phrases.Phrases(
                      bigrams_detector[unigrams], delimiter=" ", 
                      min_count=min_count, threshold=10)
      trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
      trigrams = list(trigrams_detector[bigrams])
      ngrams = trigrams
    else:
      print('ERR: ngram cannot be larger than 3. QUIT!')
      sys.exit(0)

  # write ngram file
  with open(ngram_file, "wb") as f:
      pickle.dump(ngrams, f)      

  return ngrams


def get_w2v_model(X_train, X_valid, X_test, param, rand_state):
  '''Get ngram lists and w2v model
  Args:
  Return:
  '''
  [min_count, window, ngram] = param

  print("    ngrams for training")
  ngram_train = get_ngram(X_train, ngram, min_count, "train", work_dir) 
  print("    ngrams for validation")
  ngram_valid = get_ngram(X_valid, ngram, min_count, "valid", work_dir)
  print("    ngrams for testing")
  ngram_test  = get_ngram(X_test , ngram, min_count, "test", work_dir)

  # Check if w2v model is already generated
  model_w2v_name = work_dir / f"model_cln_w2v_{min_count}-{window}-{ngram}.keras"

  if model_w2v_name.is_file():
    print("   load the w2v model")
    with open(work_dir / model_w2v_name, "rb") as f:
        model_w2v = pickle.load(f)
  else:
    print("   geneate and save w2v model")
    model_w2v = gensim.models.Word2Vec(ngram_train, vector_size=300, 
                                      window=window, min_count=min_count, 
                                      sg=1, epochs=30, seed=rand_state)
    
    with open(model_w2v_name, "wb") as f:
      pickle.dump(model_w2v, f)

  return model_w2v, model_w2v_name, ngram_train, ngram_valid, ngram_test

def get_w2v_emb_model(embeddings):
  '''Build a deep learning model with Word2Vec embeddings
  Args:
    embeddings
  '''

  ## code attention layer
  def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

  ## input
  x_in = layers.Input(shape=(500,)) ## embedding
  x = layers.Embedding(input_dim=embeddings.shape[0],  
                      output_dim=embeddings.shape[1], 
                      weights=[embeddings],
                      input_length=500, trainable=False)(x_in)

  ## apply attention
  x = attention_layer(x, neurons=500)

  ## 2 layers of bidirectional lstm
  x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
                          return_sequences=True))(x)
  x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)

  ## final dense layers
  x = layers.Dense(64, activation='relu')(x)
  y_out = layers.Dense(2, activation='softmax')(x)

  ## Initialize and compile model
  model = models.Model(x_in, y_out)
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', 
                metrics=['accuracy'])

  return model

def predict_and_output(model_emb, corpus_pred_file, X_w2v, X, y):

  # prediction probability
  print("    get prediction probability")
  y_prob  = model_emb.predict(X_w2v)

  # label mapping
  y_map   = {n:label for n,label in enumerate(np.unique(y))}
  # prediction
  print("    get predictions")
  #y_pred  = pd.Series([y_map[np.argmax(pred)] for pred in y_prob])
  y_pred  = [y_map[np.argmax(pred)] for pred in y_prob]

  # Convert y_prob to series. There are probabilities for two classes. Take
  # the column with class=1 (2nd column)
  y_prob_series = pd.Series(y_prob[:,1], index=y.index)

  # convert y_pred to series
  y_pred_series = pd.Series(y_pred, index=y.index)

  # dataframe with everything
  pred_df = pd.DataFrame({'y': y, 
                          'y_pred': y_pred_series, 
                          'y_prob': y_prob_series, 
                          'X':X})

  print("    write prediciton dataframe")
  pred_df.to_csv(corpus_pred_file, sep="\t")

  score = metrics.f1_score(y, y_pred)
  print("    F1=", score)

  return score

def run_pipeline(param, subsets, rand_state):
  '''Carry out the major steps'''

  #rand_state = config_dict['rand_state']

  [X_train, X_valid, X_test, y_train, y_valid, y_test] = subsets

  # Get list of ngrams and w2v model
  print("  get list of ngrams and w2v model")
  model_w2v, model_w2v_name, ngram_train, ngram_valid, ngram_test = \
                      get_w2v_model(X_train, X_valid, X_test, param, rand_state)
  
  # Train tokenizer
  print("  train tokenizer")
  tokenizer, dic_vocab_token = train_tokenizer(ngram_train, param)

  # Get embeddings
  print("  get embeddings")
  embeddings, X_train_w2v = get_embeddings(ngram_train, model_w2v, 
                                                    tokenizer, dic_vocab_token)
  _, X_valid_w2v = get_embeddings(ngram_valid, model_w2v, 
                                                    tokenizer, dic_vocab_token)
  _ , X_test_w2v  = get_embeddings(ngram_test , model_w2v, 
                                                    tokenizer, dic_vocab_token)
  emb_output_file = work_dir/"embeddings.pkl"
  x_train_file = work_dir/"x_train.pkl"
  x_train_w2v_file = work_dir/"x_train_w2v.pkl"

  with open(emb_output_file, "wb") as f:
      pickle.dump(embeddings, f)    
  with open(x_train_file, "wb") as f:
      pickle.dump(X_train, f)    
  with open(x_train_w2v_file, "wb") as f:
      pickle.dump(X_train_w2v, f)    
  print("saved embeddings and training set")
  # Model checkpoint path and output model file name
  cp_filepath  = Path(str(model_w2v_name) + "_dnn.keras")

  # Load model if exists
  if cp_filepath.is_dir():
    print("  load model in:", cp_filepath)
    model_emb = models.load_model(cp_filepath)

  # Train and save model if not
  else:
    print("  train model")
    model_emb    = get_w2v_emb_model(embeddings)

    # setup check points
    callback_es  = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callback_mcp = callbacks.ModelCheckpoint(filepath=cp_filepath, mode='max', 
            save_weights_only=False, monitor='val_accuracy', save_best_only=True)

    # Train model
    history = model_emb.fit(x=X_train_w2v, y=y_train, batch_size=256, 
                            epochs=20, shuffle=True, verbose=1, 
                            validation_data=(X_valid_w2v, y_valid), 
                            callbacks=[callback_es, callback_mcp])
    

  print("  output predictions of training data")
  train_pred_file = work_dir / "corpus_train_pred"
  predict_and_output(model_emb, train_pred_file, X_train_w2v, X_train, y_train)

  print("  output validation predictions and f1 score")
  valid_pred_file = work_dir / "corpus_valid_pred"
  valid_score = predict_and_output(
                     model_emb, valid_pred_file, X_valid_w2v, X_valid, y_valid)

  print("  output test predictions and f1 score")
  test_pred_file = work_dir / "corpus_test_pred"
  test_score = predict_and_output(
                     model_emb, test_pred_file, X_test_w2v, X_test, y_test)

  # provide some space between runs
  print('\n')

  return valid_score, cp_filepath, test_score  

def run_main_function():

  # Split train/validate/test for cleaned text
  #   Will not focus on original due to issues with non-alphanumeric characters
  #   and stop words.
  print("\nRead file and split train/validate/test...")
  subsets = split_train_validate_test(text['Label'],text_clean, random_seed)

  # get w2c parameter list
  #   [min_count, window, ngram]
  param_keys, param_list  = get_hyperparameters(w2v_param)

  # iterate through different parameters
  with open(work_dir / f"scores_cln_w2v", "w") as f:
    f.write("run\ttxt_flag\tlang_model\tparameters\tvalidate_f1\t" +\
            "test_f1\tmodel_dir\n")
    run_num = 0
    for param in param_list:
      print(f"\n## param: {param}")
      valid_score, model_dir, test_score = run_pipeline(param, subsets, random_seed)

      f.write(f"{run_num}\t{str(param)}\t"+\
              f"{valid_score}\t{test_score}\t{model_dir}\n")

      run_num += 1


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
    
 
    

if __name__ == '__main__':
    
 

    print("Getting config...")
    args        = get_args()
    config_file = Path(args.config_file)  # config file path
    print(f"  config_file: {config_file}\n")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    dataframe_file = config['model_hyperparams']['w2v']['dataframe_file']
    txt_clean_file = config['model_hyperparams']['w2v']['txt_clean_file']
    random_seed = config['env']['rand_seed']
    output_dir = Path(config['env']['output_dir'])
    working_dir = output_dir/config['env']['model_dir_name']
    work_dir = working_dir/config['model_hyperparams']['w2v']['run_name']
    if not os.path.exists(work_dir):
      os.makedirs(work_dir)

    w2v_param = {'min_count':config['model_hyperparams']['w2v']['min_count'],'window': config['model_hyperparams']['w2v']['window'],'ngram':config['model_hyperparams']['w2v']['ngram']}
    print("loading dataframe...")
    text = pd.read_csv(dataframe_file, sep='\t')
    with open(txt_clean_file, "rb") as f:
        print("loading text...")
        text_clean = pickle.load(f)
    run_main_function()
    print("Done!")
