#!/usr/bin/env python3
# Author: Armit
# Create Time: 2019/12/12

from os import path
import sys
import re
from time import time
import pickle
from string import punctuation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from gensim.models.word2vec import Word2Vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor

sys.path.append(path.dirname(path.abspath(__file__)))
from metrics import kappa

BASE_PATH = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
CORPUS_PATH = path.join(BASE_PATH, 'corpus')
DATA_PATH = path.join(CORPUS_PATH, 'essay_data')
MODEL_PATH = path.join(BASE_PATH, 'model')
DATA_FILE = path.join(MODEL_PATH, 'aes_dataset.pkl')
SOLUT_FN = 'MG1933029.tsv'
#SOLUT_FN = 'aes_NN.tsv'


def timer(fn):
  def wrapper(*args, **kwargs):
    t = time()
    r = fn(*args, **kwargs)
    print('[Timer] %r done in %.2f seconds' % (fn.__name__, time() - t))
    return r
  return wrapper

@timer
def load_dataset() -> { str: pd.DataFrame }:
  if path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as fp:
      ds = pickle.load(fp)
  else:
    ds = { }
    HEADERS = ['id', 'topic', 'text', 'score']
    
    # train and test/validate: [essay_set, essay_id, essay, domain1_score]
    train = pd.read_csv(path.join(DATA_PATH, 'train.tsv'), sep='\t', usecols=[0, 1, 2, 6])
    dev   = pd.read_csv(path.join(DATA_PATH, 'dev.tsv'  ), sep='\t', usecols=[0, 1, 2, 6])
    merged = pd.concat([train, dev])
    merged.columns = HEADERS
    ds['train'] = merged
    # appliance: [essay_set, essay_id, essay, -1]
    solut = pd.read_csv(path.join(DATA_PATH, 'test.tsv' ), sep='\t', usecols=[0, 1, 2])
    solut.insert(loc=len(solut.columns), column='score', value=-1)
    solut.columns = HEADERS
    ds['solut'] = solut
    
    # init save
    save_dataset(ds)
  return ds

@timer
def save_dataset(ds):
  with open(DATA_FILE, 'wb') as fp: 
    pickle.dump(ds, fp)

@timer
def wv_pre_model(ds, wv_dim=300):
  WV_MODEL_FILE = path.join(MODEL_PATH, 'aes_NN.wv')
  if path.exists(WV_MODEL_FILE):
    model = Word2Vec.load(WV_MODEL_FILE)
    wv = model.wv ; del model
  else:
    sents = [ ]
    for df in [ds['train'], ds['solut']]:
      for doc in df['DOC']:
        doc = " ".join([tok.lemma_.lower() for tok in doc])
        doc = re.split(r"[\.?!;] ", doc)
        doc = [re.sub(r"[\.,;:!?]", "", sent) for sent in doc]
        doc = [sent.split() for sent in doc]
        sents += doc
    
    print("[wv] training on %d sents" % len(sents))
    model = Word2Vec(sents, size=wv_dim, window=5, min_count=3, workers=16, sg=1)
    model.save(WV_MODEL_FILE)
    wv = model.wv ; del model
  print("[wv] %d vocabs embeded into %d dims" % (len(wv.vocab), wv_dim))
  
  return wv

def text_to_fv_avg(wv, bow, text_dim=300):
  fv_avg, cnt = np.zeros((text_dim,), dtype=np.float32), 0
  if bow != { }:
    for word in bow:
      if word in wv.vocab:
        fv_avg += wv[word]
        cnt += 1
    if cnt: fv_avg /= cnt
  return fv_avg

@timer
def abstract_featdict(df):
  nlp = spacy.load('en_core_web_sm')
  print('[Feature] abstracting doc...')
  df['DOC'] = df.apply(lambda e: nlp(e['text'], disable=['parser', 'ner']), axis=1)
  df['TOK'] = df.apply(lambda e: [tok.lemma_.lower().strip() for tok in e['DOC'] if \
      (tok.lemma_ != '-PRON-' and not tok.text in STOP_WORDS and tok.text not in punctuation)], axis=1)
  df['BOW'] = df.apply(lambda e: set(e['TOK']), axis=1)
  return df

@timer
def go():
  ds = load_dataset()
  pd_train, pd_solut = ds['train'], ds['solut']
  
  first = False
  if first:
    # traditional featvecs for each text
    pd_train = abstract_featdict(pd_train)
    pd_solut = abstract_featdict(pd_solut)

    # wv for each text
    TEXT_DIM = 300
    wv = wv_pre_model(ds, wv_dim=TEXT_DIM)
    for df in [pd_train, pd_solut]:
      df['wv'] = df.apply(lambda e: text_to_fv_avg(wv, e['BOW'], TEXT_DIM), axis=1)
    
    # save data
    save_dataset(ds)

  # featvec for each text 
  FV_NAMES = [
    'wv',         # here!!
    # 'len', 
    # 'tok', 
    # 'tok_wrong', 
    # 'tok_uniq', 
    # 'tok_long', 
    # 'tok_stop',
    # 'sent', 
    # 'sent_complex_max', 
    # 'sent_len_mean', 
    # 'sent_long',
    # 'noun',
    # 'propn', 
    # 'adj', 
    # 'pron', 
    # 'verb', 
    # 'adv', 
    # 'cconj',
    # 'det', 
    # 'part', 
    # 'punct', 
    # 'comma',
  ]
  df = pd_train
  # for col in FV_NAMES: print('%s - mean: %.2f var: %.2f' % (col, df[col].mean(), df[col].var()))

  # DNN model
  inshape = (len(FV_NAMES),)
  model = Sequential([
    # LSTM(32, input_shape=(nb_time_steps, nb_input_vector))),
    Dense(32, activation='relu', kernel_initializer='he_normal', input_shape=inshape),
    Dropout(0.2),
    # Dense(8, activation='relu', kernel_initializer='he_normal'),
    # Dropout(dropout),
    Dense(1),
  ])
  model.build()
  model.summary()
  adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(optimizer=adam, loss='mse', metrics=['mse','mae'])

  for topic, df in pd_train.groupby('topic'):
    print('[Topic] working on topic %r' % topic)
    X, y = df[FV_NAMES], df['score'].astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_test, y_test, cv=3) 
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    y_pred = model.predict(X_test)
    kp = kappa(y_pred, y_test, weights='quadratic')
    print('kappa: %r, topic %r' % (kp, topic))

  # applicational solution
  df = pd_solut[pd_solut.topic == topic]
  X = df[FV_NAMES]
  y_pred = model.predict(X)

  def trim_range(topic, score):
    RANGE = {
      1: (0, 15),
      2: (0, 8),
      3: (0, 5),
      4: (0, 5),
      5: (0, 5),
      6: (0, 5),
      7: (0, 15),
      8: (0, 70),
    }
    score = int(round(score))
    rng = RANGE[topic]
    if score < rng[0]: score = rng[0]
    elif score > rng[1]: score = rng[1]
    return score
  
  res = [ ]
  ids = list(df['id'])  # reindex column 'id'
  for i in range(len(y_pred)):
    id = ids[i]
    score = trim_range(topic, y_pred[i])
    res.append([id, topic, score])
  return res

if __name__ == "__main__":
  res = go()
  with open(path.join(MODEL_PATH, SOLUT_FN), 'w', encoding='utf8') as fp:
    for r in res:
      fp.write('\t'.join([str(x) for x in r]))
      fp.write('\n')
