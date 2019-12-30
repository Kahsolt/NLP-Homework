#!/usr/bin/env python3
# Author: Armit
# Create Time: 2019/11/21 

import sys
import csv
import random
import pickle
from os import path
from time import time
from collections import Counter, defaultdict
from pprint import pprint as pp
import numpy as np
import pandas as pd

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
#from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVR, LinearSVC
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB

sys.path.append(path.dirname(path.abspath(__file__)))
from metrics import kappa

BASE_PATH = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
CORPUS_PATH = path.join(BASE_PATH, 'corpus')
DATA_PATH = path.join(CORPUS_PATH, 'essay_data')
MODEL_PATH = path.join(BASE_PATH, 'model')
DATA_FILE = path.join(MODEL_PATH, 'aes_dataset.pkl')
DICT_FILE = path.join(MODEL_PATH, 'oxford_dict.pkl')
SOLUT_FN = 'MG1933029.tsv'
#SOLUT_FN = 'aes_ML.tsv'

def timer(fn):
  def wrapper(*args, **kwargs):
    t = time()
    r = fn(*args, **kwargs)
    print('[Timer] %r done in %.2f seconds' % (fn.__name__, time() - t))
    return r
  return wrapper

def oxford_dict() -> set:
  if path.exists(DICT_FILE):
    with open(DICT_FILE, 'rb') as fp:
      dc = pickle.load(fp)
  else:
    dc = set()
    with open(path.join(CORPUS_PATH, 'oxford_dict.txt'), encoding='utf8') as fp:
      for ln in fp.readlines():
        dc.add(ln.split(' ')[0])
    with open(DICT_FILE, 'wb') as fp:
      pickle.dump(dc, fp)
  return dc

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
def abstract_featdict(df):
  # the NLP engine and dictionary
  DICTIONARY = oxford_dict()
  nlp = spacy.load('en_core_web_sm')
  #idf = defaultdict(int)   # { 'word': cnt_in_doc }

  # parse doc
  print('[Feature] abstracting doc...')
  if not 'len' in df.columns: df['len'] = df.apply(lambda e: len(e['text']), axis=1)
  if not 'DOC' in df.columns: df['DOC'] = df.apply(lambda e: nlp(e['text']), axis=1)

  # vetoral/gloabl-statistical feature
  print('[Feature] abstracting vectoral features...')
  #for tok in doc: idf[tok.text] += 1
  #df['tf'].append(dict(Counter([tok.text for tok in doc]))) # later do tf-idf
  if not 'BOW' in df.columns: df['BOW'] = df.apply(lambda e: {tok.text.lower() for tok in e['DOC']}, axis=1)
  if not 'POS' in df.columns: df['POS'] = df.apply(lambda e: [tok.pos_ for tok in e['DOC']], axis=1)
  if not 'TAG' in df.columns: df['TAG'] = df.apply(lambda e: [tok.tag_ for tok in e['DOC']], axis=1)
  
  # scalar feature
  print('[Feature] abstracting token features...')
  if not 'tok' in df.columns: df['tok'] = df.apply(lambda e: len(e['DOC']), axis=1)
  if not 'tok_wrong' in df.columns: df['tok_wrong'] = df.apply(lambda e: sum(True for tok in e['DOC'] if not tok.lemma_.endswith('-') and tok.lemma_ not in DICTIONARY), axis=1)
  if not 'tok_uniq' in df.columns: df['tok_uniq'] = df.apply(lambda e: len(e['BOW']), axis=1) 
  if not 'tok_long' in df.columns: df['tok_long'] = df.apply(lambda e: sum(True for tok in e['DOC'] if len(tok.text) >= 10), axis=1) 
  if not 'tok_stop' in df.columns: df['tok_stop'] = df.apply(lambda e: sum(True for tok in e['DOC'] if tok.text not in STOP_WORDS), axis=1)
  
  print('[Feature] abstracting sentence features...')
  if not 'sent' in df.columns: df['sent'] = df.apply(lambda e: sum(True for sent in e['DOC'].sents), axis=1)
  if not 'sent_complex_mean' in df.columns: df['sent_complex_mean'] = df.apply(lambda e: sum([len({tok.tag_ for tok in sent}) for sent in e['DOC'].sents]), axis=1) / df['sent']
  if not 'sent_len_mean' in df.columns: df['sent_len_mean'] = df.apply(lambda e: sum(len(sent.text) for sent in e['DOC'].sents), axis=1) / df['sent']
  if not 'sent_long' in df.columns: df['sent_long'] = df.apply(lambda e: sum(True for sent in e['DOC'].sents if len(sent.text) > e['sent_len_mean']), axis=1)

  print('[Feature] abstracting word/pos features...')
  if not 'noun' in df.columns: df['noun'] = df.apply(lambda x: x['POS'].count('NOUN'), axis=1)
  if not 'propn' in df.columns: df['propn'] = df.apply(lambda x: x['POS'].count('PROPN'), axis=1)
  if not 'adj' in df.columns: df['adj'] = df.apply(lambda x: x['POS'].count('ADJ'), axis=1)
  if not 'pron' in df.columns: df['pron'] = df.apply(lambda x: x['POS'].count('PRON'), axis=1)
  if not 'verb' in df.columns: df['verb'] = df.apply(lambda x: x['POS'].count('VERB'), axis=1)
  if not 'adv' in df.columns: df['adv'] = df.apply(lambda x: x['POS'].count('ADV'), axis=1)
  if not 'cconj' in df.columns: df['cconj'] = df.apply(lambda x: x['POS'].count('CCONJ'), axis=1)
  if not 'det' in df.columns: df['det'] = df.apply(lambda x: x['POS'].count('DET'), axis=1)
  if not 'part' in df.columns: df['part'] = df.apply(lambda x: x['POS'].count('PART'), axis=1)
  if not 'punct' in df.columns: df['punct'] = df.apply(lambda x: x['POS'].count('PUNCT'), axis=1)
  if not 'comma' in df.columns: df['comma'] = df.apply(lambda x: x['TAG'].count(','), axis=1)

@timer
def go():
  res = [ ]
  ds = load_dataset()
  pd_train = ds['train']
  pd_solut = ds['solut']
  abstract_featdict(pd_train)
  abstract_featdict(pd_solut)
  # save_dataset(ds)  # SAVE MODEL

  FV_NAMES = [
    'len', 
    'tok', 
    'tok_wrong', 
    'tok_uniq', 
    # 'tok_long', 
    'tok_stop', 
    'sent', 
    # 'sent_complex_max', 
    'sent_len_mean', 
    # 'sent_long',
    'noun',
    # 'propn', 
    'adj', 
    # 'pron', 
    'verb', 
    'adv', 
    # 'cconj',
    # 'det', 
    # 'part', 
    'punct', 
    'comma', 
  ]
  df = ds['train']
  for col in FV_NAMES:
    print('%s - mean: %.2f var: %.2f' % (col, df[col].mean(), df[col].var()))
  
  for topic, df in pd_train.groupby('topic'):
    print('[Topic] working on topic %r' % topic)
    X, y = df[FV_NAMES], df['score'].astype(np.float64)

    # train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    MODLES = [
      [('linr', LinearRegression(normalize=True, copy_X=False, n_jobs=16))],
      # [('std', StandardScaler()), ('gs:svr', GridSearchCV(SVR(kernel='linear'), iid=True, param_grid={'C': [0.75, 1.5, 3, 5]}, cv=3, n_jobs=16))],
      # [('std', StandardScaler()), ('gs:lsvc', GridSearchCV(LinearSVC(), iid=True, param_grid={'C': [0.75, 1.5, 3, 5]}, cv=3, n_jobs=16))],
      [('gs:knn', GridSearchCV(KNeighborsRegressor(weights="distance"), iid=True, param_grid={'n_neighbors': [12, 16, 24, 32, 36]}, cv=3, n_jobs=16))],
      [('gs:dt', GridSearchCV(DecisionTreeRegressor(), iid=True, param_grid={'max_depth': [6, 12, 18, 24, 30]}, cv=3, n_jobs=16))],
      [('gs:et', GridSearchCV(ExtraTreesRegressor(), iid=True, param_grid={'n_estimators': [32, 64, 96, 128]}, cv=3, n_jobs=16))],
      [('gs:gb', GridSearchCV(GradientBoostingRegressor(), iid=True, param_grid={'n_estimators': [32, 64, 96, 128]}, cv=3, n_jobs=16))],
      [('gs:rf', GridSearchCV(RandomForestRegressor(), iid=True, param_grid={'n_estimators': [32, 64, 96, 128]}, cv=3, n_jobs=16))],
      [('gs:etc', GridSearchCV(ExtraTreesClassifier(), iid=True, param_grid={'n_estimators': [32, 64, 96, 128]}, cv=3, n_jobs=16))],
      [('gs:en', GridSearchCV(ElasticNet(), iid=True, param_grid={'l1_ratio': [0.01, 0.1, 0.5, 0.9], 'alpha': [0.01, 0.1, 1]}, cv=3, n_jobs=16))],
    ]
    kp_mdl = [ ]
    for pl in MODLES:
      model = Pipeline(pl)
      model.fit(X_train, y_train)
      scores = cross_val_score(model, X_test, y_test, cv=3) 
      print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

      y_pred = model.predict(X_test)
      kp = kappa(y_pred, y_test, weights='quadratic')
      print('kappa: %r, using %r' % (kp, [m[0] for m in pl]))
      
      kp_mdl.append((kp, model))

    # ok we select best k-models and use avgval of predicted
    N_MODELS = 3
    kp_mdl.sort(reverse=True)
    print('kappas: %r' % [k for k, _ in kp_mdl])
    models = [m for _, m in kp_mdl[:N_MODELS]]
    
    # applicational solution
    df = pd_solut[pd_solut.topic == topic]
    X = df[FV_NAMES]
    y_preds = [model.predict(X) for model in models]
    y_pred = [sum(y_preds[i][j] for i in range(N_MODELS)) / N_MODELS for j in range(len(X))]

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
