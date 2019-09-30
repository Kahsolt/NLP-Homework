#!/usr/bin/env python3
# Author: Armit
# Create Time: 2019/9/20
# Update Time: 2019/9/28

import re
import math
import pickle
import time

from common import *


class TrieTree:

  class Node:
    
    def __init__(self):
      self.children = { }
      self.terminal = False
      self.via = 0
      self.entropy = None
        
    def __repr__(self):
      return '<Node terminal=%r via=%d children=%r>' % (
              self.terminal, self.via, self.children.keys())
      
    def __getitem__(self, key):
      return self.children.get(key)
    
    def __setitem__(self, key, val):
      self.children[key] = val

    def get_entropy(self, cache_notice=False):
      use_cache = True
      if self.entropy is None:
        use_cache = False
        ent = 0.0
        for child in self.children.values():
          p = child.via / self.via
          if p: ent -= p * math.log2(p)
        self.entropy = ent
      return not cache_notice and self.entropy or (self.entropy, use_cache)
 
  def __init__(self, dictionary):
    self.dictionary = dictionary    # parent back-reference
    self.root = self.Node()
    
  def insert(self, word):
    cur = self.root
    for x in word:
      next = cur[x]
      if not next:
        cur[x] = next = self.Node()
      cur.via += 1
      cur.entropy = None  # clear entropy cache
      cur = next
    cur.terminal = True

  def walk_to(self, prefix):
    cur = self.root
    for x in prefix:
      if not cur: break
      cur = cur[x]
    return cur
    
  def get_entropy(self, prefix) -> float:
    cur = self.walk_to(prefix)
    if cur:
      ent, c = cur.get_entropy(True)
      if c: self.dictionary.updated = True
      return ent
    else: return 0.0
  
  def find_nearest_word(self, prefix) -> str:
    cur = self.walk_to(prefix)
    appendix = ''
    while cur and not cur.terminal:
      via_min = cur.via
      next_rune, next_child = None, None
      for rune, child in cur.children.items():
        if child.via <= via_min:
          via_min = child.via
          next_rune, next_child = rune, child
      appendix += next_rune
      cur = next_child
    return prefix + appendix

  def match_word(self, prefix, tactic='longest') -> str:
    if tactic not in ['longest', 'shortest']: raise ValueError

    cur = self.root
    m, r = '', None
    for x in prefix:
      cur = cur[x]
      if not cur: break
      m += x
      if cur.terminal:
        if tactic == 'longest': r = m
        else: break
    return r


class Dictionary:

  def __init__(self, fname=None):
    self.fp_corpus = path.join(CORPUS_PATH, fname)
    self.fp_model = path.join(MODEL_PATH, path.splitext(fname)[0] + '.pkl')
    
    self.vocabulary = { }
    self.prefix_tree = TrieTree(self)
    self.suffix_tree = TrieTree(self)
    self.updated = False

    if path.exists(self.fp_model):
      self.load_model()
    else:
      self.build_model()

  def load_model(self):
    logging.info('[Dictionary] loading...')
    s = time.time()
    with ZIP_MODULE.open(self.fp_model, 'rb') as fp:
      self.vocabulary = pickle.load(fp)
      self.prefix_tree = pickle.load(fp)
      self.suffix_tree = pickle.load(fp)
    t = time.time()
    logging.info('[Dictionary] %d items loaded (%.2fs)' % (len(self.vocabulary), t - s))

  def build_model(self):
    raise NotImplementedError
  
  def save_model(self):
    if not self.updated: return
    logging.info('[Dictionary] saving...')
    with ZIP_MODULE.open(self.fp_model, 'wb') as fp:
      pickle.dump(self.vocabulary, fp, protocol=4)
      pickle.dump(self.prefix_tree, fp, protocol=4)
      pickle.dump(self.suffix_tree, fp, protocol=4)

  def lookup(self, word) -> str:
    raise NotImplementedError

  def __contains__(self, word) -> bool:
    return word in self.vocabulary
  
  def __getitem__(self, key) -> object:
    return self.vocabulary.get(key)


class DictE2C(Dictionary):

  INSTANE = None

  def __new__(cls, *args, **kwargs):
    if not cls.INSTANE:
      cls.INSTANE = super().__new__(cls)
    return cls.INSTANE

  def __init__(self, fname):
    super().__init__(fname)

  def build_model(self):
    logging.info('[Dictionary] initial building...')
    s = time.time()
    with open(self.fp_corpus, 'rb') as fp:
      SEP_REGEX = re.compile(b'\xff')   # replace original seperator
      for line in fp.readlines():
        line = DictE2C.fuck_codec(SEP_REGEX.sub(b'\x00', line))
        if not line: continue
        word, *tagsigs = DictE2C.sanitize_list(line.split('\0'))
        self.prefix_tree.insert(word)
        self.suffix_tree.insert(word[::-1])
        self.vocabulary[word] = [(tagsigs[i], tagsigs[i + 1]) for i in range(0, len(tagsigs), 2)]
    t = time.time()
    logging.info('[Dictionary] %d items loaded (%.2fs)' % (len(self.vocabulary), t - s))
    self.updated = True
    self.save_model()
    
  def lookup(self, word) -> str:
    tagsigs = self[word]
    ret = '[%s]\n' % word
    if tagsigs:
      for tag, sig in tagsigs:
        ret += '  %s  %s\n' % (tag.ljust(6), sig)
    return ret

  @staticmethod
  def fuck_codec(bytes):
    for codec in ['utf8', 'gbk', 'gb18030', 'cp936']:
      try: return bytes.decode(codec)
      except UnicodeDecodeError: pass
    return None

  @staticmethod
  def sanitize_list(ls):
    ls = [i.strip() for i in ls]
    while '' in ls: ls.remove('')
    if len(ls) % 2 == 0: ls.append('')
    return ls


class DictC2E(Dictionary):
  
  INSTANE = None

  def __new__(cls, *args, **kwargs):
    if not cls.INSTANE:
      cls.INSTANE = super().__new__(cls)
    return cls.INSTANE

  def __init__(self, fname):
    super().__init__(fname)

  def build_model(self):
    logging.info('[Dictionary] initial building...')
    s = time.time()
    with open(self.fp_corpus, encoding='gb2312') as fp:
      for line in fp.readlines():
        def save(word, sigs):
          sigs = list(sigs)
          if word not in self.vocabulary:
            self.vocabulary[word] = sigs
            self.prefix_tree.insert(word)
            self.suffix_tree.insert(word[::-1])
          else:
            self.vocabulary[word].extend(sigs)
        
        word, sigs = None, None
        for seg in line.strip().split(','):
          if DictC2E.check_ascii(seg):
            sigs.add(seg)
          else:
            if word: save(word, sigs)
            word, sigs = seg, set()
        save(word, sigs)
    t = time.time()
    logging.info('[Dictionary] %d items loaded (%.2fs)' % (len(self.vocabulary), t - s))
    self.updated = True
    self.save_model()
  
  def lookup(self, word) -> str:
    return '[%s]\n  %s' % (word, ', '.join(self[word] or [ ]))

  @staticmethod
  def check_ascii(word) -> bool:
    for w in word:
      if ord(w) > 127 or ord(w) < 0:
        return False
    return True


def get_dict(lang):
  if lang == 'en':
    return DictE2C('dic_ec.txt')
  elif lang == 'cn':
    return DictC2E('ce（ms-word）.txt')
  else:
    raise Exception('Oh my dude, you speak %r like CaiXuKun.' % lang)


if __name__ == "__main__":
  dict_en = get_dict('en')
  print(dict_en.lookup('love'))
  print(dict_en.lookup('justice'))
  print(dict_en.lookup('evil'))
  print(dict_en.lookup('sin'))

  dict_cn = get_dict('cn')
  print(dict_cn.lookup('爱'))
  print(dict_cn.lookup('正义'))
  print(dict_cn.lookup('邪恶'))
  print(dict_cn.lookup('罪'))
