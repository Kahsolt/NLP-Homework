#/usr/bin/env python3
# Author: Armit
# Create Time: 2019/9/20
# Update Time: 2019/9/28

import re
import time
import pickle

from common import *
from dictionary import get_dict

# try searching with lower/upper/capital cases
IGNORE_CASE = True
# force lemmatize even if words are already IN dictionary
ROOTFORM_RELOCATE = False
# magic number to decide where use entropy trcik (no empirics, just magic
ENTROPY_THRESH = 8.0
# shell cmd mark
EXEC_MARK = '\\'

RULES = [
  (r'(.*)ying$', r'\1ie'),
  (r'(.*)ing$', r'\1'),
  (r'(.*)ies$', r'\1y'),
  (r'(.*)ied$', r'\1y'),
  (r'(.*)ied$', r'\1ie'),
  (r'(.*)ves$', r'\1f'),
  (r'(.*)ces$', r'\1x'),
  (r'(.*)es$', r'\1'),
  (r'(.*)es$', r'\1e'),
  (r'(.*)ed$', r'\1'),
  (r'(.*)ed$', r'\1e'),
  (r'(.*)s$', r'\1'),
  (r'(.*)i$', r'\1us'),
]

class RegexRuleset:
  
  class Rule:

    def __init__(self, find, replace):
      self.R = re.compile(find)
      self.replace = replace

    def applicable(self, word) -> bool:
      return self.R.match(word) is not None
    
    def apply(self, word) -> str:
      return self.R.sub(self.replace, word)
  
  INSTANE = None

  def __new__(cls, *args, **kwargs):
    if not cls.INSTANE:
      cls.INSTANE = super().__new__(cls)
    return cls.INSTANE

  def __init__(self):
    self.rules = [self.Rule(find, replace) for find, replace in RULES]

  def applicable(self, word) -> bool:
    for rule in self.rules:
      if rule.applicable(word):
        return True
    return False

  def apply(self, word) -> list:
    res = [ ]
    for rule in self.rules:
      if rule.applicable(word):
        res.append(rule.apply(word))
    return res or [word]


class DictionaryRuleset:

  INSTANE = None

  def __new__(cls, *args, **kwargs):
    if not cls.INSTANE:
      cls.INSTANE = super().__new__(cls)
    return cls.INSTANE

  def __init__(self, dictionary):
    self.dictionary = dictionary
    self.mappings = { }
    self.fp_model = path.join(MODEL_PATH, 
                    path.splitext(path.basename(dictionary.fp_model))[0] + '-ruleset.pkl')

  def load_model(self):
    logging.info('[DictionaryRuleset] loading...')
    s = time.time()
    with ZIP_MODULE.open(self.fp_model, 'rb') as fp:
      self.mappings = pickle.load(fp)
    t = time.time()
    logging.info('[DictionaryRuleset] %d items loaded (%.2fs)' % (len(self.mappings), t - s))

  def build_model(self):
    logging.info('[DictionaryRuleset] loading...')
    s = time.time()

    WORD_REGEX = re.compile('[a-zA-Z]+')
    RULE_REGEX = re.compile((
      '第三人称|第二人称|'
      '单数|复数|'
      '现在式|现在分词|过去式|过去分词|'
      '所有格|宾格|'
      '比较级|最高级'))
    for word, tagsigs in self.dictionary.vocabulary.items():
      for i in range(0, len(tagsigs), 2):
        _, sig = tagsigs[i], tagsigs[i + 1]
        m = RULE_REGEX.search(sig)
        if m:
          m2 = WORD_REGEX.search(sig[:m.start()])
          if m2:
            self.mappings[word] = m2.group()
    
    t = time.time()
    logging.info('[DictionaryRuleset] %d items loaded (%.2fs)' % (len(self.mappings), t - s))
    self.save_model()
   
  def save_model(self):
    with ZIP_MODULE.open(self.fp_model, 'wb') as fp:
      pickle.dump(self.mappings, fp, protocol=4)

  def applicable(self, word) -> bool:
    return word in self.mappings

  def apply(self, word) -> str:
    return self.mappings.get(word)


class Lemmatizer:
  
  def __init__(self, dictionary):
    self.dictionary = dictionary
    self.ruleset_regex = RegexRuleset()
    self.ruleset_dict = DictionaryRuleset(dictionary)
  
  def lemmatize(self, word) -> str:
    # try directly lookup in dictionary
    if not ROOTFORM_RELOCATE:
      wls = IGNORE_CASE and [word, word.lower(), word.capitalize(), word.upper()] or [word]
      for w in wls:
        if w in self.dictionary:
          return word
    
    # if static rules available
    if self.ruleset_dict.applicable(word):
      return self.ruleset_dict.apply(word)

    # try regex rules
    if self.ruleset_regex.applicable(word):
      for word in self.ruleset_regex.apply(word):
        if word in self.dictionary:
          return word
    
    # try directly lookup in dictionary
    if ROOTFORM_RELOCATE:
      wls = IGNORE_CASE and [word, word.lower(), word.capitalize(), word.upper()] or [word]
      for w in wls:
        if w in self.dictionary:
          return word
    
    # try stemize by entropy split (dangerous!)
    ent_max, idx = 0.0, -1
    for i in range(len(word), 0, -1):
      pf, sf = word[:i], word[i:]
      ent_cross = self.dictionary.prefix_tree.get_entropy(pf) \
                  * self.dictionary.suffix_tree.get_entropy(sf[::-1])
      logging.debug('>> %s %s\t%.4f' % (pf, sf, ent_cross))
      if ent_cross > ent_max:
        ent_max, idx = ent_cross, i
    if ent_max > ENTROPY_THRESH:   # FIXME: empirical magic number
      pf, sf = word[:idx], word[idx:]
      logging.debug('word: %r, prefix: %r, suffix %s' % (word, pf, sf))
      return self.dictionary.prefix_tree.find_nearest_word(pf)
    
    # just 'AS IS'
    return word


def shell(line):
  FLAG_ON = ['on', '1', 'e', 'enable', 'y', 'yes', 't', 'true']
  FLAG_OFF = ['off', '0', 'd', 'disable', 'n', 'no', 'f', 'false']
  
  try:
    cmd, *args = line.lower().split()
  except:
    cmd, *args = None, None
  if cmd in ['i', 'ignorecase']:
    if args:
      global IGNORE_CASE
      if args[0] in FLAG_ON:
        IGNORE_CASE = True
      elif args[0] in FLAG_OFF:
        IGNORE_CASE = False
    logging.info('IGNORE_CASE set to %r' % IGNORE_CASE)
  elif cmd in ['r', 'rootform']:
    if args:
      global ROOTFORM_RELOCATE
      if args[0] in FLAG_ON:
        ROOTFORM_RELOCATE = True
      elif args[0] in FLAG_OFF:
        ROOTFORM_RELOCATE = False
    logging.info('ROOTFORM_RELOCATE set to %r' % ROOTFORM_RELOCATE)
  else:
    print((
      'Options:\n'
      '  \\i [on|off]    set IGNORE_CASE\n'
      '  \\r [on|off]    set ROOTFORM_RELOCATE\n'))


def run():
  dictionary = get_dict('en')
  lemmatizer = Lemmatizer(dictionary)
  try:
    while True:
      line = input('>> Quid negoti est: ').strip()
      if line.startswith(EXEC_MARK):   # take as shellcmd
        shell(line[1:])
      else:
        for word in line.split():
          fword = lemmatizer.lemmatize(word)
          print(dictionary.lookup(fword))
  except (KeyboardInterrupt, EOFError):
    dictionary.save_model()
    logging.info('[System] bye')


def test():
  dictionary = get_dict('en')
  lemmatizer = Lemmatizer(dictionary)

  words = [
    'were', 'had', 'done', 'did', 'saith', 'thee',
    'lies', 'studies', 'flying',
    'liked', 'took', 'went', 'gone', 'sat',
    'geese', 'tomatoes', 'photos', 'friends', 'stones', 'leaves',
  ]
  for word in words:
    print('[%s] -> ' % word, end='')
    print(dictionary.lookup(lemmatizer.lemmatize(word)))

  dictionary.save_model()


if __name__ == '__main__':
  if 'test' in sys.argv:
    test()
  else:
    run()
