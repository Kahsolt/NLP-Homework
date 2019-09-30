#!/usr/bin/env python3
# Author: Armit
# Create Date: 2019/09/28 

from common import *
from dictionary import get_dict


class Tokenizer:
  
  INSTANE = None

  def __new__(cls, *args, **kwargs):
    if not cls.INSTANE:
      cls.INSTANE = super().__new__(cls)
    return cls.INSTANE

  def __init__(self, dictionary):
    self.dictionary = dictionary

  def tokenize(self, sent):
    toks1, toks2 = self._cut(sent)
    return self._disambig(toks1, toks2)

  def _match(self, sent, tactic='longest', direction=1):
    if tactic not in ['longest', 'shortest'] or direction not in [1, -1]:
      raise ValueError

    prefix = (direction == 1
              and self.dictionary.prefix_tree.match_word(sent, tactic)
              or self.dictionary.suffix_tree.match_word(sent[::-1], tactic))
    
    if not prefix: prefix = direction == 1 and sent[0] or sent[-1]
    return direction == 1 and prefix or prefix[::-1]

  def _cut(self, sent):
    # longest
    toks1, sent1 = [ ], sent
    while sent1:
      tok = self._match(sent1)
      toks1.append(tok)
      sent1 = sent1[len(tok):]

    # longest reverse
    toks2, sent2 = [ ], sent
    while sent2:
      tok = self._match(sent2, direction=-1)
      toks2.append(tok)
      sent2 = sent2[:-len(tok)]
    toks2 = toks2[::-1]

    return toks1, toks2

  def _opt_choice(self, toks1, toks2):
    EPS = 1e-5

    # for positive cut calculate suffix entropy
    ent1 = 1.0
    for tok in toks1:
      e = self.dictionary.suffix_tree.get_entropy(tok[::-1])
      if e > EPS: ent1 *= e
    
    # for reverse cut calculate prefix entropy
    ent2 = 1.0
    for tok in toks2:
      e = self.dictionary.prefix_tree.get_entropy(tok)
      if e > EPS: ent2 *= e

    return ent1 <= ent2 and toks1 or toks2

  def _disambig(self, toks1, toks2):
    ret = [ ]
    while toks1 and toks2:
      x, y = [ ], [ ]
      lenx, leny = 0, 0
      if len(toks1[0]) <= len(toks2[0]):
        x.append(toks1[0])
        lenx = len(toks1[0])
        toks1 = toks1[1:]
      else:
        y.append(toks2[0])
        leny = len(toks2[0])
        toks2 = toks2[1:]
      while lenx != leny:
        if lenx < leny:
          x.append(toks1[0])
          lenx += len(toks1[0])
          toks1 = toks1[1:]
        else:
          y.append(toks2[0])
          leny += len(toks2[0])
          toks2 = toks2[1:]
      ret.extend(x == y and x or self._opt_choice(x, y))
    return ret


def run():
  dictionary = get_dict('cn')
  tokenizer = Tokenizer(dictionary)
  try:
    while True:
      line = input('>> Quid negoti est: ').strip()
      tokens = tokenizer.tokenize(line)
      print(' '.join(tokens))
  except (KeyboardInterrupt, EOFError):
    dictionary.save_model()
    logging.info('[System] bye')


def test():
  dictionary = get_dict('cn')
  tokenizer = Tokenizer(dictionary)

  lines = [
    '南京市长江大桥',
    '独立自主和平等独立的原则',
    '讨论战争与和平等问题',
    '他骑在马上',
    '马上过来',
    '我今晚得到达南京',
    '我得到达克宁公司去',
    '从小学电脑',
    '他从小学毕业后',
    '幼儿园地节目',
    '一阵风吹过来',
    '今天有阵风',
    '他学会了解数学难题',
    '乒乓球拍卖完了',
  ]
  for line in lines:
    print(line)
    print(tokenizer.tokenize(line))
  
  dictionary.save_model()


if __name__ == "__main__":
  if 'test' in sys.argv:
    test()
  else:
    run()
