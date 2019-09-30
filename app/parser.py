#/usr/bin/env python3
# Author: Armit
# Create Time: 2019/9/28
# Update Time: 2019/9/30

from pprint import pprint as pp
from common import *
from dictionary import get_dict
from lemmatizer import Lemmatizer

# this sample dictionary is recommanded because 
# tags in `dic_ec.rar` is REALLY incomplete and disgusting
VOCABULARY = {
  'ART': ['a', 'an', 'the'],
  'PRON': ['I', 'you', 'he', 'she', 'it', 'they', 'me', 'him', 'her', 'them'],
  'PRONadj': ['my', 'your', 'his', 'her', 'its', 'their'],
  'PRONn': ['mine', 'yours', 'his', 'hers', 'its', 'theirs', 'this', 'that'],
  'Vlnk': ['be', 'am', 'is', 'are', 'was', 'were', 'look', 'feel', 'sound', 'smell', 'taste', 'seem'],
  'Vmod': ['can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'must', 'need'],
  'Vint': ['what', 'who', 'where', 'when', 'why', 'how'],
  'Vaux': ['do', 'have'],   # can be merged to Vmod
  'Vneg': ['not'],
  'Vinf': ['to'],
  'Nneg': ['no'],

  'N': ['thing', 'object', 'apple', 'bread', 'water', 'table', 'fox', 'sheep', 'book', 'nothing', 'home', 'school', 'story', 'dog', 'cat'],
  'ADJ': ['what', 'whose', 'good', 'better', 'bad', 'worse', 'beautiful', 'ugly', 'lazy', 'silly', 'interesting', 'fucking'],
  'Vt': ['eat', 'drink', 'like', 'love', 'hate', 'hit', 'read', 'help'],
  'Vi': ['study', 'swim', 'run', 'jump', 'sleep', 'happen', 'push', 'pop'],
  'ADV': ['well', 'happily', 'diligently', 'eventually', 'quickly', 'curiously', 'late', 'early', 'please', 'today', 'yesterday', 'tomorrow', 'now'],
  'PREP': ['from', 'to', 'across', 'away', 'at', 'in', 'with'],
  'CONJ': ['if', 'then', 'because', 'so', 'and', 'but', 'although'],
}
# consider fundamental sentence structures only :)
RULES = [
  # sentence
  ('S', ['SS']),
  ('S', ['SS', 'CONJ', 'SS']),     # compound sentence (no more recursivity please)
  # simple sentence
  ('SS', ['NPQ', 'Vlnk', 'NPQ']),  # declarative with linking verb
  ('SS', ['NPQ', 'VPM']),          # declarative with notional verb
  ('SS', ['Vlnk', 'NPQ', 'NPQ']),  # interrogative with linking verb
  ('SS', ['Vmod', 'NPQ', 'VP']),   # interrogative with notional verb
  ('SS', ['Vint', 'Vlnk', 'NPQ']), # interrogative (special)
  ('SS', ['Vint', 'VPM']),
  ('SS', ['VP']),                  # imperative
  # noun phrase with quantifier
  ('NPQ', ['NPA']),                # generic reference
  ('NPQ', ['ART', 'NPA']),         # specific reference
  ('NPQ', ['Nneg', 'NPA']),
  # noun phrase with attributive
  ('NPA', ['NP']),
  ('NPA', ['ADJ', 'NPA']),         # adjective (recursively)
  ('NPA', ['PRONadj', 'NPA']),     # adjectival possessive pronoun, aka. gentive case
  # noun phrase
  ('NP', ['NPp']),
  ('NP', ['NPn']),
  ('NPn', ['N']),
  ('NPn', ['N', 'NPn']),           # noun (recursively)
  ('NPp', ['PRON']),               # person pronoun
  ('NPp', ['PRONn']),              # nominal possessive pronoun
  # verbal phrase with modal/negative
  ('VPM', ['VP']),
  ('VPM', ['Vmod', 'VP']),
  ('VPM', ['Vmod', 'Vneg', 'VP']),
  ('VPM', ['Vaux', 'Vneg', 'VP']),
  # predicate phrase
  ('VP', ['Vi']),
  ('VP', ['Vi', 'ADVs']),
  ('VP', ['Vi', 'PN']),            # dative case
  ('VP', ['Vi', 'PN', 'ADVs']),
  ('VP', ['Vt', 'NPQ']),
  ('VP', ['Vt', 'NPQ', 'ADVs']),   # accusitive case
  # prepositional phrase, aka. case phrase
  ('PREPs', ['PREP']),
  ('PREPs', ['PREP', 'PREPs']),    # preposition (recursively)
  ('PN', ['PREPs']),
  ('PN', ['PREPs', 'NPQ']),
  ('PN', ['PREPs', 'NPQ', 'PN']),  # preposition phrase (recursively)
  ('PN', ['Vinf', 'VP']),          # verbal infinitive
  # adverbial phrase
  ('ADVs', ['ADV']),
  ('ADVs', ['ADV', 'ADVs']),       # adverb (recursively)
]
SYNTAX_VARIABLES = {k for k, _ in RULES}  # contrasted with lexical variables in VOCABULARY.keys()

TAG_NONE = 'none.'  # just cache this string for frequent usage
TAG_MAPPING = {
  'n.': ['N'],
  'pl.': ['N'],                 # plural of n.
  'vi.': ['Vi'],
  'vt.': ['Vt'],
  'vp.': ['Vt', 'Vi'],
  'v.': ['Vt', 'Vi'],
  'aux.': ['Vmod'],
  'vbl.': ['N', 'ADJ'],         # done / doing
  'num.': ['ADJ'],
  'adj.': ['ADJ'],
  'adv.': ['ADV'],
  'abbr.': ['N'],
  'symb.': ['N'],
  TAG_NONE: ['N', 'Vi', 'Vt' ,'ADJ', 'ADV'],   # HAIL MARY: fuck the missing tags
}

# compatiblity patch: convert dictionary tags to here we used in RULES
def fuck_dtags_to_tags(dtags:set) -> {str}:
  if not dtags: dtags.add(TAG_NONE)
  elif TAG_NONE in dtags and len(dtags) > 1:
    dtags.remove(TAG_NONE)

  ret, flag = set(), False
  for tag in set(dtags):
    # fuck typos in that dictionary
    if tag == '.dj': tag = 'adj.'
    elif tag == '.one': tag = TAG_NONE
    elif tag not in TAG_MAPPING: tag = TAG_NONE
    if tag == TAG_NONE: flag = True
    ret.update(TAG_MAPPING.get(tag))
  if flag: logging.warning('[Parser] found tag "none." in characteritics of words, '
                           'this will lead to miserable analyze result :(')
  return ret


class Parser:

  class Chart:

    class Vertex:
      
      def __init__(self, token=''):
        self.token = token

    class Edge:
      
      def __init__(self, lpos:int, rpos:int, unit:str, state:list):
        self.lpos = lpos
        self.rpos = rpos
        self.unit = unit
        self.state = state    # the right unscanned part
      
      def __lt__(self, other):
        if self.lpos != other.lpos: return self.lpos < other.lpos
        elif self.rpos != other.rpos: return self.rpos < other.rpos
        else: return self.unit <= other.unit

      def __eq__(self, other):
        return (self.lpos == other.lpos and self.rpos == other.rpos
                and self.unit == other.unit and self.state == other.state)

    def __init__(self):
      self.edges_active = [ ]
      self.edges_inactive = [ ]
      self.vertexes = [ ]   # the rank is crucial

    def __str__(self):
      lines, vertex_flow = [ ], ''
      for i, v in enumerate(self.vertexes):
        vertex_flow += ' <%d> %s' % (i + 1, v.token)
      vertex_flow += ' <%d> ' % (len(self.vertexes) + 1)
      nlen = len(vertex_flow)
      lines.append('=' * nlen)
      lines.append(vertex_flow)
      lines.append('-' * nlen)
      rel = { }  # { (int, int): [str] }
      for e in self.edges_inactive:
        span = (e.lpos, e.rpos)
        if span in rel: rel[span].append(e.unit)
        else: rel[span] = [e.unit]
      for k in sorted(rel):
        lines.append('%r: %r' % (k, rel[k]))
      lines.append('>> found %d releations/edges.' % len(self.edges_inactive))
      lines.append('')
      return '\n'.join(lines)

    def add_vertex(self, label:str):
      self.vertexes.append(self.Vertex(label))
      return len(self.vertexes)

    def add_edge(self, lpos, rpos, tag, state:list=None):
      e = self.Edge(lpos, rpos, tag, state)
      
      if state is not None:
        if e not in self.edges_active:
          self.edges_active.append(e)
      else:
        if e not in self.edges_inactive:
          self.edges_inactive.append(e)
  
  INSTANE = None

  def __new__(cls, *args, **kwargs):
    if not cls.INSTANE:
      cls.INSTANE = super().__new__(cls)
    return cls.INSTANE

  def __init__(self, dictionary):
    self.dictionary = dictionary
    self.lemmatizer = Lemmatizer(dictionary)
    self.rules = RULES
    self.tag_query_cache = { }  # runtime use for tag query in dictionary
  
  def parse(self, sent):
    tokens = sent.split()
    agenda, agenda_hist = [ ], set()  # stack and its visit record to aviod duplicate push
    chart = self.Chart()
    while agenda or tokens:
      if not agenda:
        tok, tokens = tokens[0], tokens[1:]
        tok = self.lemmatizer.lemmatize(tok)

        tags = self.tag_query_cache.get(tok)
        if not tags:
          tags = {k for k, v in VOCABULARY.items() if tok in v}
          if not tags:
            dtags = (tok in self.dictionary
                     and {tag for tag, _ in self.dictionary[tok]}
                     or set())
            tags = fuck_dtags_to_tags(dtags)
          self.tag_query_cache[tok] = tags

        idx = chart.add_vertex(tok)
        for tag in tags:
          todo = (tag, idx, idx + 1)
          agenda.append(todo)
          agenda_hist.add(todo)
      else:
        target, lpos, rpos = agenda.pop()
        for unit, unscanned in self.rules:
          if unscanned and unscanned[0] == target:
            if len(unscanned) > 1:
              chart.add_edge(lpos, rpos, unit, unscanned[1:])
            else:
              todo = (unit, lpos, rpos)
              if todo not in agenda_hist:
                agenda.append(todo)
                agenda_hist.add(todo)
        chart.add_edge(lpos, rpos, target)
        for e in chart.edges_active:
          # rule alive: str, [] => unit, unscanned
          unit, unscanned = e.unit, e.state
          if unscanned and unscanned[0] == target:
            if len(unscanned) > 1:
              chart.add_edge(e.lpos, rpos, unit, unscanned[1:])
            else:
              todo = (unit, e.lpos, rpos)
              if todo not in agenda_hist:
                agenda.append(todo)
                agenda_hist.add(todo)
      # print(agenda)
    return chart


def run():
  dictionary = get_dict('en')
  parser = Parser(dictionary)
  try:
    print('============================================================================')
    print(' You are high recommended to use these words ONLY for sentence weaving, but ')
    print(' you could still type whaterver you like, since output is not guaranteed.   ')
    print('----------------------------------------------------------------------------')
    pp(VOCABULARY, compact=True)
    print('============================================================================')
    while True:
      line = input('>> Quid negoti est: ').strip()
      print(line)
      print(parser.parse(line))
  except (KeyboardInterrupt, EOFError):
    dictionary.save_model()
    logging.info('[System] bye')


def test():
  dictionary = get_dict('en')
  parser = Parser(dictionary)

  sents = [   # no punctuations please
    'are you OK',
    'where is my knife',
    'I will push if you pop',
    'you play basketball like CaiXukun',
    'my inner heart has no fluctuations even I want to laugh',

    'would you like to swim with me tomorrow',
    'nothing happened eventually because you run away',
    'he has read an interesting book at home yesterday if you die',
  ]
  for sent in sents:
    print(sent)
    print(parser.parse(sent))

  dictionary.save_model()


if __name__ == "__main__":
  if 'test' in sys.argv:
    test()
  else:
    run()
