# parser

基本算法：CYK

基本流程：

  - 动态规划自底向上找出所有可能的子句结构
  - 规约使用预设的文法规则集
  - 最后看整个句子是否可规约到起始符号`S`

预设的文法可以支持简单句、一般疑问句、特殊疑问句和单连词双分句

非终结符有：

```python
SYNTAX_VARIABLES = {
  # noun
  'ART',
  'N', 'Nneg',
  'PRON', 'PRONadj', 'PRONn',
  'ADJ',

  # verb
  'Vt','Vi',
  'Vlnk', 'Vmod', 'Vint', 'Vaux', 'Vneg', 'Vinf',
  'ADV',
  
  # aux
  'PREP',
  'CONJ',
}
```

全部文法规则为：

```python
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
```

----

by Armit
2019年11月16日