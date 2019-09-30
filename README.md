# NLP Homework

    Collection of some no-use toy projects for NJU NLP course

----

### Quickstart
  - `make corpus` to download neccessary resource data, and extract them in place manually if you do not have `7z` installed
  - `make <app>` or `make test_<app>`, see [Makefile](/Makefile) for all target, currently we have `<app>` as follows:
    + lemmatizer
    + tokenizer
    + parser

### Toys
  - [x] dictionary-based [Lemmatizer](/app/lemmatizer.py) for English
    - primarily, query if word is exactly in dictionary; also try static rules abstracted from explanation/paraphrases in the dictionary (if you switch on `ROOTFORM_RELOCATE` option)
    - try regex rules for standard rational lemmatization
    - take ventrues to guess it's stem/root using entropy trcik, then find a nearest word in the dictionary
    - pretend to assert that the word is already in its base form :)
  - [x] dictionary-based [Tokenizer](/app/tokenizer.py) for Chinese
    - primarily, try longest match both in positive and reverse order
    - on conflicts, use entropy trcik to determine which result to accept finally
  - [x] chart-based [Parser](/app/parser.py) for English
    - dynamic programming using chart-analyzing (it's just DFS with memory in essence)
    - please test with words that appears in the sample dictionary `VOCABULARY`, because tags in corpus dictionary have much data vancancies and REALLY hard to use; however you could eventually type whatever you like since the output is not guaranteed :(
    - so-far the parser works well with simpler English sentences without sub-clause, for supported schemas see `RULES` and examples in `test()`, I do not what to add a sub-clause feature -- NO WHY -- the complexity of the ruleset is killing me :joy:
    - (one word may have several roles or characteristics or properties, to disambiguate this by  context-analyzing is the biggest challenge, or let's just say the defect of this so-called chart-analyzing method)


#### the entropy trcik?
  - for each model of a dictionary, we have tow [Trie](https://baike.baidu.com/item/%E5%AD%97%E5%85%B8%E6%A0%91) trees built for fast prefix and suffix search
  - then we could somewhat define the concept of 'entropy', see method `TrieTree.get_entropy()` in [dictionary.py](/app/dictionary.py) for details
  - however in most cases, the trick **HAS NO EGG USE** :rofl:

----

by Armit
2019/9/20
