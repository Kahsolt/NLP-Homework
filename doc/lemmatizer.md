# lemmatizer

基本算法：正则匹配 + 静态规则

基本流程：

  - 直接查字典
  - 根据静态规则，即试图根据从字典中查找释义含有变格变位的信息(如义项："backbite的过去分词")推断原型，即可直接还原
  - 根据正则规则集`RULES`改写后查字典，涵盖了绝大多数规则的名词变格和动词变位
  - 尝试用前后缀熵切出词干，寻找距离词干最近的词(错误率较高)
  - 不作还原

预设正则规则为：

```python
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
```

----

by Armit
2019年11月16日