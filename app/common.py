#!/usr/bin/env python3
# Author: Armit
# Create Date: 2019/09/28 

from os import path
import sys
import logging
import gzip

# paths for location
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__)))
CORPUS_PATH = path.join(BASE_PATH, 'corpus')
MODEL_PATH = path.join(BASE_PATH, 'model')

# module that is used to compress model
ZIP_MODULE = gzip

# logging
logging.basicConfig(
  level='-v' in sys.argv and logging.DEBUG or logging.INFO,
  format='%(asctime)s [%(levelname)s] %(message)s')
