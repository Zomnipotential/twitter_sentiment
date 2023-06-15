import sys
print('packages -> ',sys.executable)

if __name__=="sentiment_packages":
	print('hi')
else:
	print('3*\nGood bye3*\n')
import gc
import os
import re
import csv
import sys
import time
import nltk
import string
import shutil
import datetime
#import objgraph
#import threading
import threading
import functools
import sklearn
import contractions
import pandas as pd
import matplotlib.pyplot as plt
import progress_indicator as pi
from IPython.display import display
from memory_profiler import profile
from IPython.core.display import Image
from collections import Counter
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
from typing import Union, List, Dict, Tuple, Any, Optional, Callable
from varname import varname, nameof
from varname.helpers import Wrapper
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer