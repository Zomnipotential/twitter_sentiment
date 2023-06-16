import sentiment_packages as pkg
import sentiment_constants as const

import csv

def title(message):
    print(f'{3*const.nl}{(len(message)+4)*"*"}{const.nl}* {message} *{const.nl}{(len(message)+4)*"*"}{2*const.nl}')

def find_longest_row(filename):
    max_length = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            row_length = len(row)
            max_length = max(max_length, row_length)
    return max_length

# maximum number of tokens in all the df rows to decide the embedding size
class Emsize:
    file = 'sentiment_embedding_size.txt'
	
    @classmethod
    def setES(cls, size):
        with open(cls.file, 'w') as f:
        	f.write(str(size))

    @classmethod
    def getES(cls):
        with open(cls.file, 'r') as f:
        	return f.read()