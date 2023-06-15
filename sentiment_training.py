import re
import pandas as pd
import sentiment_constants as const
import sentiment_functions as func
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from IPython.display import display

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

x = pd.read_csv('nonletterclean_df.csv')
y = pd.read_csv('sentiment_df.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

func.title('x_train')
print(x_train)
func.title('x_test')
print(x_test)
func.title('y_train')
print(y_train)
func.title('y_test')
print(y_test)

func.title('embedding size')
print(const.get_constant)
		
# model = Word2Vec([str(sentence).lower().replace("'", "").replace(",", "").replace("[", "").split() for sentence in x_train.values])
model = Word2Vec([re.sub("[^a-zA-Z ]", "", str(sentence)).split() for sentence in x_train.values])
# model = Word2Vec([print(sentence) for sentence in x_train.values])

# Load the Word2Vec model
# model = Word2Vec.load('your_word2vec_model.bin')

# Access the word vectors
word_vectors = model.wv

# Get the vocabulary size
vocab_size = len(word_vectors)

# Get the vector size (dimensionality of word vectors)
vector_size = model.vector_size

# Get the alpha value
alpha = model.alpha

word_vectors_keys = word_vectors.key_to_index.keys()

for key in word_vectors_keys:
    print('KeyOne----->', key)

# Print the vocabulary, vector size, and alpha
print(f"Vocabulary size: {vocab_size}")
print(f"Vector size: {vector_size}")
print(f"Alpha value: {alpha}")

word = "i"
print(word_vectors)
#for word in word_vectors:
#    print(f"word ---->", word)
if word in word_vectors:
    vector = word_vectors[word]
    print(f"----> Word vector for '{word}': {vector}")
else:
    print(f"----> No vector found for '{word}'")

# Get the list of words in the model's vocabulary
vocabulary = list(model.wv.key_to_index.keys())

# Print the words in the vocabulary
for word in vocabulary:
    print(word)
