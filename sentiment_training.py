import re
import pandas as pd
import sentiment_constants as const
import sentiment_functions as fun
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from IPython.display import display

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

x = pd.read_csv('nonletterclean_df.csv')
y = pd.read_csv('sentiment_df.csv')

# Split the data into test and training sets both for tweets and sentiments
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

fun.title('x_train')
print(x_train.shape)
fun.title('x_test')
print(x_test.shape)
fun.title('y_train')
print(y_train.shape)
fun.title('y_test')
print(y_test.shape)

# Find out the length of all the sentences in the x_train
# and return the greatest one, to be uses as the length of
# the word vector
############### fun.title('embedding size')
############### const.MAX_LENGTH = int(fun.Emsize().getES())

# Build a model out of the words in the training set
model = Word2Vec([re.sub("[^a-zA-Z ]", "", str(sentence)).split() for sentence in x_train.values])################, vector_size=const.MAX_LENGTH, min_count=1)



# TEST AREA



#merged = x_test()
fun.title('whole x_train')
print(x_train)

fun.title('values')
print(x_train.values)

fun.title('re.sub')
print([re.sub("[^a-zA-Z ]", "", str(sentence)).split() for sentence in x_train.values])################, vector_size=const.MAX_LENGTH, min_count=1)

fun.title('Model Word')
print(model)
fun.title('Model Keys')
print(model.wv.key_to_index.keys())
fun.title('Index')
print([(word, model.wv.key_to_index[word]) for word in model.wv.index_to_key])
fun.title('Frequency')
word_frequencies = [(word, model.wv.key_to_index[word]) for word in model.wv.index_to_key]
word_frequencies_sorted = sorted(word_frequencies, key=lambda x: x[1], reverse=True)
word_frequencies_filtered = [(word, freq) for word, freq in word_frequencies_sorted]
print(word_frequencies_filtered)

fun.title('vectors')
vec = [(i+'->',model.wv[i]) for i in model.wv.key_to_index.keys()]
print(vec)



# TEST AREA




# # # Load the Word2Vec model
# # # model = Word2Vec.load('your_word2vec_model.bin')

# # Access the word vectors
# word_vectors = model.wv
# # Get the vocabulary size
# ############## const.VOCAB_SIZE = len(word_vectors)
# # # Get the vector size (dimensionality of word vectors)
# # vector_size = model.vector_size
# # # Get the alpha value
# # alpha = model.alpha
# # word_vectors_keys = word_vectors.key_to_index.keys()

# # # Print the vocabulary, vector size, and alpha
# # print(f"Vocabulary size: {const.VOCAB_SIZE}")
# # print(f"Vector size: {vector_size}")
# # print(f"Alpha value: {alpha}")
# # print(f"Model: {model}")

# # Print the words in the word vector and their frequencies
# # word_frequencies = [(word, model.wv.key_to_index[word]) for word in model.wv.index_to_key]
# # word_frequencies_sorted = sorted(word_frequencies, key=lambda x: x[1], reverse=True)

# # Minimum frequency
# # word_frequencies_filtered = [(word, freq) for word, freq in word_frequencies_sorted if freq >= const.MIN_FREQ]

# # To Keep : THE VOCABULARY
# # for word, frequency in word_frequencies_filtered:
# #     print(f"{word} {frequency}", end=" - ")
# # print()

# # 
# import tensorflow as tf
# import keras
# from tensorflow import keras
# from keras.preprocessing.text import Tokenizer
# from keras import utils

# # Tokenizer = Tokenizer()
# # tokenizer.fit_on_texts(model)

# # # Get the word counts for each token
# # word_counts = tokenizer.word_counts

# # # Get the frequency of each token
# # word_freq = tokenizer.word_docs

# # # Print the word counts and frequencies
# # for word, count in word_counts.items():
# #     print(f"{word}: {count}")

# # for word, freq in word_freq.items():
# #     print(f"{word}: {freq}")


# const.MAX_LENGTH = 100




# # # Get the list of words in the model's vocabulary
# # vocabulary = list(model.wv.key_to_index.keys())
# # print(vocabulary)



# # # Print the words in the vocabulary

# # # print(vocabulary)
# # # all_tokens = nltk.word_tokenize(text)
# # # num_tokens = len(all_tokens)
# # # print(num)
# # #
# # print("vocab size", const.VOCAB_SIZE)
# const.VOCAB_SIZE = 23

# tokenizer = Tokenizer(num_words=const.VOCAB_SIZE, oov_token='<OOV>')
# tokenizer.fit_on_texts(x_train)

# # word_index = word_frequencies_filtered

# # print('word_index -------> ',word_frequencies)

# word_index = tokenizer.word_index
# print("word_index", word_index)

# x_train_sequences = tokenizer.texts_to_sequences(x_train)
# # print(f'x train sequences {x_train_sequences}')
# x_test_sequences = tokenizer.texts_to_sequences(x_test)
# # print(f'x test sequences {x_test_sequences}')

# from tensorflow.keras.preprocessing.sequence import pad_sequences

# x_train = pad_sequences(x_train_sequences, 
#                                 maxlen=const.MAX_LENGTH,
#                                 padding='post',
#                                 truncating='post')

# x_test = pad_sequences(x_test_sequences,
#                              maxlen=const.MAX_LENGTH,
#                              padding='post',
#                              truncating='post')

# import numpy as np

# embedding_matrix = np.zeros((len(word_index)+1, const.MAX_LENGTH))

# for word, i in word_index.items():
#     try:
#         embedding_matrix[i] = model.wv[i]
#     except:
#         embedding_matrix[i] = np.zeros(const.MAX_LENGTH)

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(const.VOCAB_SIZE, const.MAX_LENGTH, input_length=const.MAX_LENGTH,
#         weights=[embedding_matrix]),
#     tf.keras.layers.LSTM(128),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# model.compile(optimizer='adam', metrics=['acc'], 
#         loss='sparse_categorical_crossentropy')

# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
