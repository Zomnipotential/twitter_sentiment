import sys
print('cleaning ->', sys.executable)

# PREAMBLE: CONTROL & INDICATOR SECTION
import time
absolute_start_time = time.time()
start_time = time.time()

# def func.title(content):
# 	nl = '\n'
# 	length = len(content)
# 	print(f"{5*nl}{(4+length)*'*'}{nl}* {content} *{nl}* {(length)*' '} *{nl}{(4+length)*'*'}{nl}")

# ACTUAL CODE STARTS HERE
import sentiment_functions as func
import sentiment_constants as const
from IPython.display import display
import pandas as pd
import nltk
from nltk.corpus import stopwords
import progress_indicator as pi
import threading
import time
import re
import contractions
from nltk.tokenize import TweetTokenizer
import csv

number_of_rows = int(sys.argv[1])

func.title('It all starts here')

def print_exec_time(start_time):
	elapsed_time = time.time() - start_time 
	# Convert the elapsed time to a human-readable format
	hours = int(elapsed_time // 3600)
	minutes = int((elapsed_time % 3600) // 60)
	seconds = int(elapsed_time % 60)
	milliseconds = int((elapsed_time % 1) * 1000)

	# Print the elapsed time
	nl = const.nl
	print(f"{5*nl}====== Elapsed time: {hours}h {minutes}m {seconds}s {milliseconds}ms ======{2*nl}")

# read the file

if __name__ == '__name__':
	print('Starting Spark Session. All data is loaded into memory')

# choose a small part of the DB to run the functions faster
# 2B REMOVED
dframe = pd.read_csv('.databases/tweets.csv', encoding='ISO-8859-1')
dframe = dframe.iloc[0:number_of_rows]
display(dframe)
dframe.columns = const.columns

# learn to know the database

func.title('the first five rows of the database')
display(dframe.head())
display(dframe)
func.title('shape of the database')
display(dframe.shape)
dframe.columns = const.columns
func.title('information')
display(dframe.info())
func.title('description')
display(dframe.describe())

# investigate the dataframe

func.title('number of unique items in each column')
display(dframe.apply(pd.Series.nunique))

func.title('number of items')
for _ in dframe.columns:
    display(_)
    display(dframe[_].value_counts())

# show the whole table without any truncations
pd.set_option('display.max_colwidth', None)

print(dframe[dframe['user'].astype('string') == 'lost_dog']['text'])

func.title('find duplicated tweets')
all_dframe_ids = dframe['id'].value_counts()
display(all_dframe_ids)

func.title('extract tweets that appear more than once')
duplicated_dframe_ids = all_dframe_ids[all_dframe_ids >= 2]
display(duplicated_dframe_ids)

func.title('find out what tweets are reapeated')
duplicated_tweet_dframe = dframe[dframe['id'].isin(duplicated_dframe_ids.index)]
display(duplicated_tweet_dframe.sort_values(by='id').head(50))

func.title('see if all repeated tweets have one 0 and one 4 as sentiment')
aggregated_sentiments = duplicated_tweet_dframe.groupby('id').agg({'sentiment': 'sum', 'text': 'first'})
display(aggregated_sentiments)

func.title('see if the estimated number of 4s matches our guess')
aggregated_sentiments['sentiment'].value_counts()

func.title('remove duplicates')
dframe_wihout_duplicates = dframe[~(dframe['id'].isin(duplicated_dframe_ids.index) & (dframe['sentiment'] == 0))]
display(dframe_wihout_duplicates)

func.title('"neutralize" the sentiments')
# change the sentiment of those rows in dframe_wihout_duplicates that are listed in duplicated_dframe_ids to 2
neutralized_dframe = dframe_wihout_duplicates.copy()
neutralized_dframe.loc[neutralized_dframe['id'].isin(duplicated_dframe_ids.index), 'sentiment'] = 2
display(neutralized_dframe)

func.title('check if neutralization successful')
neutralized_dframe.apply(pd.Series.nunique)
neutralized_dframe['sentiment'].value_counts()

func.title('clean out - remove flag')
flagless_dframe = neutralized_dframe.drop(columns=['flag'])
flagless_dframe.head(50)

func.title('drop other unnecessary columns')
df = flagless_dframe.drop(columns=['id', 'date', 'user'])
display(df)

func.title('From now on we have df to use')

func.title('info')
print(df.info())

func.title('head')
print(df.head(10))

func.title('tail')
print(df.tail())

# we import nltk here

nltk.download('punkt')

# we install the module 'contractions' here
# this is done in the terminal window using
# > pip install contractions
# then we download the stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

func.title('NLTK English stopwords are')
display(stop_words)

### These could be done separately
# func.title('wordnet lemmatizer')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

# convert stringified list to list
def strg_list_to_list(strg_list):
  	return strg_list.strip("[]").replace("'","").replace('"',"").replace(",","").split()
  
# we import re and contractions here to do some removal of hyperlinks, stop-words, ...
def remove_retweet_label(text):
  	return re.sub('RT @[\w_]+:','', text)

def remove_video_label(text):
  	return re.sub('VIDEO:','', text)

def remove_hyperlink(text):
  	return re.sub(r'http\S+','', text) # r=raw \S=string

def remove_twitterhandle(text):
  	return re.sub('@[A-Za-z0-9_]+(:)?','', text)

def remove_escape_sequence(text):
  	return re.sub(r'\n','', text)

def remove_extra_spaces(text):
  	return  re.sub(r"\s+"," ", text)  

def remove_contraction(text):
  	return ' '.join([contractions.fix(word) for word in text.split()])
  
def remove_stopwords(text):
  	return " ".join([word for word in text.split() if word not in stop_words])

def pretokenization_cleaning(text):
	ext=remove_retweet_label(text)
	text=remove_video_label(text)
	text=remove_hyperlink(text)
	text=remove_twitterhandle(text)
	text=remove_escape_sequence(text)
	text=remove_extra_spaces(text)  
	text=remove_contraction(text)
	#text=remove_stopwords(text)
	return text

# we import TweetTokenizer from nltk.tokenize
def tokenize(text):
	tknzr = TweetTokenizer(reduce_len=True)
	return tknzr.tokenize(text)

# normalizing task using Stemmer
import nltk
def stemming(unkn_input):
	porter = nltk.PorterStemmer()
	if (isinstance(unkn_input,list)):
		list_input=unkn_input
	if (isinstance(unkn_input,str)):
		list_input=strg_list_to_list(unkn_input)
	list_stemmed=[]
	for word in list_input:
		word=porter.stem(word)
		list_stemmed.append(word)
	#return " ".join(list_stemmed) #use this to return a string
	return list_stemmed #use this to return a list
	
# normalizing task using Lemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize requires list input
def lemmatize(unkn_input):
    if (isinstance(unkn_input,list)):
    	list_input=unkn_input
    if (isinstance(unkn_input,str)):
    	list_input=strg_list_to_list(unkn_input)
    list_sentence = [item.lower() for item in list_input]
    nltk_tagged = nltk.pos_tag(list_sentence)  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])),nltk_tagged)
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        #" ".join(lemmatized_sentence)
    return lemmatized_sentence
    
# (3) post tokenization task
# the following post-tokenization receives list as input parameter
# and returns list as output

def remove_punc(list_token): 
	#print(list_token)
	def process(strg_token):
		strg_numb ='''0123456789'''
		strg_3dots='...'
		strg_2dots=".."
		strg_punc = '''!()+-[]{}|;:'"\,<>./?@#$£%^&*_~“”…‘’'''
		strg_output=''
		# for idx, char in enumerate(strg_token): 
		# print(item)
		if (len(strg_token)==0): #empty char
			strg_output +=''
		else:
			if (all(char in strg_numb for char in strg_token) or
				strg_token[0] in strg_numb): #if char is a number
				strg_output +=''
			else:
				if (len(strg_token)==1 and strg_token in strg_punc): #if char is a single punc
					strg_output +=''
				else:
					if (strg_token[0]=='#'): #if char is hashtag
						strg_output +=strg_token.lower()
					elif(strg_token==strg_3dots or strg_token==strg_2dots):
						strg_output +=''
					else: 	# other than above, char could be part of word,
							# e.g key-in
						strg_output += strg_token
		return strg_output
	list_output=[process(token) for token in list_token]
	return list_output

def remove_empty_item(list_item):
	token = [token for token in list_item if len(token)>0]
	return token

def lowercase_alpha(list_token):
	return [token.lower() if (token.isalpha() or token[0]=='#') else token for token in list_token]

def posttokenization_cleaning(unkn_input):
	embedding_size = 0
	list_output=[]
	if (isinstance(unkn_input,list)):
		list_output=unkn_input
	if (isinstance(unkn_input,str)):
		list_output=strg_list_to_list(unkn_input)
	list_output=remove_punc(list_output)
	list_output=remove_empty_item(list_output)
	#list_output=lowercase_alpha(list_output)
	number_tokens = len(list_output)
	if embedding_size < number_tokens:
		const.modify_constant(number_tokens)
	return (list_output)

print_exec_time(start_time)


# PREPROCESSING

start_time = time.time()
func.title('Preprocessing')
# start the "dotter"-thread
pi.dot_thread = pi.threading.Thread(target=pi.print_dots, args=(pi.stop_progress_indicator,))
pi.start_thread()

# (4) Run Preprocessing Tasks
# (4.1) Pretokenization
# calling pretokenization_cleaning (list comprehension style)
df['pretoken']=[pretokenization_cleaning(sentence) for sentence in df['text']]
# stop the dotter-thread
pi.stop_thread()
# calculate and print elapsed time
print_exec_time(start_time)
print(df.head(10))


# TOKENIZATION

start_time = time.time()
func.title('Tokenization')
# start new thread
pi.dot_thread = pi.threading.Thread(target=pi.print_dots, args=(pi.stop_progress_indicator,))
pi.start_thread()

# (4.3) Tokenization
# calling tokenize (list comprehension style)
df['token']=[tokenize(sentence) for sentence in df['pretoken']]
# stop the dotter
pi.stop_thread()
# calculate and print execution time
print_exec_time(start_time)
print(df.head(10))


# STEMMING

start_time = time.time()
func.title('Stemming')
# start new thread
pi.dot_thread = pi.threading.Thread(target=pi.print_dots, args=(pi.stop_progress_indicator,))
pi.start_thread()

# (4.2) Stemming (optional)
#calling stemming (list comprehension style)
df['stemmed']=[stemming(tokenize(sentence)) for sentence in df['pretoken']]
# stop the dotter
pi.stop_thread()
# calculate and print execution time
print_exec_time(start_time)
print(df.head(10))


# LEMMATIZATION

start_time = time.time()
func.title('Lemmatization')
# start new thread
pi.dot_thread = pi.threading.Thread(target=pi.print_dots, args=(pi.stop_progress_indicator,))
pi.start_thread()

# (4.4) Lemmatizing (optional)
# calling stemming (list comprehension style)
df['lemmatized']=[lemmatize(tokenize(sentence)) for sentence in df['pretoken']]
# stop the dotter
pi.stop_thread()
# calculate and print execution time
print_exec_time(start_time)
print(df.head(10))


# POST-TOKENIZATION

start_time = time.time()
func.title('Post-Tokenization')
# start new thread
pi.dot_thread = pi.threading.Thread(target=pi.print_dots, args=(pi.stop_progress_indicator,))
pi.start_thread()

# (4.5) Post-Tokenization Cleaning
# calling post-tokenization_cleaning (list comprehension style)
df['posttoken']=[posttokenization_cleaning(list_sentence) for list_sentence in df['lemmatized']]
# stop the dotter
pi.stop_thread()
# calculate and print execution time
print_exec_time(start_time)
print(df.head(10))


# POST-TOKENIZATION

start_time = time.time()
func.title('String-Cleaning')
# start new thread
pi.dot_thread = pi.threading.Thread(target=pi.print_dots, args=(pi.stop_progress_indicator,))
pi.start_thread()

# (4.5) Post-Tokenization Cleaning
# calling string_cleaning (list comprehension style)
df['nonletterclean']=[re.sub("[^a-zA-Z ]", "", str(list_sentence)) for list_sentence in df['posttoken']]
# stop the dotter
pi.stop_thread()
# calculate and print execution time
print_exec_time(start_time)
print(df.head(10))

# DATA EXPORT

start_time = time.time()

final_csv='./final_csv.csv'
nonletterclean_csv = './nonletterclean_df.csv'
sentiment_csv = './sentiment_df.csv'

 
def export_csv(data_frame, csv_name):
	# get list of columns from df
	listHeader=list(data_frame.columns.values)
	print(listHeader)
	# insert index column header to the first pos in list
	listHeader.insert(0,'index')
	# export df to csv
	# set the quoting to be QUOTE_ALL
	data_frame.to_csv(csv_name,columns=data_frame.columns.values,quoting=csv.QUOTE_ALL)
	# in case pandas does not create header for the index column, do it manually.
	# create a new header line
	line_head= ','.join(listHeader)+'\n'

	with open(csv_name) as f:
		lines = f.readlines()

	# replace old header with new header
	lines[0] = line_head

	with open(csv_name, 'w') as f:
		f.writelines(lines)

export_csv(df, final_csv)
display(df.columns)
display(df)
nonletterclean_df = df.drop(['sentiment','text','pretoken','token','stemmed','lemmatized','posttoken'], axis=1)
sentiment_df = df.drop(['text','pretoken','token','stemmed','lemmatized','posttoken','nonletterclean'], axis=1)
# export the final dataset as 'input_csv'; the 
# one that is meant to be fed into the ML engine
export_csv(nonletterclean_df, nonletterclean_csv)
export_csv(sentiment_df, sentiment_csv)

# # (2) import for testing
# df_test = pd.read_csv(final_csv)
# print(df_test.info())
# 
# # (3) check empty cells
# # 46 rows
# print(len(df_test[df_test.isna().any(axis=1)]))
# 
# # (4) further inspection shows that these 46 rows become empty 
# # because their data have been wiped out during pretokenization 
# df_test[df_test.isna().any(axis=1)]
# 
# print(final_csv[1:1000])

print_exec_time(start_time)
print_exec_time(absolute_start_time)