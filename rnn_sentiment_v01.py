#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:55:57 2017

@author: ianlo
"""

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


import gensim
from gensim.utils import simple_preprocess


#import multiprocessing as mp
from multiprocessing import cpu_count

# set working directory
path = '/Users/ianlo/Documents/Data Analyitcs & Data Science/Deep Learning Developer Course/RNNProject/'
os.chdir(path)


# import custom files
import utils as utils
import global_settings as gs
from parallelproc import applyParallel


# tokenizer: can change this as needed
tokenize = lambda x: simple_preprocess(x)

# set no of groups for partitioning
_number_of_groups = int(cpu_count()*0.8)

# set no of threads
_cpu = int(cpu_count()*0.8)

# max no. of words in a review
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

# initialise global parameters
gs.init()

# -----------------------------------------------------------------------------
# extract all the reviews from the XLS files into a data frame
i=1
rows = pd.DataFrame()
while i <= 1:
    rw = utils.readXlsx("./data/Train/hotel_sentiment_v01.xlsx", sheet = i, header=True)
    # extend takes the content of another list and adds it into that list
    rows = rows.append(rw, ignore_index=True)
    i += 1
    
df = pd.DataFrame(rows)

# remove no longer used objects in memory
del rw, rows

# remove duplicate entries
df = df.dropna().drop_duplicates()

# create index for easy parallel processing
df['indx'] = df.index
df.insert(0,'grpId',df.apply(lambda row: row.indx % _number_of_groups, axis=1, raw=True))
# drop the temp indx column
df = df.drop('indx', 1)



# =============================================================================
# # -----------------------------------------------------------------------------
# # use Gensim word2vec to create embeddings and word-index map
# def create_embeddings(reviews,
#                       embeddings_path='embeddings.npz',
#                       vocab_path='map.json',
#                       **params):
#     """
#     Generate embeddings from a batch of text
#     :param embeddings_path: where to save the embeddings
#     :param vocab_path: where to save the word-index map
#     """
# 
# #    class SentenceGenerator(object):
# #        def __init__(self, dirname):
# #            self.dirname = dirname
# #
# #        def __iter__(self):
# #            for fname in os.listdir(self.dirname):
# #                for line in open(os.path.join(self.dirname, fname)):
# #                    yield tokenize(line)
# #
# #    sentences = SentenceGenerator(data_dir)
# 
#     # load a word2vec from google news
#     model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
# 
# 
# # =============================================================================
# #     # use TfIdf to get the most meaningful words
# #     # create tf-idf vectoriser to get word weightings for sentence
# #     tf = TfidfVectorizer(tokenizer=utils.tokenizer,
# #                          analyzer='word',
# #                          ngram_range=(1,1),
# #                          stop_words = 'english',
# #                          max_df = 0.95,
# #                          min_df = 0.05)
# # 
# # 
# #     # initialise the tfidf vecotrizer with the corpus to get the idf of the corpus
# #     tfidf_matrix =  tf.fit_transform(reviews["processed_text"])
# #     # get feature names
# #     feature_names = tf.get_feature_names()
# #     
# #     
# #     # check the terms with the highest tf-idf value. We can use this to find out
# #     # what terms we need to restructure
# #     from collections import defaultdict
# #     features_by_gram = defaultdict(list)
# #     for f, w in zip(tf.get_feature_names(), tf.idf_):
# #         features_by_gram[len(f.split(' '))].append((f, w))
# #     # remove additional variable
# #     del f, w
# #     
# #     
# #     top_n = 20000
# #     for gram, features in features_by_gram.items():
# #         top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
# #         top_features = [f[0] for f in top_features]
# #         print('\n{}-gram top:'.format(gram), top_features)
# # =============================================================================
# 
#     
#     max_fatures = 20000
#     tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#     tokenizer.fit_on_texts(reviews['processed_text'].values)
#     X = tokenizer.texts_to_sequences(reviews['processed_text'].values)
#     X = pad_sequences(X, padding='post', truncating='post', maxlen = MAX_SEQUENCE_LENGTH)
# 
#     print(tokenizer.word_index)
#     
# #    weights = model.syn0
# 
#     embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
#     for word, i in word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector[:embedding_dimension]
# =============================================================================

    
print('Starting pre-processing: Clean')
df = applyParallel(df.groupby(df.grpId), utils.clean_text, {"dest_col_ind": df.shape[1]-1, "dest_col": "processed_text", "src_col": "review_text"}, _cpu)


print('Starting pre-processing: Lower case')
df = applyParallel(df.groupby(df.grpId), utils.lower_case, {"dest_col_ind": df.shape[1]-1, "dest_col": "processed_text", "src_col": "processed_text"}, _cpu)


print('Starting pre-processing: Restructure text')
df = applyParallel(df.groupby(df.grpId), utils.restructureText, {"dest_col_ind": df.shape[1]-1, "dest_col": "processed_text", "src_col": "processed_text"}, _cpu)


print('Starting pre-processing: Remove custom stop words')
df = applyParallel(df.groupby(df.grpId), utils.remove_stopwords, {"dest_col_ind": df.shape[1]-1, "dest_col": "processed_text", "src_col": "processed_text"}, _cpu)



# =============================================================================
# find the word frequency and remove non frequent words

# the list of words from the corpus that has more than 10 instances
# this list will be used to tag the sentences with PAD, EOS, UNK
wordlist = utils.get_words_by_freq(df['processed_text'], 10)
wordlist.append('PAD')
wordlist.append('EOS')
wordlist.append('UNK')


# substitute everything not in wordlist with PAD, EOS, UNK
max_features = 23413
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['processed_text'].values)
X = tokenizer.texts_to_sequences(df['processed_text'].values)
X = pd.DataFrame(pad_sequences(X, padding='post', truncating='post', maxlen = MAX_SEQUENCE_LENGTH))

# get the word index from the tokenizer
word_index = tokenizer.word_index
word_index['PAD'] = 0
word_index['EOS'] = len(word_index)


# create index to word list so that we can update the padded sentences with the 
# special charaters needed
# create index to word dictionary
index_word = {}
for word in word_index.keys():
    if word in wordlist:
        index_word[word_index.get(word)] = word
    else:
        index_word[word_index.get(word)] = 'UNK'




# create special dictionary for faster replacement of tokens in sentences
temp = {}
for word in word_index.keys():
    if word in wordlist:
        temp[word_index.get(word)] = word_index.get(word)
    else:
        temp[word_index.get(word)] = word_index.get(word) # this is really meaningless but this is a short hack from experimenting

# add PAD and EOS to the index_word list
#temp[0] = 'PAD'
#temp[len(word_index)] = 'EOS' 
temp[0] = word_index['PAD']
temp[len(word_index)] = word_index['EOS']


# replace tokens in sentences
X = X.apply(lambda y: y.map(lambda x: temp.get(x,x)), axis=1)


# mark the first PAD with EOS to indicate end of sentence
def set_eos(row):
#    row.iloc[list(row).index('PAD') if 'PAD' in list(row) else -1] = word_index['EOS']
    row.iloc[list(row).index(0) if 0 in list(row) else -1] = word_index['EOS']
    return row

X = X.apply(lambda y: set_eos(y), axis=1)




# one hot encode target
enc1 = OneHotEncoder()
Y = pd.DataFrame(enc1.fit_transform(pd.DataFrame(df.iloc[:,4])).toarray())


# create train / test set
x_train, x_valid, y_train, y_valid = train_test_split(X,
                                                      Y,
                                                      test_size = 0.2,
                                                      random_state = gs.seedvalue)

# load a word2vec from google news
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  

# create a vocab based on the google news vectors model
vocab = dict([(k, v.index) for k, v in model.vocab.items()])


# create the embeddings_index so that we can get the embeddings from the google model
# for each word
embeddings_index = {}
# if word is not found in the google vector then set to 0
# otherwise the embeddings_index will have the corresponding google news vector embedding for
# the specific word found in the review text
for word in word_index.keys():
    if vocab.get(word) is not None:
        embeddings_index[word] = model.syn0[vocab.get(word)]
    else:
        embeddings_index[word] = [0] * 300 # default size of google news vector embeddings


pad_embed=np.empty(300, dtype=float); pad_embed.fill(0.001) 
eos_embed=np.empty(300, dtype=float); eos_embed.fill(0.999)
unk_embed=np.empty(300, dtype=float); unk_embed.fill(0.555) 
embeddings_index['PAD'] = pad_embed
embeddings_index['EOS'] = eos_embed
embeddings_index['UNK'] = unk_embed



# embedding_dimension is the size of the vector
embedding_dimension = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)[:embedding_dimension]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]



lstm_out = 1024

# create the model
model = Sequential()
model.add(Embedding(max_features, embedding_dimension, input_length = x_train.shape[1], weights=[embedding_matrix], trainable=True))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(5, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())


batch_size = 32
model.fit(np.array(x_train), np.array(y_train), epochs = 7, batch_size=batch_size, verbose = 1)



