#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:12:06 2017

@author: ianlo
"""
import threading, math, re
import numpy as np # linear algebra

import wordnetutils as wnu

from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import global_settings as gs


# define stop word list
stops = set(GENSIM_STOPWORDS)

# initiaise tokenizer
tokenizer = simple_preprocess



class ThreadSafeDict(dict) :
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()




def stem(row, column):
    stemlist = simple_preprocess(row[column])
    return stemlist



def lemm_question(row, column):
    vec = row[column].split(' ')
    return vec



def remv_stopwords(row, column):
    sent = ''
    for word in simple_preprocess(row[column]):
        if word not in wnu.stpwrd:
            sent += ' ' + word
    return sent



def tokenize_stem(row, column):
    sent = ' '.join(simple_preprocess(row[column]))
    return sent




def is_nan(x):
    try:
        return math.isnan(x)
    except:
        return False



def none_to_float(x):
    try:
        return float(0 if x is None else x)
    except:
        return 0
    


def to_32bit(df):
    for c in df.select_dtypes(include=['float64']):
        df[c] = df[c].astype(np.float32)
        
    for c in df.select_dtypes(include=['int64']):
        df[c] = df[c].astype(np.int32)
        
    return df



def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def readXlsx( fileName, **args ):

    import zipfile
    from xml.etree.ElementTree import iterparse

    if "sheet" in args:
       sheet=args["sheet"]
    else:
       sheet=1
    if "header" in args:
       isHeader=args["header"]
    else:
       isHeader=False

    rows   = []
    row    = {}
    header = {}
    z      = zipfile.ZipFile( fileName )

    # Get shared strings
    strings = [ el.text for e, el
                        in  iterparse( z.open( 'xl/sharedStrings.xml' ) )
                        if el.tag.endswith( '}t' )
                        ]
    value = '' 

    # Open specified worksheet
    for e, el in iterparse( z.open( 'xl/worksheets/sheet%d.xml'%( sheet ) ) ):
       # get value or index to shared strings
       if el.tag.endswith( '}v' ):                                   # <v>84</v>
           value = el.text
       if el.tag.endswith( '}c' ):                                   # <c r="A3" t="s"><v>84</v></c>

           # If value is a shared string, use value as an index
           if el.attrib.get( 't' ) == 's':
               value = strings[int( value )]

           # split the row/col information so that the row leter(s) can be separate
           letter = el.attrib['r']                                   # AZ22
           while letter[-1].isdigit():
               letter = letter[:-1]

           # if it is the first row, then create a header hash for the names
           # that COULD be used
           if rows ==[]:
               header[letter]=value
           else:
               if value != '':

                   # if there is a header row, use the first row's names as the row hash index
                   if isHeader == True and letter in header:
                       row[header[letter]] = value
                   else:
                       row[letter] = value

           value = ''
       if el.tag.endswith('}row'):
           rows.append(row)
           row = {}
    z.close()
    return rows




def clean(text, remove_stopwords=False, stem_words=False):    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # remove non-word characters
    text = re.sub(r"(\W+)", r" ", text)
    # replace numbers
    text = re.sub(r"([0-9]+)", r" ", text)
    
    # Return a list of words
    return(text)




# Change all to lower case so that frequencies will be calculated correctly
def clean_text(df, dest_col_ind, dest_col, src_col):
    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df.apply(lambda x: clean(x[src_col], remove_stopwords=True, stem_words=True), axis=1)
    else:
        df.insert(dest_col_ind, dest_col, df.apply(lambda x: clean(x[src_col], remove_stopwords=True, stem_words=True), axis=1))
    return df



# Change all to lower case so that frequencies will be calculated correctly
def lower_case(df, dest_col_ind, dest_col, src_col):
    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df[src_col].str.lower()
    else:
        df.insert(dest_col_ind, dest_col, df[src_col].str.lower())
    return df



# Separate common words that have been merged due to PDF to text conversion
def restructureText(df, dest_col_ind, dest_col, src_col):

    def restructure(x, src_col, pairs):
        text = x[src_col]
        for index, word1, word2 in pairs:
            exp = "(("+ word1 +")(" + word2 +"))"
            # split word1 and word2
            text = re.sub(exp, r"\2 \3 ", text)
        return text.replace('\x00','')


    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df.apply(lambda x: restructure(x, src_col, gs.splitPairs), axis=1).values
    else:
        df.insert(dest_col_ind, dest_col, df.apply(lambda x: restructure(x, src_col, gs.splitPairs), axis=1))
    return df



# remove stop words from the text field
def remove_stopwords(df, dest_col_ind, dest_col, src_col):
    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df.apply(lambda x: remv_stopwords(x, src_col), axis=1).values
    else:
        df.insert(dest_col_ind, dest_col, df.apply(lambda x: remv_stopwords(x, src_col), axis=1))
    return df



def get_words_by_freq(corpus, freq):
    # Note that `ngram_range=(1, 1)` means we want to extract Unigrams, i.e. tokens.
    ngram_vectorizer = CountVectorizer(analyzer='word',
                                       tokenizer=tokenizer,
                                       ngram_range=(1, 1), min_df=1)
    # X matrix where the row represents sentences and column is our one-hot vector for each token in our vocabulary
    X = ngram_vectorizer.fit_transform(corpus)
    
    # Vocabulary
    vocab = list(ngram_vectorizer.get_feature_names_out())
    
    # Column-wise sum of the X matrix.
    # It's some crazy numpy syntax that looks horribly unpythonic
    # For details, see http://stackoverflow.com/questions/3337301/numpy-matrix-to-array
    # and http://stackoverflow.com/questions/13567345/how-to-calculate-the-sum-of-all-columns-of-a-2d-numpy-array-efficiently
    counts = X.sum(axis=0).A1
    
    #freq_distribution = Counter(dict(zip(vocab, counts)))
    
    wordlist = []
    for word, count in zip(vocab, counts):
            if count > freq:
                wordlist.append(word)
    return wordlist

