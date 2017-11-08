#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:48:11 2017

@author: ianlo
"""

from nltk.corpus import stopwords
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

allow_numbers = pd.read_csv('./AllowedNumbers.csv')
allow_numbers = allow_numbers['Column1'].astype(str).tolist()

allow_stopwords = pd.read_csv('./AllowedStopwords.csv')
allow_stopwords = allow_stopwords['Column1'].astype(str).tolist()

addtn_stopwords = pd.read_csv('./AdditionalStopwords.csv')
addtn_stopwords = addtn_stopwords['Column1'].astype(str).tolist()

stpwrd = list(set(stopwords.words('english')) | set(addtn_stopwords) - set(allow_stopwords))
stpwrd.sort()

#==============================================================================
# A. WordNet only contains "open-class words": nouns, verbs, adjectives, and adverbs.
# Thus, excluded words include determiners, prepositions, pronouns, conjunctions, and particles.
# 
#==============================================================================

def is_stopword(x):
    return x not in stopwords.words('english')


def is_stopword_numbers(x):
    if x.isnumeric():
        return x in allow_numbers
    else:
        return x not in stopwords.words('english')
