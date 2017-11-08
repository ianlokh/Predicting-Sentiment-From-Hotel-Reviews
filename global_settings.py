#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:12:06 2017

@author: ianlo
"""

# import importlib
# importlib.reload(gs)

import pandas as pd


def init():
    global var_threshold
    global seedvalue
    global word2vecmodel
    global chunksize
    global maxqnstrlen
    global splitPairs
    
    # default values
    word2vecmodel = ''
    var_threshold = 0.01
    seedvalue = 37458
    chunksize = 50000
    maxqnstrlen = 1141

    # read in excel spreadsheet of pairs of concatenated words to split
    pairs = pd.read_csv('./SplitPairs.csv')
    splitPairs = [tuple(x) for x in pairs.to_records(index=True)]