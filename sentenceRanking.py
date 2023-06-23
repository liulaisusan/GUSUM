# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:06:47 2021

@author: TUBA
"""

import nltk

from nltk import sent_tokenize, word_tokenize,PorterStemmer


def allCorpusSentenceRanking(tokenizedCorpus,corpus):
    sentenceRankList=[]
    for i, sentence in enumerate(tokenizedCorpus):
        value=sentenceRanking(sentence,i,corpus)
        value=round(value,5)
        sentenceRankList.append(value)
    return sentenceRankList

def sentenceRanking(sentence,location,corpus):
    value=0
    value=sentencePosition(sentence,location,corpus)
    value=value+sentenceLength(sentence,corpus)
    value=value+properNoun(sentence,corpus)
    value=value+numericalToken(sentence,corpus)
    return value

def textWordCount(Text):
    number_of_words = word_tokenize(Text)
    count=(len(number_of_words))
    return count

def textSentenceCount(Text):
    number_of_sentences = sent_tokenize(Text)
    count=(len(number_of_sentences))
    return count

def longestSentenceLenght(Text):
    text=sent_tokenize(Text)
    lens = [textWordCount(x) for x in text]
    return max(lens)

def sentencePosition(sentence,location,corpus): 
    N=textSentenceCount(corpus)
    if location+1 == N:
        return 1.0
    elif location==0:
        return 1.0
    else:
        value=(N-location)/N
        return value
    
def sentenceLength(sentence,corpus):
    return textWordCount(sentence)/longestSentenceLenght(corpus)


def properNoun(sentence,corpus):
    text = nltk.word_tokenize(sentence)
    tagged=nltk.pos_tag(text)
    noProperNoun= tagged.count('NNP')
    return noProperNoun/len(text)

def numericalToken(sentence,corpus):
    text = nltk.word_tokenize(sentence)
    tagged=nltk.pos_tag(text)
    noNumericalToken= tagged.count('CD')
    return 1-(noNumericalToken/len(text))
