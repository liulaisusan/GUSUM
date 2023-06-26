import numpy as np
import math
from math import*
from sentence_transformers import SentenceTransformer
from decimal import Decimal
from scipy.spatial.distance import cdist

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def manhattan_distance(x,y): 
    return sum(abs(a-b) for a,b in zip(x,y))

def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)
 
def minkowski_distance(x,y):
    p_value=3
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

def createGraph(sentences):
    # bert-base-nli-mean-tokens
    # roberta-base-nli-stsb-mean-tokens
    # distilbert-base-nli-stsb-mean-tokens
    # bert-base-nli-stsb-mean-tokens
    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens') #The sentence transform models mentioned above can be used.
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    sentenceGraph = 1-cdist(sentence_embeddings, sentence_embeddings, metric='cosine')
    np.fill_diagonal(sentenceGraph, 0)
    return sentenceGraph


def findHighestSimilarityRank(similarityMatrix, initialRank):
    newRank = initialRank * np.sum(similarityMatrix, axis=1)
    return newRank