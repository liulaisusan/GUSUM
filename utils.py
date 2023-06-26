from datasets import load_dataset

from nltk import sent_tokenize
import nltk
from nltk.corpus import stopwords
import re 
import os.path
import time
from tqdm import tqdm
import pickle

from sentenceRanking import allCorpusSentenceRanking, textSentenceCount
from graph import createGraph,findHighestSimilarityRank
from Evaluation import rougeEvaluation
from sentenceRanker import SentenceRanker

def saveEvaluator(evalutor, d, path, dataname, interval):
        if not path:
            raise ValueError('Path is not defined.')
        if not os.path.exists(path):
            os.makedirs(path)
        new_path = os.path.join(path, f'evaluator_{dataname}_{d}.pkl')
        with open(new_path, 'wb') as f:
            pickle.dump(evalutor, f)
        if type(d) == int:
            old_path = os.path.join(path, f'evaluator_{dataname}_{d-interval}.pkl')
            if os.path.exists(old_path):
                os.remove(old_path)

def load_dataset_from_huggingface(dataset_name, dataset_version):
    dataset = load_dataset(dataset_name, dataset_version)
    return dataset

def cleanDocument(document):
    sentences = sent_tokenize(document)
    cleanedDocument=[]
    for sentence in sentences:
        # Removing Paranthesis
        sentence =re.sub("[\(\[].*?[\)\]]", "", sentence)

        #Removing \
        sentence.replace('\\','')

        #Removing -- and before
        if "--" in sentence: 
            splitting=sentence.split("--")
            sentence =splitting[1]

        cleanedDocument.append(sentence)
    newCorpus = ' '.join(cleanedDocument)
    return newCorpus

def saveFile(directory,filename,document):
    #directory = '/content/drive/MyDrive/Colab Notebooks/CnnDailyDataset/MySummaries'
    #filename = "a.txt"
    file_path = os.path.join(directory, filename)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    file = open(file_path, "w", encoding='utf-8')
    file.write(document)
    file.close()
    
    
def createSummary(sentences,sentencesRank,sentenceAmount):
    temp=sorted(sentencesRank)
    threshold=temp[-sentenceAmount]
    summarySentencesIndexes=[]

    for i in range(len(sentencesRank)):
        if sentencesRank[i]>=threshold:
            summarySentencesIndexes.append(i)

    #print(summarySentencesIndexes)
    summary=""
    for i in range(len(summarySentencesIndexes)):
        next_sentence = sentences[summarySentencesIndexes[i]]
        if "#Person" in next_sentence:
            summary=summary + '\n ' + next_sentence
        else:
            summary=summary + ' ' + next_sentence
  
    return summary

def mainCreateSummaries(corpus):
    corpus=cleanDocument(corpus) # Clean Paranthesis
    # sentences= sent_tokenize(corpus)
    sentences= seperateSentences(corpus)
    
    ranker = SentenceRanker()
    featurelist = ['sentencePosition', 'sentenceLength', 'properNoun', 'numericalToken']
    initialRank = ranker.rank(sentences, corpus, featurelist)
    # initialRank=allCorpusSentenceRanking(sentences,corpus)

    similarityMatrix=createGraph(sentences) # create matrix ( Graph) shows similarities of sentences

    newRank=findHighestSimilarityRank(similarityMatrix, initialRank)

    #summaryPercentage=0.3
    #sentenceNumberInSummary=int(len(sentences)*summaryPercentage)
    # sentenceNumberInSummary=1
    # if len(sentences)>2:
    #     sentenceNumberInSummary=5
    sentenceNumberInSummary=min(10,len(sentences))
    # sentenceNumberInSummary=int(0.9*len(sentences))

    lastSummary=createSummary(sentences,newRank,sentenceNumberInSummary)
    return lastSummary 

def seperateSentences(corpus):
    sentences =  re.split(r'#Person1#|#Person2#' , corpus)
    sentences = list(filter(None, sentences))
    sentences = [ '#Person1#' + sentence if i %2 == 0 else '#Person2#' + sentence for i, sentence in enumerate(sentences)]

    return sentences

def processCorpus(corpus):
        corpus = cleanDocument(corpus) # Clean Paranthesis
        sentences = seperateSentences(corpus)
        new_corpus = ""
        for sentence in sentences:
                new_corpus=new_corpus + '\n ' + sentence
        return new_corpus

def processAllCorpus(dataset, N = None):
    N = N if N else len(dataset)
    processed_sentences = []
    processed_corpus = []
    for d in tqdm(range(N)):
        corpus=cleanDocument(dataset['dialogue'][d]) # Clean Paranthesis
        sentences= seperateSentences(corpus)
        processed_sentences.append(sentences)
        new_corpus = ""
        for sentence in sentences:
                new_corpus=new_corpus + '\n ' + sentence
        processed_corpus.append(new_corpus)
  
    return processed_corpus, processed_sentences