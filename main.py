from nltk import sent_tokenize
import nltk
from nltk.corpus import stopwords
import time

from utils import load_dataset_from_huggingface, cleanDocument, saveFile, createSummary, mainCreateSummaries
from Evaluation import rougeEvaluation
from sentenceRanking import allCorpusSentenceRanking, textSentenceCount

if __name__ == "__main__":
    nltk.download('punkt')

    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    dataset = load_dataset_from_huggingface('cnn_dailymail', '3.0.0')

    startTimeforOverall = time.time()
    all_hypothesis=[]
    all_references=[]

    #documentNumber
    N=10   ## How many documents do you want summarized over the CNN/Daily Mail dataset? Limit 11490
    startN=0

    for d in range(N):
        startTimeforDocument = time.time()
        corpus=dataset['test']['article'][startN+d]
        temp=mainCreateSummaries(corpus)
        print("Document:",startN+d+1)
        print("News sentence number:",textSentenceCount(corpus))
        print("News: ",corpus)
        print("------------")
        tempHighlight=dataset['test']['highlights'][startN+d].replace('\n',' ')
        print("Highlight sentence number:",textSentenceCount(tempHighlight))
        print("Highlight: ",tempHighlight)
        print("------------")
        print("Summary sentence number:",textSentenceCount(temp))
        print("Summary: ", temp)
        elapsedTimeforDocument = time.time() - startTimeforDocument
        elapsedTimeforAll = time.time() - startTimeforOverall
        print('Document processing time: '+time.strftime("%M:%S", time.gmtime(elapsedTimeforDocument)))
        print('Total processing time: '+time.strftime("%d:%H:%M:%S", time.gmtime(elapsedTimeforAll)))
        print("###################################################")
        all_hypothesis.append(temp)
        all_references.append(tempHighlight)
        
        
    directoryDocument='data/CnnDailyDataset/Documents'
    directoryHighlights='data/CnnDailyDataset/Highlights' # references 
    directoryMySummaries='data/CnnDailyDataset/MySummaries' # hypothesis

    for p in range(N):
        temp=str(p+1)
        fileNameDocument='News'+temp+'.txt'
        fileNameHighlight='Highlight'+'.A.'+temp+'.txt' #model gold
        fileNameMySummary='MySummary.'+temp+'.txt' # system my 
        saveFile(directoryDocument,fileNameDocument,dataset['test']['article'][p])
        saveFile(directoryHighlights,fileNameHighlight,all_references[p])
        saveFile(directoryMySummaries,fileNameMySummary,all_hypothesis[p])
        print(p+1)

    print("Files are ready")


    rougeEvaluation(all_hypothesis, all_references)
