from nltk import sent_tokenize
import nltk
from nltk.corpus import stopwords
import time
from datasets import load_dataset
from tqdm import tqdm

from utils import load_dataset_from_huggingface, cleanDocument, saveFile, createSummary, mainCreateSummaries, processCorpus
from Evaluation import rougeEvaluation
from sentenceRanking import allCorpusSentenceRanking, textSentenceCount
from summarizer import GusumSummarizer, HybridSummarizer, ModelSummarizer
from sentenceRanker import SentenceRanker
from evaluator import Evaluator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



if __name__ == "__main__":
    nltk.download('punkt')

    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    dataset = load_dataset("knkarthick/dialogsum", split="train")
    # dataset = load_dataset("knkarthick/dialogsum", split="test")


    # tokenizer = AutoTokenizer.from_pretrained("gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset")
    # model = AutoModelForSeq2SeqLM.from_pretrained("gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset")

    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum")
    print("Model configurations: ", model.generation_config)

    ranker = SentenceRanker()
    featurelist = ['sentencePosition', 'sentenceLength', 'properNoun', 'numericalToken']
    gusumSummarizer = GusumSummarizer(name='gusum', processed= False, ranker=ranker, featureList=featurelist)
    hybridSummarizer = HybridSummarizer(name ='hybrid', model = model, tokenizer = tokenizer, gusumSummarizer = gusumSummarizer)
    modelSummarizer = ModelSummarizer(name ='model', processed = False, model = model, tokenizer = tokenizer)
    # summarizerList = [gusumSummarizer, hybridSummarizer, modelSummarizer]
    summarizerList = [hybridSummarizer, modelSummarizer]

    evaluator = Evaluator(summarizerList)
    evaluator.evaluate(dataset, 5, print_results=True)


    # startTimeforOverall = time.time()
    # all_hypothesis=[]
    # all_references=[]
    # all_model_summaries=[]
    # all_hybrid_summaries=[]

    
    # #documentNumber
    # N=1  ## How many documents do you want summarized over the CNN/Daily Mail dataset? Limit 11490
    # startN=0

    # for d in tqdm(range(N)):
    #     if d % 100 == 0:
    #         print(f"Processing {d}th document")
    #     startTimeforDocument = time.time()
    #     corpus=dataset['dialogue'][startN+d]
    #     ranker = SentenceRanker()
    #     featurelist = ['sentencePosition', 'sentenceLength', 'properNoun', 'numericalToken']
    #     gusumSummarizer = GusumSummarizer(name='gusum', processed= False, ranker=ranker, featureList=featurelist)
    #     temp = gusumSummarizer.summarize(corpus)
    #     # temp=mainCreateSummaries(corpus)
    #     # print("Document:",startN+d+1)
    #     # print("Dialog sentence number:",textSentenceCount(corpus))
    #     # print("Dialog: \n",corpus)
    #     # print("------------")
    #     # tempHighlight=dataset['summary'][startN+d].replace('\n',' ')
    #     tempHighlight=dataset['summary'][startN+d]

    #     # print("Highlight sentence number:",textSentenceCount(tempHighlight))
    #     # print("Highlight: ",tempHighlight)
    #     # print("------------")
    #     # print("GUSUM sentence number:",textSentenceCount(temp))
    #     # print("GUSUM: ", temp)
    #     # print("------------")

    #     hybridSummarizer = HybridSummarizer(name ='hybrid', model = model, tokenizer = tokenizer, gusumSummarizer = gusumSummarizer)
    #     hybrid_sum = hybridSummarizer.summarize(corpus, temp)
    #     # inputs = tokenizer(temp, return_tensors="pt").input_ids  # Batch size 1
    #     # outputs = model.generate(inputs,max_new_tokens=100)
    #     # hybrid_sum = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #     # print("Hybrid sum sentence number:",textSentenceCount(hybrid_sum))
    #     # print("Hybrid sum: ", hybrid_sum)

    #     modelSummarizer = ModelSummarizer(name ='model', processed = False, model = model, tokenizer = tokenizer)
    #     model_sum = modelSummarizer.summarize(corpus)
    #     # corpus_model = processCorpus(corpus)
    #     # # corpus_model = corpus
    #     # inputs_model = tokenizer(corpus_model, return_tensors="pt").input_ids  # Batch size 1
    #     # outputs_model = model.generate(inputs_model, max_new_tokens=100)
    #     # model_sum = tokenizer.decode(outputs_model[0], skip_special_tokens=True)

    #     # print("Model sum sentence number:",textSentenceCount(model_sum))
    #     # print("Model sum: ", model_sum)
    #     elapsedTimeforDocument = time.time() - startTimeforDocument
    #     elapsedTimeforAll = time.time() - startTimeforOverall
    #     # print('Document processing time: '+time.strftime("%M:%S", time.gmtime(elapsedTimeforDocument)))
    #     # print('Total processing time: '+time.strftime("%d:%H:%M:%S", time.gmtime(elapsedTimeforAll)))
    #     # print("###################################################")
    #     all_hypothesis.append(temp)
    #     all_references.append(tempHighlight)
    #     all_hybrid_summaries.append(hybrid_sum)
    #     all_model_summaries.append(model_sum)
        
        
    # directoryDocument='data/dialogsum/documents'
    # directoryHighlights='data/dialogsum/highlights' # references 
    # directoryMySummaries='data/dialogsum/summaries' # hypothesis

    # # for p in range(N):
    # #     temp=str(p+1)
    # #     fileNameDocument='dialog'+temp+'.txt'
    # #     fileNameHighlight='highlight'+'.A.'+temp+'.txt' #model gold
    # #     fileNameMySummary='summary.'+temp+'.txt' # system my 
    # #     saveFile(directoryDocument,fileNameDocument,dataset['dialogue'][p])
    # #     saveFile(directoryHighlights,fileNameHighlight,all_references[p])
    # #     saveFile(directoryMySummaries,fileNameMySummary,all_hypothesis[p])
    # #     print(p+1)

    # # print("Files are ready")


    # rougeEvaluation(all_hypothesis, all_references)
    # rougeEvaluation(all_hybrid_summaries, all_references)
    # rougeEvaluation(all_model_summaries, all_references)
    # rougeEvaluation(all_hybrid_summaries, all_model_summaries)
