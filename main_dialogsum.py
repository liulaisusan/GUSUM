import nltk
from datasets import load_dataset
import torch
import os
import pickle

from summarizer import GusumSummarizer, HybridSummarizer, ModelSummarizer
from sentenceRanker import SentenceRanker
from evaluator import Evaluator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    dataname = 'train'
    dataset = load_dataset("knkarthick/dialogsum", split=dataname)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    interval = 2 # save evaluator every this many iterations, should be larger than batch_size
    start = 0
    end = 10
    path = "./evaluator"
    print_results = True


    # tokenizer = AutoTokenizer.from_pretrained("gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset")
    # model = AutoModelForSeq2SeqLM.from_pretrained("gauravkoradiya/T5-Finetuned-Summarization-DialogueDataset")

    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum").to(device)
    model.eval()
    print("Model configurations: ", model.generation_config)

    ranker = SentenceRanker()
    featurelist = ['sentencePosition', 'sentenceLength', 'properNoun', 'numericalToken']
    gusumSummarizer = GusumSummarizer(name='gusum', processed= False, ranker=ranker, featureList=featurelist)
    hybridSummarizer = HybridSummarizer(name ='hybrid', device = device, model = model, tokenizer = tokenizer, gusumSummarizer = gusumSummarizer)
    modelSummarizer = ModelSummarizer(name ='model', device = device, processed = False, model = model, tokenizer = tokenizer)
    summarizerList = [gusumSummarizer, hybridSummarizer, modelSummarizer]
    # summarizerList = [hybridSummarizer, modelSummarizer]

    saved_evaluator = False
    if os.path.exists(path) and len(os.listdir(path)) != 0:
        for fileName in os.listdir(path):
            datatype = fileName.split('_')[1] # train, test, val
            s = fileName.split('_')[2].split('.')[0]
            if dataname == datatype and s != 'all':
                start = int(s)
                saved_evaluator = True
                break

    if saved_evaluator:
        print(f"Loading evaluator from {fileName}, starting from {start}")
        with open(os.path.join(path, fileName), 'rb') as f:
            evaluator = pickle.load(f)
            # evaluator.evaluate(dataset, dataname, start = start, end = end, print_results=print_results)
            evaluator.batch_evaluate(dataset, dataname, start = start, end = end, print_results=print_results, batch_size=batch_size)
    else:
        evaluator = Evaluator(summarizerList, saveEvaulator=True, path = path, interval=interval)
        # evaluator.evaluate(dataset, dataname, start = start, end = end, print_results=print_results)
        evaluator.batch_evaluate(dataset, dataname, start = start, end = end, print_results=print_results,batch_size=batch_size)
