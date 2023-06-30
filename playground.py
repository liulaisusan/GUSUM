import nltk
import time
from datasets import load_dataset
import torch
import os
import pickle
from tqdm import tqdm

from utils import processCorpus
from summarizer import GusumSummarizer, HybridSummarizer, ModelSummarizer
from sentenceRanker import SentenceRanker
from evaluator import Evaluator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def combine_evaluator_results():
    path = "./evaluator"
    aggregators = ['Avg']
    test_path = os.path.join(path, "evaluator_test_all.pkl")
    train_path = os.path.join(path, "evaluator_train_all.pkl")
    test_evaluator = torch.load(test_path,map_location='cpu')
    train_evaluator = torch.load(train_path,map_location='cpu')
    train_evaluator.all_goldensum = train_evaluator.all_goldensum + test_evaluator.all_goldensum
    train_evaluator.all_gusum = train_evaluator.all_gusum + test_evaluator.all_gusum
    train_evaluator.all_hybridsum = train_evaluator.all_hybridsum + test_evaluator.all_hybridsum
    train_evaluator.all_modelsum = train_evaluator.all_modelsum + test_evaluator.all_modelsum
    train_evaluator._allRougeEvaluation(aggregators)

if __name__ == "__main__":
    # nltk.download('punkt')

    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')
    # dataset = load_dataset("knkarthick/dialogsum", split="train")
    # dataname = 'train'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
    # model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum").to(device)
    # # model.eval()
    # dataset = dataset['dialogue'][:200]
    # batch_size = 10
    # for start_idx in tqdm(range(0, len(dataset), batch_size)):
    #     corpus = dataset[start_idx:start_idx+batch_size]
    #     inputs =tokenizer(corpus, return_tensors="pt", truncation = True, padding=True).to(device).input_ids  # Batch size 1
    #     outputs = model.generate(inputs, max_new_tokens=100)
    #     summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #     print(summary)
    combine_evaluator_results()