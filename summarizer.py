import re
from nltk import sent_tokenize

from graph import createGraph,findHighestSimilarityRank

class Summarizer:
    def __init__(self, name) -> None:
        self.name = name

    def cleanDocument(self, document):
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
    
    def seperateSentences(self, corpus):
        sentences =  re.split(r'#Person1#|#Person2#' , corpus)
        sentences = list(filter(None, sentences))
        sentences = [ '#Person1#' + sentence if i %2 == 0 else '#Person2#' + sentence for i, sentence in enumerate(sentences)]
        return sentences

class GusumSummarizer(Summarizer):
    def __init__(self, name, processed, ranker, featureList, maxSentence = 10) -> None:
        super().__init__(name)
        self.processed = processed
        self.ranker = ranker
        self.featureList = featureList
        self.maxSentence = maxSentence
    
    def summarize(self, corpus, sentences = None):
        if self.processed and not sentences:
            raise Exception("Corpus should be processed but separated sentences are not provided")
        if not self.processed:
            corpus=self.cleanDocument(corpus) # Clean Paranthesis
            sentences= self.seperateSentences(corpus)
        initialRank = self.ranker.rank(sentences, corpus, self.featureList)
        # initialRank=allCorpusSentenceRanking(sentences,corpus)

        similarityMatrix=createGraph(sentences) # create matrix ( Graph) shows similarities of sentences

        newRank=findHighestSimilarityRank(similarityMatrix, initialRank)
        sentenceNumberInSummary=min(self.maxSentence, len(sentences))
        # sentenceNumberInSummary=int(0.9*len(sentences))

        summary=self._createSummary(sentences, newRank, sentenceNumberInSummary)
        return summary
    
    def batch_summarize(self, batch_corpus, sentences = None):
        summaries = []
        for corpus in batch_corpus:
            summaries.append(self.summarize(corpus, sentences))
        return summaries

    def _createSummary(self, sentences, sentencesRank, sentenceAmount):
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
    
class HybridSummarizer(Summarizer):
    def __init__(self, name, device, model, tokenizer, gusumSummarizer = None):
        super().__init__(name)
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.gusumSummarizer = gusumSummarizer
    
    def summarize(self, corpus, gusumSummary = None, sentences = None):
        if not self.gusumSummarizer and not gusumSummary:
            raise Exception("Neither Gusum Summarizer or Summary is not provided")
        if not gusumSummary:
            # print(" Gusum Summary is not provides, creating one.")
            gusumSummary = self.gusumSummarizer.summarize(corpus, sentences)
        inputs = self.tokenizer(gusumSummary, return_tensors="pt", truncation = True).to(self.device).input_ids  # Batch size 1
        outputs = self.model.generate(inputs, max_new_tokens=100)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(self, batch_corpus, gusumSummaries = None, sentences = None):
        if not self.gusumSummarizer and not gusumSummaries:
            raise Exception("Neither Gusum Summarizer or Summary is not provided")
        if not gusumSummaries:
            # print(" Gusum Summary is not provides, creating one.")
            gusumSummaries = self.gusumSummarizer.batch_summarize(batch_corpus, sentences)
        inputs =self.tokenizer(gusumSummaries, return_tensors="pt", truncation = True, padding=True).to(self.device).input_ids
        outputs = self.model.generate(inputs, max_new_tokens=100)
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return summaries
    
class ModelSummarizer(Summarizer):
    def __init__(self, name, device, processed, model, tokenizer):
        super().__init__(name)
        self.device = device
        self.processed = processed
        self.model = model
        self.tokenizer = tokenizer
    
    def summarize(self, corpus):
        if not self.processed:
            corpus = self.processCorpus(corpus)
        inputs = self.tokenizer(corpus, return_tensors="pt", truncation = True).to(self.device).input_ids  # Batch size 1
        outputs = self.model.generate(inputs, max_new_tokens=100)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(self, batch_corpus):
        if not self.processed:
            batch_corpus = [self.processCorpus(corpus) for corpus in batch_corpus]
        inputs =self.tokenizer(batch_corpus, return_tensors="pt", truncation = True, padding=True).to(self.device).input_ids
        outputs = self.model.generate(inputs, max_new_tokens=100)
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return summaries
    
    def processCorpus(self, corpus):
        corpus = self.cleanDocument(corpus) # Clean Paranthesis
        sentences = self.seperateSentences(corpus)
        new_corpus = ""
        for sentence in sentences:
                new_corpus=new_corpus + '\n ' + sentence
        return new_corpus
            
        
    