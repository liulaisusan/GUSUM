import nltk

from nltk import sent_tokenize, word_tokenize,PorterStemmer

class SentenceRanker:
    def __init__(self) -> None:
        self.features = {
            "sentencePosition": self.sentencePosition,
            "sentenceLength": self.sentenceLength,
            "properNoun": self.properNoun,
            "numericalToken": self.numericalToken
        }
    
    def rank(self,tokenizedCorpus,corpus,featureList):
        sentenceRankList=[]
        for i, sentence in enumerate(tokenizedCorpus):
            value=self.sentenceRanking(sentence,i,corpus,featureList)
            value=round(value,5)
            sentenceRankList.append(value)
        return sentenceRankList
    
    def sentenceRanking(self,sentence,location,corpus, featureList):
        value = 0
        for feature in featureList:
            if "sentencePosition" == feature:
                value = value + self.features[feature](sentence,location,corpus)
            else:
                value = value + self.features[feature](sentence,corpus)
        return value
    
    def textWordCount(self,Text):
        number_of_words = word_tokenize(Text)
        count=(len(number_of_words))
        return count

    def textSentenceCount(self,Text):
        number_of_sentences = sent_tokenize(Text)
        count=(len(number_of_sentences))
        return count

    def longestSentenceLenght(self,Text):
        text=sent_tokenize(Text)
        lens = [self.textWordCount(x) for x in text]
        return max(lens)

    def sentencePosition(self,sentence,location,corpus): 
        N=self.textSentenceCount(corpus)
        if location+1 == N:
            return 1.0
        elif location==0:
            return 1.0
        else:
            value=(N-location)/N
            return value
        
    def sentenceLength(self,sentence,corpus):
        return self.textWordCount(sentence)/self.longestSentenceLenght(corpus)


    def properNoun(self,sentence,corpus):
        text = nltk.word_tokenize(sentence)
        tagged=nltk.pos_tag(text)
        noProperNoun= tagged.count('NNP')
        return noProperNoun/len(text)

    def numericalToken(self,sentence,corpus):
        text = nltk.word_tokenize(sentence)
        tagged=nltk.pos_tag(text)
        noNumericalToken= tagged.count('CD')
        return 1-(noNumericalToken/len(text))