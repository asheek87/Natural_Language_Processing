import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re 
import string

class Document(object):
    '''
    This class is the data structure for storing document info 
    and preprocesses of an document
    '''
    __seperator='.'

    def __init__(self,fileName,dataList):
        self.__saveDocID(fileName)
        self.__strContent=''.join(dataList).lower()
        self.__lemmatizer = WordNetLemmatizer()

    def getContent(self):
         return self.__strContent

    def __saveDocID(self,file):
        fileName=file.split(Document.__seperator)
        self.__docID=fileName[0]

    def getDocID(self):
        return self.__docID

    def contentToSentences(self,contentList):
        content=''.join(contentList)
        self.__sentences=sent_tokenize(content)
        return self.__sentences
    
    def removePunctuation(self,sentences):
        self.__noPuncContent=[]
        punctuation_tokenizer = RegexpTokenizer(r"\w+")
        for aSentence in sentences:
            self.__noPuncContent.append(punctuation_tokenizer.tokenize(aSentence))
        return self.__noPuncContent
    
    def removeStopWords(self,sentences,stopwordsList):
        self.__noStopWordsContent=[]
        for aSentence in sentences:
            sentence=[]
            for aWord in aSentence:
                if aWord not in stopwordsList:
                    sentence.append(aWord)
            self.__noStopWordsContent.append(sentence)       
        return self.__noStopWordsContent
    
    def lemmatizeContent(self,sentences):
        self.__lemmatizeContent=[]
        for aSentence in sentences:
            sentence=[]
            for aWord in aSentence:
                newWord=self.__lemmatizer .lemmatize(aWord) 
                sentence.append(newWord)
            self.__lemmatizeContent.append(sentence)
        return self.__lemmatizeContent

    def removeUnnecessaryWordsAndDigits(self,sentences):
        '''
        Return a list of sentences 
        
        Sentences have 
        1. no digits,
        2. words which are at least 4 characters long
        3. no empty sentences
        '''
        pattern1 = re.compile('\d+')
        self.__necessaryContent=[]
        for aSentence in sentences:
            if len(aSentence) !=0:
                sentence=[]
                strSent=' '.join(aSentence)
                noDigits = re.sub(pattern1,"", strSent)
                for aWord in noDigits.split(" "):
                    length=len(aWord)
                    if length > 2:
                        sentence.append(aWord)
                self.__necessaryContent.append(sentence)

        return self.__necessaryContent
    
    def setFinalizedContent(self,sentences):
        for aSentence in sentences:
            self.__preProcessedContent=" ".join(aSentence)
        self.__preProcessedContent=self.__preProcessedContent.strip()
        
    def getFinalizedContent(self):
        
        return self.__preProcessedContent
        
    def generateReformattedDocument(self,outputPath,encode,sentences):
        '''
        Return: None

        Generate a file in txt format to 
        list out all reformatted data line by line
        '''
        outF = open(outputPath+self.__docID+"_reformat.txt", "w",encoding=encode)
        for aSentence in sentences:
            aStr=" ".join(aSentence).strip()
            outF.write(aStr)
            outF.write("\n")
        outF.close()  

    def setDominantTopic(self,topic):
        self.__topic=topic

    def getDocumentLabel(self):
        '''
        Return: Document label 
        '''
        return self.__topic



