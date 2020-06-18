import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
import pandas as pd
import re 
import string




class Document(object):
    '''
    This class is the data structure for storing document info 
    and preprocesses of an document
    '''
    __seperator='.'
    __encoding='utf8'

    def __init__(self,fileName,data):
        self.__saveDocID(fileName)
        self.__contentList=data
       
        

    def __saveDocID(self,file):
        fileName=file.split(Document.__seperator)
        self.__docID=fileName[0]

    def contentToSentences(self):
        strContent=''.join(self.__contentList)
        self.__sentences=sent_tokenize(strContent)
    
    def getSentences(self,outputPath,encode):
        '''
        Generate a file in txt format to 
        list out all sentences line by line
        '''
        outF = open(outputPath+self.__docID+"_reformat.txt", "w",encoding=encode)
        for line in self.__sentences:
            # write line to output file
            outF.write(line)
            outF.write("\n")
        outF.close()
        

    def sentencesToNgrams(self,nGram=None):
        # input: n gram value to tokenize the words in a sentence
        self.__sentToNgramList = []
        for s in self.__sentences:
            word_tokens = word_tokenize(s)
            ngram_list = list(ngrams(word_tokens,nGram))
            self.__sentToNgramList.append(ngram_list)

    def getNGramSentences(self):
        # return: a list that contains a list of tokens
        # print(self.__sentences)
        # print ("number of sentences ", len(self.__sentences))
        # print(self.__sentences[0])
        return self.__sentToNgramList

    def printOriginalContents(self):
        
        print(self.__contentList)

