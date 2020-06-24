import os
import cv2
import numpy as np
import pandas as pd
import shutil
from Document import Document
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

class DocumentManager(object):
    '''
    This class is reponsible to handle all general document pre processing steps
    ''' 
    __RootFolder='Data'
    __Orbituaries='Orbituaries'
    __slash='//'
    __folderPath=__RootFolder + __slash + __Orbituaries
    outputFolder='output/'
    __inputFolder='input'
    __inFolderPath=__inputFolder + __slash
    __encoding='utf8'
    random=123

    def __init__(self):
        self.__doclist=[]
        self.__customizedStopWords=[]
        self.__createDir()
        self.__nltk_stop_words = stopwords.words('english')
        self.__customizeOwnStopWords()
        self.__generate_StopWords()
        self.__X_data_list=[]
        self.__y_data_list=[]

    def __generate_StopWords(self):
        '''
        Generate a file in txt format to 
        list out all stops words for reference
        '''
        outF = open(DocumentManager.outputFolder+"StopWords.txt", "w",encoding=DocumentManager.__encoding)
        for line in self.__customizedStopWords :
            # write line to output file
            outF.write(line.strip())
            outF.write("\n")
        outF.close()
    
    def __customizeOwnStopWords(self):
        '''
        Add new stop words on top of NLTK stop words
        '''
        for root, dirs, files in os.walk(DocumentManager.__inFolderPath):
            for aFile in files:
                filePath=DocumentManager.__inFolderPath+DocumentManager.__slash+aFile
                with open (filePath, "r",encoding=DocumentManager.__encoding) as myfile:
                    aList=myfile.readlines()
        
        aList.extend(self.__nltk_stop_words )
        # remove amy leading and trailing whitespace for each item in list
        self.__customizedStopWords=list(map(str.strip, aList))

    def __createDir(self):
        '''
        Return: None
        
        Create output directory 
        '''
        if os.path.exists(DocumentManager.outputFolder):
            shutil.rmtree(DocumentManager.outputFolder)
            os.makedirs(DocumentManager.outputFolder,exist_ok=True)
        else:
            os.makedirs(DocumentManager.outputFolder,exist_ok=True)

    def getDocuments(self):
        '''
        Return: a list of Document object
        '''
        return self.__doclist


    def readContents(self):
        '''
        Return: None

        Read non empty txt files.
        '''
        for root, dirs, files in os.walk(DocumentManager.__folderPath):
            for aFile in files:
                filePath=DocumentManager.__folderPath+DocumentManager.__slash+aFile
                with open (filePath, "r",encoding=DocumentManager.__encoding) as myfile:
                    data=myfile.readlines()
                    if len(data) != 0:
                        myDoc=Document(aFile,data)
                        self.__doclist.append(myDoc)
        print("Number of documents: "+str(len(self.__doclist)))

    def cleanDataInCorpus(self):
        '''
        Return: Output files in txt format chich shows n-gram words at folder /output/
        
        Will perform the following pre processing steps on all corpus:
        1. Format corpus to sentences
        2. Remove punctuation
        3. Remove stopwords
        4. Lemmatize word
        6. Remove unnecesssary words and number
        '''
        strDesc='''
        Will perform the following pre processing steps on all corpus:
        1. Format corpus to sentences
        2. Remove punctuation
        3. Remove stopwords
        4. Lemmatize word
        6. Remove unnecesssary words and number
        '''
        print(strDesc)
        print("Pre Processing....")
        for indx,aDoc in enumerate(self.__doclist):
            content=aDoc.getContent()
            sentences=aDoc.contentToSentences(content)           
            noPuncContent=aDoc.removePunctuation(sentences)
            noStopWordContent=aDoc.removeStopWords(noPuncContent,self.__customizedStopWords)
            lemmatizeContent=aDoc.lemmatizeContent(noStopWordContent)
            cleanContent=aDoc.removeUnnecessaryWordsAndDigits(lemmatizeContent)
            aDoc.setFinalizedContent(cleanContent)
            aDoc.generateReformattedDocument(DocumentManager.outputFolder,DocumentManager.__encoding,cleanContent)
        print("Pre Processing DONE \n") 

    def updateDocListWithLable(self,docList):
        self.__doclist=docList
        for aDoc in self.__doclist:
            self.__X_data_list.append(aDoc.getFinalizedContent())
            self.__y_data_list.append(aDoc.getDocumentLabel())

    def getDataLables(self):
        return self.__y_data_list
    
    def getNoOfUniqueLabel(self):
        numlist=[]
        for aDoc in self.__doclist:
            if aDoc.getDocumentLabel() not in numlist:
                numlist.append(aDoc.getDocumentLabel())
        return len(numlist)



    def getTrainAndTestData(self, stratify=False):
        X_train, X_test, y_train, y_test=[],[],[],[]
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(self.__X_data_list, self.__y_data_list, stratify=self.__y_data_list,
                                                           test_size=0.3, random_state=DocumentManager.random, shuffle= True,)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.__X_data_list, self.__y_data_list, test_size=0.3, random_state=DocumentManager.random)
                                                           
        return  X_train, X_test, y_train, y_test
    
    
    def analyseFinalizedContent(self):
        strContent=" "
        for aDoc in self.__doclist:
            strContent=strContent+aDoc.getFinalizedContent()+" "
        wordslist=[]
        for aWords in strContent.split():
            if aWords not in wordslist:
                wordslist.append(aWords)
        wordslist.sort()
        outF = open("output/"+"ALL_Words.txt", "w",encoding=DocumentManager.__encoding)
        for aWord in wordslist:
            outF.write(aWord)
            outF.write("\n")
        outF.close() 

        

     



    