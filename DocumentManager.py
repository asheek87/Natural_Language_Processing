import os
import cv2
import numpy as np
import pandas as pd
import shutil
from Document import Document
from nltk.corpus import stopwords

class DocumentManager(object):
    '''
    This class is reponsible to handle all general document processing steps
    ''' 
    __RootFolder='Data'
    __Orbituaries='Orbituaries'
    __slash='//'
    __folderPath=__RootFolder + __slash + __Orbituaries
    __outputFolder='output/'
    __encoding='utf8'

    def __init__(self):
        self.__doclist=[]
        self.__createDir()

        
    def __createDir(self):
       if os.path.exists(DocumentManager.__outputFolder):
           shutil.rmtree(DocumentManager.__outputFolder)
           os.makedirs(DocumentManager.__outputFolder,exist_ok=True)
       else:
           os.makedirs(DocumentManager.__outputFolder,exist_ok=True)

        
    def readContents(self):
        for root, dirs, files in os.walk(DocumentManager.__folderPath):
            for aFile in files:
                filePath=DocumentManager.__folderPath+DocumentManager.__slash+aFile
                with open (filePath, "r",encoding=DocumentManager.__encoding) as myfile:
                    data=myfile.readlines()
                    myDoc=Document(aFile,data)
                    self.__doclist.append(myDoc)
        print("Number of documents: "+str(len(self.__doclist)))

    def tokenizeData(self):
        for aDoc in self.__doclist:
            aDoc.contentToSentences()
            aDoc.sentencesToNgrams(3)
            aDoc.getSentences(DocumentManager.__outputFolder, DocumentManager.__encoding)

    def printDoc(self):
        # self.__doclist[1].printOriginalContents()
        # self.__doclist[0].getSentences()
        self.__doclist[0].sentences_to_ngrams(3)
    