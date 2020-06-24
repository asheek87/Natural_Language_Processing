from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
from datetime import datetime
import numpy as np
import pandas as pd


class ModelManager(object):
    '''
    This class is reponsible to handle all model related process
    such as creating model for topic  nd classication modelling
    ''' 
    def __init__(self,doclist,randomSeed):
        self.__doclist=doclist
        # self.__dfCcorpus=self.__convertFinalizedContentToDataFrame()
        self.__random=randomSeed
        self.__X_data_list=[]
        self.__docIDList=[]
        for aDoc in self.__doclist:
            self.__X_data_list.append(aDoc.getFinalizedContent())
            self.__docIDList.append(aDoc.getDocID())
    
    def countVectorizeContent(self,nGram_min,nGram_max):
        '''
        Return: A data frame of vectorize content

        Vectorization done using  CountVectorizer
        '''
        self.__countVect = CountVectorizer(ngram_range=(nGram_min,nGram_max))
        # self.__dtmCV_vectorized = self.__countVect .fit_transform(self.__dfCcorpus.Finalize_Content)
        self.__dtmCV_vectorized = self.__countVect .fit_transform(self.__X_data_list)
        self.__dtmCV_arr = self.__dtmCV_vectorized.toarray()
        # self.__df_count = pd.DataFrame(self.__dtmCV_arr, columns=self.__countVect .get_feature_names()).set_index(self.__dfCcorpus.Document_ID)
        self.__df_count = pd.DataFrame(self.__dtmCV_arr, columns=self.__countVect .get_feature_names()).set_index(np.array(self.__docIDList))
        return self.__df_count
    
    # Not using Gridsearch as the score returned is only Log_Likelihood
    # In addition, the best Loglikehood value returned by gridsearch is not the same
    # when taking the best model and apply the score function directly. There is discrepencies between the 2 values
    # Perplexity is also not returned using Gridsearch.
    def findBestParamAndLDAmodel(self,iterationList,topicList):
        '''
        Return: A dataframe to show the results

        Will find the best LDA model when tuning hyperparam 'iterations' and 'number of topics'
        '''
        
        iterList=[]
        compList=[]
        modelList=[]
        modelOutList=[]
        logLikeList=[]
        perplexList=[]

        for iterNum in iterationList:
            for topic in range(topicList[0],topicList[1]):
                    lda_model = LatentDirichletAllocation(n_components= topic , max_iter=iterNum, learning_method='online',learning_offset=5.,random_state=self.__random,learning_decay=0.50)
                    lda_output = lda_model.fit_transform(self.__dtmCV_vectorized )  
                    log_likelihood = lda_model.score(self.__dtmCV_vectorized )
                    perplexity = lda_model.perplexity(self.__dtmCV_vectorized )

                    iterList.append(iterNum)
                    compList.append(topic)
                    modelList.append(lda_model)
                    modelOutList.append(lda_output)
                    logLikeList.append(log_likelihood)
                    perplexList.append(perplexity)

        resultDictFull={ 'Iteration':iterList,'Topic':compList,'Log_Likelihood':logLikeList,
                     'Perplexity':perplexList,'LDA_Model':modelList,'LDA_Model_Out':modelOutList}

        dfResultFull=pd.DataFrame(resultDictFull)

        bestIndex=dfResultFull.index[dfResultFull.Perplexity == dfResultFull.Perplexity.min()]
        bestParamModelList=dfResultFull.loc[bestIndex, :].values.tolist()#returns a list of list
        self.__bestLDA_Model=bestParamModelList[0][4]#save model
        self.__bestModelLDA_Output=bestParamModelList[0][5]#save model output

        return dfResultFull
    
    def findDominantTopic_LDA(self):
        '''
        Return: A dataframe to show the results and A list of documeny objects

        Will find the domiant topic for each model
        '''
        
        topicnames = ["Topic_" + str(i) for i in range(self.__bestLDA_Model.n_components)]
        df_document_topic = pd.DataFrame(np.round(self.__bestModelLDA_Output, 3), columns=topicnames)
        # find dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        
        docnames = [aDoc.getDocID() for aDoc in self.__doclist]
        df_document_topic.insert(0,'Docmument_ID',docnames)
        df_document_topic.insert(1,'Dominant_Topic',dominant_topic)

        #Save dominant topic value into Doc obj
        for indx,value in enumerate(dominant_topic):
            self.__doclist[indx].setDominantTopic(value)
        
        return df_document_topic,self.__doclist

    def getLDAmodel(self):
        return self.__bestLDA_Model

    def getCountVectorizer(self):
        return self.__countVect 

    def getCountVectorisedData(self):
        return self.__dtmCV_vectorized

    def trainAndPedict_NaiveBayesClassificationModel(self,X_train, X_test, y_train ):
        '''
        Return: an Array of predicted result
        '''
        X_train_cv = self.__countVect.fit_transform(X_train) 
        print(X_train_cv.toarray().shape)
        X_test_cv  = self.__countVect.transform(X_test)
        print(X_test_cv.toarray().shape)

        self.__mnb = MultinomialNB(alpha=1.0)
        # Train the model
        self.__mnb .fit(X_train_cv, y_train)
        y_pred_cv = self.__mnb .predict(X_test_cv)
        return y_pred_cv

    def saveVectorizerAndClassificationModel(self,outputFolder):
        '''
        Save the vectorizer and classifier model to output file
        '''
        dateTime=datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        vetorizerName="CountVect_"
        classifyModelname="NaiveBayes_"
        extension='.pickle'

        pathVect=outputFolder+vetorizerName+dateTime+extension
        pathClassify=outputFolder+classifyModelname+dateTime+extension

        with open(pathVect ,'wb+') as out_file:
            pickle.dump(self.__countVect, out_file)

        with open(pathClassify, 'wb+') as out_file:
            pickle.dump(self.__mnb, out_file)
 

        
        

    


    



        

