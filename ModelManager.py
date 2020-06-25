from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,NMF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from itertools import combinations
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import gensim
import nltk


class ModelManager(object):
    '''
    This class is reponsible to handle all model related process
    such as creating model for topic  nd classication modelling
    ''' 
    def __init__(self,doclist,randomSeed):
        self.__doclist=doclist
        self.__random=randomSeed
        self.__X_data_list=[]
        self.__docIDList=[]
        for aDoc in self.__doclist:
            self.__X_data_list.append(aDoc.getFinalizedContent())
            self.__docIDList.append(aDoc.getDocID())
    
    def tfIDFVectorizeContent(self,nGram_min,nGram_max):
        '''
        Return: A data frame of vectorize content

        Vectorization done using TF-IDFVectorizer 
        '''
        self.__tfidf_Vect = TfidfVectorizer(ngram_range=(nGram_min,nGram_max))
        self.__dtm_tfidf_vectorized_A = self.__tfidf_Vect.fit_transform(self.__X_data_list)
        self.__dtm_tfidf_arr = self.__dtm_tfidf_vectorized_A.toarray()
        self.__df_tfidf = pd.DataFrame(self.__dtm_tfidf_arr, columns=self.__tfidf_Vect .get_feature_names()).set_index(np.array(self.__docIDList))
        return self.__df_tfidf

    def getTFIDFVectorizer(self):
        return self.__tfidf_Vect 

    def getTFIDFVectorisedData(self):
        return self.__dtm_tfidf_vectorized_A
    
    def findBestParamAnd_NMFmodel(self,iterationList,topicsList,topKTerms):
        '''
        Return: A dataframe to show the results

        Will find the best NMF model when tuning hyperparam 'iterations' and 'number of topics'
        
        topKTerms: no of terms which has the highest weight to calculate coherence for each topic
        '''
        # prep data input suitable for word2vec model
        fullStr=''
        for aSentence in self.__X_data_list:
            fullStr=fullStr+" "+aSentence
        tokens=nltk.word_tokenize(fullStr.strip())#returns a list of indiv words
      
        # word2vex takes in a list of lists =>[tokens]
        w2v_model = gensim.models.Word2Vec([tokens], size=600, sg=1,min_count=1)

        iterList=[]
        compList=[]
        coherenceList=[]
        modelList=[]
        modelOutList=[]
        modelCompArrList=[]
        for iterNum in iterationList:
            for topics in range(topicsList[0],topicsList[1]+1):
                    nmf_model = NMF(init='nndsvd',n_components= topics , max_iter=iterNum,random_state=self.__random)
                    nmf_doc_topic_W = nmf_model.fit_transform(self.__dtm_tfidf_vectorized_A)
                    nmf_topic_term_H = nmf_model.components_
                    coherance = self.__calculate_overall_coherence(nmf_topic_term_H,w2v_model,topKTerms)
                    #save results
                    iterList.append(iterNum)
                    compList.append(topics)
                    modelList.append(nmf_model)
                    modelOutList.append(nmf_doc_topic_W)
                    modelCompArrList.append(nmf_topic_term_H)
                    coherenceList.append(coherance)

        resultDictFull={ 'Iteration':iterList,'Topic':compList,'Coherence':coherenceList,
                     'NMF_Model':modelList,'NMF_Model_Out':modelOutList,'TermsWeights':modelCompArrList}

        dfResultFull=pd.DataFrame(resultDictFull)

        bestIndex=dfResultFull.index[dfResultFull.Coherence == dfResultFull.Coherence.max()]
        bestParamModelList=dfResultFull.loc[bestIndex, :].values.tolist()#returns a list of list
        self.__bestNMF_Model=bestParamModelList[0][3]#save model
        self.__bestModelNMF_DocTopic_W=bestParamModelList[0][4]#save model doc topics weights array
        self.__bestModelNMF_TopicTerm_H=bestParamModelList[0][5]#save model topic term weights array


        return dfResultFull

    def getNMFmodel(self):
        return self.__bestNMF_Model
    
    def getNMFmodel_DocTopicArr(self):
        return self.__bestModelNMF_DocTopic_W

    def getNMFmodel_TopicTermArr(self):
        return self.__bestModelNMF_TopicTerm_H

    def __calculate_overall_coherence(self,topicTermArr,w2v_model, topKTerms):
        
        termList=self.__tfidf_Vect .get_feature_names()
        numOfRows=len(topicTermArr)
        topicsCoherence=[]

        for aTopicIndx in range(numOfRows):
            termWeightDict = self.mapTermToWeight(aTopicIndx,topicTermArr,termList)
            topKTermList,topKWeightList = self.topKtermsPerTopic(aTopicIndx,topicTermArr,topKTerms,termWeightDict)
            aTopicCoherence = self.__calculateTopicCoherance(topKTermList,w2v_model)
            topicsCoherence.append(aTopicCoherence)

        overallCoherance =sum(topicsCoherence)/numOfRows
        return overallCoherance

    def mapTermToWeight(self,aTopicIndx,topicTermArr,termList):
        termWeightDict={}
        # map term to weights for each topic
        for indexCol,aTerm in enumerate(termList):
            termWeightDict[aTerm]=topicTermArr[aTopicIndx][indexCol]
        return termWeightDict

    def topKtermsPerTopic(self,aTopicIndx,topicTermArr,topKTerms,termWeightDict):

        #get a row/array of weights for each topic
        aTopicArr = topicTermArr[aTopicIndx,:]
        #sort arr of weights into descending order
        aTopicArrSorted = np.sort(aTopicArr)[::-1]
        #Find K terms which has the highest weight value in a topic
        i=0
        topKTermList=[]
        topKWeightsList=[]
        for aWeight in aTopicArrSorted:
            if  i< topKTerms:
                aTerm=self.__get_keyFromValue_Dict(aWeight,termWeightDict)
                topKTermList.append(aTerm)
                topKWeightsList.append(aWeight)
            else:
                break
            i+=1
        return topKTermList,topKWeightsList

    def __get_keyFromValue_Dict(self,val,aDict): 
        for key, value in aDict.items(): 
            if val == value: 
                return key 
        return "key doesn't exist"
    
    def __calculateTopicCoherance(self,topKTermList,w2v_model):
        #calculate pair score of terms within each topic
        pairScores=[]
        for pairTerm in combinations(topKTermList,2):
            # find cosine similarity for each pair
            cosSim=w2v_model.wv.similarity(pairTerm[0], pairTerm[1])
            pairScores.append(cosSim)
            #Calculate the coherence score for each topic    
            aTopicCoherence = sum(pairScores)/len(pairScores)
        return aTopicCoherence
 
    def countVectorizeContent(self,nGram_min,nGram_max):
        '''
        Return: A data frame of vectorize content

        Vectorization done using CountVectorizer
        '''
        self.__countVect = CountVectorizer(ngram_range=(nGram_min,nGram_max))
        self.__dtmCV_vectorized = self.__countVect.fit_transform(self.__X_data_list)
        self.__dtmCV_arr = self.__dtmCV_vectorized.toarray()
        self.__df_count = pd.DataFrame(self.__dtmCV_arr, columns=self.__countVect .get_feature_names()).set_index(np.array(self.__docIDList))
        return self.__df_count

    def getCountVectorizer(self):
        return self.__countVect 

    def getCountVectorisedData(self):
        return self.__dtmCV_vectorized


    # Not using Gridsearch as the score returned is only Log_Likelihood
    # In addition, the best Loglikehood value returned by gridsearch is not the same
    # when taking the best model and apply the score function directly. There is discrepencies between the 2 values
    # Perplexity is also not returned using Gridsearch.
    def findBestParamAnd_LDAmodel(self,iterationList,topicList):
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
            for topics in range(topicList[0],topicList[1]):
                    lda_model = LatentDirichletAllocation(n_components= topics , max_iter=iterNum, learning_method='online',learning_offset=5.,random_state=self.__random,learning_decay=0.50)
                    lda_output = lda_model.fit_transform(self.__dtmCV_vectorized )  
                    log_likelihood = lda_model.score(self.__dtmCV_vectorized )
                    perplexity = lda_model.perplexity(self.__dtmCV_vectorized )

                    iterList.append(iterNum)
                    compList.append(topics)
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


    def getLDAmodel(self):
        return self.__bestLDA_Model


    def findDominantTopic(self,modelType):
        '''
        Return: A dataframe to show the results and A list of documeny objects

        Will find the domiant topic for each model

        modelType: nmf or lda
        '''
        if modelType.lower()=='lda':
            model=self.__bestLDA_Model
            modelOutput=self.__bestModelLDA_Output
        else:
            model=self.__bestNMF_Model
            modelOutput=self.__bestModelNMF_DocTopic_W
        
        topicnames = ["Topic_" + str(i) for i in range(model.n_components)]
        df_document_topic = pd.DataFrame(np.round(modelOutput, 3), columns=topicnames)
        # find dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        
        docnames = [aDoc.getDocID() for aDoc in self.__doclist]
        df_document_topic.insert(0,'Docmument_ID',docnames)
        df_document_topic.insert(1,'Dominant_Topic',dominant_topic)

        #Save dominant topic value into Doc obj
        for indx,value in enumerate(dominant_topic):
            self.__doclist[indx].setDominantTopic(value)
        
        return df_document_topic,self.__doclist

    def trainAndPedict_NaiveBayesClassificationModel(self,X_train, X_test, y_train,vectorizer):
        '''
        Return: an Array of predicted result
        '''
        X_train_cv = vectorizer.fit_transform(X_train) 
        print(X_train_cv.toarray().shape)
        X_test_cv  = vectorizer.transform(X_test)
        print(X_test_cv.toarray().shape)

        self.__mnb = MultinomialNB(alpha=1.0)
        # Train the model
        self.__mnb .fit(X_train_cv, y_train)
        y_pred_cv = self.__mnb .predict(X_test_cv)
        return y_pred_cv

    def saveVectorizerAndClassificationModel(self,outputFolder,vectType):
        '''
        Save the vectorizer and classifier model to output file

        vectType: CountVectorizer or TfidfVectorizer
        '''
        if vectType.lower()=='CountVectorizer':
            vetorizerName="CountVect_"
            vectorizer=self.__countVect

        else:
            vetorizerName="TfidfVect_"
            vectorizer=self.__tfidf_Vect

        dateTime=datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        classifyModelname="NaiveBayes_"
        extension='.pickle'

        pathVect=outputFolder+vetorizerName+dateTime+extension
        pathClassify=outputFolder+classifyModelname+dateTime+extension

        with open(pathVect ,'wb+') as out_file:
            pickle.dump(vectorizer, out_file)

        with open(pathClassify, 'wb+') as out_file:
            pickle.dump(self.__mnb, out_file)
 

        
        

    


    



        

