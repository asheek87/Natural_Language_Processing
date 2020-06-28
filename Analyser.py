'''
Acknowledgement:
Some codes are from 
1) Derek Greene, University College Dubin
   Original source: https://github.com/derekgreene/topic-model-tutorial

2) Tan Poh Keam,Repulbic Polytechnic 
   Source TIPP lab 01 to 08 
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.sklearn
import pandas as pd
import numpy as np
class Analyser():
    '''
    This class handles data plotting and visualizing 
    '''
    def __init__(self):
        self.__color='OrRd'

    def displayAndSummarizeTopics_LDA(self,countVect,model,no_top_words=10):
        '''
        Return: None

        Display the top K words in a topic
        '''
        feature_names = countVect.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print ("Topic %d:" % (topic_idx))
            print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def plt_pyLDAvis(self,model,vectData,countVect):
        pyLDAvis.enable_notebook()
        panel = pyLDAvis.sklearn.prepare(model , vectData, countVect)
        return panel

    def distplotlabel(self,targetLabel,xLabelName):
        ax=sns.distplot(targetLabel,axlabel=xLabelName)

    def plotConfusionMatrix(self,arr,noOfLable):
        df_cm = pd.DataFrame(arr, range(noOfLable), range(noOfLable))
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap=self.__color) # font size
    
    def plot_top_term_weights(self, modelManager,topKTerms ):
        termList=modelManager.getTFIDFVectorizer().get_feature_names()
        topicTermArr=modelManager.getNMFmodel_TopicTermArr()
        numOfRows=len(topicTermArr)
        
        for aTopicIndx in range(numOfRows):
            termWeightDict = modelManager.mapTermToWeight(aTopicIndx,topicTermArr,termList)
            topKTermList,topKWeightList = modelManager.topKtermsPerTopic(aTopicIndx,topicTermArr,topKTerms,termWeightDict)

            # create the plot
            fig = plt.figure(figsize=(13,8))
            # add the horizontal bar chart
            ypos = np.arange(topKTerms)
            ax = plt.barh(ypos, topKWeightList, align="center", color="green",tick_label=topKTermList)
            plt.xlabel("Term Weight",fontsize=14)
            plt.title("Topic: "+str(aTopicIndx))
            plt.tight_layout()
            plt.show()

