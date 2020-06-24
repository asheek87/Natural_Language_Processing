#%%
from DocumentManager import DocumentManager
from ModelManager import ModelManager
from Analyser import Analyser
import time as t
import pandas as pd


#%%
#Create document manager obj
docMan=DocumentManager()
#%%
# Read non empty corpus
start=t.time()
docMan.readContents()
end=t.time()
print('Time taken for reading all corpus is {:.4f}'.format((end-start))+" seconds")
#%%
#Perform text pre-processing on corpus
start=t.time()
docMan.cleanDataInCorpus()
docMan.analyseFinalizedContent()
end=t.time()
print('Time taken for cleaning all corpus is {:.4f}'.format((end-start))+" seconds")
#%%
#Create model manager obj
modMan=ModelManager(docMan.getDocuments(),DocumentManager.random)
#%%
# Count Vectorize content using unigram
df_countVect=modMan.countVectorizeContent(1,1)
df_countVect.head()
#%%
iterList=[10,15]
componentListRange=[10,20]
start=t.time()
df_resultFull=modMan.findBestParamAndLDAmodel(iterList,componentListRange)
end=t.time()
print('Time taken to find best LDA model is {:.4f}'.format((end-start))+" seconds")
#%%
# Display overall result on Log likelihood and Perplexity value
print('Display overall result on Log likelihood and Perplexity value: ')
df_resultFull.iloc[:,0:4]
#%%
# Display result with highest Log likelihood value
print('Display result with highest Log likelihood value: ')
df_resultFull[df_resultFull.Log_Likelihood == df_resultFull.Log_Likelihood.max()].iloc[:,0:4]

#%%
# Display result with lowest perplexity value
print('Display result with lowest perplexity value: ')
df_resultFull[df_resultFull.Perplexity == df_resultFull.Perplexity.min()].iloc[:,0:4] 

# %%
# Display overall result on the Dominant topic per document
print('Display overall result on the Dominant topic per document: ')
df_topic_doc, updateDocList=modMan.findDominantTopic_LDA()
docMan.updateDocListWithLable(updateDocList)
df_topic_doc
#%%
# Display overall top K terms for each topic for analysis
analyser=Analyser()
analyser.displayAndSummarizeTopics_LDA(modMan.getCountVectorizer(),modMan.getLDAmodel(),20)
#%%
# Display overall result on the topics and terms using pyLDAvis for analysis
panel=analyser.plt_pyLDAvis(modMan.getLDAmodel(),modMan.getCountVectorisedData(),modMan.getCountVectorizer())
panel

#%%
# Analyse the distibution of the target lable. 
print('Lable is not evenly distributed,so have to stratify preparing training data model classification')
analyser.distplotlabel(docMan.getDataLables(),"Dominant_Topic")

#%%
# Get the data and train the Classification model
X_train, X_test, y_train, y_test = docMan.getTrainAndTestData(True)
y_pred=modMan.trainAndPedict_NaiveBayesClassificationModel(X_train, X_test, y_train )
#%% 
# Evaluate model
from sklearn import metrics
cmArr=metrics.confusion_matrix(y_test, y_pred)
numOfUniqueLables=docMan.getNoOfUniqueLabel()
analyser.plotConfusionMatrix(cmArr,numOfUniqueLables)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted'))
print(metrics.classification_report(y_test, y_pred))

#%%
modMan.saveVectorizerAndClassificationModel(DocumentManager.outputFolder)
#%%