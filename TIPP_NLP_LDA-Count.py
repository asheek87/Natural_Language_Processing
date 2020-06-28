#%%
# Step 1: Decide on the dataset of choice
# Selecting the Orbituaries data set
#%%
from DocumentManager import DocumentManager
from ModelManager import ModelManager
from Analyser import Analyser
import time as t
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#%%
# Step 2: Prepare to load the dataset into the notebook
# Document manager obj will read in data from Orbituaries folder.
# Read non empty corpus

# Create document manager obj
docMan=DocumentManager()

# Read non empty corpus
start=t.time()
docMan.readContents()
end=t.time()
print('Time taken for reading all corpus is {:.4f}'.format((end-start))+" seconds")
#%%
# Step 3: Perform an inititate exploratory data analysis on the original dataset
# Since there are over 300 corpusus. Perform a selective visual inspection of data in txt. 
#%%
# Step 4: Clean and prepare the data
# Perform text pre-processing on corpus
# The cleaned data, contyaining all words in all documents,will also be generated in txt format
# at folder output/ALL_Words.txt. The txt is used for visual alaysis to check if stopwords needs to be updated
# If there is a need, perform step 4 gain.
start=t.time()
docMan.cleanDataInCorpus()
docMan.analyseFinalizedContent()
end=t.time()
print('Time taken for cleaning all corpus is {:.4f}'.format((end-start))+" seconds")
#%%
# Step 5: Prepare the document term matrix
# Create model manager obj
modMan=ModelManager(docMan.getDocuments(),DocumentManager.random)
# Count Vectorize content using unigram
df_countVect=modMan.countVectorizeContent(1,1)
df_countVect.head()
#%%
# Step 6 to 9
# This cell will use method findBestParamAnd_NMFmodel() to perform the following steps:
# Step 6: Prepare the topic model object
# Step 7: Determine a suitable range of topic numbers
# Step 8: Fit and transform the document term matrix into topic model object
# Step 9: Select on the "optimal" number of topics
iterList=[10,15]
componentListRange=[10,20]
start=t.time()
df_resultFull=modMan.findBestParamAnd_LDAmodel(iterList,componentListRange)
end=t.time()
print('Time taken to find best LDA model is {:.4f}'.format((end-start))+" seconds")
#%%
# Step 10: Evaluate the topic and terms. Determine if there is a need to go back prior steps.

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
df_topic_doc, updateDocList=modMan.findDominantTopic('lda')
docMan.updateDocListWithLable(updateDocList)
df_topic_doc
#%%
# Step 11: Label the original documents with the the most dominant topic.
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
# Step 12: Split the original documents into test and train sets
# Get the data and train the Classification model
X_train, X_test, y_train, y_test = docMan.getTrainAndTestData(True)
#%%
# Step 13: Carry out classification modelling
y_pred=modMan.trainAndPedict_NaiveBayesClassificationModel(X_train, X_test, y_train,modMan.getCountVectorizer() )
#%% 
# Step 14: Evaluate the results. Determine if there is a need to return to prior steps.
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
modMan.saveVectorizerAndClassificationModel(DocumentManager.outputFolder,"CountVectorizer")
#%%