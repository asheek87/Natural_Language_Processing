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
# The cleaned data, containing all words and its frequency in all documents,will also be generated in txt format
# at folder output/ALL_Words.xlsx. The excel is used for visual analysis to check if stopwords needs to be updated
# If there is a need, update input/MyStopWprds.txt and perform step 4 gain.

#Perform text pre-processing on corpus
start=t.time()
docMan.cleanDataInCorpus()
docMan.analyseFinalizedContent()
end=t.time()
print('Time taken for cleaning all corpus is {:.4f}'.format((end-start))+" seconds")
#%%
# Step 5: Prepare the document term matrix
# Create model manager obj
modMan=ModelManager(docMan.getDocuments(),DocumentManager.random)
# Count Vectorize content using unigram and create DTM
df_countVect=modMan.tfIDFVectorizeContent(1,1)
df_countVect.head()
# df_countVect.to_excel("output.xlsx")  
#%%
# This cell will use method findBestParamAnd_NMFmodel() to perform the following steps:
# Step 6: Prepare the topic model object
# Step 7: Determine a suitable range of topic numbers
# Step 8: Fit and transform the document term matrix into topic model object
# Step 9: Select on the "optimal" number of topics
iterList=[10,15]
topicListRange=[2,10]# no of topics
topKterms=5
start=t.time()
df_resultFull=modMan.findBestParamAnd_NMFmodel(iterList,topicListRange,topKterms)
end=t.time()
print('Time taken to find best LDA model is {:.4f}'.format((end-start))+" seconds")
#%%
# Step 10: Evaluate the topic and terms.
# Determine if there is a need to go back prior steps.
# Base on evaluation in step 10 below, the number of topics which has the highest coherence is 2 irregardless of the number of iterations set.

# Display overall result on No of topics vs coherence value
print('Display overall result on coherence value: ')
df_resultFull.iloc[:,0:3].T
#%%
# Display result with highest coherence value
print('Display overall result on No of topics vs coherence value: ')
df_resultFull[df_resultFull.Coherence == df_resultFull.Coherence.max()].iloc[:,0:3]
# %%
# Step 11: Label the original documents with the the most dominant topic.
# Base on evaluation in step 11 below, it can be assumed that there are 5 main topics in the Obituary data set
# Topic 0: Family
# Topic 1: Type of funeral service
# Display overall result on the Dominant topic per document
print('Display overall result on the Dominant topic per document: ')
df_topic_doc, updateDocList=modMan.findDominantTopic('nmf')
docMan.updateDocListWithLable(updateDocList)
# Generate a xlxs file to see complete info of dataframe: df_topic_doc
df_topic_doc.to_excel("output/Doc_Topic.xlsx")  
df_topic_doc
#%%
# Display overall top K terms for each topic for analysis
analyser=Analyser()
analyser.plot_top_term_weights(modMan,topKterms)
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
y_pred=modMan.trainAndPedict_NaiveBayesClassificationModel(X_train, X_test, y_train,modMan.getTFIDFVectorizer() )
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
modMan.saveVectorizerAndClassificationModel(DocumentManager.outputFolder,'TfidfVectorizer')
#%%