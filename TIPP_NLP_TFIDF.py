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
df_countVect=modMan.tfIDFVectorizeContent(1,1)
df_countVect.head()
# df_countVect.to_excel("output.xlsx")  
#%%
iterList=[10,15]
componentListRange=[5,10]
topKterms=10

start=t.time()
df_resultFull=modMan.findBestParamAnd_NMFmodel(iterList,componentListRange,topKterms)
end=t.time()
print('Time taken to find best LDA model is {:.4f}'.format((end-start))+" seconds")
#%%
# Display overall result on coherencevalue
print('Display overall result on coherence value: ')
df_resultFull.iloc[:,0:3].T
#%%
# Display result with highest coherence value
print('Display result with highest coherence value: ')
df_resultFull[df_resultFull.Coherence == df_resultFull.Coherence.max()].iloc[:,0:3]


# %%
# Display overall result on the Dominant topic per document
print('Display overall result on the Dominant topic per document: ')
df_topic_doc, updateDocList=modMan.findDominantTopic('nmf')
docMan.updateDocListWithLable(updateDocList)
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
# Get the data and train the Classification model
X_train, X_test, y_train, y_test = docMan.getTrainAndTestData(True)
y_pred=modMan.trainAndPedict_NaiveBayesClassificationModel(X_train, X_test, y_train,modMan.getTFIDFVectorizer() )
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
modMan.saveVectorizerAndClassificationModel(DocumentManager.outputFolder,'TfidfVectorizer')
#%%