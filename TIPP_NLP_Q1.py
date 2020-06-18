#%%
from DocumentManager import DocumentManager
import time as t
import pandas as pd

#%%
docMan=DocumentManager()
docMan.readContents()
docMan.tokenizeData()

#%%