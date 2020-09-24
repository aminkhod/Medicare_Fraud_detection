
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import array
from sklearn.model_selection import KFold


# In[2]:


# data 
data = pd.read_csv('provider features and fraud labels.csv')
data.head()


# In[ ]:


def sim(x, datat1, data0, similarity='cosine'):
    


# In[3]:


# prepare cross validation
kfold = KFold(10, True, 1)
# enumerate splits
K_foldPredicts = []
for train, test in kfold.split(data['npi']):
    positiveData = data.iloc[train,2:][data['fraud_label']==1]
    negativeData = data.iloc[train,2:][data['fraud_label']==0]
    testLabel =  data.iloc[test,1]
    testData =  data.iloc[test,2:]
    print(negativeData.shape,positiveData.shape)
    predicList = []
    for i in range(len(testData.iloc[:,0])):
        predict = sim(testData.iloc[i,:],positiveData,negativeData,similarity='cosine')
        predicList.append(predict)
    K_foldPredicts.append(predicList)
    break
    
#     print('train: %s, test: %s' % (data.iloc[train,2:], data.iloc[test,2:]))

