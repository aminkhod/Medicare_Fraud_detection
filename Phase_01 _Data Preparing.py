
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter


# In[2]:


data = pd.read_csv('../Medicare_Provider_Util_Payment_PUF_CY2017.csv', sep='\t')
print(data.shape)
data.head()


# In[3]:


data.columns


# In[4]:


data.drop(['nppes_provider_last_org_name', 'nppes_provider_first_name', 'nppes_provider_mi', 'nppes_credentials',          'nppes_entity_code', 'nppes_provider_street1', 'nppes_provider_street2', 'nppes_provider_city',          'nppes_provider_zip', 'nppes_provider_state', 'nppes_provider_country', 'medicare_participation_indicator',           'place_of_service','hcpcs_drug_indicator', 'hcpcs_code', 'hcpcs_description'],inplace=True,axis=1)
data.drop([0],inplace=True,axis=0)


# In[5]:


print(data.shape)
data.head()


# In[6]:


provider_gender = []
m = f = n = 0
for x in data.iloc[:,1]:
    if x =='M' or x == 'm':
        provider_gender.append(1)
        m += 1
    elif x =='F' or x == 'f':
        provider_gender.append(-1)
        f += 1
    else:
        n += 1
        if np.random.random()>0.7:
            provider_gender.append(-1)
        else:
            provider_gender.append(+1)
print(m,f,n)
print(sum(provider_gender))
# M = 6535107 F = 2881018 Null = 431318
# So we randomly and with ratio of 2 to 1 (number of Ms to number of Fs) replace null gender with M and F
data['nppes_provider_gender'] = provider_gender


# In[7]:


6836855 /(3010588 + 6836855)


# In[8]:


data.dropna(inplace=True)
print(data.shape)
data.head()


# In[9]:


# i = 0
# countHCPCS = Counter(data['hcpcs_code'])

# # for x in data.iloc[:,2]:
# #     count.
# print(countHCPCS)
# len(countHCPCS)


# In[11]:


i = 0
countType = Counter(data['provider_type'])
types = list(dict(countType).keys())
intType = []
for x in data.iloc[:,2]:
    intType.append(types.index(x))
len(intType)
# print(count)
# len(count)


# In[12]:


# data[data['provider_type']=='Medical Toxicology']


# In[13]:


data['provider_type'] = intType


# In[14]:


data.to_csv('../Medicare_Provider_Util_Payment_PUF_CY2017_revised.csv',index=False)


# In[15]:


del data


# In[16]:


data = pd.read_csv('../Medicare_Provider_Util_Payment_PUF_CY2017_revised.csv')


# In[17]:


fraudData = pd.read_csv('data/UPDATED.csv')
print(fraudData.shape)
fraudData.head()


# In[18]:


index = []
i = 0
for x in fraudData['NPI']:
    try:
        int(float(x))
    except:
        i += 1
    if len(str(x))==10:
        index.append(i)
    i += 1
print(len(index))
cleaFraud = fraudData.loc[index,['NPI','EXCLTYPE']]


# In[19]:


fraudLabel = list(cleaFraud['EXCLTYPE'])

i = 0
for x in fraudLabel:
    fraudLabel[i] = x.replace(' ','')
    i += 1
print(len(fraudLabel))
Counter(fraudLabel)


# In[20]:


classlabel = []
labels = ['1128a1','1128a2','1128a3','1128b4','1128c3gi','1128c3gi']
for x in fraudLabel:
    if str(x) in labels:
        classlabel.append(1)
    else:
        classlabel.append(0)
print(len(classlabel), sum(classlabel))
cleaFraud['fraud_label'] = classlabel
cleaFraud.drop(['EXCLTYPE'],inplace=True,axis=1)
cleaFraud.columns = ['npi', 'fraud_label']
cleaFraud.head()


# In[21]:


cleaFraud.to_csv('LEIE Exclusion.csv',index=False)


# In[22]:


mergedData = pd.merge(cleaFraud, data, on='npi',how='inner')
print(mergedData.shape, sum(mergedData['fraud_label']))
mergedData.head()


# In[23]:


mergedData.to_csv('provider features and fraud labels.csv', index=False)

