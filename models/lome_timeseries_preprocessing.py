#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/lome_preprocessed.csv',index_col='date_time',parse_dates=True)


# In[3]:


df.describe().transpose()


# In[4]:


n = len(df)

# Split 70:20:10 (train:validation:test)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_df.shape, val_df.shape, test_df.shape


# In[5]:


#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()
#scaler.fit(train_df)

#train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
#val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
#test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])


# In[6]:


train_df.describe().transpose()


# In[ ]:


train_df.to_csv('data/lome_train1.csv')
val_df.to_csv('data/lome_val1.csv')
test_df.to_csv('data/lome_test1.csv')


# In[ ]:




