#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[5]:


gym = pd.read_excel(r'D:\Python\python tutorials\dataGYM.xlsx')


# In[6]:


gym


# In[7]:


gym['Class'] = LabelEncoder().fit_transform(gym['Class'])


# In[8]:


gym['Class'].value_counts()


# In[9]:


X = gym.iloc[:,:3]


# In[10]:


X


# In[11]:


Y = gym.iloc[:,5:]


# In[12]:


Y


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


# In[14]:


model_GYM = RandomForestClassifier(n_estimators=20)

model_GYM.fit(X_train,Y_train)


# In[15]:


# make predictions
expected = Y_test
predicted = model_GYM.predict(X_test)


# In[16]:


#Correction
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[17]:


import pickle

pickle.dump(model_GYM, open("Model_GYM.pkl", "wb"))

model = pickle.load(open("Model_GYM.pkl", "rb"))

print(model.predict([[40,5.6,70]]))


# In[ ]:




