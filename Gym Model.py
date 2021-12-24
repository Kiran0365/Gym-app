#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[12]:


gym = pd.read_excel(r'D:\Python\python tutorials\dataGYM.xlsx')


# In[13]:


gym


# In[16]:


df = gym.drop('Unnamed: 6', 1)


# In[17]:


df


# In[18]:


df['Class'] = LabelEncoder().fit_transform(df['Class'])


# In[35]:


df['Class'].value_counts()


# In[37]:


X = df.iloc[:,:3]


# In[38]:


X


# In[39]:


Y = df.iloc[:,5:]


# In[40]:


Y


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


# In[42]:


model_GYM = RandomForestClassifier(n_estimators=20)
model_GYM.fit(X_train,Y_train)


# In[43]:


# make predictions
expected = Y_test
predicted = model_GYM.predict(X_test)


# In[44]:


#Correction
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[45]:


import pickle

pickle.dump(model_GYM, open("Model_GYM.pkl", "wb"))

model = pickle.load(open("Model_GYM.pkl", "rb"))

print(model.predict([[40,5.6,70]]))


# In[ ]:




