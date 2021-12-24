#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask
from flask import Flask, request , jsonify, render_template
#import jinja2
import numpy as np
import pandas as pd
import pickle

from flask import Flask, request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
get_ipython().system('pip install gunicorn')
get_ipython().system('pip install -U itsdangerous')
get_ipython().system('pip install -U MarkupSafe')



# In[9]:


app = Flask(__name__)
filename = 'Model_GYM.pkl'
model = pickle.load(open(filename, 'rb'))


# In[10]:



@app.route('/')
def man():
    return render_template('home.html')


# In[11]:



@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


# In[ ]:


if __name__ == "__main__":
    app.run()    

