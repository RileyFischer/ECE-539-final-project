#!/usr/bin/env python
# coding: utf-8

# In[383]:


import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


# In[384]:


#read in data
df=pd.read_csv("breast-cancer.csv")
#encode our output variable 
df['diagnosis'] = df['diagnosis'].map({'B' : 0, 'M' : 1})
df['diagnosis'] = pd.to_numeric(df['diagnosis'])
X=np.array(df)[:,2:]
target=np.array(df)[:,1]
df


# In[419]:


#visualize data
for i in range(30):
    plt.scatter(X[:, (i)], target, 1);
    plt.xlabel(df.columns[i+2]) 
    plt.ylabel("diagnosis") 
    plt.show()


# In[ ]:





# In[434]:


X_train, X_test, y_train,  y_test = train_test_split(X, target, test_size=0.2)

pModel = Perceptron(random_state=2).fit(X_train, y_train)

predictions_test = pModel.predict(X_test)
test_score = accuracy_score(predictions_test, y_test)
print("score on test data: ", test_score)


# In[437]:


Cmat=confusion_matrix(y_test,predictions_test)
Recall=Cmat[0][0]/(Cmat[0][0]+Cmat[0][1])
Precision=Cmat[0][0]/(Cmat[0][0]+Cmat[1][0])
print(("recall: ",recall))
print(("precision: ",Precision))


# In[ ]:




