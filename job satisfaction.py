#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[10]:


data = pd.read_excel('data.xlsx')
data.salary = data.salary*1000


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.sample(10)


# In[18]:


X = data.iloc[:,:10]
y = data.iloc[:,-1]


# In[19]:


X


# In[20]:


y


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train , X_test , Y_train , Y_test = train_test_split(X,y,test_size=0.2,random_state=100)


# In[ ]:





# # Linear Regression

# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


reg = LinearRegression().fit(X_train, Y_train)


# In[42]:


score_lr = reg.score(X_test, Y_test)
score_lr= score_lr * 100


# # Random Forest

# In[31]:


from sklearn.ensemble import RandomForestRegressor


# In[32]:


regr = RandomForestRegressor(max_depth=2, random_state=0)


# In[39]:


regr.fit(X_train, Y_train)


# In[41]:


score_rf = regr.score(X_test,Y_test)
score_rf =score_rf*100


# In[44]:


scores = [score_lr,score_rf]
algorithems = ['linear regressor','random regressor']

for i in range(len(algorithems)):
    print('the accuracy score achieved usinf'+algorithems[i]+'is :'+str(scores[i])+" %")


# In[48]:


import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel('algo')
plt.ylabel('score')

sns.barplot(algorithems,scores)


# In[ ]:




