#!/usr/bin/env python
# coding: utf-8

# 1. Loading Data:
# 

# In[11]:


#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df = pd.read_csv('Downloads/iris.csv')


# In[13]:


df.head(5)


# In[15]:


df.shape


# In[16]:


df.describe()


# In[17]:


#Listing the features of the dataset

df.columns


# In[18]:


df.info()


# In[19]:


#checking for number of null value

df.isna().sum()


# In[20]:


#calculating correlation among data

df.corr().abs()


# In[21]:


#count by Species

df.groupby(['Species']).count()


# 2. Visualizing the data:

# In[22]:


df.hist(bins = 20,figsize = (15,10));


# In[23]:


#Correlation heatmap

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True,cmap="RdYlGn")
plt.show()


# In[25]:


#countplot for Species

sns.countplot(x='Species',data=df)
plt.title('Species')
plt.show()


# In[26]:


sns.set(style = 'whitegrid')
sns.stripplot(x ='Species',y = 'SepalLengthCm',data = df);
plt.title('Iris Dataset')
plt.show()


# In[27]:


#scatterplot based on Species

sns.set(style = 'whitegrid')
sns.scatterplot(x ='PetalLengthCm',y = 'PetalWidthCm',hue="Species",data = df);
plt.title('Iris Dataset')
plt.show()


# In[28]:


sns.set(style = 'whitegrid')
sns.stripplot(x ='Species',y = 'PetalLengthCm',data = df);
plt.title('Iris Dataset')
plt.show()


# In[29]:


#scatterplot based on Species

sns.set(style = 'whitegrid')
sns.scatterplot(x ='SepalLengthCm',y = 'SepalWidthCm',hue="Species",data = df);
plt.title('Iris Dataset')
plt.show()


# In[30]:


sns.boxplot(x='Species',y='PetalLengthCm',data=df)
plt.title("Iris Dataset")
plt.show()


# In[31]:


sns.boxplot(x='Species',y='PetalWidthCm',data=df)
plt.title("Iris Dataset")
plt.show()


# 3. Splitting the Data:
# 

# In[32]:


# Sepratating & assigning features and target columns to X & y

y = df['Species']
X = df.drop(['Species',"Id"],axis=1)
X.shape, y.shape


# In[33]:


# Splitting the dataset into train and test sets: 80-20 split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, X_test.shape


# 4. Model Building & Training:

# In[34]:


# Decision Tree regression model 
from sklearn.tree import DecisionTreeClassifier

# instantiate the model 
decision = DecisionTreeClassifier()
# fit the model 
decision.fit(X_train, y_train)


# In[35]:


#predicting the target value from the model for the samples
y_test_tree = decision.predict(X_test)
y_train_tree = decision.predict(X_train)


# In[36]:


print(decision.score(X_test,y_test_tree))
print(decision.score(X_train,y_train_tree))


# In[37]:


# performance calculation
from sklearn.metrics import classification_report
print(classification_report(y,decision.predict(X)))


# In[38]:


# confusion_matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,decision.predict(X),labels=df['Species'].unique()))


# 5. Tree Visualization:
# 

# In[39]:


#importing tree function for Visualization

from sklearn import tree
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
tree.plot_tree(decision,feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'] ,
               class_names = df['Species'].unique(), filled=True)
plt.show()


# In[ ]:




