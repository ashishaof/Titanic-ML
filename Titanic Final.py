
# coding: utf-8

# In[1]:


## Loading Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# In[3]:


##Loading Datasets
gender=pd.read_csv('C:/Users/BAKSHI/Downloads/gender_submission.csv')
train=pd.read_csv('C:/Users/BAKSHI/Downloads/train.csv')
test=pd.read_csv('C:/Users/BAKSHI/Downloads/test.csv')


# In[5]:


##Printing first 5 rows of the train dataset.
train.head()


# In[6]:


##Total rows and columns
train.shape


# In[8]:


##Describing training dataset
##describe() method can show different values like count, mean, standard deviation, etc. of numeric data types.
train.describe()


# In[9]:


##describe(include = ['O']) will show the descriptive statistics of object data types.
train.describe(include=['O'])


# In[10]:


##We use info() method to see more information of our train dataset.
##We can see that Age value is missing for many rows.Out of 891 rows, the Age value is present only in 714 rows.Similarly,
## Cabin values are also missing in many rows. Only 204 out of 891 rows have Cabin values.
train.info()


# In[11]:


##There are 177 rows with missing Age, 687 rows with missing Cabin and 2 rows with missing Embarked information.
train.isnull().sum()


# In[12]:


##Looking into the testing dataset
##Test data has 418 rows and 11 columns.
##Train data rows = 891
##Test data rows = 418
##Total rows = 891+418 = 1309
##We can see that around 2/3 of total data is set as Train data and around 1/3 of total data is set as Test data.
test.shape


# In[14]:


##Survived column is not present in Test data. We have to train our 
##classifier using the Train data and generate predictions (Survived) on Test data.
test.head()


# In[15]:


test.info()


# In[16]:


##There are missing entries for Age in Test dataset as well.

##Out of 418 rows in Test dataset, only 332 rows have Age value.

##Cabin values are also missing in many rows. Only 91 rows out ot 418 have values for Cabin column.
##There are 86 rows with missing Age, 327 rows with missing Cabin and 1 row with missing Fare information.
test.isnull().sum()


# In[17]:


##Relationship between Features and Survival
##In this section, we analyze relationship between different features with respect to Survival. We see how 
##different feature values show different survival chance. We also plot different kinds of diagrams to visualize
##our data and findings.
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))


# In[18]:


##Pclass vs. Survival
##Higher class passengers have better survival chance.
train.Pclass.value_counts()


# In[19]:


train.groupby('Pclass').Survived.value_counts()


# In[20]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[21]:


#train.groupby('Pclass').Survived.mean().plot(kind='bar')
sns.barplot(x='Pclass', y='Survived', data=train)


# In[22]:


##Sex vs. Survival
##Females have better survival chance.
train.Sex.value_counts()


# In[23]:


train.groupby('Sex').Survived.value_counts()


# In[24]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[25]:


#train.groupby('Sex').Survived.mean().plot(kind='bar')
sns.barplot(x='Sex', y='Survived', data=train)


# In[26]:


##Pclass & Sex vs. Survival
##Below, we just find out how many males and females are there in each Pclass. We then plot a stacked bar 
##diagram with that information. We found that there are more males among the 3rd Pclass passengers.
tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[27]:


sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)

##From the  plot, it can be seen that:

##Women from 1st and 2nd Pclass have almost 100% survival chance.
##Men from 2nd and 3rd Pclass have only around 10% survival chance.


# In[28]:


##Pclass, Sex & Embarked vs. Survival
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)

##From the  plot, it can be seen that:

##Almost all females from Pclass 1 and 2 survived.
##Females dying were mostly from 3rd Pclass.
##Males from Pclass 1 only have slightly higher survival chance than Pclass 2 and 3.


# In[29]:


##Embarked vs. Survived
train.Embarked.value_counts()


# In[30]:


train.groupby('Embarked').Survived.value_counts()


# In[31]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[32]:


#train.groupby('Embarked').Survived.mean().plot(kind='bar')
sns.barplot(x='Embarked', y='Survived', data=train)


# In[34]:


##Age vs. Survival
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)


# In[35]:


##Classification & Accuracy
##classifying algorithms


# In[36]:


##Logistic Regression


# In[ ]:


##Logistic Regression
##Support Vector Machines (SVC)
##Linear SVC
##k-Nearest Neighbor (KNN)
##Decision Tree
##Random Forest
##Naive Bayes (GaussianNB)
##Perceptron
##Stochastic Gradient Descent (SGD)

