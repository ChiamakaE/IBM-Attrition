#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pandas-profiling


# In[10]:


import pandas as pd
import pandas_profiling


# In[15]:


import sqlite3
conn = sqlite3.connect("attr_data.db")
cursor = conn.cursor()
print("Cursor created successfully")
conn.commit()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


pip install plotly


# In[12]:


import sys


# In[ ]:


pip install cufflinks


# In[13]:


import plotly.offline
import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[97]:


df = pd.read_sql_query("SELECT * from attrition_records", conn)
df.head()


# In[18]:


df.info();


# In[19]:


df.describe()


# In[99]:


attr_dummies = pd.get_dummies(df['Attrition'])
attr_dummies.head()


# In[122]:


df = pd.concat([df,attr_dummies], axis =1)
df.head()


# In[22]:


df=df.drop(['No'], axis =1 )
df.head()


# In[23]:


df.hist(figsize=(25, 25), bins = 30);


# In[106]:


df.loc[:,"MaritalStatus"]


# In[108]:


df.loc[:,"MaritalStatus"]


# ## What do you think are the 3 factors behind employee attrition?¶

# #### Converting the non-numeric columns in the dataset to numeric columns

# In[107]:


# Numerise all attributes
df_numerised = df
for col in df_numerised.columns:
    if (df_numerised[col].dtype == 'object'):
        df_numerised[col] = df_numerised[col].astype('category').cat.codes


# In[27]:


# Look at the correlation between attrition and all other factors
stacked_df_numerised = df_numerised.corr().stack().sort_values(ascending=False)
stacked_df_numerised['Attrition']


# <div class="alert alert-block alert-info">
# 
# I can infer that the factors behind Employee attrition is Over time, Marital Status and Distance From Home.
# 
# There is low positive correlation between overtime and attrition, an increase in overtime will lead to an increase attrition
# 
# There is very low positive correlation between MaritalStatus and attrition, an increase in MaritalStatus will lead to an increase attrition
# 
# There is very low positive correlation between Distance From Home and attrition, an increase in the employee's distance from home will lead to an increase attrition
# 
# </div>

# In[110]:


df.MaritalStatus.unique()             


# In[33]:


df.corr()


# In[102]:


# replacing the values
df['Attr_num'] = df['Attrition'].replace([0, 1],['No', 'Yes'])
df['Attr_num'].head


# In[109]:


df['MaritalStatus'].replace(0,'Divorced', inplace=False)
df['MaritalStatus'].replace(1,'Married', inplace=False)
df['MaritalStatus'].replace(2,'Single', inplace=False)


# In[113]:


df['MaritalStatus'] = df['MaritalStatus'].replace([0], 'Divorced')


# In[114]:


df['MaritalStatus'] = df['MaritalStatus'].replace([1], 'Married')


# In[115]:


df['MaritalStatus'] = df['MaritalStatus'].replace([2], 'Single')


# In[151]:


plt.figure(figsize=(8,5), dpi=80)
sns.set_theme(style="darkgrid", font_scale = 1.1)
ax = sns.barplot(x='MaritalStatus',y='Attrition', palette='rocket',data=df).set(title= "Attrition and Marital Status")
("")


# <div class="alert alert-block alert-info">
# single employees are on average much more likely to leave the company while married and especially divorced have on average much smaller attrition (given all other parameters to be the same)
# </div>

# In[141]:


plt.figure(figsize=(8,5), dpi=80)
sns.set_theme(style="darkgrid", font_scale = 1.1)
ax = sns.barplot(x='DistanceFromHome',y='Attrition', palette='rocket',data=df).set(title= "Attrition and Distance From Home")
("")


# <div class="alert alert-block alert-info">
# Relatively large distance from home is also considered a significant factor to leave the company: Most of the people who leave the company are located more than 12 km away from the company
# </div>

# ## What is the relationship between Education and Monthly Income?
# 

# In[131]:


corr_heat=df[['Education','MonthlyIncome']].corr()
corr_heat


# <div class="alert alert-block alert-info">
# There is a very low positive correlation between Education and Monthly Income. An increase in the Education level lead to a slight increase an in the income level  
# </div>

# In[63]:


plt.figure(figsize=(8,5), dpi=80)
sns.set_theme(style="darkgrid", font_scale = 1.1)
ax = sns.barplot(x='Education',y='MonthlyIncome', palette='rocket',data=df)
plt.xticks(rotation=90);


# ## What is the effect of age on attrition?
# 

# In[68]:


corr_heat=df[['Age','Attrition']].corr()
corr_heat


# <div class="alert alert-block alert-info">
# I can infer that there is low negative relationship between Age and Attrition. Therefore as employees increase in age the level of attrition reduce, but proper analyses below will show the proper breakdown
# </div>

# In[71]:


plt.figure(figsize=(8,5), dpi=80)
sns.set_theme(style="darkgrid", font_scale = 1.1)
ax = sns.barplot(x='Age',y='Attrition', palette='rocket',data=df)
plt.xticks(rotation=90);


# <div class="alert alert-block alert-info">
# 1- The largest percentage of Attrition is from the age of 18 to 21 years
# 
# 2- The percentage of employees leaving the company between the ages of 27 and 50 years is low
# 
# 3- The percentage increases again after the age of 60 years.
# 
# From this we conclude that
# 
# it is possible that the young people left as a result the pressure of work or their lack of responsibility
# 
# And those over the age of 60 leave due to illness problems or advancing age
# </div>

# ## Is Income the main factor in employee attrition?
# 

# In[72]:


stacked_df_numerised = df.corr().stack().sort_values(ascending=False)
stacked_df_numerised['Attrition']


# <div class="alert alert-block alert-info">
# We can infer that income is not the main factor in employee attrition
# </div>

# In[150]:


plt.figure(figsize=(10,6))
sns.histplot(x='MonthlyIncome',hue='Attrition', data=df);


# ### How does work-life balance impact the overall attrition rate?
# 

# In[81]:


income = df['WorkLifeBalance'].corr(df['Attrition'])
income


# In[83]:


sns.set(font_scale=1)
plt.figure(figsize = (5,3))
sns.barplot(x='WorkLifeBalance', y = 'Attrition', data=df);


# <div class="alert alert-block alert-info">
# From the analysis, it is found that poor work-life balance leads to higher attrition rate
# </div>

# In[ ]:




