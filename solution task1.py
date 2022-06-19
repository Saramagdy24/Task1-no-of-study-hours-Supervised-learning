#!/usr/bin/env python
# coding: utf-8

# ### Sara Magdy Mohamed
# 

# ### Task #1 : Explore Supervised Machine learning
# 

# In[23]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split 
from sklearn import metrics   
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


# Import data
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# ### Data Preprocessing

# In[25]:


s_data.describe()


# In[26]:


s_data.info()


# In[27]:


s_data.isnull().sum()


# ### Exploring the Dataset

# In[28]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### Preparing the Dataset 

# In[29]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                           test_size=0.2, random_state=0) 


# In[31]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[32]:


print("Coefficent percentage", regressor.score(X_test,y_test)*100)


# In[33]:


y_pred= regressor.predict(X_test)
y_pred


# In[34]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ### the Prediction regression line

# In[35]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.title('Hours Vs Score')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.plot(X, line);
plt.show()


# ### Predict Score if student studies for 9.25 hrs/day

# In[40]:


score_prediction=regressor.predict([[9.25]])
float(score_prediction)


# ### Evaluation the model 

# In[41]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

