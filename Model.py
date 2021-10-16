#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv('data.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50 , figsize=(20,15))
plt.show()


# ## Train Test Split 

# In[8]:


import numpy as np


# In[9]:


#the basics of train test split
#def split_train_test(data,test_ratio):
    #np.random.seed(42)
    #shuffled = np.random.permutation(len(data))
    #test_set_size = int(len(data)*test_ratio)
    #test_indices = shuffled[:test_set_size]
    #train_indices = shuffled[test_set_size:]
    #return data.iloc[train_indices],data.iloc[test_indices]


# In[10]:


#train_set,test_set = split_train_test(housing,0.2)


# In[11]:


#print(f"train set number of rows:{len(train_set)}\ntest set number of rows:{len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size = 0.2,random_state = 42)
print(f"train set number of rows:{len(train_set)}\ntest set number of rows:{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1 ,test_size = 0.2 , random_state = 42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_train_set.head()


# In[15]:


strat_test_set.head()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


strat_test_set['CHAS'].value_counts()


# In[18]:


housing = strat_train_set.copy()


# # Looking For Co Relation

# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()


# In[21]:


housing.plot(kind = "scatter",x = "RM", y = "MEDV",alpha = 0.9)
plt.show()


# In[22]:


housing.plot(kind = "scatter",x = "LSTAT", y = "MEDV",alpha = 0.9)
plt.show()


# # Trying out attribute combination

# In[23]:


housing['TaxRM']= housing['TAX']/housing['RM']
housing['TaxRM'].head()


# In[24]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[25]:


housing.plot(kind = "scatter",x = "TaxRM", y = "MEDV",alpha = 0.9)
plt.show()


# In[26]:


housing.describe()


# In[27]:


housing = strat_train_set.drop('MEDV',axis =1)
housing_labels = strat_train_set['MEDV'].copy()


# # Sci-Kit Learn Design

# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = 'median')),
    ('std_scaler',StandardScaler()),
])


# In[29]:


housing_tr = my_pipeline.fit_transform(housing)


# In[30]:


housing_tr.shape


# # Random Forest Implementation

# In[31]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_tr,housing_labels)


# In[32]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[33]:


prepared_data[0]


# # Model Evaluation

# In[34]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr)
rand_mse = mean_squared_error(housing_labels,housing_predictions)
rand_rmse = np.sqrt(rand_mse)
rand_rmse


# # Cross Validation - Random Forest

# In[35]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_tr,housing_labels,scoring = "neg_mean_squared_error",cv = 10)
rand_rmse_scores = np.sqrt(-scores)
rand_rmse_scores


# # Model Dump - Random Forest

# In[36]:


from joblib import dump,load
dump(model , 'DragonFinal.joblib')


# # Score Details Function

# In[37]:


def print_scores(scores):
    print('scores: ',scores)
    print('mean: ',scores.mean())
    print('standard deviation: ',scores.std())


# # Random Forest Score Details

# In[38]:


print_scores(rand_rmse_scores)


# # Model Testing

# In[39]:


x_test = strat_test_set.drop('MEDV',axis =1)
y_test = strat_test_set['MEDV'].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# # Using the Model - Price Prediction Based on features

# In[40]:


from joblib import dump,load
import numpy as np
model = load('DragonFinal.joblib')


# In[41]:


import pickle
filename = 'DragonFinal.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)


# In[42]:


features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




