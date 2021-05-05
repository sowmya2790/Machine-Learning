#!/usr/bin/env python
# coding: utf-8

# ## Kaggle Competition for House Prices: Advanced Regression Techniques 

# In[283]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[284]:


df=pd.read_csv('train.csv')


# In[285]:


df.head()


# In[286]:


df['MSZoning'].value_counts()


# In[287]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[288]:


df.shape


# In[289]:


df.info()


# In[290]:


## Fill Missing Values

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[291]:


df.drop(['Alley'],axis=1,inplace=True)


# In[292]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[293]:


df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[294]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[295]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[296]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[297]:


df.shape


# In[298]:


df.drop(['Id'],axis=1,inplace=True)


# In[299]:


df.isnull().sum()


# In[300]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[301]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[302]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[303]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[304]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[305]:


df.dropna(inplace=True)


# In[306]:


df.shape


# In[192]:


df.head()


# In[ ]:


##HAndle Categorical Features


# In[307]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[308]:


len(columns)


# In[309]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[310]:


main_df=df.copy()


# In[311]:


## Combine Test Data 

test_df=pd.read_csv('formulatedtest.csv')


# In[312]:


test_df.shape


# In[313]:


test_df.head()


# In[314]:


final_df=pd.concat([df,test_df],axis=0)


# In[316]:


final_df['SalePrice']


# In[317]:


final_df.shape


# In[318]:


final_df=category_onehot_multcols(columns)


# In[319]:


final_df.shape


# In[320]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[321]:


final_df.shape


# In[256]:


final_df


# In[322]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[324]:


df_Train.head()


# In[325]:


df_Test.head()


# In[327]:


df_Train.shape


# In[326]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[328]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# ## Prediciton and selecting the Algorithm

# In[329]:


import xgboost
classifier=xgboost.XGBRegressor()


# In[147]:


import xgboost
regressor=xgboost.XGBRegressor()


# In[260]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[261]:


## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[262]:


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[263]:


random_cv.fit(X_train,y_train)


# In[264]:


random_cv.best_estimator_


# In[99]:


random_cv.best_estimator_


# In[353]:


regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[354]:


regressor.fit(X_train,y_train)


# In[224]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[280]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[282]:


df_Test.shape


# In[347]:


df_Test.head()


# In[349]:


df_Test.drop(['SalePrice'],axis=1).head()


# In[355]:


y_pred=regressor.predict(df_Test.drop(['SalePrice'],axis=1))


# In[356]:


y_pred


# In[391]:


##Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(ann_pred)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)


# ## Step2

# In[335]:


pred.columns=['SalePrice']


# In[336]:


temp_df=df_Train['SalePrice'].copy()


# In[337]:


temp_df.column=['SalePrice']


# In[338]:


df_Train.drop(['SalePrice'],axis=1,inplace=True)


# In[339]:


df_Train=pd.concat([df_Train,temp_df],axis=1)


# In[253]:


df_Test.head()


# In[340]:


df_Test=pd.concat([df_Test,pred],axis=1)


# In[276]:





# In[341]:


df_Train=pd.concat([df_Train,df_Test],axis=0)


# In[343]:


df_Train.shape


# In[345]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# ## Artificial Neural Network Implementation

# In[389]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu',input_dim = 174))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'he_uniform'))

# Compiling the ANN
classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)


# In[390]:


ann_pred=classifier.predict(df_Test.drop(['SalePrice'],axis=1).values)


# In[382]:


from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[ ]:





