##import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import pickle
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

##import libraries

df=pd.read_csv(r'D:\Vardhan\ML\MLU\DATASETS\Algerian_forest_fires_dataset_UPDATE.csv',header=1)

df_org=df

##DATA CLEANING
df[df.isnull().any(axis=1)]
##LOC IS BASICALLY USED FOR SELECTING ROWS,ACCESING ROWS AND COLS USING INDEX AND LABELS
df.loc[:122,"Region"]=0
df.loc[122:,"Region"]=1
df=df.dropna().reset_index(drop=True)
df.isnull().sum()

df.iloc[122:123]
df=df.drop(122).reset_index(drop=True)

df.columns
##FIX SPACES IN COLUMNS
df.columns=df.columns.str.strip()
df.columns
for i in df.columns:
    if i!='Classes':
        df[i]=df[i].astype(float)

df.to_csv('ALGERian datset cleaned.csv',index=False)

##fwi is our output
df_copy=df
df=df.drop(['day','month','year'],axis=1)

##encoding of the catergories
df['Classes'].value_counts()
df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)
df['Classes'].value_counts()



##monthly anaalusis

df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),'not fire','fire')
dftemp=df_copy.loc[df_copy['Region']==0]




x=df.drop('FWI',axis=1)
y=df['FWI']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


def correlation(dataset,threshold):
    col_cor=set()
    cor_mat=dataset.corr()
    for i in range(len(cor_mat.columns)):
        for j in range(i):
            if abs(cor_mat.iloc[i][j])>threshold:
                colname=cor_mat.columns[i]
                col_cor.add(colname)
    return col_cor
cor_features=correlation(x_train,threshold=0.85)
##thresold is set by domain expert//..
x_train=x_train.drop(cor_features,axis=1)
x_test=x_test.drop(cor_features,axis=1)
##standarization

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
lasso=Lasso()
lasso.fit(x_train_scaled,y_train)
y_scaled=scaler.transform([[
            29.0, 
            57.0, 
            18,   
            0.0, 
            65.67, 
            3.4,
            1.0,  # ISI (Initial Spread Index)
            1.0,   # Classes (Fire Class)
            1.0    # Region (Region identifier)
        ]])
print(type(y_scaled),y_scaled.shape)

pickle.dump(lasso,open('lasso.pkl','wb'))
