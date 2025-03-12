#%% import packages 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from time_series_helper import WindowGenerator
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import pandas as pd 
import numpy as np 
import mlflow
import keras 
from keras import layers,losses,optimizers,regularizers,callbacks,initializers
import os 

os.environ['KERAS_BACKEND']='tensorflow'
print(keras.backend.backend())

mpl.rcParams['figure.figsize']=(8,6)
mpl.rcParams['axes.grid']=False

#%% data loading 
train_path=os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
df_train=pd.read_csv(train_path)
test_path=os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
df_test=pd.read_csv(test_path)

#%% EDA
print(df_train.head())
print(df_test.head())
#%% check data type
print(df_train.info()) # there are 2 object columns: date and cases_new
print(df_test.info()) # there are 1 object columns: date
#%% check statistic summary
print(df_train.describe().transpose()) # possibility of outlliers
print(df_test.describe().transpose()) # possibility of outlliers
#%% plot data distribution
# for train data
for i in df_train.columns:
    sns.histplot(df_train[i],kde=True)
    plt.show()
# for test data
for i in df_test.columns:
    sns.histplot(df_train[i],kde=True)
    plt.show()

#%% inspect the data
# check null values
print(df_train.isna().sum()) # there are null values
print(df_test.isna().sum()) # there are null values
# replace object with numeric
df_train['cases_new']=pd.to_numeric(df_train['cases_new'],errors='coerce')
df_train=df_train.fillna(0)
print(df_train.info())
# check null value again 
print(df_train.isna().sum()) # return 0
print(df_test.isna().sum()) # return 0

#%% check duplicated values
print(df_train.duplicated().sum()) # return 0
print(df_train.duplicated().sum()) # return 0

#%% extract and convert the date time column
date=pd.to_datetime(df_train['date'],format='%d/%m/%Y')
print(date.dtypes)
# plot 
plot_cols=['cases_new']
plot_features=df_train[plot_cols]
plot_features.index=date
_= plot_features.plot(subplots=True)

#%% drop date column
df_train=df_train.drop(columns=['date','cases_boost'],axis=1)
df_train.head()
df_test=df_test.drop(columns=['date','cases_boost'],axis=1)
df_test.head()

#%% split df_train to train and val
n=df_train.shape[0]
train_df=df_train[0:int(0.8*n)]
val_df=df_train[int(0.8*n):]
test_df=df_test

#%% data normalization
train_mean=train_df.mean()
train_std=train_df.std()
train_df=(train_df-train_mean)/train_std
val_df=(val_df-train_mean)/train_std
test_df=(test_df-train_mean)/train_std

#%% create a window for single-time-step single-output prediction
from time_series_helper import WindowGenerator
wide_window=WindowGenerator(30,30,1,train_df=train_df,val_df=val_df,test_df=test_df,batch_size=64,label_columns=['cases_new'])
wide_window.plot(plot_col='cases_new')

#%% create an RNN (LSTM) for the wide window
n_layer=5
model=keras.Sequential()
for i in range(n_layer):
    model.add(keras.layers.LSTM(units=32,return_sequences=True))
model.add(keras.layers.Dense(units=1))
model.summary()

#%% model compilation
model.compile(optimizer='adam',loss='mse',metrics=['mae','mape'])

#%% setup mlflow experiment
experiment=mlflow.set_experiment('Covid19 New Cases Prediction')
# model training
with mlflow.start_run() as run:
    mlflow_callback=mlflow.keras.MlflowCallback(run)
    history=model.fit(wide_window.train,validation_data=wide_window.val,epochs=200,batch_size=32,callbacks=[mlflow_callback])
    # save model
    mlflow.keras.save.log_model(model,artifact_path='model')

# test prediction
wide_window.plot(model=model,plot_col='cases_new')
#%%
y_pred=model.predict(test_df)

