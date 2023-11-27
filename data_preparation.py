# This file is for preparing the data before training
#%%
import pickle
import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
#%%
####################################################################################################################
# load parquet data set
path = './DataCollection'

os.chdir(path)
print("Current Working Directory " , os.getcwd())

all_parquets = glob.glob("total_df_2023*.parquet.gzip")

print(all_parquets)

#%%
# find the latest df
df = max(all_parquets, key=os.path.getctime)
print(df)
#%%
# load df from parquet
df = pd.read_parquet(df)
print(df.shape)

#%%
# Select draws which does not habe winter barley prodution
wb_condition = df['WinterBarley_Quantity_'] > 0
df = df.drop(df[wb_condition].index)
print(df.shape)

#%%
####################################################################################################################
# Define inputs and outputs
path = '../DataPreparation'
os.chdir(path)
print('Current Working Directory ' , os.getcwd())

#Load Input and outout table
InputOutputArable = pd.read_excel('InputOutput.xlsx', index_col=0)  

# Assign these whose Input = 1 to in_col
Input = InputOutputArable.query('Input==1')
Output = InputOutputArable.query('Output==1')

# Get name of indexs
in_col = Input.index.values.tolist() 
out_col = Output.index.values.tolist()

X_all = df.reindex(columns = in_col)
Y_all = df.reindex(columns = out_col)

# print('X:', X_all)
# print('Y:', Y_all)
print('shape of X_all:', X_all.shape)
print('shape of Y_all:', Y_all.shape)

#%%
# define the name of dir to be created 
path = './DataPreparation'+ datetime.now().strftime('_%Y%m%d%H%M')

try:
    os.makedirs(path)
except OSError:
    print ('Creation of the directory %s failed' % path)
else:
    print ('Successfully created the directory %s' % path)

# change working directory in order to save all data later
os.chdir(path)
print('Current Working Directory ' , os.getcwd())

#%%
X_min = X_all.min(axis=0)
X_max = X_all.max(axis=0)
X_empty_count = X_all.isnull().sum()

Y_min = Y_all.min(axis=0)
Y_max = Y_all.max(axis=0)
Y_empty_count = Y_all.isnull().sum()

X_min_max = pd.concat([X_min, X_max, X_empty_count], axis=1)
X_min_max = X_min_max.rename(columns={0: "min", 1: "max", 2: "empty_count"})

Y_min_max = pd.concat([Y_min, Y_max, Y_empty_count], axis=1)
Y_min_max = Y_min_max.rename(columns={0: "min", 1: "max", 2: "empty_count"})

X_min_max.to_excel('X_min_max_FarmDyn_'+ datetime.now().strftime('_%Y%m%d')+'.xlsx')
Y_min_max.to_excel('Y_min_max_FarmDyn_'+ datetime.now().strftime('_%Y%m%d')+'.xlsx')




#%%
###################################################################################################################
# Change column names for input and output sides
# Read new column names for X_all and Y_all
All_names = InputOutputArable['Names in FarmLin'].values

# Replace with new column names
X_all.columns = All_names[:len(in_col)]
Y_all.columns = All_names[len(in_col):]

# Round up the coloum of machinery
for col in Y_all.iloc[:, 1:58]:
    print(Y_all[col])
    Y_all[col] = Y_all[col].round(decimals = 0)
    print(Y_all[col])

#%%
# Fill the rest of NA with 0
X_all= X_all.fillna(0)
Y_all= Y_all.fillna(0)

#%%
# plot histogram for variables
X_all.hist(bins=30, figsize=(100, 100))
plt.savefig('X_all.png')
Y_all.hist(bins=30, figsize=(100, 100))
plt.savefig('Y_all.png')

#%%

#%%
X_min = X_all.min(axis=0)
X_max = X_all.max(axis=0)
X_empty_count = X_all.isnull().sum()

Y_min = Y_all.min(axis=0)
Y_max = Y_all.max(axis=0)
Y_empty_count = Y_all.isnull().sum()

X_min_max = pd.concat([X_min, X_max, X_empty_count], axis=1)
X_min_max = X_min_max.rename(columns={0: "min", 1: "max", 2: "empty_count"})

Y_min_max = pd.concat([Y_min, Y_max, Y_empty_count], axis=1)
Y_min_max = Y_min_max.rename(columns={0: "min", 1: "max", 2: "empty_count"})

X_min_max.to_excel('X_min_max_NN_'+ datetime.now().strftime('_%Y%m%d')+'.xlsx')
Y_min_max.to_excel('Y_min_max_NN_'+ datetime.now().strftime('_%Y%m%d')+'.xlsx')


#%%
# ####################################################################################################################
# Train-test split
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_all, Y_all, test_size=0.1)

print('shape of X_train_raw:', X_train_raw.shape)
print('shape of Y_train_raw:', Y_train_raw.shape)
print('shape of X_test_raw:', X_test_raw.shape)
print('shape of Y_test_raw:', Y_test_raw.shape)

#%%
# X_train_raw.hist(bins=30, figsize=(100, 100))
# Y_train_raw.hist(bins=30, figsize=(100, 100))

#%%
####################################################################################################################
# Normalize/standardize X and Y
# test set must be normalized with X_scaler of training set
X_scaler = MinMaxScaler() # 0-1
Y_scaler = MinMaxScaler()

# X_scaler = StandardScaler() # z = (x - u) / s
# Y_scaler = StandardScaler()

#%%
X_train = X_scaler.fit_transform(X_train_raw)
X_test  = X_scaler.transform(X_test_raw)
Y_train = Y_scaler.fit_transform(Y_train_raw)
Y_test  = Y_scaler.transform(Y_test_raw)

# After transformation, names of columns are lost. Now we put them back
X_train = pd.DataFrame(X_train, columns = X_train_raw.columns)
X_test = pd.DataFrame(X_test, columns = X_test_raw.columns)
Y_train = pd.DataFrame(Y_train, columns = Y_train_raw.columns)
Y_test = pd.DataFrame(Y_test, columns = Y_test_raw.columns)

print('X_train:', X_train)
print('X_test:', X_test)
print('Y_train:', Y_train)
print('Y_test:', Y_test)

#%%
# Pickle scaler X_scaler Y_scaler
dump(X_scaler, open('X_scaler.pkl', 'wb'))
dump(Y_scaler, open('Y_scaler.pkl', 'wb'))


#%%
####################################################################################################################
# save the save the data in parquet form

my_list = [X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, X_train, Y_train, X_test, Y_test]

filenames = ['X_train_raw', 'Y_train_raw', 'X_test_raw', 'Y_test_raw', 'X_train', 'Y_train', 'X_test', 'Y_test']

# parallel iteration of two list

for df, j in zip(my_list, range(8)):

    filename = filenames[j]

    #  save all data as parquet files
    df.to_parquet(filename+'.parquet.gzip', compression='gzip')


#%%