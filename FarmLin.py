# This is the code of deploying FarmLin
# Copy paste the trained FarmLin.h5 from the model folder to the data preparation folder.
# Note: FarmLin takes pandas.core.frame.DataFrame or numpy.array, and produces numpy.ndarray
# Conda activate FarmLin
#%%
from math import fabs
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle
from pickle import load
from pickle import dump
from datetime import datetime
import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, Lambda
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler # @Changxing: This line is new

#%%
# find the latest DataPreparation
path ='./DataPreparation'

all_folders = glob.glob(os.path.join(path + '/Da*/'))

# find the latest Train Data
DataPreparation = max(all_folders, key=os.path.getctime)

# set DataPreparation as work dir
path = DataPreparation
os.chdir(path)
print("Current Working Directory " , os.getcwd())

#%%
# load the raw data into DataFrame
X_train_raw = pd.read_parquet('X_train_raw.parquet.gzip') 
Y_train_raw = pd.read_parquet('Y_train_raw.parquet.gzip') 
X_test_raw = pd.read_parquet('X_test_raw.parquet.gzip') 
Y_test_raw = pd.read_parquet('Y_test_raw.parquet.gzip')

print('shape of X_train raw:', X_train_raw.shape)
print('shape of Y_train raw:', Y_train_raw.shape)
print('shape of X_test raw:', X_test_raw.shape)
print('shape of Y_test raw:', Y_test_raw.shape)


#%%
# load X_scaler and Y_scaler, load FarmLin
X_scaler = load(open('X_scaler.pkl', 'rb'))
Y_scaler = load(open('Y_scaler.pkl', 'rb'))
FarmLin= load_model('FarmLin.h5')
print(FarmLin.summary())

#%%
# Precit for train and test dataset: FarmLin produces numpy.ndarray
start_time = time.time()
yhat_train_raw = FarmLin.predict(X_train_raw)
end_time = time.time()
time_cost_train =  end_time - start_time
print(time_cost_train)
per_train = time_cost_train/X_train_raw.shape[0]
print(per_train)



start_time = time.time()
yhat_test_raw = FarmLin.predict(X_test_raw)
end_time = time.time()
time_cost_test =  end_time - start_time
print(time_cost_test)
per_test = time_cost_test/X_test_raw.shape[0]
print(per_test)

#%%

# r2 of train dataset compa
r2_train = r2_score(Y_train_raw, yhat_train_raw)
print("r2 of training set", r2_train)

# r2 for test dataset
r2_test = r2_score(Y_test_raw, yhat_test_raw)
print("r2 of test set", r2_test)

# %%
# Note: FarmLin takes pandas.core.frame.DataFrame or numpy.array, and produces numpy.ndarray
