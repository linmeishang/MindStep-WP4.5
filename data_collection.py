# open gdx file in python and automatically add read folders and add it to exitsing dataframe 

#%%
import gdxpds
from gdxpds import gdx
from collections import OrderedDict
import pandas as pd
import numpy as np
import glob
import os
import time
from datetime import datetime
import pickle

#%%
# mapperInstance is a function to rename columns so that we can select them
def mapperInstance():

    def gen_i():
        i = 0
        while True:
            yield str(i)
            i = i+1
    gen = gen_i()

    def mapper(s):
        return (s + gen.__next__())
    
    return mapper

#%%
# get_df is a function to read a single gdx file of a single draw, and transform it into a data frame
def get_df(gdx_file):
   
    p_res = gdxpds.read_gdx.to_dataframe(gdx_file, symbol_name='p_sumFarmLin', gams_dir='N:\soft\GAMS28.3', old_interface=False)
    # print(p_res)
    
    # Rename Columns (1st pass)
    mpr = mapperInstance()
    
    p_res.rename(mpr, axis="columns", inplace=True)
    # print('here is p_res 1:', p_res)
    
    # p_res = p_res.drop(['*4'], axis=1) # drop the column *4
    # add a column and give the new column a name (combine all the names before)
    p_res['concat']= pd.Series(p_res[['*2', '*3', '*4']].fillna('').values.tolist()).map(lambda x: '_'.join(map(str,x)))
    # print('here is p_res 2:', p_res)
    
    df = p_res[['Value6','concat']].T # change 5 to 6
    # print('here is p_res 3:', p_res)
    
    # print(df)
    df = df.rename(columns=df.iloc[1])
    # print("renamed df:", df)
    
    df = df.drop(['concat']) # drop the row "concat"
    # print(df.columns)
    
    df = df.loc[:,~df.columns.duplicated()]
    # print("final df:", df)    

    return df

# %%
# get_df_parquet is function to read all gdx files in a folder, concate them and store in as a parquet file
def get_df_parquet(folder):

    all_files = glob.glob(os.path.join(folder + "\\*.gdx"))
    
    df = pd.concat([get_df(gdx_file) for gdx_file in all_files])
   
    # print(df)
    
    df.index = [f'draw_{i}' for i in range(df.shape[0])]

    
    df.to_parquet(folder[len(folder)-12:]+".parquet.gzip",  compression="gzip")
   
    # print('file saved')
    return df


# %%
# Get all folders in DataCollection
path = './DataCollection'
os.chdir(path) 

print('Current working directory is:', os.getcwd())

all_folders = [folder for folder in os.listdir() if os.path.isdir(folder)]
print('All folders in the current directory:', all_folders)


#%%
# python finds the latest .parquet.gzip
all_parquets = glob.glob("total_df_2023*.parquet.gzip")

print('all total parquets:', all_parquets)


if len(all_parquets) == 0:

    print("No total_df yet")
    
    # Create a empty total df
    total_df = pd.DataFrame()

else: 

    print ("Total df exists")
    
    total_df = max(all_parquets, key=lambda f: os.path.getmtime(f))

    total_df = pd.read_parquet(total_df) 

    
print ("Latest total_df is:", total_df)

print("Shape of the latest total_df:", total_df.shape)


#%%

for folder in all_folders:
    

    filename = folder[len(folder)-12:]
    
    print(filename)
    
    # if .parquet.gzip exist, do nothing; if not, creat a parquet file by get_df_parquet
    if os.path.isfile(filename+".parquet.gzip"):
        
        print(filename, "File exist")
        
    else:
        print(filename, "File not exist")

        df = get_df_parquet(folder)
        
        # print("df:", df)
        # append it into total_df
        total_df = pd.concat([total_df, df], axis=0)
       

        print(filename, "File is created")


#%%
# Rename the indexs of total_df
total_df.index = [f'draw_{i}' for i in range(total_df.shape[0])]

print("shape of total_df:", total_df.shape)
#  print("Total df: ", total_df)


#%%
# Rename total_df according to time YYMMDD
Date = datetime.now().strftime("%Y%m%d") # use ("%Y%m%d-%H%M%S") for hour-minute-second

total_df.to_parquet("total_df_"+Date+".parquet.gzip",  compression="gzip")

print("new total_df saved")



#%%
# Get the name of columns so that we can define it later
# ColumnNames = list(total_df.columns)
# print(ColumnNames)
# with open('ColumnNames.pkl', 'wb') as f:
# pickle.dump(ColumnNames, f)

