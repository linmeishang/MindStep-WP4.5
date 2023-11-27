# MindStep-WP4.5

Working Package 4.5 is a sub-project of the EU project MindStep (https://mind-step.eu/). Here, we aim to develop surrogate models of the detailed farm-level model FarmDyn (https://farmdyn.github.io/documentation/) using deep learning approach. 

## Overview

This repository contains 5 folders. Their functions are decribed below.

### 1. Data Collection

This folder contains the raw data generated from the farm-level model. These data was generated at different times and stored each time in a seperated folder (e.g. 202308141151). In each folder, there are many gdx files. Each gdx file is a farm draw. 

Due to storagte limitation, we only keep an example folder with a few example gdx files.

The file "data_collection.py" reads the gdx files in each folder and stores the data of each folder in a parquet file (e.g. 202308141151.parquet.gzip).

At the end, data from all parquet files was combined together into the file, named "total_df_(date).parquet.gzip". 

The "data_collection.py" can automatically add the newly added folder of farm data to the "total_df_(date).parquet.gzip" file, without needing to read all gdx files again.

In this application, the file total_df_20230818.parquet.gzip is used as a raw data file. 

### 2. Data Preparation

This folder prepares the data before training. The "data_preparation.py" file does the following steps of prepration:

(1) Load the newest "total_df_(date).parquet.gzip" file from the DataCollection folder.

(2) Seperate inputs (X) and outputs (Y) based on the "InputOutputArable.xlsx" file.

(3) Split train and test datasets.

(4) Normalize the train and test datasets.

(5) Store the datasets (including raw data and normalized data) and the scalers in a "DataPreparation_(date)" folder.



### 3. AutomatedFarmDyn

The python and R code in this folder is used to automatically generate data points from FarmDyn, without having to open and close GUI manually. 


### 4. AgriPoliS
Links to source code used in AgriPoliS to integrate the trained model into AgriPoliS.


## Features and Targets

The model takes 83 input variables and predicts for 122 output variables. See the InputOutput.xlsx under the folder DataPreparation for more detail. The inputs include remaining capacities of machinery and buildings (No. 1-57), crop prices (No. 58-68), crop yields (No. 69-79), farm size (No. 80), labor availability (No. 81), liquidity (No. 82) and interest rate (No. 83). Crops included in the model are winter wheat, winter rye, summer cereal, summer peas, winter rape, sugar beet, winter barley, summer triticale, corn maize, summer beans, and potatoes. The outputs include the objective variable (No. 1), investment amount of machinery and buildings (No. 1-58), labor attribution (No. 59, 60, 63, 64), short-term borrowed capital (No. 61), interest gain (No. 62), purchased agriculture services (No. 65), production level of crops or idled land (ha) (No. 66-78), use of farming inputs (No. 93-101), farm-level environmental indicators (No. 102- 122). 


## Getting Started


### Step 1: Clone the repository

git clone https://gitlab.com/test5623246/bike-demand.git


Go to your local path using:

cd <your path>\MindStepWP4_5



### Step 2: Create a new python environment with python 3.10 if you don't have one

conda create -n FarmLin python=3.10

conda activate FarmLin


### Step 3: Install the requirments

pip install -r requirements.txt


### Step 4: Train models or use our trained model to predict


You can train new models by running:

python train.py

### Step 5: Employ the trained surrogate model

python FarmLin.py

## Funding

This project has received funding from the European Union's research and innovation programme under grant agreement Mind Step No 817566. 

## Contact information

Dr. Linmei Shang, Institute for Food and Resource Economics (ILR), University of Bonn, Nußallee 21, 53115 Bonn, Germany (E-mail address: linmei.shang@ilr.uni-bonn.de)