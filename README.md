# Flight_Cancellation_Prediction

<div align="center">
  <img src="https://paulstravelnotes.com/wp-content/uploads/2022/01/a.jpg" width="1000">
</div>

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [ETL Pipeline](#etl-pipeline)
- [Data Pre-Processing](#data-pre-processing)
- [Class Imbalance](#class-imbalance)
- [Classification Models](#classification-models)
- [Results](#results)
- [Future Scope](#future-scope)


## 1. Introduction

This GitHub repository hosts an end-to-end data science project that focuses on the analysis and prediction of flight status using binary classification techniques. The project comprises two main components:


  
-  #### ETL (Extract, Transform, Load) pipeline


Our ETL pipeline has been meticulously crafted to ingest, cleanse, and preprocess raw data, culminating in a refined dataset primed for in-depth analysis and the creation of purposeful reports.
  
The aim of this project is to perform binary classication into following classes:

- Flight is Cancelled (Positive class or 1)
- Fligh isn't Cancelled ( Negative class or 0)



Machine learning classifier: 

- ####  XGBoost
- #### Bernoullie Naive Base
- #### Random Forest

This study rigorously explores and contrasts the performance of three prominent classification models: XGBoost, Random Forest, and Bernoulli Naive Bayes. Our objective is to discern the model that excels in diverse contexts, thus enhancing the efficiency and accuracy of classification tasks.



This project's code is also available on Kaggle, where it was collaboratively developed by Khushal Sharma and Venkat Anand Sai Duggi. You can access and interact with the codes on Kaggle: <a href="https://www.kaggle.com/code/venkatanandsaid/airline-on-time-data-preprocessing">Data-Preprocessing</a>, <a href="https://www.kaggle.com/code/kms025/airline-on-time-classification">Classification-Task</a>


## 2. Dependencies

- Extracting Parquet data from Kaggle's API


```bash
from kaggle.api.kaggle_api_extended import KaggleApi
```


- Data Preprocessing

  
```bash
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
```


- Machine Learning Operations


```bash
# Basic Libraries
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

# Class Imbalance 
from imblearn.over_sampling import SMOTE

# Combining features for Visualization 
from sklearn.decomposition import PCA

# Visualization Purpose
import matplotlib.pyplot as plt
import seaborn as sns

# Classifiers
import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

# Model Evaluation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
```



## 3. Dataset

- The data consists of flight arrival and departure details for all commercial flights within the USA, from October 1987 to April 2008. This is a large dataset: there are nearly 120 million records in total and takes up 1.6 gigabytes of space when compressed and 12 gigabytes when uncompressed.
- Data was extracted using Kaggle's API 
- You can find the dataset <a href="https://www.kaggle.com/datasets/ahmedelsayedrashad/airline-on-time-performance-data/code?datasetId=3670668&sortBy=dateRun&tab=profile">here</a>


[Link to Kaggle Project](https://www.kaggle.com/your-kaggle-username/your-project-name) 

## 4. Data Pre-processing


