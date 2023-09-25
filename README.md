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

- #### Dataset : Airline-on-time-performance-data,
- You can find the dataset <a href="https://www.kaggle.com/datasets/ahmedelsayedrashad/airline-on-time-performance-data/code?datasetId=3670668&sortBy=dateRun&tab=profile">here</a>
- The data consists of flight arrival and departure details for all commercial flights within the USA, from October 1987 to April 2008. This is a large dataset: there are nearly 120 million records in total and takes up 1.6 gigabytes of space when compressed and 12 gigabytes when uncompressed.

  
- #### Features: There are 31 attributes shown in the following table

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>year</td>
      <td>1987-2008</td>
    </tr>
    <tr>
      <td>month</td>
      <td>1-12</td>
    </tr>
    <tr>
      <td>day of month</td>
      <td>1-31</td>
    </tr>
    <tr>
      <td>day of week</td>
      <td>1 (Monday) - 7 (Sunday)</td>
    </tr>
    <tr>
      <td>DepTime</td>
      <td>actual departure time (minutes)</td>
    </tr>
    <tr>
      <td>CRSDepTime</td>
      <td>scheduled departure time (minutes)</td>
    </tr>
    <tr>
      <td>ArrTime</td>
      <td>actual arrival time (minutes)</td>
    </tr>
    <tr>
      <td>CRSArrTime</td>
      <td>scheduled arrival time (minutes)</td>
    </tr>
    <tr>
      <td>UniqueCarrier</td>
      <td>unique carrier code</td>
    </tr>
    <tr>
      <td>FlightNum</td>
      <td>flight number</td>
    </tr>
    <tr>
      <td>TailNum</td>
      <td>plane tail number</td>
    </tr>
    <tr>
      <td>ActualElapsedTime</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>CRSElapsedTime</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>AirTime</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>ArrDelay</td>
      <td>arrival delay, in minutes</td>
    </tr>
    <tr>
      <td>DepDelay</td>
      <td>departure delay, in minutes</td>
    </tr>
    <tr>
      <td>Origin</td>
      <td>origin IATA airport code</td>
    </tr>
    <tr>
      <td>Dest</td>
      <td>destination IATA airport code</td>
    </tr>
    <tr>
      <td>Distance</td>
      <td>in miles</td>
    </tr>
    <tr>
      <td>TaxiIn</td>
      <td>taxi in time, in minutes</td>
    </tr>
    <tr>
      <td>TaxiOut</td>
      <td>taxi out time in minutes</td>
    </tr>
    <tr>
      <td>Cancelled</td>
      <td>was the flight cancelled?</td>
    </tr>
    <tr>
      <td>CancellationCode</td>
      <td>reason for cancellation (A = carrier, B = weather, C = NAS, D = security)</td>
    </tr>
    <tr>
      <td>Diverted</td>
      <td>1 = yes, 0 = no</td>
    </tr>
    <tr>
      <td>CarrierDelay</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>WeatherDelay</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>NASDelay</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>SecurityDelay</td>
      <td>in minutes</td>
    </tr>
    <tr>
      <td>LateAircraftDelay</td>
      <td>in minutes</td>
    </tr>
  </tbody>
</table>

## 4. Data Pre-processing


