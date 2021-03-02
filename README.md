# TITANIC--Analysis-and-Prediction
Titanic: Machine Learning from Disaster. (The notebook is written in Python.The tragic disaster of 1912. The TItanic a ship that sank leading to the deaths of more than 1500 passengers and crew. "Of the 2,240 passengers and crew on board, more than 1,500 lost their lives"
![image](https://user-images.githubusercontent.com/71189710/109598447-51013600-7ae7-11eb-8494-cae5c925b5c7.png)
In this notebook, based on real data about the disaster, our task is to predict whether a person survived the tragedy or not.

Libraries
Firs, we need to install necessary tools. We'll start by importing libraries:

import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
print('Setup complete')
Load Data

#The next step is to load the data

train = pd.read_csv('../input/titanic/train.csv') 
test = pd.read_csv('../input/titanic/test.csv')
train.head()
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450
