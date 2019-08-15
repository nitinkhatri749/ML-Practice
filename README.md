# ML-Practice

Entire work is done on Spyder(Scientific Python Development Environment). So many Introduction videos are available on [Spyder](https://www.youtube.com/results?search_query=spyder+introduction). 

###### Prerequisites-
One should have knowledge of - Python and it's libraries - Numpy (Python package used for Scientific Computing), Matplotlib (Data Visualization library), Pandas(used for data maniulation and importing data), scikit-learn/sklearn (Python library for machine learning & predictive modeling)

## Data Preprocessing-
Before starting we should have basic knowledge of Numpy, matplotlib, Pandas & sklearn which has been covered(or still working on) in other [repositories](https://github.com/nitinkhatri749).



### 1) Importing the Libraries & Loading the data

Data needs to be numeric and stored as Matrices. NumPy is a Python package which stands for 'Numerical Python'. It is the core library for scientific computing. Matplotlib is an amazing visualization library in Python for 2D plots of arrays. Matplotlib is a multi-platform data visualization library built on NumPy arrays. Pandas is a python library for data manipulation and analysis. We can load our data using pandas.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
To read a csv file-
```python
dataset = pd.read_csv('data.csv')
```


### 2) Taking Care of Missing Data/ Imputr the Data

Before starting with ML, we have to learn how to prepare data and we call it Data_Preprocessing.
Data Preprocessing is used to take care of missing values in our data. 
Sometimes we may have missing data in our dataset so we have two things to take care of -
a) We can simply remove those rows where data is missing but this could be dangerous as they may have crucial data.
b) We can fill the missing values with the mean of that column values.
So we proceed with 2nd point everytime, 1st point was just to give the possibility.
We use Preprocessing module of sklearn to do this job.

```python
from sklearn.preprocessing import Imputer  
imp = Imputer(missing_values ='Nan',  strategy='mean', axis=0)  
imp.fit_transform(x)
```


### 3) Encoding Categorial Features

Since Machine Learning models are based on mathematical equation, therefore categorial features are encoded.

```python
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
```
But in case if there are categorial features in our independent variable by which we are going to predict dependent variable then this method won't help that much because suppose we have a categorial column Country and after encoding countries will be encoded with integers - 0,1,2... and so on so to take care of this we use 'OneHotEncoder' class. Let's call it dummy encoding.

* Encoding the Independent Variable/ Creating dummy variables
```python
from sklearn.preprocessing import OneHotEncoder
enc_x = LabelEncoder()
x[:,a] = enc.fit_transform(x[:,a]) 
onehotencoder = OneHotEncoder(categorical_features=[a])
x = onehotencoder.fit_transform(x).toarray()
# a-independent feature matrix column to be encoded or there can be more than 1 collumn
```
##### Note -
1) We don't need dummy encoding(OneHotEncoder) on dependent variable, since Machine Learning Model will know that it's a category.
2) When we are adding dummy variables, then only n-1 features will be included(will be covered later in Multiple Linear Regression) in our dataset when fitting the model where n is number of features.



### 4) Splitting the dataset into the Training set and Test set
```python
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split (newer versions of anaconda)'''


### 5) Feature Scaling

Features scaling is very important in machine learning. Most of the time our independent features are not on the same scale. This will cause some issues in your Machine Learning models. It's because a lot of machine learning models are based on Euclidean distance. The Euclidean distance between two points is the square root of the sum of the square coordinates.

```python
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)```
