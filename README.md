# ML-Practice

## Data Preprocessing-

1) Taking Care of Missing Data

Before starting we should have basic knowledge of Numpy, Pandas & sklearn.
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

imp.fit_transform(x_train)
```


2) Encoding Categorial Features

Since Machine Learning models are based on mathematical equation, therefore categorial features are encoded.

```python
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

y = enc.fit_transform(y)
```
But in case if there are categorial features in our independent variable by which we are going to predict dependent variable then this method won't help that much because suppose we have a categorial column Country and after encoding countries will be encoded with integers - 0,1,2... and so on so to take care of this we use 'OneHotEncoder' class. Let's call it dummy encoding.

```python
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder

x = onehotencoder.fit_transform(x).toarray()
```
