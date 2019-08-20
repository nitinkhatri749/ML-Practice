# ML-Practice

Entire work is done on Spyder(Scientific Python Development Environment). So many Introduction videos are available on [Spyder](https://www.youtube.com/results?search_query=spyder+introduction). 

###### Prerequisites-
One should have prior knowledge of Python and it's libraries -<br/>
Numpy (Python package used for Scientific Computing) <br/>
Matplotlib (Data Visualization library) <br/>
Pandas(used for data maniulation and importing data) <br/>
scikit-learn/sklearn (Python library for machine learning & predictive modeling)<br/>
And also basic knowledge of Regression, Classification, Supervised Ml, Unsupervised ML and differences.

## 1) Data Preprocessing-
Before starting we should have basic knowledge of Numpy, matplotlib, Pandas & sklearn which has been covered(or still working on) in other [repositories](https://github.com/nitinkhatri749).



### a) Importing the Libraries & Loading the data

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
##### Note - 
The data to be imported should be in cwd(Current Working Directory) and we have to specify cwd in spyder, and to do that-

```python
import os
os.getcwd() # To know about current working direcctory
os.chdir('Desired path to be changed') # to change the working directory
```


### b) Taking Care of Missing Data/ Impute the Data

Before starting with ML, we have to learn how to prepare data and we call it Data_Preprocessing.<br/>
Data Preprocessing is used to take care of missing values in our data.<br/>
Sometimes we may have missing data in our dataset so we have two things to take care of-.<br/>
a) We can simply remove those rows where data is missing but this could be dangerous as they may have crucial data.<br/>
b) We can fill the missing values with the mean of that column values.<br/>
So we proceed with 2nd point everytime, 1st point was just to give the possibility.<br/>
We use Preprocessing module of sklearn to do this job.<br/>

```python
from sklearn.preprocessing import Imputer  
imp = Imputer(missing_values ='Nan',  strategy='mean', axis=0)  
imp.fit_transform(x)
```


### c) Encoding Categorial Features

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
# a-independent feature matrix column to be encoded or there can be more than 1 column
```
##### Note -
1) We don't need dummy encoding(OneHotEncoder) on dependent variable, since Machine Learning Model will know that it's a category.<br/>
2) When we are adding dummy variables, then only n-1 features will be included(will be covered later in Multiple Linear Regression) in our dataset when fitting the model where n is number of features.



### d) Splitting the dataset into the Training set and Test set

Training set- on which we build Machine Learning model.<br/>
Test set-a set on which we test the performane of this machine learning model.<br/>
and the performance on the test set shouldn't be different from the performance on the training set.
```python
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split (newer versions of anaconda)
```


### e) Feature Scaling

Features scaling is very important in machine learning. Most of the time our independent features are not on the same scale. This will cause some issues in your Machine Learning models. It's because a lot of machine learning models are based on Euclidean distance. The Euclidean distance between two points is the square root of the sum of the square coordinates.

```python
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
```
We are doing feature scaling on dummy variables too because algorithm converge much faster.<br/>
fit & transform training set but only transform test set.

##### Note-
we don't need to apply feature scaling to y(y_train, y_test--dependent variable) in case of classification.<br/>
But we will see for regression when the dependent variable is huge range of values, we will need to apply feature scaling to dependent variable y as well



## 2) Simple Linear Regression- 
We have a data of Past Employees with Salaries and Years of experience. We have to build a Simple Linear Model using the same data and predict the salaries and find the correlations between dependent variable(Salary) and indepdendent variable(Years of Experience). 

```python
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Checking cwd
os.getcwd()
os.chdir('E:\Machine Learning A-Z\Part 2 - Regression\Section 4 - Simple Linear Regression')

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

we'll train simple linear regression model that will learn the correlations between dependent and independent variables and then fit simple linear regression to training set.<br/>
Simple linear regressor learnt the correlations of the training set i.e. learnt the relation between independent and dependent variable.<br/>
In my pc the Salary_Data.csv was in 'E:\Machine Learning A-Z\Part 2 - Regression\Section 4 - Simple Linear Regression' path so I changed the path. Feature Scaling was not required as Python library takes care of it by itself.



## 3) Multiple Linear Regresion
Suppose we have data of 50 Startups which include- R&D Spend, Administration,	Marketing Spend, State,	Profit. We have to analyse the data and figure out If we want to invest in one of the startup then which startup should we be funding so that we get good returns and which one of the above features contribute maximum into profit.

We want to build a model to see if there is some linear dependencies between all these variables.<br/>
Before we get started we should have knowledge about P-value.

Before starting, we should have knowledge of the following topics.-

* P- value - 
P value is a statistical measure that helps us determine whether or not their hypotheses are correct.<br/>
Usually, if the P value of a data set is below a certain pre-determined amount (like, for instance, 0.05), 
we will reject the "null hypothesis" of their experiment - in other words, they'll rule out the hypothesis that the variables of their experiment had no meaningful effect on the results. 

* Building a model- We'll be doing it by Backward Elimination
Let's say we have lots of columns and not all of these columns are potential predictors for a dependent variable, 
so we need to decide which ones to keep and wich ones to throw out the columns or get rid of the data.<br/>
Why can't we just use everything in our model, 2 reasons-.<br/>
1) garbage in = garbage out.<br/>
2) Explain those variables/features.<br/>

To construct a model I'm using [Backward Elimination](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Step-by-step-Blueprints-For-Building-Models.pdf)

* Backward Elimination -<br/>
STEP 1 - Select a significance level to stay in the model(eg - sl: 0.5).<br/>
STEP 2 - Fit the model with all possible predictors.<br/>
STEP 3- Consider the predictor with the heighest P-value. If P > SL, go to STEP-4 otherwise go to FIN.<br/>
STEP 4- Remove the predicttor.<br/>
STEP 5- Fit the model without this variable (Refit the model) & GO BACK to STEP -3.<br/>
FIN : Model is Ready. 

H0 - Opposite of what we are testing.<br/>
H1 - The claim we are testing

H0(null hypothesis) = the dependent and the independent variables are not associated.<br/>
H1(alternative hypothesis) = They both are associated.<br/>
Here H1 has to prove his statement that they both(dependent and independent variables) are associated. So in order to prove that he need to show that the p-value is less than the significant level which is 0.05.<br/>
In backward elimination we need to keep only the predictors whose p-value is less than the significant level. So we are eliminating the predictors whose p-values are higher than the significant level, which in turn helps us to retain only the predictors whose p-value is less than 0.05.

for example- p-value at 84% says that the probability of null hypothesis is true is 84% (and 16% that alternative hypothesis is true, which is positive ie. the data is good)

```python
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
```

### Building optimal model using backward elimination
Now if we want to find the features which contribute the most then we'll use P-values. Having Significance value decided(0.05) we will only keep those variables/features whose value is less than 0.05 (SL) & then we eliminate the features with P-value greater than 0.05. In our data we find that the 'R&D Spend' contribute the most. With higher R&D higher the profit, so one should invest in the startup with higher 'R&D Spend'. We'll use statsmodels module to get the P-values of all the features and other important terms.

since pthon library include x0 = 1 (b0x0 + b1x1  and so on) and statsmodel library doesn't so we need to add a column of ones in independent features matrix(in the begining/ index=0)
```python
import statsmodels.formula.api as sm
x = np.append(np.ones((50,1),dtype=int), x, axis = 1)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
# to fit usng stat library we will use OLS(Ordinary least square)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
```


## 4) Polynomial Regression-
We are building regresors which are not linear.<br/>
We just add polynomial term in the multiple linear regression equation.

```python
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
```
Feature scaling was not required as linearmodel module takes care of that part by itself.

## 5) Support Vector Regression(SVR)

SVR is a type of support vector machine that supports linear and non-linear regression.
I have found some useful videos on SVR - [Video-1](https://www.youtube.com/watch?v=Y6RRHw9uN9o), [Video-2](https://www.youtube.com/watch?v=xLkk6MUrvrw)

SVR uses the same basic idea as Support Vector Machine (SVM), a classification algorithm, but applies it to predict real values rather than a class.
SVR is a type of support vector machine that supports linear and non-linear regression.

* Basics Of SVM(Support Vector Machine)-<br/>
-SVM can be used to do binary classification(the task of classifying the elements of a given set into two groups (predicting which group each one belongs to)).<br/>
-SVM finds a hyper-plane (line in 2d, plane in 3d, etc) that separates its training data in such a way that the distance between the hyper plane and the closest points from each class is maximized.<br/>
-once SVM finds this hyper-plane, you can classify new data points by seeing which side of this hyper-plane they land on.<br/>
-SVM can only be used on data that is linearly separable (i.e. a hyper-plane can be drawn between the two groups).<br/>
-Fear not though, as a common way to make data linearly separable is to map it to a higher dimension.<br/>
-SVMs simply tell you what class a new data point falls in, not the probability that it's in that class(disadvantage).<br/>

One of the prominent kernels for SVR is Gaussian kernel.

```python
# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

Feature Scaling becomes important in this case as this library doesn't include feature scaling in it, so in previous cases we skipped feature scaling, but not here.





