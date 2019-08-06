# ML-Practice

# Data Preprocessing-
Before starting we should have basic knowledge of Numpy, Pandas & sklearn.
Before starting with ML, we have to learn how to prepare data and we call it Data_Preprocessing.
Data Preprocessing is used to take care of missing values in our data. 
Sometimes we may have missing data in our dataset so we have two options to take care of -
1) We can simply remove those rows where data is missing but this could be dangerous as they may have crucial data.
2) We can fill the missing values with the mean of that column values.
So we proceed with 2nd point everytime, 1st point was just to give the possibility.

We use Preprocessing module of sklearn to do this job.

1) from sklearn.preprocessing import Imputer
2) imp = Imputer(missing_values ='Nan',  strategy='mean', axis=0)
3) imp.fit_transform(x_train)


# Encoding Categorial
