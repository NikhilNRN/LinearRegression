#Loading data and importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
advert = pd.read_csv('Advertising.csv')
advert.head()
advert.info()

#Made a change here

#Remove the index column 
#We remove this as it is redundant data
advert.columns
advert.drop(['Unnamed: 0'], axis=1, inplace = True)
advert.head()

#Data Analysis (EDA)
import seaborn as sns
sns.distplot(advert.sales)
sns.distplot(advert.newspaper)
sns.distplot(advert.radio)
sns.distplot(advert.TV)

#Explored relationships between predictors and responses
sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, aspect=0.7, kind='reg']
advert.TV.corr(advert.sales)
advert.corr()
sns.heatmap(advert.corr(), annot=True)

#Make a simple linear regression model
X = advert[['TV']]
X.head()
print(type(X))
print(X.shape)

y = advert.sales
print(type(y))
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#Interpret coefficients of model
print(linreg.intercept_)
print(linreg.coef_)

#Make predictions with our model
y_pred = linreg.predict(X_test)
y_pred[:5]

#Evaluate model using evaluation metrics
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]

print((10+0+20+10) / 4)
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))

print((10**2 + 0 + +20**2 + 10**2) / 4)
print(metrics.mean_squared_error(true, pred))

print(np.sqrt((10**2 + 0 + +20**2 + 10**2) / 4)
print(np.sqrt(metrics.mean_squared_error(true, pred)))

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))