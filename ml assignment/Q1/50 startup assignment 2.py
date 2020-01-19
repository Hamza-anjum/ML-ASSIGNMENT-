# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:05:21 2020

@author: Dell 5530
"""

#Importing Libraries
 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
# Importing the dataset
 
dataset = pd.read_csv('50_Startups.csv') 
p = dataset.loc[dataset['State']=='New York'] 
print(p) 
y = p.iloc[:, -1].values 
q =np.arange(17) 
X = q.reshape(-1, 1) 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0) 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 

 
# Fitting Polynomial Regression to the dataset
 
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 3) 
X_poly = poly_reg.fit_transform(X) 
poly_reg.fit(X_poly, y) 
lin_reg_2 = LinearRegression() 
lin_reg_2.fit(X_poly, y) 

 
# Visualising the Polynomial Regression results
 
plt.scatter(X, y, color = 'red') 
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') 
plt.title('Startups Profit In New York (Polynomial Regression)') 
plt.xlabel('(City: New York)') 
plt.ylabel('Profit') 
plt.show() 

 
print('Startups Profit In New York (10)') 
print(regressor.predict([[10]])) 


 

 
#now florida turn ######
 

 
q = dataset.loc[dataset['State']=='Florida'] 
print(q) 
b = q.iloc[:, -1].values 
r =np.arange(16) 
a = r.reshape(-1, 1) 

 
from sklearn.model_selection import train_test_split 
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.20,random_state = 0) 

 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(a_train, b_train) 

 
# Fitting Polynomial Regression to the dataset
 
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 3) 
a_poly = poly_reg.fit_transform(a) 
poly_reg.fit(a_poly, b) 
lin_reg_2 = LinearRegression() 
lin_reg_2.fit(a_poly, b) 

 
# Visualising the Polynomial Regression results
 
plt.scatter(a, b, color = 'red') 
plt.plot(a, lin_reg_2.predict(poly_reg.fit_transform(a)), color = 'blue') 
plt.title('Startups Profit In Florida (Polynomial Regression)') 
plt.xlabel('(City: Florida)') 
plt.ylabel('Profit') 
plt.show() 

 
print('Startups Profit In Florida (10)') 
print(regressor.predict([[10]])) 
print("starting profit of newyork is less than florida ")
print("thanks to Allah....done it")


 

 