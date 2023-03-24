# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:51:18 2023

@author: Amina Muhammad
Samsung Capstone: Supervised Learning, 
Multi-Linear Regression (Eng Type Excluded)
"""

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import os
#import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')              # Turn off the warnings.
# %matplotlib inline


# Step 2: Import Dataset
path1 = 'C:/Users/amina/Downloads/Samsung_Capstone_Python/GES_DATA_ANALYSIS.xlsx'
ges1 = pd.read_excel(open(path1,'rb')) #data frame

shapeGes1 = ges1.shape
#print(ges1.head(3)) #checking first three rows of ges1 df


# Get the columns and rows for the first emission type: HC
gesNO = ges1[['B/P Ratio', 'Fuel LTO Cycle (kg)','NOx LTO Total Mass (g)']]

# # Option 1: Set NaN/missing values to zero
# gesNO = gesNO.replace(np.nan,0) 

# Option 2: Remove rows with NaN and and missing values
gesNO[gesNO==np.inf] = np.nan #remove NaN
gesNO = gesNO.dropna(axis=0) #to remove rows with missing values
# #print(gesHC.shape)

# 2 independent variables (x values)
BpNO = gesNO['B/P Ratio']
FuelLto = gesNO['Fuel LTO Cycle (kg)']

#dependent variable I think is better (y values)
LtoMassNO = gesNO['NOx LTO Total Mass (g)']

desc = gesNO.describe(include='all').T
#print(desc)


# Step 3: Define x and y
#get x and y values
x=gesNO.drop(['NOx LTO Total Mass (g)'],axis=1).values
#print(x)
y=LtoMassNO.values
# print(y)
# y=y/1000
# print(y)


# Step 4: Split dataset into training set and test set
from sklearn.model_selection import train_test_split
# choosing a 70-30 split where 70% of data if for training, 30% for testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
#80-20 was a better dataset split


# Step 5: Train the model on the training set
from sklearn.linear_model import LinearRegression as lr
ml=lr()
ml.fit(x_train,y_train)


# Step 6: Predict the test set results
y_pred = ml.predict(x_test)
lenY = len(y_pred)
#print(y_pred)


# Check if prediction is correct
# out1 = ml.predict([[2.64,85]])
# print(out1) 
# out1 = 1420.2 and y_actual = 823
# this is better than the value w Eng Type included, but still way off


# If things were looking good, the next steps would be:
    
# Step 7: Evaluate the model
# Step 8: Plot the results
# Step 9: Predicted Values

# Going to still plot y-pred and y-test

# Scatterplot
list2 = np.linspace(1,lenY,lenY)
plt.scatter(list2,y_pred,label='Predicted Y Values')
plt.scatter(list2,y_test,label='Test Y Values')

# Label Plot
plt.title('Scatterplot for Y_Pred vs Y_Test')
plt.legend()
plt.show()