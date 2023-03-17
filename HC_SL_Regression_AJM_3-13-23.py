# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:23:18 2023

@author: Amina Muhammad
Samsung Capstone: Supervised Learning, 
Multi-Linear Regression (Eng Type Included)"""



# Step 1: Import Libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import os
#import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')                  # Turn off the warnings.
# %matplotlib inline


# Step 2: Import Dataset
path1 = 'C:/Users/amina/Downloads/Samsung_Capstone_Python/GES_DATA_ANALYSIS.xlsx'
ges1 = pd.read_excel(open(path1,'rb')) #data frame

shapeGes1 = ges1.shape
#print(ges1.head(3)) #checking first three rows of ges1 df


# Get the columns and rows for the first emission type: HC
gesHC = ges1[['Eng Type','B/P Ratio', 'Fuel LTO Cycle (kg)','HC LTO Total Mass (g)']]
## kept getting this error: KeyError: "['Total HC El LTO g/kg'] not in index" 
## when trying to bring in 'Total HC El LTO (g/kg)'

gesHC[gesHC==np.inf] = np.nan #remove NaN

gesHC = gesHC.dropna(axis=0) #to remove rows with missing values
#print(gesHC.shape)

# 3 independent variables (x values)
EngTypeHC = gesHC['Eng Type'] 
BpHC = gesHC['B/P Ratio']
FuelLto = gesHC['Fuel LTO Cycle (kg)']

#dependent variable I think is better (y values)
LtoMassHC = gesHC['HC LTO Total Mass (g)']

desc = gesHC.describe(include='all').T
#print(desc)

#dummy encoding for engine type
gesHC1 = pd.get_dummies(gesHC, columns=['Eng Type'])
#print(gesHC1)

desc2 = gesHC1.describe(include='all').T
#print(desc2)


# Step 3: Define x and y
#get x and y values
x=gesHC1.drop(['HC LTO Total Mass (g)'],axis=1).values
#print(x)
y=LtoMassHC.values
# print(y)
# y=y/1000
# print(y)


# Step 4: Split dataset into training set and test set
from sklearn.model_selection import train_test_split
# choosing a 70-30 split where 70% of data if for training, 30% for testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)


# Step 5: Train the model on the training set
from sklearn.linear_model import LinearRegression as lr
ml=lr()
ml.fit(x_train,y_train)


# Step 6: Predict the test set results
y_pred = ml.predict(x_test)
#print(y_pred)

out1 = ml.predict([[2.64,85,0,1]])
print(out1) 
# out1 = 2167.2 and y_actual = 823
# it is not predicting well at all...


# If things were looking good, the next steps would be:
    
# Step 7: Evaluate the model
# Step 8: Plot the results
# Step 9: Predicted Values

