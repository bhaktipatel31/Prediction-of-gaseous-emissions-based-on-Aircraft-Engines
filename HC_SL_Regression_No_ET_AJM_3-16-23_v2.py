# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:23:18 2023

@author: Amina Muhammad
Samsung Capstone: Supervised Learning, 
Multi-Linear Regression (Eng Type Excluded)"""



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


# Get the columns and rows for each emission type
gesALL = ges1[['B/P Ratio','HC LTO Total Mass (g)', 'CO LTO Total Mass (g)', 'NOx LTO Total Mass (g)','Fuel LTO Cycle (kg)']]
## kept getting this error: KeyError: "['Total HC El LTO g/kg'] not in index" 
## when trying to bring in 'Total HC El LTO (g/kg)'

gesALL[gesALL==np.inf] = np.nan #remove NaN

gesALL = gesALL.dropna(axis=0) #to remove rows with missing values
#print(gesALL.shape)

# 4 independent variables (x values)

#dependent variable (y values)
FuelLto = gesALL['Fuel LTO Cycle (kg)']

desc = gesALL.describe(include='all').T
#print(desc)


# Step 3: Define x and y
#get x and y values
x=gesALL.drop(['Fuel LTO Cycle (kg)'],axis=1).values
#print(x)
y=FuelLto.values
#print(y)
# #convert y values from kg to g
# y=y*1000
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

out1 = ml.predict([[2.64,823,2612,630]])
print(out1) 
# out1 = 239.02 and y_actual = 85
# However, if you compare y_pred to y_test, the values are not too far off.

### Comments: 
# Doing the 3 emission LTO total mass and the B/P ratio as the independent variables
# (4 total) gave me the best predictions. And none of the predictions were negative
# like in the other two cases. I think this is the best bet, at least so far.

# If things were looking good, the next steps would be:
    
# Step 7: Evaluate the model
# Step 8: Plot the results
# Step 9: Predicted Values

