# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:27:07 2023

@author: Amina Muhammad
Samsung Capstone: Supervised Learning, 
Multi-Linear Regression (Eng Type Included)
"""


## 3/30/23: This code does have the outlier treatment applied.


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
path1 = 'C:/Users/amina/Downloads/Samsung_Capstone_Python/ges_original.xlsx'
ges1 = pd.read_excel(open(path1,'rb')) #data frame

shapeGes1 = ges1.shape
#print(ges1.head(3)) #checking first three rows of ges1 df


# Get the columns and rows for the first emission type: NO
gesCO = ges1[['Eng Type','B/P Ratio', 'Fuel LTO Cycle (kg)','CO LTO Total Mass (g)']]
gesCO.to_excel('gesCO_test.xlsx')


gesTest3 = gesCO
#print(gesTest3)
print('NaN Count =', gesTest3.isna().sum().sum())


# ### Option 1: Set NaN/missing values to zero
# gesCO = gesCO.replace(np.nan,0) 
# #print(gesCO.shape)

### Option 2: Remove rows with NaN and and missing values
# gesCO[gesCO==np.inf] = np.nan #remove NaN
# gesCO = gesCO.dropna(axis=0) #to remove rows with missing values
#print(gesCO.shape)


### Option 3: Median Imputation
gesCO['B/P Ratio'].fillna(gesCO['B/P Ratio'].median(), inplace=True)
gesCO['Fuel LTO Cycle (kg)'].fillna(gesCO['Fuel LTO Cycle (kg)'].median(), inplace=True)
gesCO['CO LTO Total mass (g)'].fillna(gesCO['CO LTO Total mass (g)'].median(), inplace=True)


# ### Option 4: Mean Imputation
# gesCO['B/P Ratio'].fillna(gesCO['B/P Ratio'].mean(), inplace=True)
# gesCO['Fuel LTO Cycle (kg)'].fillna(gesCO['Fuel LTO Cycle (kg)'].mean(), inplace=True)
# gesCO['HC LTO Total mass (g)'].fillna(gesCO['CO LTO Total mass (g)'].mean(), inplace=True)


# 3 independent variables (x values)
#dummy encoding for engine type
gesCO1 = pd.get_dummies(gesCO, columns=['Eng Type'])
print(gesCO1)
BpCO = gesCO1['B/P Ratio']
FuelLto = gesCO1['Fuel LTO Cycle (kg)']

#dependent variable (y values)
LtoMassCO = gesCO1['CO LTO Total Mass (g)']

desc = gesCO1.describe(include='all').T
#print(desc)


######## Outlier Treament Applied: IQR Method

### Choose either TF or MTF:


# ## Select rows based on condition: MTF
# gesCO2 = gesCO1[(gesCO1['Eng Type_MTF'] == 1)]
# #print(gesCO2)
# ## Drop the Eng Type_TF column from df
# gesCO2 = gesCO2.drop(['Eng Type_TF'],axis=1).values
# print(gesCO2)


## Select rows based on condition: TF
gesCO2 = gesCO1[(gesCO1['Eng Type_TF'] == 1)]
#print(gesCO2)
## Drop the Eng Type_MTF column from df
gesCO2 = gesCO2.drop(['Eng Type_MTF'],axis=1).values
#print(gesCO2)


## Detecting outliers using the Inter Quantile Range(IQR)
## (1) Doing this for LTO Mass for HC
LtoMassCO2 = gesCO2[:,2]
outliers = []
def detect_outliers_iqr(LtoMassCO2):
    LtoMassCO2 = sorted(LtoMassCO2)
    q1 = np.percentile(LtoMassCO2, 25)
    q3 = np.percentile(LtoMassCO2, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in LtoMassCO2: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code
COMass_outliers = detect_outliers_iqr(LtoMassCO2)
print("Mass CO Outliers from IQR method: ", COMass_outliers)



## Version 1: Dropping Outlier Values for Mass NOx
### Actually, not recommended. Going to do Version 2 for now.


## Version 2: Quantile Based Flooring & Capping for Mass NOx
ten_per = np.percentile(LtoMassCO2,10)
ninety_per = np.percentile(LtoMassCO2,90)
gesCO3 = np.where(gesCO2>ninety_per, ninety_per, gesCO2)
#print("Sample:", LtoMassCO2)
#print("New array:",gesCO3)

## Detecting outliers using the Inter Quantile Range(IQR)
## (2) Doing this for B/P Ratio for NOx
BpCO2 = gesCO3[:,0]
outliers2 = []
def detect_outliers_iqr(BpCO2):
    BpCO2 = sorted(BpCO2)
    q1_2 = np.percentile(BpCO2, 25)
    q3_2 = np.percentile(BpCO2, 75)
    # print(q1, q3)
    IQR2 = q3_2-q1_2
    lwr_bound_2 = q1_2-(1.5*IQR2)
    upr_bound_2 = q3_2+(1.5*IQR2)
    # print(lwr_bound, upr_bound)
    for i in BpCO2: 
        if (i<lwr_bound_2 or i>upr_bound_2):
            outliers2.append(i)
    return outliers2 # Driver code
BpCO2_outliers = detect_outliers_iqr(BpCO2)
print("B/P Ratio Outliers from IQR method: ", BpCO2_outliers)


## Version 2: Quantile Based Flooring & Capping for B/P Ratio
ten_per2 = np.percentile(BpCO2,10)
ninety_per2 = np.percentile(BpCO2,90)
gesCO4 = np.where(gesCO3>ninety_per, ninety_per, gesCO3)
#print("Sample:", LtoMassCO2)
#print("New array:",gesCO4)


## Detecting outliers using the Inter Quantile Range(IQR)
## (2) Doing this for Fuel LTO Cycle for NOx
FuelLto3 = gesCO3[:,1]
outliers3 = []
def detect_outliers_iqr(FuelLto3):
    FuelLto3 = sorted(FuelLto3)
    q1_3 = np.percentile(FuelLto3, 25)
    q3_3 = np.percentile(FuelLto3, 75)
    # print(q1, q3)
    IQR3 = q3_3-q1_3
    lwr_bound_3 = q1_3-(1.5*IQR3)
    upr_bound_3 = q3_3+(1.5*IQR3)
    # print(lwr_bound, upr_bound)
    for i in FuelLto3: 
        if (i<lwr_bound_3 or i>upr_bound_3):
            outliers3.append(i)
    return outliers3 # Driver code
FuelLto3_outliers = detect_outliers_iqr(FuelLto3)
print("Fuel LTO Cycle Outliers from IQR method: ", FuelLto3_outliers)


## Version 2: Quantile Based Flooring & Capping for B/P Ratio
ten_per3 = np.percentile(FuelLto3,10)
ninety_per3 = np.percentile(FuelLto3,90)
gesCO5 = np.where(gesCO4>ninety_per, ninety_per, gesCO4)
#print("Sample:", LtoMassNO2)
#print("New array:",gesCO5)



### Step 3: Define x and y
# #get x and y values
gesCO6 = np.delete(gesCO5,2,axis=1)
x = gesCO6
y = gesCO5[:,2]



# x=gesNO.drop(['NOx LTO Total mass (g)'],axis=1).values
# #print(x)
# y=LtoMassNO.values
# # print(y)
# # y=y/1000
# # print(y)


## Step 4: Split dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
# Initially chose a 70-30 split where 70% of data if for training, 30% for testing
#80-20 was a better dataset split


## Step 5: Train the model on the training set
from sklearn.linear_model import LinearRegression as lr
ml=lr()
ml.fit(x_train,y_train)


## Step 6: Predict the test set results
y_pred = ml.predict(x_test)
lenY = len(y_pred)
#print(y_pred)


# # Check if prediction is correct
# # out1 = ml.predict([[2.64,85]])
# # print(out1) 
# # out1 = 1420.2 and y_actual = 823



# # If things were looking good, the next steps would be:
    
# # Step 7: Evaluate the model
# # Step 8: Plot the results
# # Step 9: Predicted Values

# Going to still plot y-pred and y-test

# Scatterplot
list2 = np.linspace(1,lenY,lenY)
plt.scatter(list2,y_pred,label='Predicted Y Values')
plt.scatter(list2,y_test,label='Test Y Values')

# Label Plot
plt.title('Scatterplot for Y_Pred vs Y_Test')
plt.legend()
plt.show()

# Get the RMSE and R2 Score
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
from sklearn.metrics import r2_score
r2score = r2_score(y_test,y_pred)
#r2score = ml.score(x_test,y_test) #another way to calculate R2

print('RSME = ', rmse)
print('R2 = ', r2score)

### Changing the NaN/missing values to zero performed better than removing them

### For CO
### Option 1: Using MTF and Setting NaN to Zero
    ## RSME =  1649.2111257196339
    ## R2 =  0.5242535710469363 ## Eh, not the best...
    
### Option 2: Using TF and Setting NaN to Zero
    ## RSME =  2792.1129775419454
    ## R2 =  0.15108316037154013 ## This is horrible...