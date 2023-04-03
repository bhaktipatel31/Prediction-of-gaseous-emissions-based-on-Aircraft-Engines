# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:37:59 2023

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
gesNO = ges1[['Eng Type','B/P Ratio', 'Fuel LTO Cycle (kg)','NOx LTO Total mass (g)']]
gesNO.to_excel('gesNO_test.xlsx')


gesTest3 = gesNO
#print(gesTest3)
print('NaN Count =', gesTest3.isna().sum().sum())


# ## Option 1: Set NaN/missing values to zero
# gesNO = gesNO.replace(np.nan,0) 
# ## print(gesNO.shape)

# ## Option 2: Remove rows with NaN and and missing values
# gesNO[gesNO==np.inf] = np.nan #remove NaN
# gesNO = gesNO.dropna(axis=0) #to remove rows with missing values
# print(gesNO.shape)


### Option 3: Median Imputation
gesNO['B/P Ratio'].fillna(gesNO['B/P Ratio'].median(), inplace=True)
gesNO['Fuel LTO Cycle (kg)'].fillna(gesNO['Fuel LTO Cycle (kg)'].median(), inplace=True)
gesNO['NOx LTO Total mass (g)'].fillna(gesNO['NOx LTO Total mass (g)'].median(), inplace=True)


# ### Option 4: Mean Imputation
# gesNO['B/P Ratio'].fillna(gesNO['B/P Ratio'].mean(), inplace=True)
# gesNO['Fuel LTO Cycle (kg)'].fillna(gesNO['Fuel LTO Cycle (kg)'].mean(), inplace=True)
# gesNO['NOx LTO Total mass (g)'].fillna(gesNO['NOx LTO Total mass (g)'].mean(), inplace=True)



# 3 independent variables (x values)
#dummy encoding for engine type
gesNO1 = pd.get_dummies(gesNO, columns=['Eng Type'])
print(gesNO1)
BpNO = gesNO1['B/P Ratio']
FuelLto = gesNO1['Fuel LTO Cycle (kg)']

#dependent variable (y values)
LtoMassNO = gesNO1['NOx LTO Total mass (g)']

desc = gesNO1.describe(include='all').T
#print(desc)


######## Outlier Treament Applied: IQR Method

### Choose either TF or MTF:

# ## Select rows based on condition: MTF
# gesNO2 = gesNO1[(gesNO1['Eng Type_MTF'] == 1)]
# #print(gesNO2)
# ## Drop the Eng Type_TF column from df
# gesNO2 = gesNO2.drop(['Eng Type_TF'],axis=1).values
# print(gesNO2)


### Select rows based on condition: TF
gesNO2 = gesNO1[(gesNO1['Eng Type_TF'] == 1)]
#print(gesNO2)
## Drop the Eng Type_MTF column from df
gesNO2 = gesNO2.drop(['Eng Type_MTF'],axis=1).values
#print(gesNO2)


## Detecting outliers using the Inter Quantile Range(IQR)
## (1) Doing this for LTO Mass for NOx
LtoMassNO2 = gesNO2[:,2]
outliers = []
def detect_outliers_iqr(LtoMassNO2):
    LtoMassNO2 = sorted(LtoMassNO2)
    q1 = np.percentile(LtoMassNO2, 25)
    q3 = np.percentile(LtoMassNO2, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in LtoMassNO2: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code
NOxMass_outliers = detect_outliers_iqr(LtoMassNO2)
print("Mass NOx Outliers from IQR method: ", NOxMass_outliers)



## Version 1: Dropping Outlier Values for Mass NOx
### Actually, not recommended. Going to do Version 2 for now.


## Version 2: Quantile Based Flooring & Capping for Mass NOx
ten_per = np.percentile(LtoMassNO2,10)
ninety_per = np.percentile(LtoMassNO2,90)
gesNO3 = np.where(gesNO2>ninety_per, ninety_per, gesNO2)
#print("Sample:", LtoMassNO2)
#print("New array:",gesNO3)

## Detecting outliers using the Inter Quantile Range(IQR)
## (2) Doing this for B/P Ratio for NOx
BpNO2 = gesNO3[:,0]
outliers2 = []
def detect_outliers_iqr(BpNO2):
    BpNO2 = sorted(BpNO2)
    q1_2 = np.percentile(BpNO2, 25)
    q3_2 = np.percentile(BpNO2, 75)
    # print(q1, q3)
    IQR2 = q3_2-q1_2
    lwr_bound_2 = q1_2-(1.5*IQR2)
    upr_bound_2 = q3_2+(1.5*IQR2)
    # print(lwr_bound, upr_bound)
    for i in BpNO2: 
        if (i<lwr_bound_2 or i>upr_bound_2):
            outliers2.append(i)
    return outliers2 # Driver code
BpNO2_outliers = detect_outliers_iqr(BpNO2)
print("B/P Ratio Outliers from IQR method: ", BpNO2_outliers)


## Version 2: Quantile Based Flooring & Capping for B/P Ratio
ten_per2 = np.percentile(BpNO2,10)
ninety_per2 = np.percentile(BpNO2,90)
gesNO4 = np.where(gesNO3>ninety_per, ninety_per, gesNO3)
#print("Sample:", LtoMassNO2)
#print("New array:",gesNO4)


## Detecting outliers using the Inter Quantile Range(IQR)
## (2) Doing this for Fuel LTO Cycle for NOx
FuelLto3 = gesNO3[:,1]
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
gesNO5 = np.where(gesNO4>ninety_per, ninety_per, gesNO4)
#print("Sample:", LtoMassNO2)
#print("New array:",gesNO5)



### Step 3: Define x and y
# #get x and y values
gesNO6 = np.delete(gesNO5,2,axis=1)
x = gesNO6
y = gesNO5[:,2]


### X and Y Values w/out Outlier Treatment
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

### Scatterplot
list2 = np.linspace(1,lenY,lenY)
plt.scatter(list2,y_pred,label='Predicted Y Values')
plt.scatter(list2,y_test,label='Test Y Values')
## plt.plot(list2,y_pred,color='red')

# ## B/P Ratio vs Y Pred and Y Test
# plt.scatter(x_test[:,0],y_pred,label='Predicted Y Values')
# plt.scatter(x_test[:,0],y_test,label='Test Y Values')
# ##plt.plot(x_test[:,0],y_pred,color='red')


### Label Plot
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

### For NOx
### Option 1: Using MTF and Setting NaN to Zero
    ## RSME =  1123.6010335231365
    ## R2 =  0.7996109267808469 ##Hmm, not too bad
    
### Option 2: Using TF and Setting NaN to Zero
    ## RSME =  1948.1633125729545
    ## R2 =  0.9135377559708425 ## Wow, this is really good
    
    

### Option 3: Using TF and Setting NaN to Median Imputation
    ## RSME =  1916.8952451825182
    ## R2 =  0.9139600902675283
    
    
### Option 4: Using TF and Setting NaN to Mean Imputation
    ## RSME =  2031.0980547618208
    ## R2 =  0.9030167535396607    
