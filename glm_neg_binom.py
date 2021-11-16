# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:03:05 2021

@author: willis
"""
#########
#IMPORTS#
#########
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

##############
#LOADING DATA#
##############
#changes to project directory
os.chdir(r'D:\BEST\FOLDER\EVER')
#loads data from csv.
df = pd.read_csv('marketing_campaign_data.csv')
#creates copy of original dataset
df_orig = df
###############
#DATA CLEANING#
###############
'''
EDUCATION:
    needs label encoding
'''
# numerical representation of education
def edu_clean(barg):
    i = -1
    if barg == 'Basic':
        i = 0
    elif barg == 'Graduation':
        i = 1
    elif barg == '2n Cycle':
        i = 2
    elif barg == 'Master':
        i = 2
    elif barg == "PhD":
        i = 3
    else:
        i = i
    return  i
# applies function to dataframe
df['edu_cat'] = df['Education'].apply(edu_clean)
'''
AGE:
    There was no variable for age. As remedy the customer's birth year was
    subtracted from the year 2021.
'''
# Uses the Year_Birth column to calculate customers age
df['age'] = 2021 - df['Year_Birth']
'''
MARITAL STATUS:
    needs label encoding
'''
# numerical representation of marital status
def mari_clean(carg):
    i = -1
    if carg == 'Single': 
        i = 0
    elif carg == 'Divorced':
        i = 1
    elif carg == 'Widow':
        i = 2
    elif carg == 'Together':
        i = 3
    elif carg == "Married":
        i = 4
    else:
        i = i
    return i
# applies function to dataframe
df['mari_status'] = df['Marital_Status'].apply(mari_clean)
'''
NUMBER OF CHILDREN:
    Kids and Teens are split in this dataset.  There are added together to 
    create a count of the number of children in a hh.
    Darg: (ARG)number of "kids"
    Farg: (ARG)number of "teens"
    RETURNS: sum of darg and farg
'''
def youth_calc(darg, farg):
    i = darg + farg
    return i
#applies function of df
df['youth_calc'] = df.apply(lambda x: youth_calc(x['Kidhome'], x['Teenhome']), axis=1)
#dropping rows with missing values for two_person_hh_di which was stored as -1
df = df.drop(df[df.mari_status < 0].index)
#renaming income variable
df['inc'] = df['Income']
#combining advert engagement
df['advert_engagement'] = df['AcceptedCmp5'] + df['AcceptedCmp4'] + df['AcceptedCmp3'] + df['AcceptedCmp2'] + df['AcceptedCmp1']
#these variables have no information about them, and were dropped due to lack of interpretability
df = df.drop(['AcceptedCmp5', 'AcceptedCmp4', 'AcceptedCmp3', 'AcceptedCmp2', 'AcceptedCmp1', 'Response', 'Complain','Z_Revenue', 'Z_CostContact', 'MntGoldProds'], axis = 1)
#separating working variables
x1 = df[df.columns[8:25]]
'''
GBMs are sensitive to outliers in the DV/Target. As such, observations 
distinct from the larger distribution are culled from the dataset.
'''
#plotting possible targets
plt.figure(figsize = (20,15))
plt.subplot(1,5,1)
sns.boxplot(data = x1['MntWines'], color='orange')
plt.subplot(1,5,2)
sns.boxplot(data = x1['MntFruits'], color='purple')
plt.subplot(1,5,3)
sns.boxplot(data = x1['MntMeatProducts'], color='brown')
plt.subplot(1,5,4)
sns.boxplot(data = x1['MntFishProducts'], color='green')
plt.subplot(1,5,5)
sns.boxplot(data = x1['MntSweetProducts'], color='red')
plt.show()
'''
From the boxplots plotted above it appears that MntMeatProducts and 
MntSweetProducts have distinct breaks in their tails. They are also all 
severely postively skewed. 

MntMeatProducts > 1500 were dropped
MntSweetProdcuts > 250 were dropped
'''
#dropping rows with extreme values for MntMeatProducts.
x1 = x1.drop(x1[x1.MntMeatProducts > 1500].index)
#dropping rows with extreme values for MntSweetProducts.
x1 = x1.drop(x1[x1.MntSweetProducts > 250].index)
#plotting age and inc to check for outliers
plt.figure(figsize = (20,15))
plt.subplot(1,2,1)
sns.boxplot(data = x1['age'], color='orange')
plt.subplot(1,2,2)
sns.boxplot(data = x1['inc'], color='purple')
plt.show()
'''
The boxplots constructed above show that there are some extreme outliers in 
age and inc. These variables were effectively limited to their IQR.
Age > 100
Income > 140,000
'''
#dropping rows with extreme values for income.
x1 = x1.drop(x1[x1.inc > 140000].index)
#dropping rows with extreme values for age.
x1 = x1.drop(x1[x1.age > 100].index)
'''
Education and marital status are measured categorically and have to be broken 
out into dummy variables
*I SHOULD DO THIS EARLIER!*
'''
#education dummies
edu_dum = pd.get_dummies(x1['edu_cat'])
edu_dum = edu_dum.rename(columns={0: 'basic_edu', 1: 'graduation_edu', 2: 'master_edu', 3: 'doc_edu'})
x1 = pd.concat([x1, edu_dum], axis=1)
x1 = x1.drop(['edu_cat'], axis=1)
#marital status dummies
mari_dum = pd.get_dummies(x1['mari_status'])
mari_dum = mari_dum.rename(columns={0: 'single', 1: 'divorced', 2: 'widow', 3: 'together', 4: 'married'})
x1 = pd.concat([x1, mari_dum], axis=1)
x1 = x1.drop(['mari_status'], axis=1)
x1 = x1.drop(['basic_edu', 'single'], axis=1)

'''
I removed all observations without complete informations
'''
x1 = x1.dropna(axis=0, how='any')
'''
To simplify life I am creating a new variable that will be the regressors' 
final target it is equal to the log of the sum of the 5 amount features.
doing it here just affixes it to the end of the df and makes future calls easier    
'''
#log amount target
x1['total_amount'] = x1['MntWines']+x1['MntFruits']+x1['MntMeatProducts']+x1['MntFishProducts']+x1['MntSweetProducts']
#checking the identity of the sum
sum_total_amount = x1['MntWines']+x1['MntFruits']+x1['MntMeatProducts']+x1['MntFishProducts']+x1['MntSweetProducts']
#checking what a square root transformation does
sqrt_total_amount = np.sqrt(x1['MntWines']+x1['MntFruits']+x1['MntMeatProducts']+x1['MntFishProducts']+x1['MntSweetProducts'])
#dropping individual amount features
x1 = x1.drop(['MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 'MntSweetProducts'], axis=1)
#plotting target for safety check
plt.figure(figsize=(20,15))
sns.displot(x1['total_amount'])
plt.show
#correlation matrix just for fun
def correlation_heatmap(dataframe,l,w):
    correlation = dataframe.corr()
    plt.figure(figsize=(l,w))
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
    plt.title('Correlation between different features')
    plt.show();
correlation_heatmap(x1, 30,15)
################
#SPLITTING DATA#
################
#checking target column location for feature/target split
print(x1.columns.get_loc("total_amount"))
#gets column values and separates them into features and target
dataset = x1.values
X = dataset[:,0:17]
Y = dataset[:,17]
# split data into train and test sets
seed = 2448 #seed for replication
test_size = 0.30 #train on 70% of observations test on 30%
#first split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

y_train_df = pd.DataFrame(y_test)
y_train_df = y_train_df.rename(columns={0: 'amount'})

df_train_x = pd.DataFrame(X_train)
df_train_xy = pd.concat([df_train_x, y_train_df], axis=1)

poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

df_train_xy['MNT_LAMBDA'] = poisson_training_results.mu

df_train_xy['AUX_OLS_DEP'] = df_train_xy.apply(lambda x: ((x['amount'] - x['MNT_LAMBDA'])**2 - x['MNT_LAMBDA']) / x['MNT_LAMBDA'], axis=1)


aux_olsr_results = smf.ols(formula="AUX_OLS_DEP ~ MNT_LAMBDA - 1", data=df_train_xy).fit()

print(aux_olsr_results.tvalues)
print('=============')
print(aux_olsr_results.params[0])
print('=============')
print(np.mean(df_train_xy['amount']))
print('=============')
print(np.var(df_train_xy['amount']))
print('=============')
nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
print(nb2_training_results.summary())


#make some predictions using our trained NB2 model
nb2_predictions = nb2_training_results.get_prediction(X_test)

#print out the predictions
predictions_summary_frame = nb2_predictions.summary_frame()
print(predictions_summary_frame)


predicted_counts = predictions_summary_frame['mean']
actual_counts = y_test
residuals = y_test - predicted_counts


print('R2 Value:',metrics.r2_score(actual_counts, predicted_counts))

fig = plt.figure().suptitle('Predicted versus actual')
plt.scatter(actual_counts, predicted_counts)
plt.show()

plt.figure(figsize= (20,15)).suptitle('residuals')
plt.scatter(y_test, residuals)
plt.show()

#distribution of predictions
plt.figure(figsize= (20,15))
sns.displot(data = predicted_counts, kde=True)
plt.show()
#distribution of true values
plt.figure(figsize= (20,15))
sns.displot(data = actual_counts, kde=True)
plt.show()

