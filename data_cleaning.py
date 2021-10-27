# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:15:48 2021

@author: he
"""

#importing the libraries
import pandas as pd
import numpy as np

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

#checking for the missing value
train.isnull().sum()

'''
    For categorical let's impute using mode and 
# =============================================================================
#     For numerical use mean or median.
# =============================================================================
'''
#for the train dataset.
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train.Married.fillna(train.Married.mode()[0], inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0], inplace=True)
train.Self_Employed.fillna(train.Self_Employed.mode()[0], inplace=True)
train.Credit_History.fillna(train.Credit_History.mode()[0], inplace=True)

train.Loan_Amount_Term.value_counts()

train.Loan_Amount_Term.fillna(train.Loan_Amount_Term.mode()[0], inplace=True)

#since mean is affected by outliers we are going to use median.
train.LoanAmount.fillna(train.LoanAmount.median(), inplace=True)

train.isnull().sum()

# =============================================================================
# #for the test datset.
# =============================================================================
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test.Married.fillna(test.Married.mode()[0], inplace=True)
test.Dependents.fillna(test.Dependents.mode()[0], inplace=True)
test.Self_Employed.fillna(test.Self_Employed.mode()[0], inplace=True)
test.Credit_History.fillna(test.Credit_History.mode()[0], inplace=True)
test.Loan_Amount_Term.fillna(test.Loan_Amount_Term.mode()[0], inplace=True)

#Outlier treament.
'''
    Log transformation to make the data set normally distributed on the 
    LoanAmount to remove right skewness which is brought by outliers.
'''

train['LoanAmount_log'] = np.log(train.LoanAmount)
test['LoanAmount_log'] = np.log(test.LoanAmount)

# =============================================================================
# # visualizing the results.
# =============================================================================

train.LoanAmount_log.hist(bins=20)



