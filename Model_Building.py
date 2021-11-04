# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:20:30 2021

@author: he
"""

import pandas as  pd
import numpy as np
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt

train =  pd.read_csv('Data/train_new.csv')
test = pd.read_csv('Data/test_new.csv')

# =============================================================================
# #Removing the Loan_ID since it has no effect.
# =============================================================================

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

X = train.drop('Loan_Status', 1)
y = train.Loan_Status

X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

# =============================================================================
# LOGISTIC REGRESSION.
# =============================================================================

model = LogisticRegression()
model.fit(x_train, y_train)

#preddicting the Loan_status.

pred_cv = model.predict(x_cv)

#now let's calculate the accuracy of the model.

log_reg = accuracy_score(y_cv, pred_cv)

# let's make the predection for the test dataset
pred_test = model.predict(test)

print('The Accuracy of Logistic Regression Model:',log_reg*100)

# =============================================================================
# The Accuracy of Logistic Regression is 79.45945945945945
# =============================================================================

submission = pd.read_csv('Data/submission.csv')

submission.Loan_Status = pred_test
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)

pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('submission.csv', index=False)

# =============================================================================
# Cross Validation metrics Using Stratified-K-Folds 
# =============================================================================

'''
    Now Let's  make a cross validation logistic model with stratified 5 folds
    And make prediction on the test dataset.    
'''

i = 1

KF = StratifiedKFold(n_splits = 5, random_state=1,shuffle=True)

for train_index, test_index in KF.split(X,y):
    print('\n{} of Kfold {} '.format(i,KF.n_splits))
    xtrain = X.iloc[train_index]
    ytrain = y.iloc[train_index]
    xvalidation = X.iloc[test_index]
    yvalidation = y.iloc[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtrain,ytrain)
    pred_test = model.predict(xvalidation)
    score = accuracy_score(yvalidation, pred_test)
    print('accuracy_score', score)
    i+=1 
    pred_test = model.predict(test)
    pred = model.predict_proba(xvalidation)[:,1]
    

False_Positive_Rate, True_Positive_Rate, _ = metrics.roc_curve(yvalidation, pred, pos_label='Y')
AUC = metrics.roc_auc_score(yvalidation, pred)
plt.figure(figsize=(12,8))
plt.plot(False_Positive_Rate, True_Positive_Rate, label='validation, AUC='+str(AUC))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)#Location of the legend on the position on the screen represented by numbers from 1 to 4.
plt.show()

# =============================================================================
# Feature Engineering
# =============================================================================

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

train.columns
train['Total_Amount'] = train.ApplicantIncome + train.CoapplicantIncome
test['Total_Amount'] = test.ApplicantIncome + test.CoapplicantIncome

train['Total_Amount_log'] = np.log(train.Total_Amount)
test['Total_Amount_log'] = np.log(test.Total_Amount)

sns.distplot(train.Total_Amount_log)

train['EMI'] = train.LoanAmount / train.Loan_Amount_Term
test['EMI'] = test.LoanAmount / train.Loan_Amount_Term

sns.distplot(train.EMI)

train['Balance_Income'] = train['Total_Amount'] -(train['EMI']*1000)
test['Balance_Income'] = test['Total_Amount'] -(test['EMI']*1000)

sns.displot(train.Balance_Income)

train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1)

# =============================================================================
# Decision Tree.
# =============================================================================
