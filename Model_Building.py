# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:20:30 2021

@author: he
"""

import pandas as  pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train =  pd.read_csv('Data/train_new.csv')
test = pd.read_csv('Data/test_new.csv')

train_og = pd.read_csv('Data/train.csv')
test_og = pd.read_csv('Data/test.csv')

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

