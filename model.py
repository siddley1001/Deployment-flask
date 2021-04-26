# Importing the libraries
import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import os

dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.dropna(axis=0, inplace=True)
dataset = dataset.drop('id', axis=1)

'''Manipulate Features so they are machine readable'''

# Gender
dataset = dataset.loc[dataset.gender != 'Other']
dataset['is_male'] = dataset['gender'].apply(lambda row: 1 if row == 'Male' else 0).astype(bool)
dataset = dataset.drop('gender', axis=1)

# Hypertension
dataset.hypertension = dataset['hypertension'].astype(bool)

# Heart Disease
dataset.heart_disease = dataset['heart_disease'].astype(bool)

# Married
dataset.ever_married = dataset['ever_married'].astype(bool)

# Work Type
work_df = pd.get_dummies(dataset['work_type'], drop_first=True)
dataset = dataset.drop('work_type', axis=1)
dataset = pd.concat([dataset, work_df], axis=1)

# Residence Type
dataset['is_urban'] = dataset['Residence_type'].apply(lambda res: 1 if res == "Urban" else 0).astype(bool)
dataset = dataset.drop('Residence_type', axis=1)

# BMI: Generate a log feature and keep the original feature
dataset['ln_bmi'] = np.log(dataset['bmi'])

# Glucose: Generate a log feature and keep the original feature
dataset['ln_glucose_lvl'] = np.log(dataset['avg_glucose_level'])

smokes_df = pd.get_dummies(dataset['smoking_status'], 'smoke', drop_first=True)
dataset = dataset.drop('smoking_status', axis=1)
dataset = pd.concat([dataset, smokes_df], axis=1)

print(dataset.head())

k, seed = 1, 42
X = dataset.drop('stroke', axis=1)
y = dataset.stroke

# increases variety of training samples
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
X_res, y_res = sm.fit_resample(X, y)

# Splits data into train, validation, and test
X_train, X_valid, y_train, y_valid = train_test_split(X_res, y_res, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)

searched_params = {'subsample': 0.8, 'min_child_weight': 1, 'max_depth': 7, 'learning_rate': 0.25,
                   'grow_policy': 'lossguide', 'gamma': 1.5, 'colsample_bytree': 0.6}
xgbc = XGBClassifier(**searched_params)

X_train = np.concatenate([X_train, X_valid], axis=0)
y_train = np.concatenate([y_train, y_valid], axis=0)

xgbc.fit(X_train, y_train)
# final_preds = xgbc.predict(X_test)
# print(final_preds.columns)
# #
# # print(recall_score(y_test, final_preds))
# #
dirr = '/Users/vanamsid/Deployment-flask/'
xgbc.save_model(dirr + 'xgbc_model.json')
