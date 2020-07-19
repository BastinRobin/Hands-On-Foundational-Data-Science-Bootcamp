import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump, load


# Create three clasifiers
lr = LogisticRegression()
# rfc = RandomForestClassifier(n_estimators=1000)


# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }

# rf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

pima = pd.read_csv('pimadiabetes.csv')


### Data Cleaning 
# Removing 0 sinces bp and bmi Cant be zero

# Replace 0 with Median not Mean
pima['bp'] = pima['bp'].replace(to_replace=0,value=pima['bp'].median())

# Replace 0 with Median not Mean
pima['bmi'] = pima['bmi'].replace(to_replace=0,value=pima['bmi'].median())


# Replace 0 with Median not Mean
pima['serum'] = pima['serum'].replace(to_replace=0,value=pima['serum'].median())


# Replace 0 with Median not Mean
pima['glu'] = pima['glu'].replace(to_replace=0,value=pima['glu'].median())

pima['skin'] = pima['skin'].fillna(pima['skin'].median())

X = pima.iloc[:,:8]
y = pima['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lr.fit(X_train, y_train)

predicted = lr.predict(X_test)

print(accuracy_score(predicted, y_test))

dump(lr, 'trained_logit.joblib') 