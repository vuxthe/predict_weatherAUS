import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

data = pd.read_csv('postprocess_weatherAUS.csv')

# phan chia train test
x = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

print(x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

lgbm1 = lgbm.LGBMClassifier()
lgbm1.fit(x_train, y_train)

pickle.dump(lgbm1, open('model.pkl', 'wb'))

