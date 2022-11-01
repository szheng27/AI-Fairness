import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


import copy
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import plotly
# Importing packages for SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN

def rf(df):
	train,test = train_test_split( df, test_size=0.2, random_state=20,shuffle=True)
	X_train = train.drop("Y",axis =1).values
	y_train=train['Y'].values

	X_test  =test.drop("Y",axis =1).values
	y_test=test['Y'].values

	# Trying out Random Forest Classifier
	rf=RandomForestClassifier()
	rf.fit(X_train,y_train)
	acc = accuracy_score(y_test,rf.predict(X_test))
	print('the accuracy of the model is:',acc*100,'%')
	prediction = rf.predict(df.drop("Y",axis =1).values)
	
	return [np.reshape(prediction,len(prediction)),acc]