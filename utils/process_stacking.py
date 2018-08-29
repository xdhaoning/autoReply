import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import config
df = pd.read_csv('.' + config.stacking_data)
x1 = df['average_vector_score']
x2 = df['bm25_score']
x3 = df['miss_word_num']
x4 = df['sentence_len']
x5 = df['hit_num']
x6 = df['hit_ratio']
y = df['label']
x = np.vstack((np.array(x1),np.array(x2),np.array(x3),np.array(x4),np.array(x5),np.array(x6)))
# y = df['label']
# x = np.vstack((np.array(x1),np.array(x2),np.array(x3),np.array(x4)))
# x = np.vstack((np.array(x1),np.array(x2)))
x = np.transpose(x)
y = np.array(y)

# from sklearn import svm
# model_SVR = svm.LinearSVR(C=0.1)
# model_SVR.fit(x,y)
# print(model_SVR.predict(x))
# from sklearn import tree
# model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
# model_DecisionTreeRegressor.fit(x,y)
# print(model_DecisionTreeRegressor.predict(x))
# cls_r =  Ridge(alpha=5)
# cls_r.fit(x,y)
# print(cls_r.predict(x))
cls_l = LogisticRegression(C=10)
cls_l.fit(x,y)
print(cls_l.predict_proba(x))
print(cls_l.score(x,y))


joblib.dump(cls_l,'.' + config.stacking_model)