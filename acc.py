import pandas as pd
df = pd.read_csv('dr.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,19]

from sklearn.decomposition import PCA
pca = PCA(n_components=19,whiten=True, svd_solver='randomized')
X = pca.fit_transform(X)
print('Shape of the feature matrix is \t',X.shape)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(C=20, warm_start=True)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
from sklearn.metrics import accuracy_score
lr = accuracy_score(y_test, y_pred)
print('Accuracy of Logistic Regression is \t',lr)

import numpy as np
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
TN=np.float(cm[0][0])
FN=np.float(cm[1][0])
TP=np.float(cm[1][1])
FP=np.float(cm[0][1])

sens=TP/(TP+FN)
spec=TN/(TN+FP)
print(sens)
print(spec)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20, criterion='entropy')
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
rf_acc = accuracy_score(y_test, rf_pred)
print('Accuracy of Random Forest is \t',rf_acc)

from sklearn.svm import SVC
s = SVC(C=20,kernel='linear')
s.fit(X_train,y_train)
s_pred = s.predict(X_test)
from sklearn.metrics import accuracy_score
s_acc = accuracy_score(y_test, s_pred)
print('Accuracy of SVC is \t',s_acc)


from sklearn.svm import NuSVC
n = NuSVC(nu=0.4,kernel='linear')
n.fit(X_train,y_train)
n_pred = n.predict(X_test)
from sklearn.metrics import accuracy_score
n_acc = accuracy_score(y_test, n_pred)
print('Accuracy of NuSVC is \t',n_acc)

from sklearn import linear_model
sgd = linear_model.SGDClassifier(loss='perceptron',tol=None,warm_start=True
                                 ,average=True,class_weight=None)
sgd.fit(X_train,y_train)
sgd_pred = sgd.predict(X_test)
from sklearn.metrics import accuracy_score
sgd_acc = accuracy_score(y_test, sgd_pred)
print('Accuracy of Stohastic Gradient Descent Classifier is \t',sgd_acc)
