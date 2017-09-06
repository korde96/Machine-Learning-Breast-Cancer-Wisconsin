import numpy as np
from sklearn import preprocessing, cross_validation, metrics
from sklearn import svm, linear_model, neighbors
from sklearn.model_selection import KFold,train_test_split
import pandas as pd


def work(X,y,clf_class,**kwargs):
	traindf, testdf = train_test_split(, test_size = 0.3)
	for train,test in kf.split(X):
		'''X_train, X_test = X[train],X[test]
		y_train = y[train]
		clf = clf_class(**kwargs)
		clf.fit(X_train,y_train)
		
		y_pred[test] = clf.predict(X_test)
			'''
		#clf = clf_class(**kwargs)	
		#print(clf.fit(X[train], y[train]).score(X[test], y[test]))				
	#print(clf_class,' = ',metrics.accuracy_score(y_test,y_test))
	a = clf_class(**kwargs)
	print(cross_validation.cross_val_score(a, X, y, cv=n_folds, n_jobs=-1)	)


df = pd.read_csv('data.csv')
df.drop(df.columns[[32]], axis=1, inplace=True)

#print(df.describe())

X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1,random_state=100)

neigh = neighbors.KNeighborsClassifier(algorithm='auto')

neigh.fit(X_train, y_train) 
print(neigh.score(X_test,y_test))

y_pred = neigh.predict(X_test)
print("Accuracy is ", metrics.accuracy_score(y_test,y_pred)*100)


#work(X,y,svm.SVC)
work(X,y,neighbors.KNeighborsClassifier)
