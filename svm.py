
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib

def main():
   ID_map, X_train, Y_train = read.read_sample_train()
   print 'Training rbf C= 0.1 svm..'
   model = svm.SVC(C=0.01, kernel='rbf', gamma=1, tol=1e-7, shrinking=True, verbose=True)
   #model = svm.SVC(C=0.1, kernel='linear', shrinking=True, verbose=True)
   model.fit(X_train, Y_train)
   print 'Store the model...'
   joblib.dump(model, 'model_rbf.pkl')
   print 'Using model to predict'
   predict = model.predict(X_train)
   Ein = np.count_nonzero(predict != Y_train)
   print Ein/float(len(predict))

if __name__ == '__main__':
    main()
