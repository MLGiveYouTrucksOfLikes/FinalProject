
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib
import sys
import time

def main():
   c, k = handleArgv()
   ID_map, X_train, Y_train = read.read_sample_train()
   model = doSVM((X_train, Y_train), (c, k))
   print 'Store the model...'
   current = str(time.time())
   joblib.dump(model, 'model/model_'+k+'_'+current+'.pkl')
   print 'Using model to predict'
   predict = model.predict(X_train)
   Ein = np.count_nonzero(predict != Y_train)
   print Ein/float(len(predict))

def handleArgv():
    if len(sys.argv) < 3:
        print 'Error: There should be two argv'
        sys.exit(0)
    if sys.argv[2] != 'rbf' and sys.argv[2] != 'linear' and sys.argv[2] != 'poly':
        print 'Error: Second argv should be rbf or linear'
        sys.exit(0)
    return float(sys.argv[1]), sys.argv[2]

def doSVM(data, arg):
    print 'Training kernel = ',arg[1],', C = ',arg[0], 'SVM...'
    model = 0
    if arg[1] == 'rbf':
        model = svm.SVC(C=arg[0], kernel=arg[1], gamma=1, tol=1e-7, shrinking=True, verbose=True)
    elif arg[1] == 'linear':
        model = svm.SVC(C=arg[0], kernel=arg[1], shrinking=True, verbose=True)
    else:
        model = svm.SVC(C=arg[0], kernel=arg[1], degree=2, gamma=1, coef0=1, tol=1e-4, shrinking=True, verbose=True)
    model.fit(data[0], data[1])
    return model

if __name__ == '__main__':
    main()
