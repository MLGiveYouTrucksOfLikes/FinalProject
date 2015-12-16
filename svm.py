
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib
import sys
import time

def main():
   c, k , fold, save= handleArgv()
   ID_map, X_train, Y_train = read.read_sample_train()
   '''
    Cross Validation
   '''
   arr = np.arange(len(Y_train))
   np.random.shuffle(arr)
   X_cross_train, Y_cross_train, X_cross_val, Y_cross_val = 0,0,0,0
   perWindow = len(Y_train) / fold
   bestEval, bestModel = 1.0, 0
   for fold_i in xrange(fold):
       print '@ Fold ',fold_i
       X_cross_train = np.concatenate((X_train[arr[:fold_i*perWindow]], X_train[arr[(fold_i+1)*perWindow:]]), axis=0)
       Y_cross_train = np.concatenate((Y_train[arr[:fold_i*perWindow]], Y_train[arr[(fold_i+1)*perWindow:]]), axis=0)
       X_cross_val = X_train[arr[fold_i*perWindow+1: (fold_i+1)*perWindow]]
       Y_cross_val = Y_train[arr[fold_i*perWindow+1: (fold_i+1)*perWindow]]
       currentModel = doSVM((X_cross_train, Y_cross_train), (c, k))
       cross_predcit = currentModel.predict(X_cross_val)
       currentEval = np.count_nonzero(cross_predcit != Y_cross_val) / float(len(Y_cross_val))
       print 'Eval = ',currentEval
       if currentEval < bestEval:
           bestEval = currentEval
           bestModel = currentModel
   current = str(int(time.time()))
   if save == 1:
       print 'Store the model...'
       joblib.dump(bestModel, 'model/model_'+k+'_c'+str(c)+'_fold'+str(fold)+'.pkl')
   print '*********************************************************************'
   print '                            Best Eval = ', bestEval
   print '                            Current Time = ', current
   print '*********************************************************************'

def handleArgv():
    if len(sys.argv) < 5:
        print 'Error: There should be four argvs'
        printArgv()
        sys.exit(0)
    if sys.argv[2] != 'rbf' and sys.argv[2] != 'linear' and sys.argv[2] != 'poly' and sys.argv[2] != 'linearSVC':
        print 'Error: Second argv should be [rbf||linear||poly||linearSVC]'
        printArgv()
        sys.exit(0)
    return float(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def printArgv():
    print 'python svm.py [C] [model type] [Cross validation folds] [Enable save model]'

def doSVM(data, arg):
    print 'Training kernel(or type) = ',arg[1],', C = ',arg[0], 'SVM...'
    model = 0
    if arg[1] == 'rbf':
        model = svm.SVC(C=arg[0], kernel=arg[1], gamma=1, tol=1e-7, shrinking=True, verbose=True)
    elif arg[1] == 'linear':
        model = svm.SVC(C=arg[0], kernel=arg[1], shrinking=True, verbose=True)
    elif arg[1] == 'linearSVC':
        model = svm.LinearSVC(C=arg[0], verbose=True, max_iter = 1000)
    else:
        model = svm.SVC(C=arg[0], kernel=arg[1], degree=2, gamma=1, coef0=1, tol=1e-4, shrinking=True, verbose=True)
    model.fit(data[0], data[1])
    return model

if __name__ == '__main__':
    main()
