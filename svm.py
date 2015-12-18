
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib
import sys
import time
import os

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
   saveModel(bestModel, save)
   print '*********************************************************************'
   print '                            Best Eval = ', bestEval
   print '                            Current Time = ', current
   print '*********************************************************************'

def saveModel(model, enableSave = False):
    if enableSave:
        print 'Saving the model...'
        checkDir('model')
        # python svm.py c kernel fold save gamma
        command = sys.argv
        modelName = 'model/model_' + command[2] + '_c' + command[1] + '_fold' + command[3]
        if command[2] == 'rbf' or command[2] == 'poly':
            modelName += ('_gamma' + command[5])
        modelName += '.pkl'
        joblib.dump(model, modelName)

def checkDir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def handleArgv():
    if len(sys.argv) < 5:
        print 'Error: There should be at least four argvs'
        printArgv()
    if sys.argv[2] != 'rbf' and sys.argv[2] != 'linear' and sys.argv[2] != 'poly' and sys.argv[2] != 'linearSVC':
        print 'Error: Second argv should be [rbf||linear||poly||linearSVC]'
        printArgv()
    if (sys.argv[2] == 'rbf' or sys.argv[2] == 'poly') and len(sys.argv) < 6:
        print len(sys.argv)
        print 'Error: When model = rbf or poly, should input gamma'
        printArgv()
    return float(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def printArgv():
    print 'python svm.py [C] [model type] [Cross validation folds] [Enable save model] [gamma if kernel = rbf or poly]'
    sys.exit(0)

def doSVM(data, arg):
    print 'Training kernel(or type) = ',arg[1],', C = ',arg[0], 'SVM...'
    model = 0
    if arg[1] == 'rbf':
        gamma = float(sys.argv[5])
        model = svm.SVC(C=arg[0], kernel=arg[1], gamma=gamma, tol=1e-7, shrinking=True, verbose=True)
    elif arg[1] == 'linear':
        model = svm.SVC(C=arg[0], kernel=arg[1], shrinking=True, verbose=True)
    elif arg[1] == 'linearSVC':
        model = svm.LinearSVC(C=arg[0], verbose=True, max_iter = 1000)
    else:
        gamma = float(sys.argv[5])
        model = svm.SVC(C=arg[0], kernel=arg[1], degree=3, gamma=gamma, coef0=3, tol=1e-4, shrinking=True, verbose=True)
    model.fit(data[0], data[1])
    return model

if __name__ == '__main__':
    main()
