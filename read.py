
import numpy as np
import csv
import sys
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

def read_sample_train():
    ID_map = []
    ID_check = []
    X_train = []
    Y_train = []
    '''
        Limit is used to limit the maximum data size
    '''
    readDepthLimit = 90000
    
    print 'Reading Sample_train_x.csv & Truth_train.csv...'
    with open('../ML_final_project/sample_train_x.csv', 'r') as f:
        reader = csv.reader(f)
        X_train = np.array(list(reader)[1:]).astype(float)
        ID_map = X_train[:, 0]
        readDepthLimit = len(ID_map)
        '''
            Limit could be canceled by setting readDepthLimit = len(ID_map) here
        '''
        #X_train = normalize(X_train[:, 5:][:readDepthLimit])
        X_train = scale(X_train[:, 5:])
        #X_train= X_train[:, 5:]
    f.close()
    with open('../ML_final_project/truth_train.csv', 'r') as f:
        reader = csv.reader(f)
        Y_train = np.array(list(reader)).astype(int)
        ID_check = Y_train[:, 0]
        Y_train = np.ravel(Y_train[:, 1:])[:readDepthLimit]
    if len(Y_train) != len(X_train):
        print 'Error: Y"s len should equal to X"s len!'
        sys.exit(0)
    for i in xrange(len(ID_map)):
        if ID_map[i] != ID_check[i]:
            print 'Error: ID ckeck failed!'
            sys.exit(0)
    return ID_map, X_train, Y_train

def read_truth():
    ID_map = []
    test = []
    print 'Reading Truth.csv'
    with open('../ML_final_project/sample_test_x.csv', 'r') as f:
        reader = csv.reader(f)
        test = np.array(list(reader)[1:]).astype(float)
        ID_map = test[:, 0]
        #test = normalize(test[:, 5:])
        test = scale(test[:, 5:])
        #test = test[:, 5:]
    f.close()
    if len(ID_map) != len(test):
        print 'Error: ID_map len. != test len.'
        sys.exit(0)
    return ID_map, test
