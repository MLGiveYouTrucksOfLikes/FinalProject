
import numpy as np
import csv
import sys
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

def return_picked_list():
    return [True, False, False, False, True, True, True, False, True, True, True, True, False, False, True, True, True]

def pick_out_data(x):
    picked = return_picked_list()
    if x.shape[1] != len(picked):
        print 'Error: picking failed'
        print 'x shape[1]  = ', x.shape[1]
        print 'len picked = ', len(picked)
        sys.exit(0)
    print 'Original X"s column size = ',x.shape[1]
    for i in xrange(len(picked) - 1, -1, -1):
        if not picked[i]:
            x = np.delete(x, i, axis = 1)
    print 'Selected column size = ',x.shape[1]
    return x

def do_normalize(data, normalization_way):
    if normalization_way != 'normal' and normalization_way != 'scale':
        print 'Error: normalization_way should be normal or scale'
        print 'normalization_way = ',normalization_way
        sys.exit(0)
    elif normalization_way == 'scale':
        data = scale(data)
    else:
        data = normal(data)
    return data


def read_sample_train(normalization_way):
    ID_map = []
    ID_check = []
    X_train = []
    Y_train = []
    
    print 'Reading Sample_train_x.csv & Truth_train.csv...'
    with open('../ML_final_project/sample_train_x.csv', 'r') as f:
        reader = csv.reader(f)
        X_train = np.array(list(reader)[1:]).astype(float)
        ID_map = X_train[:, 0]
        X_train = do_normalize(pick_out_data(X_train[:, 1:]), normalization_way)
    f.close()
    with open('../ML_final_project/truth_train.csv', 'r') as f:
        reader = csv.reader(f)
        Y_train = np.array(list(reader)).astype(int)
        ID_check = Y_train[:, 0]
        Y_train = np.ravel(Y_train[:, 1:])#[:readDepthLimit]
    if len(Y_train) != len(X_train):
        print 'Error: Y"s len should equal to X"s len!'
        sys.exit(0)
    for i in xrange(len(ID_map)):
        if ID_map[i] != ID_check[i]:
            print 'Error: ID ckeck failed!'
            sys.exit(0)
    return ID_map, X_train, Y_train

def read_truth(normalization_way):
    ID_map = []
    test = []
    print 'Reading Truth.csv'
    with open('../ML_final_project/sample_test_x.csv', 'r') as f:
        reader = csv.reader(f)
        test = np.array(list(reader)[1:]).astype(float)
        ID_map = test[:, 0]
        test = do_normalize(pick_out_data(test[:, 1:]) , normalization_way)
    f.close()
    if len(ID_map) != len(test):
        print 'Error: ID_map len. != test len.'
        sys.exit(0)
    return ID_map, test

