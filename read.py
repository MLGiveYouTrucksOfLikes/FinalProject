
import numpy as np
import csv
import sys

def read_sample_train():
    ID_map = []
    ID_check = []
    X_train = []
    Y_train = []
    with open('../ML_final_project/sample_train_x.csv', 'r') as f:
        reader = csv.reader(f)
        X_train = np.array(list(reader)[1:]).astype(int)
        ID_map = X_train[:, 0]
        X_train = X_train[:, 1:]
    f.close()
    with open('../ML_final_project/truth_train.csv', 'r') as f:
        reader = csv.reader(f)
        Y_train = np.array(list(reader)).astype(int)
        ID_check = Y_train[:, 0]
        Y_train = Y_train[:, 1:]
    for i in xrange(len(ID_map)):
        if ID_map[i] != ID_check[i]:
            print 'Error: ID ckeck failed!'
            sys.exit(0)
    print Y_train
read_sample_train()
