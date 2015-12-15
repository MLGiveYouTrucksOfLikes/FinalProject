
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib
import sys

def predict():
    if len(sys.argv) < 2:
        print 'Error: Short of argv -> python predict.py [model]'
        sys.exit(0)
    Test_ID_map, test = read.read_truth()
    print 'Load model...'
    model = joblib.load(sys.argv[1])
    print 'Using model to predict...'
    predict = model.predict(test)
    print 'Making output csv...'
    make_csv(Test_ID_map, predict)
    

def make_csv(ID, answer):
    if len(ID) != len(answer):
        print 'Error: make failed due to diff. len'
        return
    with open(sys.argv[1]+'_predict.csv', 'wb') as f:
        for index in xrange(len(ID)):
            f.write(str(int(ID[index]))+','+str(answer[index])+'\n')
    f.close()

if __name__ == "__main__":
    predict()
