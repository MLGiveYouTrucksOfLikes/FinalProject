
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib
import sys
from svm import checkDir 

def predict_withoutARGV(ID, X_test, model):
    print 'Using model to predict...'
    predict = model.predict(X_test)
    print 'Making output csv...'
    make_csv_withoutARGV(ID, predict)

def make_csv_withoutARGV(ID, ans):
    checkDir('csv')
    dirPath = './csv/'
    with open(dirPath+'WithoutARGV_predict.csv', 'wb') as f:
        for index in xrange(len(ID)):
            f.write(str(int(ID[index]))+','+str(answer[index])+'\n')
    f.close()

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
    checkDir('csv')
    dirPath = './csv/'
    with open(dirPath+sys.argv[1][5:]+'_predict.csv', 'wb') as f:
        for index in xrange(len(ID)):
            f.write(str(int(ID[index]))+','+str(answer[index])+'\n')
    f.close()

if __name__ == "__main__":
    #TODO
    print 'Current predict.py will not work !'
    #predict()
