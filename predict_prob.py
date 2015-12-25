
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib
import sys
from svm import checkDir 

def predict():
    if len(sys.argv) < 2:
        print 'Error: Short of argv -> python predict.py [model]'
        sys.exit(0)
    normalization_way = 'scale'
    Test_ID_map, test = read.read_truth(normalization_way)
    print 'Load model...'
    model = joblib.load(sys.argv[1])
    print 'Using model to predict...'
    predict = model.predict_proba(test)
    print 'Making output csv...'
    make_csv(Test_ID_map, predict)
    

def make_csv(ID, answer):
    if len(ID) != len(answer):
        print 'Error: make failed due to diff. len'
        return
    checkDir('csv')
    dirPath = './csv/'
    with open(dirPath+sys.argv[1][5:]+'_proba_predict.csv', 'wb') as f:
        for index in xrange(len(ID)):
            f.write(str(int(ID[index]))+','+str(answer[index][1])+'\n')
    f.close()

if __name__ == "__main__":
    predict()
