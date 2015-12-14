
import numpy as np
import read
from sklearn import svm
from sklearn.externals import joblib

def predict():
    Train_ID_map, X_train, Y_train = read.read_sample_train()
    Test_ID_map, test = read.read_truth()
    print 'Load model...'
    model = joblib.load('model_rbf.pkl')
    print 'Using model to predict...'
    predict = model.predict(X_train)
    print 'Making output csv...'
    make_csv(Train_ID_map, predict)
    

def make_csv(ID, answer):
    if len(ID) != len(answer):
        print 'Error: make failed due to diff. len'
        return
    with open('ans.csv', 'wb') as f:
        for index in xrange(len(ID)):
            f.write(str(ID[index])+','+str(answer[index])+'\n')
    f.close()

if __name__ == "__main__":
    predict()
