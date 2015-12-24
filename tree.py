
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import read
import numpy as np
from sklearn.externals import joblib
from sklearn import cross_validation

def do_cross_validation(x, y , percent):
    print 'Cutting cross val. data...'
    x_train, x_test, y_train, y_test = \
            cross_validation.train_test_split(\
            x , y , test_size=percent
                    )
    return x_train, x_test, y_train, y_test


def tree():
    normalization_way = 'scale'

    ID_map, X_train, Y_train = read.read_sample_train(normalization_way)
    X_cross_train, X_cross_test, Y_cross_train, Y_cross_test = do_cross_validation(X_train, Y_train, 0.2)
    print 'Making Extreme Random Forest...'
    clf = ExtraTreesClassifier(n_estimators = 1000, verbose = True)
    clf = clf.fit(X_cross_train, Y_cross_train)
    Y_cross_predict = clf.predict(X_cross_test)
    Eval = np.count_nonzero(Y_cross_predict != Y_cross_test)/float(len(Y_cross_test)) 
    print 'Eval = ',Eval 



if __name__ == '__main__':
    tree()
