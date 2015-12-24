
import numpy as np
from sklearn.preprocessing import scale
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from read import read_sample_train_withoutReduce
from read import read_truth_withoutReduce

def feature_select():
    ID, X_train, Y_train = read_sample_train_withoutReduce()
    print 'Original X shape = ',X_train.shape
    clf = ExtraTreesClassifier()
    clf = clf.fit(X_train, Y_train)
    model = SelectFromModel(clf, prefit=True)
    X_new_train = model.transform(X_train)
    print 'Seleceted X shape = ',X_new_train.shape
    _ , X_test = read_truth_withoutReduce()
    print 'Original X_test shape = ', X_test.shape
    index = model.get_support()
    for i in xrange(len(index)-1, -1, -1):
        if not index[i]:
            X_test = np.delete(X_test, i, axis=1)
    print 'Seleceted X_test shape = ',X_test.shape

    return ID,scale(X_new_train), Y_train, scale(X_test)

if __name__ == '__main__':
    feature_select()
