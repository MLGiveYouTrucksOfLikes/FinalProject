
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
    X_new_test = model.transform(X_test)
    print X_test[0]
    print X_new_test[0]

    return ID,scale(X_new_train), Y_train




if __name__ == '__main__':
    feature_select()
