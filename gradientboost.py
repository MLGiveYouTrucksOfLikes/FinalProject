
from sklearn.ensemble import GradientBoostingClassifier
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


def grad_tree_adaboost():
    # Argument to change
    normalization_way = 'scale'
    fold = 8
    n_estimators, learning_rate = 140, 1
    #
    ID_map, X_train, Y_train = read.read_sample_train(normalization_way)
    kf = cross_validation.KFold(len(X_train), n_folds=fold, shuffle=True)
    i = 0
    bestmodel, bestscore = 0, 0
    for train_index, test_index in kf:
        i += 1
        print '@ Fold ', i
        print 'Cutting cross val. data...'
        X_cross_train, X_cross_test = X_train[train_index], X_train[test_index]
        Y_cross_train, Y_cross_test = Y_train[train_index], Y_train[test_index]
        print 'Making Gradient Tree Adaboost...'
        clf = GradientBoostingClassifier(n_estimators=n_estimators,verbose=True, learning_rate=learning_rate, loss='exponential')
        clf = clf.fit(X_cross_train, Y_cross_train)
        Y_cross_predict = clf.predict(X_cross_test)
        testScore = clf.score(X_cross_test, Y_cross_test)
        print 'Eval',i,' = ', 1.0 - testScore
        print 'Train Score = ', clf.score(X_cross_train, Y_cross_train)
        print 'Test Score = ', testScore
        if bestscore < testScore:
            bestscore = testScore
            bestmodel = clf
    print 'Best score = ', bestscore
    print 'Dumping best model...'
    joblib.dump(bestmodel,'model/Grad_Tree_Adaboost_n'+str(n_estimators)+'_rate'+str(learning_rate)+'_fold'+str(fold)+'.pkl')

if __name__ == '__main__':
    grad_tree_adaboost()
