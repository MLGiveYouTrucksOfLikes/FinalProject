
from sklearn.ensemble import GradientBoostingRegressor
import read
import numpy as np
from sklearn.externals import joblib
from sklearn import cross_validation

'''
    n_estimators = 200 , rate = 0.3 , Score ~= .8701

'''


def do_cross_validation(x, y , percent):
    print 'Cutting cross val. data...'
    x_train, x_test, y_train, y_test = \
            cross_validation.train_test_split(\
            x , y , test_size=percent
                    )
    return x_train, x_test, y_train, y_test

def train_argu():
    n_ = [50, 100, 200, 500, 1000]
    rate = [.01, .1, .3, .5]
    bestscore, bestn, bestrate, bestmodel = 0, 0, 0, 0
    for n in n_:
        for r in rate:
            print 'Training n = ',n,' r = ',r
            score, model = grad_tree_adaboost(n, r, False)
            if score > bestscore:
                bestscore = score
                bestn = n
                bestrate = r
                bestmodel =  model
                
    print '////////////////////////// Train done. //////////////////////////'
    print 'Bestscore = ',bestscore
    print 'Best n = ',bestn
    print 'Best rate = ',bestrate
    joblib.dump(bestmodel, 'MyBestModel.model')


def do_grid_search():
    normalization_way = 'scale'
    from sklearn.grid_search import GridSearchCV
    print 'Grid search'
    param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],'max_depth': [4, 5, 6], 'min_samples_leaf': [3, 5, 9, 17], 'max_features': [1., .5, .3, .1],
            'verbose': [1]}
    clf = GradientBoostingClassifier(n_estimators=3000)
    _ , x, y = read.read_sample_train(normalization_way)

    gs_cv = GridSearchCV(clf, param_grid, verbose=20).fit(x,y)

    print gs_cv.best_params_

def grad_tree_adaboost(n_estimators=100, learning_rate=.1, saveModel = False):
    # Argument to change
    # normal Best score =  0.860492124555
    # scale Best score =  0.871733449477
    normalization_way = 'scale'
    fold = 7
    #
    ID_map, X_train, Y_train = read.read_sample_train(normalization_way)
    kf = cross_validation.KFold(len(X_train), n_folds=fold, shuffle=True)
    i = 0
    print ' Hello Grad. Tree Adaboost World !'
    print ' I\'m n = ',n_estimators, ' rate = ',learning_rate, ' fold = ', fold, ' Boost !' 
    bestmodel, bestscore = 0, 0
    for train_index, test_index in kf:
        i += 1
        print '******************************************************************************************'
        print '@ Fold ', i
        print 'Cutting cross val. data...'
        X_cross_train, X_cross_test = X_train[train_index], X_train[test_index]
        Y_cross_train, Y_cross_test = Y_train[train_index], Y_train[test_index]
        print 'Making Gradient Tree Adaboost...'
        clf = GradientBoostingRegressor(n_estimators=n_estimators,verbose=True,\
                learning_rate=learning_rate)
        clf = clf.fit(X_cross_train, Y_cross_train)
        Y_cross_predict = clf.predict(X_cross_test)
        testScore = clf.score(X_cross_test, Y_cross_test)
        print 'Eval',i,' = ', 1.0 - testScore
        print 'Train Score = ', clf.score(X_cross_train, Y_cross_train)
        print 'Test Score = ', testScore
        if bestscore < testScore:
            bestscore = testScore
            bestmodel = clf
        print '******************************************************************************************'
    print 'Best score = ', bestscore
    print 'Dumping best model...'
    if saveModel or bestscore >= .871:
        joblib.dump(bestmodel,'model/Grad_Tree_Adaboost_regress_n'+str(n_estimators)+'_rate'+str(learning_rate)+'_fold'+str(fold)+'.pkl')
    return bestscore, bestmodel

if __name__ == '__main__':
    #do_grid_search()
    grad_tree_adaboost(200, .01, True)
