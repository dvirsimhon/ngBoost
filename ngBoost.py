from sklearn.pipeline import Pipeline

from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import optuna
import sklearn
import databases
import boosting_alg
import time
from sklearn.model_selection import GridSearchCV

def basic_ngBoost():
    db = databases.get_db("laptop")
    X_train = db.X_train
    Y_train = db.y_train
    X_test = db.X_test
    Y_test = db.y_test

    start_time = time.time()
    ngb = NGBRegressor().fit(X_train, Y_train)
    time_elapsed = time.time() - start_time
    print("Time elapsed: ", time_elapsed)
    '''    Y_preds = ngb.predict(X_test)
        Y_dists = ngb.pred_dist(X_test)
    
        # test Mean Squared Error
        test_MSE = mean_squared_error(Y_preds, Y_test)
        print('Test MSE', test_MSE)
    
        # test Negative Log Likelihood
        test_NLL = -Y_dists.logpdf(Y_test).mean()
        print('Test NLL', test_NLL)
    '''
    boosting_alg.training_results(ngb, X_train, Y_train)
    boosting_alg.testing_results(ngb, X_test, Y_test)

    return

# tune parameters - hyper parameter optimizer
def optimize_with_grid_search():
    db = databases.get_db("diamonds")
    X_train = db.X_train
    y_train = db.y_train
    rfr_clf = Pipeline([('clf', NGBRegressor())])
    parameters =  {'clf__n_estimators': (70,100), 'clf__learning_rate': (0.001, 0.01)}
    gs_clf = GridSearchCV(rfr_clf, parameters, n_jobs=1, cv=KFold(n_splits=10, shuffle=True, random_state=0))
    gs_clf = gs_clf.fit(X_train,y_train)
    print('Best score: ',gs_clf.best_score_)
    print('Best params: ',gs_clf.best_params_)


def objective(trial):
    db = databases.get_db("metro")
    X_train = db.X_train
    y_train = db.y_train
    minibatch_frac = float(trial.suggest_loguniform('minibatch_frac', 0.1, 1))
    n_estimators = int(trial.suggest_loguniform('n_estimators', 1, 100))
    learning_rate = float(trial.suggest_loguniform('learning_rate', 0.001, 1))
    clf = NGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                        minibatch_frac=minibatch_frac)
    return sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_dt():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def ngBoost_comparision():
    db = databases.get_db("laptop")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # life exp. best: learning rate = 0.974, max depth = 3, n estimators = 57
    # car prices best: learning rate = 0.521, max depth = 8, n_estimators = 51
    complex_ngBoost_model = NGBRegressor(n_estimators=99, learning_rate=0.119,
                                        minibatch_frac=0.81)

    start_time = time.time()
    complex_ngBoost_model.fit(X_train, y_train)
    complex_time = time.time() - start_time
    print("basic:")
    basic_ngBoost()
    print("complex:")
    boosting_alg.training_results(complex_ngBoost_model,X_train, y_train)
    print("Time elapsed: ", complex_time)
    boosting_alg.testing_results(complex_ngBoost_model,X_test, y_test)