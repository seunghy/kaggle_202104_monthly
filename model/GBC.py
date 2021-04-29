# from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import datetime

gbc_params = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_samples_split": [3,5,7],
    "min_samples_leaf": [3,5,7],
    "max_depth":[3,5,7],
    "max_features":["sqrt"],
    "criterion": ["friedman_mse"],
    "subsample":[0.6, 0.8],
    "n_estimators":[300,500],
    'verbose' : [1],
    'random_state':[0]
    }

# GradientBoosting
def GBoost_gridsearch(x_train, y_train):
    GBC = GradientBoostingClassifier()
    GBC.fit(x_train, y_train)
    tune = GridSearchCV(estimator = GBC, param_grid=gbc_params,
    scoring='accuracy', cv=5)
    tune.fit(x_train, y_train)
    print(tune.cv_results_['mean_test_score'].mean())
    return tune


def tuned_GBoost():
    # 500, 0.4148
    gbc_fit = GradientBoostingClassifier(criterion='friedman_mse',learning_rate=0.05, loss='deviance',
    max_depth=3, max_features='sqrt',
    min_samples_leaf=3, min_samples_split=3, n_estimators=500, subsample=0.6)
    gbc_fit.fit(x_train, y_train)
    return gbc_fit


def gbc_make_log(gbc_grid):
    gbc_dict = {}
    gbc_dict['GBC']= {'time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'name': 'GradientBoosting', 
    'best_param':gbc_grid.best_params_,
    'cross_val_score_mean':cross_val_score(gbc_fit, x_val, y_val).mean()}
    
    update_dict = {}
    update_dict.update(gbc_dict)
    return update_dict
    


