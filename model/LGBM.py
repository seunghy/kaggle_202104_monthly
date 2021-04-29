import lightgbm as lgbm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

lgbm_params = {
    'objective':['binary'],
    'boost_from_average': [False],
    'verbose':[1],
    'boosting': ['dart','gbdt'], # traditional gradient boosting decision tree
    'num_iterations': [300,500], 
    'learning_rate': [0.01,0.05,0.1,0.2],
    'num_leaves': [50,100],
    'device': ['cpu'], 
    'max_depth': [-1], 
    'max_bin': [510], 
    # 'lambda_l1': [3,5], # L1 regularization
    # 'lambda_l2': [3,5], # L2 regularization
    'metric' : ['binary_error'],
    'subsample_for_bin': [100,200], 
    'colsample_bytree': [0.6,0.8], 
    'min_split_gain': [0.5,0.6], 
    'min_child_weight': [1], 
    'min_child_samples': [5,7],
    'early_stopping_rounds':[20]
}


# LGBM
def lgbm_gridsearch(x_train, y_train):
    Lgbm = lgbm.LGBMClassifier()
    Lgbm.fit(x_train, y_train)
    tune = GridSearchCV(estimator = Lgbm, param_grid=lgbm_params,
    scoring='accuracy', cv=5)
    tune.fit(x_train, y_train)
    print(tune.cv_results_['mean_test_score'].mean())
    return tune

def tuned_Lgbm():
    # 500, 0.4148
    lgbm_fit = lgbm.LGBMClassifier(boost_from_average=False,boosting='dart',colsample_bytree=0.6,early_stopping_rounds=20,
learning_rate=0.1, max_bin=510, max_depth=-1, metric='binary_error',min_child_samples=7, min_child_weight=1,
min_split_gain=0.6, num_iterations=300, num_leavees=50, objective='binary', subsample_for_bin=200)
    lgbm_fit.fit(x_train, y_train)
    return lgbm_fit


# def lgbm_make_log(lgbm_grid):
#     lgbm_dict = {}
#     lgbm_dict['GBC']= {'time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'name': 'LGBM', 
#     'best_param':lgbm_grid.best_params_,
#     'cross_val_score_mean':cross_val_score(gbc_fit, x_val, y_val).mean()}
    
#     update_dict = {}
#     update_dict.update(gbc_dict)
#     return update_dict
    
