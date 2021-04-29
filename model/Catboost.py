from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


cat_params = {'iterations':[150,300,500], 'depth':[3,5,7], 
'random_seed':[0], 'learning_rate':[0.005,0.01,0.1,0.2], 'l2_leaf_reg':[3,5,7,9],'leaf_estimation_iterations':[10,30,50]}

# Catboosting gridsearch
def CatBoost_gridsearch(x_train, y_train, categorical_indexes):
    cat = CatBoostClassifier(cat_features=categorical_indexes)
    tune = cat.grid_search(cat_params, cv=5, stratified=True, shuffle=True, serch_by_train_test_split=True, 
X=x_train, y=y_train, plot=True)
    return tune

def tuned_Cat():
    # 500, 0.4148
    cat_fit = CatBoostClassifier(cat_features=categorical_indexes, leaf_estimation_iterations=50, depth=5, 
random_seed=0, l2_leaf_reg=7, iterations=300, learning_rate=0.2)
    cat_fit.fit(x_train, y_train)
    return cat_fit


# def gbc_make_log(gbc_grid):
#     gbc_dict = {}
#     gbc_dict['GBC']= {'time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'name': 'GradientBoosting', 
#     'best_param':gbc_grid.best_params_,
#     'cross_val_score_mean':cross_val_score(gbc_fit, x_val, y_val).mean()}
    
#     update_dict = {}
#     update_dict.update(gbc_dict)
#     return update_dict
    
