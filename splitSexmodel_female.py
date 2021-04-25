import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pycaret.classification import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce
import lightgbm as lgbm
from catboost import CatBoostClassifier
import re


train = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/test.csv")

delete_v = ['PassengerId'] 
##################################################################
# For model_female
train = train[train.Sex=='female'].reset_index(drop=True)
test = test[test.Sex=='female']
##################################################################

# label encoding
def labeling(df, columns):
    encoder = LabelEncoder()
    for c in columns:
        col = encoder.fit_transform(df[c])
        df[c] = col
    return df

# if Ticket/Cabin/Embarked.isnull -> fill "X"
def fillNA(df,cols, replaceString='X'):
    df[cols] = df[cols].fillna(replaceString).astype('string')
    return df[cols]

train[['Ticket','Cabin','Embarked']] = fillNA(train, ['Ticket','Cabin','Embarked'])
test[['Ticket','Cabin','Embarked']] = fillNA(test, ['Ticket','Cabin','Embarked'])

# add family size variable
train['famsize'] = train.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
test['famsize'] = test.apply(lambda x: x['SibSp']+x['Parch'], axis=1)

delete_v.extend(['SibSp','Parch'])

# Ticket type
train['Ticket_alpha'] = train['Ticket'].apply(lambda x: x.split(' ')[0])
train['Ticket_alpha'] = train['Ticket_alpha'].apply(lambda x: x if x.upper().isupper() else "X")
train['Ticket_num'] = train['Ticket'].apply(lambda x: x.split(' ')[1][:2] if x != 'X' and x.upper().isupper() else x[:2])

test['Ticket_alpha'] = test['Ticket'].apply(lambda x: x.split(' ')[0])
test['Ticket_alpha'] = test['Ticket_alpha'].apply(lambda x: x if x.upper().isupper() else "X")
test['Ticket_num'] = test['Ticket'].apply(lambda x: x.split(' ')[1][:2] if x != 'X' and x.upper().isupper() else x[:2])

# compare survival ratio by Ticket - BY TRAIN DATA  
alpha_ratio = train.groupby('Ticket_alpha').mean()[['Survived']].sort_values(by='Survived', ascending=False)
'''
plot = sns.barplot(alpha_ratio.index, alpha_ratio.Survived)
plot.tick_params(labelsize=5)
plt.show() # --> add Ticket_alpha variable (survival ratio between 0.1~0.6 : difference)
'''

num_ratio = train.groupby('Ticket_num').mean()[['Survived']].sort_values(by='Survived', ascending=False)
'''
plot = sns.barplot(num_ratio.index, num_ratio.Survived)
plot.tick_params(labelsize=7)
plt.show() # --> drop Ticket_num varible (survival ratio between 0.3~0.5 : similar)
'''

delete_v.extend(['Ticket','Ticket_num'])

# Cabin type -> A,B,C,D,E,F,G,S
# train['HaveCabin'] = train['Cabin'].apply(lambda x :0 if x == "X" else 1)
train['Cabin_alpha'] = train['Cabin'].apply(lambda x: x[:2])
test['Cabin_alpha'] = test['Cabin'].apply(lambda x: x[:2])

# compare survival ratio by cabin 
cabin_ratio = train.groupby(['Cabin_alpha']).mean()[['Survived']].sort_values(by='Survived', ascending=False)
'''
plot = sns.barplot(cabin_ratio.index, cabin_ratio.Survived)
plot.tick_params(labelsize=7)
plt.show() 
'''

delete_v.extend(['Cabin'])

# # get Family name
train['familyname'] = train['Name'].apply(lambda x:x.split(' ')[1])
familyname_ratio = train.groupby('familyname').mean()[['Survived']].sort_values(by='Survived', ascending=False)
'''
sns.barplot(familyname_ratio.index, familyname_ratio.Survived)
plt.show() # --> have huge difference
'''

test['familyname'] = test['Name'].apply(lambda x: x.split(' ')[1])

# drop Name, familyname and add survival ratio by familyname -->  Target encoding instead of familyname
target_encoder = ce.TargetEncoder()
train['survival_ratio'] = target_encoder.fit_transform(train.familyname, train.Survived).values

ratio_dict = pd.Series(familyname_ratio.Survived.values, index=familyname_ratio.index).to_dict()
test['survival_ratio'] = test['familyname'].map(ratio_dict)
test['survival_ratio'] = test['survival_ratio'].fillna(np.median(familyname_ratio))

delete_v.extend(['Name','familyname'])

# labeling categorical variable
train = labeling(train, ['Pclass', 'Sex','Embarked','Ticket_alpha','Cabin_alpha'])
test = labeling(test, ['Pclass', 'Sex','Embarked','Ticket_alpha','Cabin_alpha'])

## categorical variable correlation with target variable
# survived / pclass / sex / embarked vs age : boxplot , sibsp / parch / fare vs age : scatter plot
fig, axes = plt.subplots(2,3,figsize=(10,5), sharey=True)

sns.boxplot(x='Survived', y='Age',data=train, ax=axes[0,0])
axes[0,0].set_title("Survived(categorical)")
sns.boxplot(x='Pclass', y='Age',data=train, ax=axes[0,1])
axes[0,1].set_title("Pclass(categorical)")
sns.boxplot(x='Sex', y='Age',data=train, ax=axes[0,2])
axes[0,2].set_title("Sex(categorical)")
sns.boxplot(x='Embarked', y='Age',data=train, ax=axes[1,0])
axes[1,0].set_title("Embarked(categorical)")
sns.histplot(x='Fare', y='Age', data=train, ax=axes[1,1])
axes[1,1].set_title("Fare(continuous)")
sns.boxplot(x='famsize', y='Age', data=train, ax=axes[1,2])
axes[1,2].set_title("famsize(categorical)")
plt.show() #----------------> Age and Pclass

# check correlation with each other
sns.pairplot(train)
plt.show()
sns.heatmap(train.corr(), annot=True)
plt.show() # Fare: Pclass와 0.4 correlation (Pclass별로 Fare를 impute)

# 1. IMPUTE FARE
fig, axes = plt.subplots(3,1,figsize=(10,5), sharey=True)
for i in range(3):
    sns.histplot(train[train['Pclass']==i]['Fare'], label='{} class'.format(i), legend=True,ax=axes[i])
    plt.legend()
plt.show() # right_skewed --> use median instead of mean

# impute Fare with groupby median(Pclass)
train['Fare'] = train['Fare'].fillna(train.groupby(['Pclass']).transform('median')['Fare'])
test['Fare'] = test['Fare'].fillna(test.groupby(['Pclass']).transform('median')['Fare'])

train['Fare'] = np.log(train['Fare'])
test['Fare'] = np.log(test['Fare'])


# 2. IMPUTE AGE
sns.displot(data=train, x='Age',col='Pclass', multiple='dodge')
plt.show()

train['Age'] = train['Age'].fillna(train.groupby(['Pclass']).transform('mean')['Age'])
train['Age'] = train['Age'].fillna(np.mean(train['Age']))

test['Age'] = test['Age'].fillna(test.groupby(['Pclass']).transform('mean')['Age'])
test['Age'] = test['Age'].fillna(np.mean(test['Age']))

# Age to category
pd.qcut(train.Age, 4, labels=['kid','adult1','adult2','elderly'])

# check missing value -> do not exist missing value
msno.bar(train)
plt.show()
msno.bar(test)
plt.show()

# drop col
delete_v.extend(['Sex'])
train.drop(delete_v, axis=1, inplace=True)
test.drop(delete_v, axis=1, inplace=True)

''' ----------------------------------'''
''' FIT THE MODELS -------------------'''
''' ----------------------------------'''

# # for submission
# x_train, y_train = train.iloc[:,1:],train.Survived 
# x_val = test

# for validation
x_train, x_val, y_train, y_val = train_test_split(train.iloc[:,1:],train.Survived, test_size=0.3 ) 


# gridsearch
def grid_search(model, param, cvnum=5):
    tune = GridSearchCV(estimator = model,
    param_grid=param, scoring='accuracy',cv=cvnum)
    tune.fit(x_train, y_train)
    return tune

# pred func
def predmodel(model, x_val, y_val):
    pred = model.predict(x_val)
    pred_p = model.predict_proba(x_val)
    return pred, pred_p

# GradientBoosting
def GBoost(x_train, y_train):
    GBC = GradientBoostingClassifier()
    GBC.fit(x_train, y_train)
    return GBC

GBC = GBoost(x_train, y_train)
# pred = GBC.predict(x_val[x_val.Sex==0])
# pred_p = GBC.predict_proba(x_val)
# GBC.score(x_val, y_val) #0.7889
# cross_val_score(GBC, x_val, y_val).mean() # 0.7830667 # 0.7881

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
gbc_grid = grid_search(GBC, gbc_params)
gbc_grid.cv_results_['mean_test_score'].mean()

# 500, 0.4148
gbc_fit = GradientBoostingClassifier(criterion='friedman_mse',learning_rate=0.05, loss='deviance',max_depth=3, max_features='sqrt',
min_samples_leaf=3, min_samples_split=3, n_estimators=500, subsample=0.6)
gbc_fit.fit(x_train, y_train)

import datetime
gbc_dict = {}
gbc_dict['GBC']= {'time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'name': 'GradientBoosting', 
'best_param':gbc_grid.best_params_,
'cross_val_score_mean':cross_val_score(gbc_fit, x_val, y_val).mean()}

# LIGHTGBM
def lightgmb(x_train,y_train):
    lgb = lgbm.LGBMClassifier()
    lgb.fit(x_train, y_train)
    return lgb

lgb = lgbm.LGBMClassifier().fit(x_train, y_train)
# lgb_pred = lgb.predict(x_val)
# lgb_pred_p = lgb.predict_proba(x_val)
# lgb.score(x_val,y_val) # 0.7896666666666666
# cross_val_score(lgb, x_val, y_val).mean() #0.7824333333333333

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
lgbm_grid = grid_search(lgb, lgbm_params)


lgbm_fit = lgbm.LGBMClassifier(boost_from_average=False,boosting='dart',colsample_bytree=0.6,early_stopping_rounds=20,
learning_rate=0.1, max_bin=510, max_depth=-1, metric='binary_error',min_child_samples=7, min_child_weight=1,
min_split_gain=0.6, num_iterations=300, num_leavees=50, objective='binary', subsample_for_bin=200)
lgbm_fit.fit(x_train, y_train)

lgbm_dict = {}
lgbm_dict['lgbm']= {'time':str(datetime.datetime.now()),'name': 'Lightbm', 
'best_param':lgbm_grid.best_params_,
'cross_val_score_mean':cross_val_score(lgbm_fit, x_val, y_val).mean(), 'best_score':lgbm_grid.best_score_}



# Catboost
def catb(x_train, y_train):
    cat = CatBoostClassifier()
    cat.fit(x_train, y_train)
    return cat

# cat_pred = cat.predict(x_val)
# cat.score(x_val, y_val)
# cross_val_score(cat, x_val, y_val).mean() #0.7828333333333334

# cat_params = {'iterations': [300, 500],
#           'depth': [3, 4, 5, 6],
#           'loss_function': ['Logloss', 'CrossEntropy'],
#           'l2_leaf_reg': np.logspace(-20, -19, 3),
#           'leaf_estimation_iterations': [20],
#           'eval_metric': ['Accuracy'],
#           'use_best_model': ['True'],
#           'logging_level':['Silent'],
#           'random_seed': [0]
#          }

categorical_indexes = [0,3,5,6]
cat = CatBoostClassifier(cat_features=categorical_indexes).fit(x_train, y_train)
cross_val_score(cat,x_val, y_val).mean() 

cat_grid = {'iterations':[150,300,500], 'depth':[3,5,7], 
'random_seed':[0], 'learning_rate':[0.005,0.01,0.1,0.2], 'l2_leaf_reg':[3,5,7,9],'leaf_estimation_iterations':[10,30,50]}
cat = CatBoostClassifier(cat_features=categorical_indexes)
cat_grid = cat.grid_search(cat_grid, cv=5, stratified=True, shuffle=True, serch_by_train_test_split=True, 
X=x_train, y=y_train, plot=True) # test의 logloss, std 확인하여 iteration=?? 정함

cat_grid['cv_results'].keys()


cat_fit = CatBoostClassifier(cat_features=categorical_indexes, leaf_estimation_iterations=50, depth=5, 
random_seed=0, l2_leaf_reg=7, iterations=300, learning_rate=0.2)
cat_fit.fit(x_train, y_train)

cat_dict = {}
cat_dict['Catboost']= {'time':str(datetime.datetime.now()),'name': 'Catboosting', 
'best_param':cat_grid['params'],
'cross_val_score_mean':cross_val_score(cat_fit, x_val, y_val).mean()}


# kfold testing
def kfold(model, splitnum, shuffleox=True, x_train, y_train)
    folds = StratifiedKFold(n_splits=splitnum, shuffle=shuffleox, random_state=0)

    for f, (tr_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
        print("[fold num]: ",f)
        tr_x, tr_y = x_train[tr_idx], y_train[tr_idx]
        val_x, val_y = x_train[val_idx], y_train[val_idx]

        ml = model
        ml.fit(tr_x, tr_y, eval_set=[val_x,val_y])
        pred_y = ml.predict(val_x)
        print("---val acc: ", accuracy_score(val_y, pred_y))

kfold(lgb, 7, shuffleox=True, x_train, y_train)


# most voting
temp = pd.DataFrame({'gbc':pred, 'lgbm':lgb_pred, 'cat':cat_pred})


result_survival = np.argmax((pred_p + lgb_pred_p)/2, axis=1)
result_survival

submission = pd.read_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/sample_submission.csv')
submission['Survived'] = temp

submission.to_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/submission_files/20210413_GBC_lgbm_cat_freqvoting.csv', index=False)



'''
json 저장
'''
update_dict = {}
update_dict.update(gbc_dict)
update_dict.update(lgbm_dict)
update_dict.update(cat_dict)

make_log(update_dict,path_='C:/Users/10188/local_git/titanic/data/logging_female.json')
import json
from collections import OrderedDict
import os

def make_log(update_dict,  path_):
    if os.path.exists(path_):
        with open(path_,mode='r+') as f:
            data=json.load(f)
        data.update(update_dict)
        with open(path_,'w+') as f:
            json.dump(data,f)
    else:
        with open(path_, mode='w+') as f:
            json.dump(update_dict,f)