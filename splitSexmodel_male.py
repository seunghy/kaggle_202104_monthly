import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pycaret.classification import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce
import lightgbm as lgbm
from catboost import CatBoostClassifier, Pool
import re
import datetime


train = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/test.csv")

delete_v = ['PassengerId'] 
##################################################################
# For model_male
train = train[train.Sex=='male'].reset_index(drop=True)
test = test[test.Sex=='male']
maleidx = test.PassengerId
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
plt.show() # Fare: Pclass??? 0.4 correlation (Pclass?????? Fare??? impute)

# 1. IMPUTE FARE
fig, axes = plt.subplots(3,1,figsize=(10,5), sharey=True)
for i in range(3):
    sns.histplot(train[train['Pclass']==i]['Fare'], label='{} class'.format(i), legend=True,ax=axes[i])
    plt.legend()
plt.show() # right_skewed --> use median instead of mean

# impute Fare with groupby median(Pclass)
train['Fare'] = train['Fare'].fillna(train.groupby(['Pclass','Cabin_alpha']).transform('median')['Fare'])
test['Fare'] = test['Fare'].fillna(test.groupby(['Pclass','Cabin_alpha']).transform('median')['Fare'])

train['Fare'] = np.log(train['Fare'])
test['Fare'] = np.log(test['Fare'])


# 2. IMPUTE AGE
sns.displot(data=train, x='Age',col='Pclass', multiple='dodge')
plt.show()

train['Age'] = train['Age'].fillna(train.groupby(['Pclass','Cabin_alpha']).transform('mean')['Age'])
train['Age'] = train['Age'].fillna(np.mean(train['Age']))

test['Age'] = test['Age'].fillna(test.groupby(['Pclass','Cabin_alpha']).transform('mean')['Age'])
test['Age'] = test['Age'].fillna(np.mean(test['Age']))

'''
# Age to category
train['Age'] = pd.qcut(train.Fare, 5, labels=[1,2,3,4,5])
'''

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
x_train, y_train = train.iloc[:,1:],train.Survived 
x_val = test

# for validation
x_train, x_val, y_train, y_val = train_test_split(train.iloc[:,1:],train.Survived, test_size=0.3 ) 

import datetime

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

# gg = GradientBoostingClassifier(gbc_grid.best_params_)
GBC = GBoost(x_train, y_train)
# pred = GBC.predict(x_val[x_val.Sex==0])
# pred_p = GBC.predict_proba(x_val)
# GBC.score(x_val, y_val) #0.7889
# cross_val_score(GBC, x_val, y_val).mean() # 0.7830667 # 0.7881 #0.8189486189486189

gbc_params = {
    "random_state" : [0],
    "loss":["deviance"],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_samples_split": [5,9],
    "min_samples_leaf": [3,5],
    "max_depth":[2,3,5,7],
    "max_features":["sqrt"],
    "criterion": ['friedman_mse'],
    "subsample":[0.6, 0.8],
    "n_estimators":[300],
    "verbose":[1]
    }
gbc_grid = grid_search(GBC, gbc_params)

# {'criterion': 'friedman_mse', 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 9, 'n_estimators': 300, 'random_state': 0, 'subsample': 0.8, 'verbose': 1}
# 0.8206929829223931 / 0.8157410157410159
gbc_fit = GradientBoostingClassifier(criterion='friedman_mse',learning_rate=0.05, loss='deviance',
max_depth=3, max_features='sqrt',
min_samples_leaf=3, min_samples_split=9, n_estimators=300, subsample=0.8)
gbc_fit.fit(x_train, y_train)


gbc_dict = {}
gbc_dict['GBC']= {'time':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'name': 'GradientBoosting', 
'best_param':{'learning_rate': 0.05, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 9, 'n_estimators': 300, 'subsample': 0.8},
'cross_val_score_mean':cross_val_score(gbc_fit, x_val, y_val).mean()}



# LIGHTGBM
def lightgmb(x_train,y_train):
    lgb = lgbm.LGBMClassifier()
    lgb.fit(x_train, y_train)
    return lgb

lgb = lightgmb(x_train, y_train)

cross_val_score(lgb, x_val, y_val).mean() #0.8143748143748144
# lgb_pred = lgb.predict(x_val)
# lgb_pred_p = lgb.predict_proba(x_val)
# lgb.score(x_val,y_val) # 0.7896666666666666
# cross_val_score(lgb, x_val, y_val).mean() #0.7824333333333333

import warnings
warnings.filterwarnings(action='ignore')

lgbm_params = {
    'application': ['binary'], # for binary classification
    'boosting': ['dart'], # traditional gradient boosting decision tree
    'num_iterations': [100,300], 
    'learning_rate': [0.05,0.1],
    'num_leaves': [50,100],
    'device': ['cpu'], 
    'max_depth': [-1], 
    'max_bin': [510], 
    # 'lambda_l1': [0,1], # L1 regularization
    # 'lambda_l2': [0,1], # L2 regularization
    'metric' : ['accuracy'],
    'subsample_for_bin': [100,200], 
    'subsample': [0.7], 
    'colsample_bytree': [0.6,0.8], 
    'min_split_gain': [0.5], 
    'min_child_weight': [1], 
    'min_child_samples': [5,7],
    'early_stopping_rounds':[30],
    'verbose' :[1]
}
lgbm_grid = grid_search(lgb, lgbm_params)

lgbm_fit = lgbm.LGBMClassifier(application='binary',boosting='dart',colsample_bytree=0.6, early_stopping_rounds=30,
learning_rate=0.05, max_bin=510, max_depth=-1, metric='accuracy', min_child_samples=5, min_child_weight=1,
min_split_gain=0.5, num_iterations=100, num_leaves=50, subsample=0.7, subsample_for_bin=100, verbose=1)
lgbm_fit.fit(x_train, y_train)

lgbm_dict = {}
lgbm_dict['lgbm']= {'time':str(datetime.datetime.now()),'name': 'Lightbm', 
'best_param':lgbm_grid.best_params_,
'cross_val_score_mean':cross_val_score(lgbm_fit, x_val, y_val).mean(), 'best_score':lgbm_grid.best_score_}


# Catboost
def catb(category, x_train, y_train, x_val, y_val):
    cat = CatBoostClassifier(cat_features=category)
    cat.fit(x_train, y_train, eval_set=(x_val, y_val) )
    return cat

categorical_indexes = [0,3,5,6]

# params {'leaf_estimation_iterations': 30, 'depth': 3, 'random_seed': 0, 'l2_leaf_reg': 9, 'iterations': 200, 'learning_rate': 0.2}
cat = catb(categorical_indexes, x_train, y_train, x_val, y_val)
cross_val_score(cat,x_val, y_val).mean() #0.815028215028215, 0.8145530145530145, 0.8174042174042174
# cat_pred = cat.predict(x_val)
# cat.score(x_val, y_val)
# cross_val_score(cat, x_val, y_val).mean() #0.7828333333333334
'''
cat_params = {'iterations': [700],
          'depth': [3, 4, 5, 6],
          'loss_function': ['Logloss', 'CrossEntropy'],
          'l2_leaf_reg':[3,5,7,9],
          'leaf_estimation_iterations': [10],
          'logging_level':['Silent'],
          'cat_features': categorical_indexes,
          'random_seed': [0]
         }
''''
cat_grid = {'iterations':[200], 'depth':[3,5,7], 
'random_seed':[0], 'learning_rate':[0.005,0.01,0.1,0.2], 'l2_leaf_reg':[3,5,7,9],'leaf_estimation_iterations':[10,30]}
cat = CatBoostClassifier(cat_features=categorical_indexes)
cat_grid = cat.grid_search(cat_grid, cv=5, stratified=True, shuffle=True, search_by_train_test_split=True, 
X=x_train, y=y_train, plot=True) # test??? logloss, std ???????????? iteration=200 ??????

# {'depth': 3, 'random_seed': 0, 'l2_leaf_reg': 9, 'iterations': 700, 'learning_rate': 0.2}
# 0.8168696168696169

cat_fit = CatBoostClassifier(cat_features=categorical_indexes, leaf_estimation_iterations=30, depth=5, random_seed=0,
l2_leaf_reg=9, iterations=200, learning_rate=0.2)
cat_fit.fit(x_train, y_train)

cat_dict = {}
cat_dict['Catboost']= {'time':str(datetime.datetime.now()),'name': 'Catboosting', 
'best_param':cat_grid['params'],
'cross_val_score_mean':cross_val_score(cat_fit, x_val, y_val).mean()}


# kfold testing
def kfold(model, splitnum, x_train, y_train):
    acc = []
    folds = StratifiedKFold(n_splits=splitnum, shuffle=True, random_state=0)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    for f, (tr_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
        print("fold num: ",f)
        tr_x, tr_y = x_train.iloc[tr_idx,:], y_train[tr_idx]
        val_x, val_y = x_train.iloc[val_idx,:], y_train[val_idx]
        ml = model
        ml.fit(tr_x, tr_y)
        pred_y = ml.predict(val_x)
        acc.append(accuracy_score(val_y, pred_y))
        print("---val acc: ", accuracy_score(val_y, pred_y))
    print("mean of acc: ", np.mean(acc))

kfold(gg, 5, x_train, y_train)
'''
lgb: 0.81801971834784
gbc: 0.8187325493106139
cat: 0.8158557516563053
'''

# most voting
temp = pd.DataFrame({'gbc':pred, 'lgbm':lgb_pred, 'cat':cat_pred})


result_survival = np.argmax((pred_p + lgb_pred_p)/2, axis=1)
result_survival

submission = pd.read_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/sample_submission.csv')
submission['Survived'] = temp

submission.to_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/submission_files/20210413_GBC_lgbm_cat_freqvoting.csv', index=False)


result_survival = np.argmax((gbc_fit.predict_proba(x_val) + lgbm_fit.predict_proba(x_val) + 
cat_fit.predict_proba(x_val))/3, axis=1)
result_survival_male = pd.DataFrame({'PassengerId':maleidx, 'Survived':result_survival})

result_survival_total = pd.concat([result_survival_male, result_survival_female], axis=0).sort_values('PassengerId')
result_survival_total.to_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/submission_files/20210426_GBC_lgbm_cat_splitdata.csv', index=False)

'''
json ??????
'''
update_dict = {}
update_dict.update(gbc_dict)
update_dict.update(lgbm_dict)
update_dict.update(cat_dict)

make_log(update_dict,path_='C:/Users/10188/local_git/titanic/data/logging.json')
import json
from collections import OrderedDict
import os

def make_log(update_dict,  path_):
    if os.path.exists(path_):
        with open(path_,mode='r+') as f:
            data=json.load(f)
        data.update(update_dict)
        with open(path_,'w+') as f:
            json.dump(update_dict,f)
    else:
        with open(path_, mode='w+') as f:
            json.dump(update_dict,f)



## submisiion


# get submission file
malesub = pd.DataFrame({'PassengerId':maleidx, 'Survived':gbc_fit.predict(x_val)})
subfile = pd.concat([malesub, femalesub], axis=0).sort_values('PassengerId')

subfile.to_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/submission_files/20210426_GBC_splitdata.csv', index=False)
