import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from pycaret.classification import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, f1_score, confusion_matrix
# from sklearn.ensemble import GradientBoostingClassifier


train = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/test.csv")
delete_v = []

'''
preprocessing.py에서 get
'''
######
from src.preprocessing import *
from util.preprocess_module import *
train = fillNA(train, ['Ticket','Cabin','Embarked'] )
train = get_familysize(train)
train = ticket_alpha(train)
train = cabin_alpha(train)

# get Family name
train['familyname'] = train['Name'].apply(lambda x:x.split(' ')[1])
familyname_ratio = train.groupby('familyname').mean()[['Survived']].sort_values(by='Survived', ascending=False)
ratio_dict = pd.Series(familyname_ratio.Survived.values, index=familyname_ratio.index).to_dict()

test['familyname'] = test['Name'].apply(lambda x: x.split(' ')[1])

train = cal_survivalratio(train, ratio_dict, familyname_ratio)


# labeling categorical variable
train = labeling(train, ['Pclass', 'Sex','Embarked','Ticket_alpha','Cabin_alpha'])
test = labeling(test, ['Pclass', 'Sex','Embarked','Ticket_alpha','Cabin_alpha'])


#1. IMPUTE FARE
# impute Fare with groupby median(Pclass)   
train['Fare'] = np.log(imputation(train, 'Fare', ['Pclass'], 'median'))
test['Fare'] = np.log(imputation(test, 'Fare', ['Pclass'], 'median'))

# 2. IMPUTE AGE
train['Age'] = np.log(imputation(train, 'Age', ['Pclass'], 'mean'))
test['Age'] = np.log(imputation(test, 'Age', ['Pclass'], 'mean'))

######

delete_v.extend(['PassengerId','SibSp','Parch'])
delete_v.extend(['Ticket','Ticket_num'])
delete_v.extend(['Cabin'])
delete_v.extend(['Name','familyname'])

# drop col
# delete_v.extend(['Sex'])

train.drop(delete_v, axis=1, inplace=True)
test.drop(delete_v, axis=1, inplace=True)


''' ----------------------------------'''
''' FIT THE MODELS -------------------'''
''' ----------------------------------'''
# for validation
from model.GBC import *
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
x_train, x_val, y_train, y_val = train_test_split(train.iloc[:,1:],train.Survived, test_size=0.3 ) 

gbc_grid = gg.GBoost_gridsearch(x_train = x_train, y_train = y_train)
update_dict = gbc_make_log(gbc_grid)
make_log(update_dict,path_='C:/Users/10188/local_git/titanic/data/logging_female.json')
