import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pycaret.classification import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce
import lightgbm as lgbm
from catboost import CatBoostClassifier
import re
from sklearn.neighbors import KNeighborsClassifier



train = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/test.csv")

msno.matrix(train)
msno.bar(train) #Age, Ticket, Fare, Cabin, Embarked have null 

msno.bar(test) #Age, Ticket, Fare, Cabin(to drop), Embarked have null

numeric_v = ['Pclass','Age','SibSp','Parch','Fare']
string_v = ['Survived','Sex','Embarked','Ticket','Cabin','Name']
delete_v = ['PassengerId'] 
##################################################################
# - > name(family name), ticket(앞글자), cabin(a,b...)이용해보기 : Name은 의미x, ticket, cabin 이용
# - > famsize, name freq 를 categorical vari.로 이용해보기 & 남녀나눠서 모델링
# 2021.04.13 - > cabin을 영어만 가져오기, 남녀 나눠서 모델링, module별 구분
##################################################################

# label encoding
def labeling(df, columns):
    encoder = LabelEncoder()
    for c in columns:
        col = encoder.fit_transform(df[c])
        df[c] = col
    return df

train.drop(delete_v, axis=1, inplace=True)
test.drop(delete_v, axis=1, inplace=True)

# if Ticket/Cabin/Embarked.isnull -> fill "X"
def fillNA(df,cols, replaceString='X'):
    df[cols] = df[cols].fillna(replaceString).astype('string')
    return df[cols]

train[['Ticket','Cabin','Embarked']] = fillNA(train, ['Ticket','Cabin','Embarked'])
test[['Ticket','Cabin','Embarked']] = fillNA(test, ['Ticket','Cabin','Embarked'])

# add family size variable
train['famsize'] = train.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
train.drop(['SibSp','Parch'], axis=1, inplace=True)

test['famsize'] = test.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
test.drop(['SibSp','Parch'], axis=1, inplace=True)


# Ticket type
train['Ticket_alpha'] = train['Ticket'].apply(lambda x: x.split(' ')[0])
train['Ticket_alpha'] = train['Ticket_alpha'].apply(lambda x: x if x.upper().isupper() else "X")
train['Ticket_num'] = train['Ticket'].apply(lambda x: x.split(' ')[1][:2] if x != 'X' and x.upper().isupper() else x[:2])

test['Ticket_alpha'] = test['Ticket'].apply(lambda x: x.split(' ')[0])
test['Ticket_alpha'] = test['Ticket_alpha'].apply(lambda x: x if x.upper().isupper() else "X")
test['Ticket_num'] = test['Ticket'].apply(lambda x: x.split(' ')[1][:2] if x != 'X' and x.upper().isupper() else x[:2])

# compare survival ratio by Ticket - BY TRAIN DATA  
alpha_ratio = train.groupby('Ticket_alpha').mean()[['Survived']].sort_values(by='Survived', ascending=False)
plot = sns.barplot(alpha_ratio.index, alpha_ratio.Survived)
plot.tick_params(labelsize=5)
plt.show() # --> add Ticket_alpha variable (survival ratio between 0.1~0.6 : difference)

num_ratio = train.groupby('Ticket_num').mean()[['Survived']].sort_values(by='Survived', ascending=False)
plot = sns.barplot(num_ratio.index, num_ratio.Survived)
plot.tick_params(labelsize=7)
plt.show() # --> drop Ticket_num varible (survival ratio between 0.3~0.5 : similar)

train.drop(['Ticket','Ticket_num'], axis=1, inplace=True)
test.drop(['Ticket','Ticket_num'], axis=1, inplace=True)


train.drop(['Ticket_alpha'], axis=1, inplace=True)
test.drop(['Ticket_alpha'], axis=1, inplace=True)


# Cabin type -> A,B,C,D,E,F,G,S
train['HaveCabin'] = train['Cabin'].apply(lambda x :0 if x == "X" else 1)
train['Cabin_alpha'] = train['Cabin'].apply(lambda x: x[:2])
test['Cabin_alpha'] = test['Cabin'].apply(lambda x: x[:2])


# compare survival ratio by cabin 
cabin_ratio = train.groupby(['Cabin_alpha']).mean()[['Survived']].sort_values(by='Survived', ascending=False)
plot = sns.barplot(cabin_ratio.index, cabin_ratio.Survived)
plot.tick_params(labelsize=7)
plt.show() 

train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)


# # get Family name
train['familyname'] = train['Name'].apply(lambda x:x.split(' ')[1])
familyname_ratio = train.groupby('familyname').mean()[['Survived']].sort_values(by='Survived', ascending=False)
sns.barplot(familyname_ratio.index, familyname_ratio.Survived)
plt.show() # --> have huge difference

test['familyname'] = test['Name'].apply(lambda x: x.split(' ')[1])

# drop Name, familyname and add survival ratio by familyname -->  Target encoding instead of familyname
target_encoder = ce.TargetEncoder()
train['survival_ratio'] = target_encoder.fit_transform(train.familyname, train.Survived).values

ratio_dict = pd.Series(familyname_ratio.Survived.values, index=familyname_ratio.index).to_dict()
test['survival_ratio'] = test['familyname'].map(ratio_dict)
test['survival_ratio'] = test['survival_ratio'].fillna(np.median(familyname_ratio))

train.drop(['Name','familyname'], axis=1, inplace=True)
test.drop(['Name','familyname'], axis=1, inplace=True)


# # Age: categorical, Embarked: Categorical
# # Age: [,16] [17,40][41,60][61,]
# merge_data['Age'] = pd.cut(merge_data.Age,bins=[0,16,40,60,120],labels=['kid','youth','middle-aged','elderly'])
# # Age labeling---------/

# merge_data = labeling(merge_data, ['Pclass']) 
# merge_data = labeling(merge_data, ['Sex'])
# # merge_data = labeling(merge_data, ['Ticket'])
# merge_data = labeling(merge_data, ['Ticket_alpha'])
# # merge_data = labeling(merge_data, ['Cabin'])
# merge_data = labeling(merge_data, ['Cabin_alpha'])
# merge_data = labeling(merge_data, ['Embarked'])
# # merge_data = labeling(merge_data, ['Name'])
# # merge_data = labeling(merge_data, ['Age'])

train = labeling(train, ['Pclass', 'Sex','Embarked','Ticket_alpha','Cabin_alpha'])
test = labeling(test, ['Pclass', 'Sex','Embarked','Ticket_alpha','Cabin_alpha'])

train = labeling(train, ['Pclass', 'Sex','Embarked','Cabin_alpha'])
test = labeling(test, ['Pclass', 'Sex','Embarked','Cabin_alpha'])

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

# check missing value -> do not exist missing value
msno.bar(train)
plt.show()
msno.bar(test)
plt.show()



# for submission
x_train, y_train = train.iloc[:,1:],train.Survived 
x_val = test
# for validation
x_train, x_val, y_train, y_val = train_test_split(train.iloc[:,1:],train.Survived, test_size=0.3 ) 
'''
남녀 나눠서 모델링
'''
temp2 = train[train.Sex==0]
temp = train[train.Sex==1]
temp2 = pd.concat([temp2, pd.get_dummies(temp2.Embarked, prefix = 'Embarked')], axis=1).drop(['Embarked'], axis=1)

x_train_0, x_val_0, y_train_0, y_val_0 = train_test_split(temp2.iloc[:,1:], temp2.Survived, test_size=0.3)
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(temp.iloc[:,1:], temp.Survived, test_size=0.3)

# GradientBoosting
GBC = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=1328, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
GBC.fit(x_train, y_train)

pred = GBC.predict(x_val[x_val.Sex==0])
pred_p = GBC.predict_proba(x_val)
GBC.score(x_val, y_val) #0.7889
cross_val_score(GBC, x_val, y_val).mean() # 0.7830667 # 0.7881

print(classification_report(pred, y_val))

# LIGHTGBM
lgb = lgbm.LGBMClassifier()
lgb.fit(x_train, y_train)

lgb_pred = lgb.predict(x_val)
lgb_pred_p = lgb.predict_proba(x_val)
lgb.score(x_val,y_val) # 0.7896666666666666
cross_val_score(lgb, x_val, y_val).mean() #0.7824333333333333

# Catboost
cat = CatBoostClassifier()
cat.fit(x_train, y_train)

cat_pred = cat.predict(x_val)
cat.score(x_val, y_val)
cross_val_score(cat, x_val, y_val).mean() #0.7828333333333334

# most voting
temp = pd.DataFrame({'gbc':pred, 'lgbm':lgb_pred, 'cat':cat_pred})


result_survival = np.argmax((pred_p + lgb_pred_p)/2, axis=1)
result_survival

submission = pd.read_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/sample_submission.csv')
submission['Survived'] = temp

submission.to_csv('C:/Users/10188/local_git/tabular-playground-series-apr-2021/submission_files/20210413_GBC_lgbm_cat_freqvoting.csv', index=False)


# PYCARET
set1 = setup(data=pd.concat([x_train, y_train], axis=1), target='Survived',
categorical_features=['Pclass','Sex','Embarked','Ticket_alpha','Cabin_alpha'],
numeric_features=['Age','Fare','famsize','survival_ratio'])
best = compare_models(n_select = 5)

model1 = create_model('gbc')
plot_model(model1, plot='feature') #->Gradient

model2 = create_model('lightgbm')

model3 = create_model('catboost')

# stacking model
bagged_m = ensemble_model(model1)
boosted_m = ensemble_model(model1, method='Boosting')

stacked_m = stack_models([model1, model2, model3], meta_model=model3)

print(classification_report(predict_model(stacked_m)['Label'], predict_model(stacked_m)['Survived'])) #0.77 -> 0.78 -> 0.79(0.789)


print(confusion_matrix(predict_model(stacked_m)['Label'].astype('int'),predict_model(stacked_m)['Survived'].astype('int')))
