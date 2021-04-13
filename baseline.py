import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce
import lightgbm as lgbm
import re


train = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("C:/Users/10188/local_git/tabular-playground-series-apr-2021/test.csv")
train.head()
train.info()
train.describe()

msno.matrix(train)
msno.bar(train) #Age, Ticket, Fare, Cabin, Embarked have null 

msno.bar(test) #Age, Ticket, Fare, Cabin(to drop), Embarked have null

numeric_v = ['Pclass','Age','SibSp','Parch','Fare']
string_v = ['Survived','Sex','Embarked','Ticket','Cabin','Name']
delete_v = ['PassengerId'] 
##################################################################
# - > name(family name), ticket(앞글자), cabin(a,b...)이용해보기 : Name은 의미x, ticket, cabin 이용
# - > famsize, name freq 를 categorical vari.로 이용해보기 & 남녀나눠서 모델링
##################################################################

# label encoding
def labeling(df, columns):
    encoder = LabelEncoder()
    cols = encoder.fit_transform(df[columns])
    df[columns] = cols
    return df

train.drop(delete_v, axis=1, inplace=True)
test.drop(delete_v, axis=1, inplace=True)

merge_data = pd.concat([train, test], axis=0, join='outer')
merge_data.reset_index(drop=True, inplace=True)

# if Ticket/Cabin/Embarked.isnull -> fill "X"
merge_data[['Ticket']] = merge_data[['Ticket']].fillna('X').astype('string')
merge_data[['Cabin']] = merge_data[['Cabin']].fillna('X').astype('string')
merge_data['Embarked'] = merge_data['Embarked'].fillna('X').astype('string')

# add family size variable
merge_data['famsize'] = merge_data.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
merge_data.drop(['SibSp','Parch'], axis=1, inplace=True)


# def famsize_cat(x):
#     if x==0:
#         return "single"
#     elif x>0 and x<=4:
#         return "normal"
#     elif x>4 and x<7:
#         return "middle"
#     else:
#         return "large"
# merge_data['famsize'] = merge_data['famsize'].apply(famsize_cat )


# Ticket type
merge_data['Ticket_alpha'] = merge_data['Ticket'].apply(lambda x: x.split(' ')[0])
merge_data['Ticket_alpha'] = merge_data['Ticket_alpha'].apply(lambda x: x if x.upper().isupper() else "X")

merge_data['Ticket_num'] = merge_data['Ticket'].apply(lambda x: x.split(' ')[1][:2] if x != 'X' and x.upper().isupper() else x[:2])


# compare survival ratio by Ticket
alpha_ratio = merge_data.groupby('Ticket_alpha').mean()[['Survived']].sort_values(by='Survived', ascending=False)
plot = sns.barplot(alpha_ratio.index, alpha_ratio.Survived)
plot.tick_params(labelsize=5)
plt.show() # --> add Ticket_alpha variable (survival ratio between 0.1~0.6 : difference)

num_ratio = merge_data.groupby('Ticket_num').mean()[['Survived']].sort_values(by='Survived', ascending=False)
plot = sns.barplot(num_ratio.index, num_ratio.Survived)
plot.tick_params(labelsize=7)
plt.show() # --> drop Ticket_num varible (survival ratio between 0.3~0.5 : similar)

merge_data.drop(['Ticket','Ticket_num'], axis=1, inplace=True)

# Cabin type -> A,B,C,D,E,F,G,S
merge_data['Cabin_alpha'] = merge_data['Cabin'].apply(lambda x: x[:2])

# compare survival ratio by cabin 
cabin_ratio = merge_data.groupby(['Cabin_alpha']).mean()[['Survived']].sort_values(by='Survived', ascending=False)
plot = sns.barplot(cabin_ratio.index, cabin_ratio.Survived)
plot.tick_params(labelsize=7)
plt.show() 

merge_data.drop(['Cabin'], axis=1, inplace=True)


# # get Family name
# merge_data['Name'] = merge_data['Name'].apply(lambda x: x.split(', ')[1])
merge_data['familyname'] = merge_data['Name'].apply(lambda x:x.split(' ')[1])
familyname_ratio = merge_data.groupby('familyname').mean()[['Survived']].sort_values(by='Survived', ascending=False)
sns.barplot(familyname_ratio.index, familyname_ratio.Survived)
plt.show() # --> have huge difference

# drop Name, familyname and add survival ratio by familyname -->  Target encoding instead of familyname
target_encoder = ce.TargetEncoder()
merge_data['survival_ratio'] = target_encoder.transform(merge_data.familyname, merge_data.Survived).values

merge_data.drop(['Name','familyname'], axis=1, inplace=True)



# # Age: categorical, Embarked: Categorical
# # Age: [,16] [17,40][41,60][61,]
# merge_data['Age'] = pd.cut(merge_data.Age,bins=[0,16,40,60,120],labels=['kid','youth','middle-aged','elderly'])
# # Age labeling---------/


merge_data = labeling(merge_data, ['Pclass']) 
merge_data = labeling(merge_data, ['Sex'])
# merge_data = labeling(merge_data, ['Ticket'])
merge_data = labeling(merge_data, ['Ticket_alpha'])
# merge_data = labeling(merge_data, ['Cabin'])
merge_data = labeling(merge_data, ['Cabin_alpha'])
merge_data = labeling(merge_data, ['Embarked'])
# merge_data = labeling(merge_data, ['Name'])
# merge_data = labeling(merge_data, ['Age'])


# trian/test split again
train = merge_data[merge_data['Survived'].notnull()]
train['Survived'] = train['Survived'].astype('int')

test = merge_data[merge_data['Survived'].isnull()].reset_index(drop=True)
test.drop(['Survived'],axis=1, inplace=True)

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
sns.heatmap(merge_data.corr(), annot=True)
plt.show() # Fare: Pclass와 0.4 correlation (Pclass별로 Fare를 impute)

# 1. IMPUTE FARE
fig, axes = plt.subplots(3,1,figsize=(10,5), sharey=True)
for i in range(3):
    sns.histplot(train[train['Pclass']==i]['Fare'], label='{} class'.format(i), legend=True,ax=axes[i])
    plt.legend()
plt.show() # right_skewed --> use median instead of mean

# impute Fare with groupby median(Pclass)
train['Fare'] = train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'))
test['Fare'] = test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'))

train['Fare'] = np.log(train['Fare'])
test['Fare'] = np.log(test['Fare'])


# 2. IMPUTE AGE
sns.displot(data=train, x='Age',col='Pclass', multiple='dodge')
plt.show()

train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'))
test['Age'] = test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'))

# 3. STANDARDIZE Ticket
# train['Fare'] = (train['Fare']-np.mean(train['Fare']))/np.std(train['Fare'])

# check missing value -> do not exist missing value
msno.bar(train)
plt.show()
msno.bar(test)
plt.show()


# GradientBoosting
x_train, x_val, y_train, y_val = train_test_split(train.iloc[:,1:],train.Survived, test_size=0.3 )

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

pred = GBC.predict(x_val)
pred_p = GBC.predict_proba(x_val)
GBC.score(x_val, y_val)

print(classification_report(pred, y_val))

# LIGHTGBM
lgb = lgbm.LGBMClassifier()
lgb.fit(x_train, y_train)

lgb_pred = lgb.predict(x_val)
lgb_pred_p = lgb.predict_proba(x_val)
lgb.score(x_val,y_val)


confusion_matrix(np.argmax((pred_p + lgb_pred_p)/2, axis=1), y_val)

# PYCARET
set1 = setup(data=pd.concat([x_train, y_train], axis=1), target='Survived')
best = compare_models(n_select = 5)

model1 = create_model('gbc')
plot_model(model1, plot='feature') #->Gradient

model2 = create_model('lightgbm')

model3 = create_model('catboost')

# stacking model
bagged_m = ensemble_model(model1)
boosted_m = ensemble_model(model1, method='Boosting')

stacked_m = stack_models([model1, model2, model3], meta_model=model1)

print(classification_report(predict_model(stacked_m)['Label'], predict_model(stacked_m)['Survived'])) #0.77 -> 0.78 -> 0.79(0.789)


print(confusion_matrix(predict_model(stacked_m)['Label'],predict_model(stacked_m)['Survived']))
