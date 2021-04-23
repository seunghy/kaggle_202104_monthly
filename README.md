# kaggle_202104_monthly

## dataset _ aumented Titanic data

Private folder as storage

[기록]
numeric_v = ['Pclass','Age','SibSp','Parch','Fare']
string_v = ['Survived','Sex','Embarked','Ticket','Cabin','Name']
delete_v = ['PassengerId'] 

- Survived
- Pclass
- Age
- SibSp
- Parch
- Fare
- Sex
- Embarked
- Ticket
- Cabin
- Name
- PseengerId

2021.04.13
[V] - 제출 : val 0.78xx , public 0.792
    SibSp + Parch : familysize변수 파생
    Cabin : Cabin_alpha변수 파생(알파벳 + 숫자한자리)
    Age : Pclass + Cabin_alpha 별 median imputation
    Fare : Pclass + Cabin_alpha 별 median imputation
    Ticket : Ticket_alpha변수 파생(알파벳만 추출, 숫자만 있으면 X)
    Name : family name추출하여 family name별 survival ratio 계산

    모델: lgbm+bgc -> 확률평균 이용
[V]
성별에 따라 나눠서 fit
남성(1)의 경우 lgbm val: 0.8160974160974162
여성(0)의 경우 Embarked에서 남성과 차이가 있어 dummy함.
-->[] 남성/여성 나누어서 gbc, lgbm, catboost 로 ensemble하여 fit (각각 parameter tuning)


