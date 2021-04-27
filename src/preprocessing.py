import pandas as pd

def fillNAs(df, col):
    '''import labeling and fillNA'''
    df[col] = fillNA(df,col)
    return df

def get_familysize(df):
    # add family size variable
    df['famsize'] = df.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
    return df

# get ticket type by alphabet and number
def ticket_alpha(df):    
    df['Ticket_alpha'] = df['Ticket'].apply(lambda x: x.split(' ')[0])
    df['Ticket_alpha'] = df['Ticket_alpha'].apply(lambda x: x if x.upper().isupper() else "X")
    df['Ticket_num'] = df['Ticket'].apply(lambda x: x.split(' ')[1][:2] if x != 'X' and x.upper().isupper() else x[:2])
    return df
    
def cabin_alpha(df):
    # get cabin alphabet(by 2 char)
    df['Cabin_alpha'] = df['Cabin'].apply(lambda x : x[:2])
    return df

def cal_survivalratio(df, ratio_dict, familyname_ratio):
    # # get Family name
    df['survival_ratio'] = df['familyname'].map(ratio_dict)
    df['survival_ratio'] = df['survival_ratio'].fillna(familyname_ratio.median())
    return df