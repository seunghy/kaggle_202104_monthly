from sklearn.preprocessing import LabelEncoder

# label encoding
def labeling(df, columns):
    encoder = LabelEncoder()
    for c in columns:
        col = encoder.fit_transform(df[c])
        df[c] = col
    return df

def fillNA(df,cols, replaceString='X'):
    df[cols] = df[cols].fillna(replaceString).astype('string')
    return df

def imputation(df, cols, groupby=None, value='mean'):
    if groupby == None:
        df[col] = df[cols].fillna(np.mean(df[cols]))
    else:
        df[cols] = df[cols].fillna(df.groupby(groupby).transform(value)[cols])
    return df[cols]