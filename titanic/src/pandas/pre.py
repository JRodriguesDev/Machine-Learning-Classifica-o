import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
import pickle as pk

data = pd.read_csv('./db/train.csv')
dataT = pd.read_csv('./db/test.csv')
data = data.drop(columns=['PassengerId'])
dataT_id = dataT['PassengerId']
dataT = dataT.drop(columns='PassengerId')


standard = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

data['Survived'] = data['Survived'].astype('Int64')

data['Title'] = data['Name'].str.extract(r'\s([A-Za-z]+)\.')
rareTitles = data['Title'].value_counts()[data['Title'].value_counts() < 10].index
ticketCounts = data['Ticket'].value_counts()
data['Deck'] = data['Cabin'].str[0]
data['Deck'] = data['Deck'].fillna('U')
rare_decks = data['Deck'].value_counts()[data['Deck'].value_counts() < 10].index
ageM = data['Age'].median()
fareM = data['Fare'].median()
embarkM = data['Embarked'].mode()[0]
max_age_train = data['Age'].max() + 1

datas = [data, dataT]

for i, dt in enumerate(datas):
    dt = dt.replace('', np.nan)
    dt = dt.replace('-', np.nan)
    dt.loc[(dt['Age'] == 0) | (dt['Age'] < 0), 'Age'] = np.nan
    dt.loc[dt['SibSp'] < 0, 'SibSp'] = np.nan
    dt.loc[dt['Parch'] < 0, 'Parch'] = np.nan
    dt.loc[(dt['Fare'] < 0) | (dt['Fare'] == 0), 'Fare'] = np.nan
    dt['Title'] = dt['Name'].str.extract(r'\s([A-Za-z]+)\.')
    dt['FamilySize'] = dt['Parch'] + dt['SibSp'] + 1
    dt['Title'] = dt['Title'].replace(rareTitles, 'Other')
    dt['IsAlone'] = (dt['FamilySize'] == 1).astype('int').astype('category')
    dt['TicketGroupSize'] = dt['Ticket'].map(ticketCounts)
    dt['TicketGroupSize'] = dt['Ticket'].map(ticketCounts).fillna(1)
    dt['HasCabin'] = dt['Cabin'].notna().astype('int')
    dt['Deck'] = dt['Cabin'].str[0]
    dt['Deck'] = dt['Deck'].fillna('U')
    dt['Deck'] = dt['Deck'].replace(rare_decks, 'OtherDeck')
    dt['Age'] = dt['Age'].fillna(ageM)
    dt['Fare'] = dt['Fare'].fillna(fareM)
    dt['Embarked'] = dt['Embarked'].fillna(embarkM)

    dt['FareBand'] = pd.qcut(dt['Fare'], q=4, labels=['VeryLow', 'Low', 'Medium', 'High'], duplicates='drop').astype('category')
    dt['AgeBand'] = pd.cut(dt['Age'], bins=[0, 12, 18, 35, 60, max_age_train], labels=['Child', 'Teenager', 'YoungAdult', 'Adult', 'Senior'], right=True)
    dt['Title'] = dt['Title'].astype('category')
    dt['Pclass'] = dt['Pclass'].astype('category')
    dt['Sex'] = dt['Sex'].astype('category')
    dt['Age'] = dt['Age'].astype('int64').astype('Int64')
    dt['SibSp'] = dt['SibSp'].astype('Int64')
    dt['Parch'] = dt['Parch'].astype('Int64')
    dt['Fare'] = dt['Fare'].astype('Float64')
    dt['Embarked'] = dt['Embarked'].astype('category')
    dt['FamilySize'] = dt['FamilySize'].astype('Int64')
    dt['TicketGroupSize'] = dt['TicketGroupSize'].astype('Int64')
    dt['HasCabin'] = dt['HasCabin'].astype('category')
    dt['Deck'] = dt['Deck'].astype('category')
    dt = dt.drop(columns=['Name', 'Cabin', 'Ticket'])
    datas[i] = dt

dataY = data['Survived']
data = data.drop(columns='Survived')

#X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.25, random_state=42, stratify=dataY)

dataX, dataT = datas

categoric_cols = ['Sex', 'Title', 'Embarked', 'FareBand', 'AgeBand', 'IsAlone', 'HasCabin', 'Deck']
numeric_cols = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch', 'TicketGroupSize']

X_train_categoric_cols = encoder.fit_transform(dataX[categoric_cols])
X_train_numeric_cols = standard.fit_transform(dataX[numeric_cols])
features_cat = encoder.get_feature_names_out(categoric_cols)

X_train_categoric_df = pd.DataFrame(X_train_categoric_cols, columns=features_cat, index=dataX.index)
X_train_numeric_df = pd.DataFrame(X_train_numeric_cols, columns=numeric_cols, index=dataX.index)
X_train_processed = pd.concat([X_train_categoric_df, X_train_numeric_df], axis=1)

X_test_categoric_cols = encoder.transform(dataT[categoric_cols])
X_test_numeric_cols = standard.transform(dataT[numeric_cols])

X_test_categoric_df = pd.DataFrame(X_test_categoric_cols, columns=features_cat, index=dataT.index)
X_test_numeric_df = pd.DataFrame(X_test_numeric_cols, columns=numeric_cols, index=dataT.index)
X_test_processed = pd.concat([X_test_categoric_df, X_test_numeric_df], axis=1)

with open('./pre_process/titanicF.pkl', mode='wb') as f:
    pk.dump([X_train_processed, dataY], f)

with open('./pre_process/titanic_testF.pkl', mode='wb') as f:
    pk.dump([X_test_processed, dataT_id], f)



