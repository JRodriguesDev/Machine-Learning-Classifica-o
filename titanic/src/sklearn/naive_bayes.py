import pickle as pk
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

with open('./pre_process/titanic.pkl', mode='rb') as f:
    X_train, X_test, Y_train, Y_test = pk.load(f)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
standard = StandardScaler()
model = GaussianNB()

categoricCols = ['Sex', 'Title', 'Embarked', 'FareBand', 'AgeBand', 'IsAlone', 'HasCabin', 'Deck']
numericCols = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch', 'TicketGroupSize']

X_trainCategorics = encoder.fit_transform(X_train[categoricCols])
X_trainNumerics = standard.fit_transform(X_train[numericCols])

featuresNameCat = encoder.get_feature_names_out(categoricCols)
print(featuresNameCat)

X_train_cat_df = pd.DataFrame(X_trainCategorics, columns=featuresNameCat, index=X_train.index)
X_train_num_df = pd.DataFrame(X_trainNumerics, columns=numericCols, index=X_train.index)
X_train_processed = pd.concat([X_train_cat_df, X_train_num_df], axis=1)

X_testCategorics = encoder.transform(X_test[categoricCols])
X_testNumerics = standard.transform(X_test[numericCols])

X_test_cat_df = pd.DataFrame(X_testCategorics, columns=featuresNameCat, index=X_test.index)
X_test_num_df = pd.DataFrame(X_testNumerics, columns=numericCols, index=X_test.index)
X_test_processed = pd.concat([X_test_cat_df, X_test_num_df], axis=1)

model.fit(X_train_processed, Y_train)

prev = model.predict(X_test_processed)

print(accuracy_score(Y_test, prev))
print(classification_report(Y_test, prev))

#with open('./models/naive_bayes.pkl', mode='wb') as f:
#    pk.dump([encoder, standard, model], f)