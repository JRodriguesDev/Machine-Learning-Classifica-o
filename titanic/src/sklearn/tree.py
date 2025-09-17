import pickle as pk
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier 

with open('./pre_process/titanic.pkl', mode='rb') as f:
    X_train, X_test, Y_train, Y_test = pk.load(f)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
standard = StandardScaler()
model = MLPClassifier(hidden_layer_sizes=(13,13), tol=0.0000100, verbose=True, max_iter=2000)

categoric_cols = ['Sex', 'Title', 'Embarked', 'FareBand', 'AgeBand', 'IsAlone', 'HasCabin', 'Deck']
numeric_cols = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch', 'TicketGroupSize']

X_trainCaregorics = encoder.fit_transform(X_train[categoric_cols])
X_TrainNumerics = standard.fit_transform(X_train[numeric_cols])
featuresNameCat = encoder.get_feature_names_out(categoric_cols)

X_train_cat_df = pd.DataFrame(X_trainCaregorics, columns=featuresNameCat, index=X_train.index)
X_train_num_df = pd.DataFrame(X_TrainNumerics, columns=numeric_cols, index=X_train.index)
X_train_processed = pd.concat([X_train_cat_df, X_train_num_df], axis=1)

X_testCategorics = encoder.transform(X_test[categoric_cols])
X_testNumerics = standard.transform(X_test[numeric_cols])

X_test_cat_df = pd.DataFrame(X_testCategorics, columns=featuresNameCat, index=X_test.index)
X_test_num_df = pd.DataFrame(X_testNumerics, columns=numeric_cols, index=X_test.index)
X_test_procesed = pd.concat([X_test_cat_df, X_test_num_df], axis=1)

model.fit(X_train_processed, Y_train)
prev = model.predict(X_train_processed)

print(accuracy_score(Y_train, prev))
print(classification_report(Y_train, prev))