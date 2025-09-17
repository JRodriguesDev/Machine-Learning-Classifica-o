import pickle as pk
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    with open('./pre_process/titanicF.pkl', mode='rb') as f:
        #X_train_processed, X_test_processed, Y_train, Y_test = pk.load(f)
        X_train_processed, dataY = pk.load(f)

    model_neural = MLPClassifier(random_state=42, max_iter=3000)
    model_tree = DecisionTreeClassifier()
    model_random = RandomForestClassifier()
    model_KNN = KNeighborsClassifier()
    model_regression = LogisticRegression(max_iter=1000)
    model_SVM = SVC()

    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    param_grid_neural = {
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'hidden_layer_sizes': [(8,8), (10,10), (15,15), (18,18), (20,20), (30,30), (40,40), (50,50), (100, 100)],
        'tol': [0.01, 0.001, 0.0001, 0.00001],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    }

    param_grid_tree = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 8, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': [None, 'sqrt', 'log2']
    }

    param_grid_random = {
        'n_estimators': [50, 100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    param_grid_KNN = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    param_grid_regress = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'saga'],
    }

    param_grid_SVM = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1, 1.0]
    }

    list_models = {
        'Rede Neural': [model_neural, param_grid_neural],
        'Arvore': [model_tree, param_grid_tree],
        'Floresta Randomica': [model_random, param_grid_random],
        'KNN': [model_KNN, param_grid_KNN],
        'Regressão': [model_regression, param_grid_regress],
        'SVM': [model_SVM, param_grid_SVM]
    }

    results = []

    for name, (model, param) in list_models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param, cv=cv_strategy, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(X_train_processed, dataY)
        #predict = grid_search.predict(X_test_processed)

        results.append({
            'Model': name,
            'Accuracy': grid_search.best_score_,
            'Best_estimator': grid_search.best_estimator_,
            'Best_params': grid_search.best_params_,
            #'Accuracy_score': accuracy_score(Y_test, predict),
            #'Classification_report': classification_report(Y_test, predict)
        })

        with open(f'./models/{name}F.sav', mode='wb') as f:
            pk.dump(grid_search.best_estimator_, f)

    results_df = pd.DataFrame(results)
    results_df.to_csv('./db/modelsF.csv', sep=',')
