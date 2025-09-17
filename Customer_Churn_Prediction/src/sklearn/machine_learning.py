import pandas as pd
import pickle as pk

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

with open('./db/pre_process/data_pre_process.pkl', mode='rb') as f:
    X_train_processed, Y_train_cols = pk.load(f)

model_tree = DecisionTreeClassifier(random_state=42)
model_random_forest = RandomForestClassifier()
model_KNN = KNeighborsClassifier()
model_regression = LogisticRegression(max_iter=5000)
model_SVM = SVC()
model_neural_network = MLPClassifier(max_iter=5000)

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

params_tree = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5 ,10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

params_random_forest = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

params_KNN = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}

params_regression = [
    {'penalty': ['l1'], 'solver': ['liblinear'], 'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']},
    {'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.9], 'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']},
    {'penalty': [None], 'solver': ['lbfgs'], 'class_weight': [None, 'balanced']}
]

params_SVM = [
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'class_weight': [None, 'balanced']},
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100], 'class_weight': [None, 'balanced']}
]

params_neural_network = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'learning_rate_init': [0.001, 0.01, 0.1, 1.0],
    'early_stopping': [True, False]
}

training_models = {
    'tree': [model_tree, params_tree],
    'random_forest': [model_random_forest, params_random_forest],
    'KNN': [model_KNN, params_KNN],
    'regression': [model_regression, params_regression],
    'SVM': [model_SVM, params_SVM],
    'neural_network': [model_neural_network, params_neural_network]
}

results = []

for name, (model, params) in training_models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv_strategy, n_jobs=-1, verbose=2, scoring='f1')
    grid_search.fit(X_train_processed, Y_train_cols)
    #prev = grid_search.predict(X_test_processed)

    results.append({
        'model': name,
        'f1_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        #'accuracy_score': accuracy_score(Y_test_cols, prev),
        #'classification_report': classification_report(Y_test_cols, prev, output_dict=True)
    })

    with open(f'./db/models/{name}_best_model.pkl', mode='wb') as f:
        pk.dump(grid_search.best_estimator_, f)

results_df = pd.DataFrame(results)
results_df.to_csv('./db/final_data/models_results.csv', sep=',', index=False)



