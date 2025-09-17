import pandas as pd

models = pd.read_csv('./db\models.csv')


print(models.loc[[2,5], ['Model', 'Accuracy', 'Accuracy_score']])