import pandas as pd
import pickle as pk

model = pk.load(open('./models/Floresta_RandomicaF.sav', mode='rb'))

with open('./pre_process/titanic_testF.pkl', mode='rb') as f:
    X_test_processed, dataT_id = pk.load(f)

prev = model.predict(X_test_processed)

submission_df = pd.DataFrame({
    'PassengerId': dataT_id,
    'Survived': prev
})

submission_df['Survived'] = submission_df['Survived'].astype('int64').astype('Int64')

submission_df.to_csv('./db/submission.csv', index=False)

