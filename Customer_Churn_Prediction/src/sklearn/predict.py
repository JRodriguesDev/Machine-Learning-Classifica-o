import pandas as pd
import pickle as pk

data_IDs = pd.read_csv('./db/final_data/test_IDs.csv', sep=',')

with open('./db/final_data/test.pkl', mode='rb') as f:
    X_test_processed, L_encoder = pk.load(f)

with open('./db/models/tree_best_model.pkl', mode='rb') as f:
    model_tree = pk.load(f)

prev = model_tree.predict(X_test_processed)

submission_df = pd.DataFrame({
    'id': data_IDs.iloc[:, 0],
    'churn': L_encoder.inverse_transform(prev)
})

submission_df.to_csv('./db/final_data/submission.csv', sep=',', index=False)