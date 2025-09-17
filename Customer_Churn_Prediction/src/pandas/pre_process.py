import pandas as pd
import numpy as np
import pickle as pk
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('./db/datas/train.csv', sep=',')
data_test = pd.read_csv('./db/datas/test.csv', sep=',')

data_trainY = data_train['churn']
data_train = data_train.drop('churn', axis=1)
data_test_id = data_test['id']
data_test = data_test.drop('id', axis=1)

standard = StandardScaler()
O_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
L_encoder = LabelEncoder()

data_trainY = data_trainY.astype('string').astype('category')

datas = [data_train, data_test]

for i, data in enumerate(datas):
    data = data.replace('', np.nan)
    data = data.replace('-', np.nan)
    data.loc[data['account_length'] < 0, 'account_length'] = np.nan
    data.loc[data['number_vmail_messages'] < 0, 'number_vmail_messages'] = np.nan
    data.loc[data['total_day_minutes'] < 0, 'total_day_minutes'] = np.nan
    data.loc[data['total_day_calls'] < 0, 'total_day_calls'] = np.nan
    data.loc[data['total_day_charge'] < 0, 'total_day_charge'] = np.nan
    data.loc[data['total_eve_minutes'] < 0, 'total_eve_minutes'] = np.nan
    data.loc[data['total_eve_calls'] < 0, 'total_eve_calls'] = np.nan
    data.loc[data['total_eve_charge'] < 0, 'total_eve_charge'] = np.nan
    data.loc[data['total_night_minutes'] < 0, 'total_night_minutes'] = np.nan
    data.loc[data['total_night_calls'] < 0, 'total_night_calls'] = np.nan
    data.loc[data['total_night_charge'] < 0, 'total_night_charge'] = np.nan
    data.loc[data['total_intl_minutes'] < 0, 'total_intl_minutes'] = np.nan
    data.loc[data['total_intl_calls'] < 0, 'total_intl_calls'] = np.nan
    data.loc[data['total_intl_charge'] < 0, 'total_intl_charge'] = np.nan
    data.loc[data['number_customer_service_calls'] < 0, 'number_customer_service_calls'] = np.nan

    data['state'] = data['state'].astype('string').astype('category')
    data['account_length'] = data['account_length'].astype('int64')
    data['area_code'] = data['area_code'].astype('string').astype('category')
    data['international_plan'] = data['international_plan'].astype('string').astype('category')
    data['voice_mail_plan'] = data['voice_mail_plan'].astype('string').astype('category')
    data['number_vmail_messages'] = data['number_vmail_messages'].astype('int64')
    data['total_day_calls'] = data['total_day_calls'].astype('int64')
    data['total_eve_calls'] = data['total_eve_calls'].astype('int64')
    data['total_night_calls'] = data['total_night_calls'].astype('int64')
    data['total_intl_calls'] = data['total_intl_calls'].astype('int64')

    data['total_minutes'] = data['total_day_minutes'] + data['total_eve_minutes'] + data['total_night_minutes'] + data['total_intl_minutes'].astype('float64')
    data['total_calls'] = data['total_day_calls'] + data['total_eve_calls'] + data['total_night_calls'] + data['total_intl_calls'].astype('int64')
    data['total_charge'] = data['total_day_charge'] + data['total_eve_charge'] + data['total_night_charge'] + data['total_intl_charge'].astype('float64')
    data['has_vmail_messages'] = (data['number_vmail_messages'] > 0).astype('int64').astype('category')
    data['no_day_usage'] = ((data['total_day_minutes'] == 0) & (data['total_day_calls'] == 0) & (data['total_day_charge'] == 0)).astype('int64').astype('category')
    data['no_eve_usage'] = ((data['total_eve_minutes'] == 0) & (data['total_eve_calls'] == 0) & (data['total_eve_charge'] == 0)).astype('int64').astype('category')
    data['no_night_usage'] = ((data['total_night_minutes'] == 0) & (data['total_night_calls'] == 0) & (data['total_night_charge'] == 0)).astype('int64').astype('category')
    data['no_intl_usage'] = ((data['total_intl_minutes'] == 0) & (data['total_intl_calls'] == 0) & (data['total_intl_charge'] == 0)).astype('int64').astype('category')
    data['no_total_usage'] = (data['total_minutes'] == 0).astype('int64').astype('category')
    data['day_perc_total_minutes'] = np.where(data['total_minutes'] == 0, 0, data['total_day_minutes'] / data['total_minutes']).astype('float64')
    data['eve_perc_total_minutes'] = np.where(data['total_minutes'] == 0, 0, data['total_eve_minutes'] / data['total_minutes']).astype('float64')
    data['night_perc_total_minutes'] = np.where(data['total_minutes'] == 0, 0, data['total_night_minutes'] / data['total_minutes']).astype('float64')
    data['intl_perc_total_minutes'] = np.where(data['total_minutes'] == 0, 0, data['total_intl_minutes'] / data['total_minutes']).astype('float64')
    data['avg_charge_per_minute'] = np.where(data['total_minutes'] == 0, 0, data['total_charge'] / data['total_minutes']).astype('float64')
    data['calls_per_minute'] = np.where(data['total_minutes'] == 0, 0, data['total_calls'] / data['total_minutes']).astype('float64')
    data['calls_to_cs_ratio'] = np.where(data['total_minutes'] == 0, 0, data['number_customer_service_calls'] / data['total_calls']).astype('float64')
    data['high_cs_calls'] = (data['number_customer_service_calls'] >= 4).astype('Int64').astype('category')
    data['plan_complexity'] = ((data['voice_mail_plan'] == 'yes') & (data['international_plan'] == 'yes')).astype('int64').astype('category')
    datas[i] = data

data_train, data_test = datas

#X_train, X_test, Y_train, Y_test = train_test_split(data_train, data_trainY, test_size=0.20, random_state=42, stratify=data_trainY)

categoric_cols = data_train.select_dtypes(include='category').columns.to_list()
numeric_cols = data_train.select_dtypes(include=np.number).columns.to_list()

X_train_categoric_cols = O_encoder.fit_transform(data_train[categoric_cols])
features_cols = O_encoder.get_feature_names_out(categoric_cols)
X_train_numeric_cols = standard.fit_transform(data_train[numeric_cols])
X_test_categoric_cols = O_encoder.transform(data_test[categoric_cols])
X_test_numeric_cols = standard.transform(data_test[numeric_cols])

Y_train_cols = L_encoder.fit_transform(data_trainY)
#Y_test_cols = L_encoder.transform(Y_test)

X_train_categoric_df = pd.DataFrame(X_train_categoric_cols, columns=features_cols, index=data_train.index)
X_train_numeric_df = pd.DataFrame(X_train_numeric_cols, columns=numeric_cols, index=data_train.index)
X_test_categoric_df = pd.DataFrame(X_test_categoric_cols, columns=features_cols, index=data_test.index)
X_test_numeric_df = pd.DataFrame(X_test_numeric_cols, columns=numeric_cols, index=data_test.index)

X_train_processed = pd.concat([X_train_categoric_df, X_train_numeric_df], axis=1)
X_test_processed = pd.concat([X_test_categoric_df, X_test_numeric_df], axis=1)

with open('./db/pre_process/data_pre_process.pkl', mode='wb') as f:
    pk.dump([X_train_processed, Y_train_cols], f)

with open('./db/final_data/test.pkl', mode='wb') as f:
    pk.dump([X_test_processed, L_encoder], f)

data_test_id.to_csv('./db/final_data/test_IDs.csv', sep=',', index=False)

