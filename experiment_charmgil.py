import pandas as pd
from pandas import DataFrame as df
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import missingno as msno


from WeightedRMSE import wrmse

print('data loading...')

column_names = ['year', 'doy', 'hr', 'min', 'Np', 'Vp', 'Tp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt', 'Kp']
#data = pd.read_csv(r'D:\charmgil\2019space_data\new_data_3hour_stats.csv',  delimiter=',')
data_tr = pd.read_csv('../organized_data/new_data_3hour_stats.csv',  delimiter=',')
data_ts = pd.read_csv('../organized_data/new_data_long_3hour_stats_with_nan.csv',  delimiter=',')

'''
data_tr = data[data['year'] <= 2012]    ## originally 2010
data_ts = data[data['year'] > 2012]     ## originally 2010
'''

'''
for val in range(0,10):
    data_tr[data_tr['Kp'] == val] = data_tr[data_tr['Kp'] == val].fillna(data_tr[data_tr['Kp'] == val].mean(axis=0, skipna=True))

for val in range(0,10):
    data_ts[data_ts['Kp'] == val] = data_ts[data_ts['Kp'] == val].fillna(data_ts[data_ts['Kp'] == val].mean(axis=0, skipna=True))
'''

data_tr = data_tr[data_tr.columns[4:]]
print(data_tr)
data_ts = data_ts[data_ts.columns[2:]]
print(data_ts)

for cols in range(0,len(data_tr.columns)):
    data_tr[data_tr.columns[cols]].fillna(data_tr[data_tr.columns[cols]].mean(skipna=True), inplace=True)
    #data_tr[data_tr.columns[cols]].replace(np.nan, np.nanmean(data_tr[data_tr.columns[cols]], axis=0))

#print(data_tr['B_gsm_x'][12253])
for cols in range(0,len(data_ts.columns)):
    data_ts[data_ts.columns[cols]].fillna(data_ts[data_ts.columns[cols]].mean(skipna=True), inplace=True)
    #data_ts[data_tr.columns[cols]].replace(np.nan, np.nanmean(data_ts[data_tr.columns[cols]], axis=0))

imputer = SimpleImputer(strategy="median")

'''
data_tr['Np'] = imputer.fit_transform(data_tr.Np.values.reshape(-1,1))
data_tr['Tp'] = imputer.fit_transform(data_tr.Tp.values.reshape(-1, 1))
data_tr['Vp'] = imputer.fit_transform(data_tr.Vp.values.reshape(-1, 1))
data_tr['B_gsm_x'] = imputer.fit_transform(data_tr.B_gsm_x.values.reshape(-1, 1))
data_tr['B_gsm_y'] = imputer.fit_transform(data_tr.B_gsm_y.values.reshape(-1, 1))
data_tr['B_gsm_z'] = imputer.fit_transform(data_tr.B_gsm_z.values.reshape(-1, 1))
data_tr['Bt'] = imputer.fit_transform(data_tr.Bt.values.reshape(-1, 1))

data_ts['Np'] = imputer.fit_transform(data_ts.Np.values.reshape(-1, 1))
data_ts['Tp'] = imputer.fit_transform(data_ts.Tp.values.reshape(-1, 1))
data_ts['Vp'] = imputer.fit_transform(data_ts.Vp.values.reshape(-1, 1))
data_ts['B_gsm_x'] = imputer.fit_transform(data_ts.B_gsm_x.values.reshape(-1, 1))
data_ts['B_gsm_y'] = imputer.fit_transform(data_ts.B_gsm_y.values.reshape(-1, 1))
data_ts['B_gsm_z'] = imputer.fit_transform(data_ts.B_gsm_z.values.reshape(-1, 1))
data_ts['Bt'] = imputer.fit_transform(data_ts.Bt.values.reshape(-1, 1))
'''

data_tr.to_csv('../organized_data/data_tr_nan.csv')
data_ts.to_csv('../organized_data/data_ts_nan.csv')
mean_list = {}

'''
for i in range(0, 8):
    mean_list[str(data_tr.columns[i])] = np.mean(data_tr[data_tr.columns[i]])
    print('mean: ', mean_list[str(data_tr.columns[i])])
'''

#for val in range(0,10):
    #data_tr[data_tr['Kp'] == val] = data_tr[data_tr['Kp'] == val].fillna(data_tr[data_tr['Kp'] == val].mean(axis=0, skipna=True))


print('downsampling...')
count_per_kp = data_tr.groupby('Kp')[data_tr.columns[0]].nunique()
print(count_per_kp)

undersampling_rate = 0.95
data_tr_sample = pd.DataFrame(columns=data_tr.columns)

for i in range(0, 5):
    tmp_df = data_tr.loc[data_tr['Kp'] == i].sample(n=int(np.round(count_per_kp[i]*undersampling_rate)), replace=False)
    data_tr_sample = data_tr_sample.append(tmp_df, ignore_index=True)

for i in range(5, 9):
    tmp_df = data_tr.loc[data_tr['Kp'] == i]
    if len(tmp_df) > 0:
        data_tr_sample = data_tr_sample.append(tmp_df, ignore_index=True)
data_tr = data_tr_sample

print(data_tr.groupby('Kp')[data_tr.columns[0]].nunique())

Y_tr = data_tr.loc[:, data_tr.columns == 'Kp']
Y_ts = data_ts.loc[:, data_ts.columns == 'Kp']

X_tr = data_tr.loc[:, data_tr.columns != 'Kp']
X_ts = data_ts.loc[:, data_ts.columns != 'Kp']

print(X_tr.shape)
print(X_ts.shape)
print(Y_tr.shape)
print(Y_ts.shape)
'''
for i in range(0,8):
    X_ts[data_tr[data_tr.columns[i]]] = X_ts[data_tr[data_tr.columns[i]]].replace(np.nan, mean_list[str(data_tr.columns[i])])
'''
# normalize
normalizer = StandardScaler()
normalizer.fit(X_tr)
print('normalizing...')
print(X_ts)
X_tr_normalized = normalizer.transform(X_tr)
X_ts_normalized = normalizer.transform(X_ts)

# base_model = ElasticNet(fit_intercept=True, normalize=False, max_iter=3000)
# parameters = {#'alpha':[0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000000],
#               'alpha':[0.0001, 0.1, 1, 1000],
#               'l1_ratio':[1.0, 0.5, 0.]}



### Linear Regression
print('** LinearRegression() **')

base_model = LinearRegression()
parameters = {'fit_intercept':[True]}

print('training...')
model = GridSearchCV(base_model, param_grid=parameters, cv=5, scoring='neg_mean_absolute_error')
model.fit(X_tr_normalized, Y_tr)
print(model.best_params_)

# test
print('making predictions...')
Y_ts_pred_linr = model.predict(X_ts_normalized)

print('RESULT(wrmse):')
#wrmse(Y_ts_pred_linr,Y_ts)

print('RESULT(wrmse)-rounded:')
#wrmse(np.clip(np.round(Y_ts_pred_linr), a_min=0, a_max=9),Y_ts)

### ridge_regression
print('** ridge_regression() **')

weight_tr = Y_tr / Y_tr.sum(axis=0)

base_model = Ridge()
parameters = {'alpha':[.0001, .01, .1, 1., 10, 100, 10000]}

print('training...')
model = GridSearchCV(base_model, param_grid=parameters, cv=5, scoring='neg_mean_absolute_error')
model.fit(X_tr_normalized, Y_tr, weight_tr) # fit_params={'sample_weight': weight_tr}
print(model.best_params_)

# test
print('making predictions...')
Y_ts_pred_ridge = model.predict(X_ts_normalized)
Y_ts_pred_ridge = df(Y_ts_pred_ridge.round(0).astype(int))
Y_ts_pred_ridge.to_csv('../organized_data/y_answer_finally_ridge.csv', index = False)

print('RESULT(wrmse):')
wrmse(Y_ts_pred_ridge,Y_ts)

print('RESULT(wrmse)-rounded:')
wrmse(np.clip(np.round(Y_ts_pred_ridge), a_min=0, a_max=9),Y_ts)


### RandomForestRegressor
print('** RandomForestRegressor **')
rf_model = RandomForestRegressor(n_estimators=150, criterion='mse', max_depth=None,
                                   min_samples_split=2, min_samples_leaf=25, min_weight_fraction_leaf=0.0,
                                   max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
                                   random_state=None, verbose=0, warm_start=False)
print('training...')
rf_model.fit(X_tr_normalized, Y_tr.values.ravel())

print('making predictions...')
Y_ts_pred_rf = rf_model.predict(X_ts_normalized)
Y_ts_pred_rf = Y_ts_pred_rf.reshape(len(Y_ts_pred_rf),1)

print('RESULT(wrmse):')
#wrmse(Y_ts_pred_rf, Y_ts)

print('RESULT(wrmse)-rounded:')
wrmse(np.clip(np.round(Y_ts_pred_rf), a_min=0, a_max=9),Y_ts)

### Support Vector Regression
print('** SVR() **')

base_model = SVR(gamma='scale')
parameters = {'C':[.0001, .01, .1, 1., 10, 100, 10000],
              'epsilon': [.05, .1, .5, 1.]}

print('training...')
model = GridSearchCV(base_model, param_grid=parameters, cv=5, scoring='neg_mean_absolute_error')
model.fit(X_tr_normalized, Y_tr.values.ravel())

# test
print('making predictions...')
Y_ts_pred_svr = model.predict(X_ts_normalized)
Y_ts_pred_svr = Y_ts_pred_svr.reshape(len(Y_ts_pred_svr),1)

print('RESULT(wrmse):')
wrmse(Y_ts_pred_svr,Y_ts)


