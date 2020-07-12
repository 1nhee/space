import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def wrmse(Y_pred, Y_true, verbose=True):
    w = Y_true / Y_true.sum(axis=0)
    wrmse = np.sqrt((w * ((Y_pred - Y_true) ** 2)).sum())
    
    return np.sqrt(((Y_pred - Y_true) ** 2).mean()), wrmse

file = open('./result_ML.txt', 'w')
print('data loading...')

# column_names = ['year', 'doy', 'hr', 'min', 'Np', 'Vp', 'Tp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt', 'Kp']
data = pd.read_csv('../organized_data/new_data_3hour_stats.csv',  delimiter=',')


# for val in range(0,10):
#     data[data['Kp'] == val] = data[data['Kp'] == val].fillna(data[data['Kp'] == val].mean(axis=0, skipna=True))

data_tr = data[data['year'] <= 2006]    ## originally 2010
data_ts = data[data['year'] > 2006]     ## originally 2010

Y_tr = data_tr.loc[:, data_tr.columns == 'Kp']
Y_ts = data_ts.loc[:, data_ts.columns == 'Kp']

X_tr = data_tr.loc[:, data_tr.columns != 'Kp']
X_ts = data_ts.loc[:, data_ts.columns != 'Kp']

X_tr = X_tr[X_tr.columns[4:]]
print(X_tr)
X_ts = X_ts[X_ts.columns[4:]]
print(X_ts)

for cols in range(0,len(X_tr.columns)):
    X_tr[X_tr.columns[cols]].fillna(X_tr[X_tr.columns[cols]].mean(skipna=True), inplace=True)
    #data_tr[data_tr.columns[cols]].replace(np.nan, np.nanmean(data_tr[data_tr.columns[cols]], axis=0))

#print(data_tr['B_gsm_x'][12253])
for cols in range(0,len(X_ts.columns)):
    X_ts[X_ts.columns[cols]].fillna(X_ts[X_ts.columns[cols]].mean(skipna=True), inplace=True)
    #data_ts[data_tr.columns[cols]].replace(np.nan, np.nanmean(data_ts[data_tr.columns[cols]], axis=0))

# normalize
normalizer = StandardScaler()
normalizer.fit(X_tr)
print('normalizing...')
X_tr_normalized = normalizer.transform(X_tr)
X_ts_normalized = normalizer.transform(X_ts)


# base_model = ElasticNet(fit_intercept=True, normalize=False, max_iter=3000)
# parameters = {#'alpha':[0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000000],
#               'alpha':[0.0001, 0.1, 1, 1000],
#               'l1_ratio':[1.0, 0.5, 0.]}



### Linear Regression
print('\n** LinearRegression() **')
file.write('** LinearRegression() **')
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
file.write('RESULT(wrmse):')
rmse, wrmse = wrmse(Y_ts_pred_linr,Y_ts)

print('RESULT(wrmse)-rounded:')
file.write('RESULT(wrmse)-rounded:')
rmse_round, wrmse_round = wrmse(np.clip(np.round(Y_ts_pred_linr), a_min=0, a_max=9),Y_ts)


### ridge_regression
print('\n** ridge_regression() **')
file.write('** ridge_regression() **')
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

print('RESULT(wrmse):')
file.write('RESULT(wrmse):')
rmse, wrmse = wrmse(Y_ts_pred_ridge,Y_ts)

print('RESULT(wrmse)-rounded:')
file.write('RESULT(wrmse)-rounded:')
rmse_round, wrmse_round = wrmse(np.clip(np.round(Y_ts_pred_ridge), a_min=0, a_max=9),Y_ts)





### RandomForestRegressor
print('\n** RandomForestRegressor **')
file.write('** RandomForestRegressor **')
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
file.write('RESULT(wrmse):')
rmse, wrmse = wrmse(Y_ts_pred_rf, Y_ts)

print('RESULT(wrmse)-rounded:')
file.write('RESULT(wrmse)-rounded:')
rmse_round, wrmse_round = wrmse(np.clip(np.round(Y_ts_pred_rf), a_min=0, a_max=9),Y_ts)

### Support Vector Regression
print('\n** SVR() **')
file.write('** SVR() **')

base_model = SVR(gamma='scale')
parameters = {'C':[.0001, .01, .1, 1., 10, 100, 10000],
              'epsilon': [.05, .1, .5, 1.]}

print('training...')
model = GridSearchCV(base_model, param_grid=parameters, cv=5, scoring='neg_mean_absolute_error')
model.fit(X_tr_normalized, Y_tr.values.ravel())

# test
print('making predictions...')
Y_ts_pred_svr = model.predict(X_ts_normalized)

print('RESULT(wrmse):')
file.write('RESULT(wrmse):')
rmse, wrmse = wrmse(Y_ts_pred_svr,Y_ts)

file.close()

