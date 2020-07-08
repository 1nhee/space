# single target prediction testbed

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from prev_code import WeightedRMSE as WRMSE

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#### PROCEDURES ####
def rmse(y_true, y_pred):
    rmse = (np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def print_eval(model, X_tr, y_tr, X_ts, y_ts):
    y_tr_predict = model.predict(X_tr)
    # rmse_tr = (np.sqrt(mean_squared_error(y_tr, y_tr_predict)))
    # r2_tr = r2_score(y_tr, y_tr_predict)
    rmse_tr, r2_tr = rmse(y_tr, y_tr_predict)
    wrmse_tr = WRMSE.wrmse(y_tr_predict, y_tr)

    y_ts_predict = model.predict(X_ts)
    # rmse_ts = (np.sqrt(mean_squared_error(y_ts, y_ts_predict)))
    # r2_ts = r2_score(y_ts, y_ts_predict)
    rmse_ts, r2_ts = rmse(y_ts, y_ts_predict)
    wrmse_ts = WRMSE.wrmse(y_ts_predict, y_ts)

    print(' TR: RMSE is {}'.format(rmse_tr))
    print(' TR: R2 score is {}'.format(r2_tr))
    print(" --")
    print(' TS: RMSE is {}'.format(rmse_ts))
    print(' TS: R2 score is {}'.format(r2_ts))


#### DATA ####
DIR_DATAHOME = '../../data'

# data load - space
D = pd.read_csv('../organized_data/new_data_3hour_stats_with_nan.csv')
X = D.iloc[:, 4:(D.shape[1] - 1)]
y = D.iloc[:, -1]

# X.isna().sum()

# nan: carryover the last val
for j in range(0, X.shape[1]):
    xval_prev = 0
    for i in X.index:
        if np.isnan(X.iloc[i, j]):
            X.iloc[i, j] = xval_prev
        else:
            xval_prev = X.iloc[i, j]

# # nan: interpolate
# for j in range(0, X.shape[1]):
#     X.iloc[:, j] = pd.Series(X.iloc[:, j]).interpolate().to_numpy()


# dataset split
from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.3, random_state=777)
print(X_tr.shape)
print(X_ts.shape)
print(y_tr.shape)
print(y_ts.shape)

# normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
normzer = StandardScaler().fit(X_tr)
X_tr = normzer.transform(X_tr)
X_ts = normzer.transform(X_ts)




#### MODELS ####

## With sklearn

# linear regression
from sklearn.linear_model import LinearRegression

print('** Lin Reg **')

model = LinearRegression()
model.fit(X_tr, y_tr)

# linear regression - model evaluation
print_eval(model, X_tr, y_tr, X_ts, y_ts)

# Ridge reg
from sklearn.linear_model import Ridge

print('** Ridge Reg **')

params = {'alpha': [0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000000]
          }
base_model = Ridge(max_iter=10000)
model = GridSearchCV(base_model, params, cv=5)
model.fit(X_tr, y_tr)
print(model.best_params_)

# Ridge reg - model evaluation
print_eval(model, X_tr, y_tr, X_ts, y_ts)


# SGD reg
from sklearn.linear_model import SGDRegressor

print('** SGD Reg **')

params = {'loss': ['squared_loss', 'huber'],
          'l1_ratio': [0., .25, .5, .75, 1.],
          'alpha': [0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000000]
          }
            # 'huber' -> epsilon=0.1 (default)

base_model = SGDRegressor(penalty='elasticnet', fit_intercept=True, max_iter=10000)
model = GridSearchCV(base_model, params, cv=5)
model.fit(X_tr, y_tr)
print(model.best_params_)
print_eval(model, X_tr, y_tr, X_ts, y_ts)



# # Kernel ridge reg
# from sklearn.kernel_ridge import KernelRidge
# print('** Kernel Ridge Reg **')
# params = {'alpha': [0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000000],
#           'gamma': np.logspace(-2, 2, 5)
#           }
#           # 'kernel' = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
# base_model = KernelRidge(kernel='rbf', gamma=0.1)
#
# model = GridSearchCV(base_model, params, cv=5)
# model.fit(X_tr, y_tr)
# print(model.best_params_)
# print_eval(model, X_tr, y_tr, X_ts, y_ts)



# # SVR
# from sklearn.svm import SVR
# print('** Support Vector Reg **')
# params = {'C': [0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1000000],
#           'gamma': np.logspace(-2, 2, 5)
#           }
#           # 'kernel' = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
# base_model = SVR(kernel='rbf', gamma=0.1)
#
# model = GridSearchCV(base_model, params, cv=5)
# model.fit(X_tr, y_tr)
# print(model.best_params_)
# print_eval(model, X_tr, y_tr, X_ts, y_ts)



# RF
from sklearn.ensemble import RandomForestRegressor
print('** Random Forest **')
params = {'n_estimators': [50, 100, 200, 300], 'min_samples_leaf': [1, 10, 30]}
base_model = RandomForestRegressor(criterion='mse')

model = GridSearchCV(base_model, params, cv=5)
model.fit(X_tr, y_tr)

feature_importances = pd.DataFrame(model.best_estimator_.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

print(model.best_params_)
print_eval(model, X_tr, y_tr, X_ts, y_ts)

# MLP
from sklearn.neural_network import MLPRegressor
print('** MLP **')
params = {'alpha': [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
          'hidden_layer_sizes': [(5,), (50,), (100,), (50, 50), (100, 50),
                                 (50, 30, 10), (50, 50, 30, 10),
                                 (30, 30, 30, 10, 10, 10, 10), (10, 10, 10, 10, 10, 10, 10),
                                 (10, 10, 7, 7, 5, 5, 3, 3)]}
base_model = MLPRegressor(activation='identity', learning_rate='adaptive', max_iter=5000)

model = GridSearchCV(base_model, params, cv=5)
model.fit(X_tr, y_tr)
print(model.best_params_)
print_eval(model, X_tr, y_tr, X_ts, y_ts)

# XGBoost
import xgboost as xgb
print('** XGBoost **')
params = {'n_estimators': [5, 10, 25, 50, 100, 200],
          'subsample': [.8, 1]}
base_model = xgb.XGBRegressor(silent=1, n_jobs=8)

model = GridSearchCV(base_model, params, cv=5)
model.fit(X_tr, y_tr)

feature_importances = pd.DataFrame(model.best_estimator_.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

print(model.best_params_)
print_eval(model, X_tr, y_tr, X_ts, y_ts)

# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
print('** GradientBoostingRegressor **')
params = {'loss': ['ls', 'lad', 'huber'],
          'n_estimators': [50, 100, 200, 300],
          'subsample': [.8, 1.],
          'criterion': ['friedman_mse', 'mse']}
base_model = GradientBoostingRegressor()

model = GridSearchCV(base_model, params, cv=5)
model.fit(X_tr, y_tr)

feature_importances = pd.DataFrame(model.best_estimator_.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

print(model.best_params_)
print_eval(model, X_tr, y_tr, X_ts, y_ts)




