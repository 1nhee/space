import numpy as np

def wrmse(Y_pred, Y_true, verbose=True):
    w = Y_true / Y_true.sum(axis=0)
    wrmse = np.sqrt((w * ((Y_pred - Y_true) ** 2)).sum())

    if verbose:
        print('RMSE=%f' % (np.sqrt(((Y_pred - Y_true) ** 2).mean())))
        print('WRMSE=%f' % wrmse)

    return wrmse

# import pandas as pd
# Y_pred = pd.DataFrame([0.3, 0.7, 3.2, 5.])
# Y_true = pd.DataFrame([0, 1, 3, 5])
# print(wrmse(Y_pred, Y_true))
