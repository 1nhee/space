### 1. 각 파일 합치기
### 2. 분 단위로 나눠져 있는 데이터를 평균을 내어 3시간 단위의 데이터로 만들기
###     각 3시간의 데이터는 데이터의 mean, std, min, max, median 값을 갖는다.
### 3. -9999.99와 같은 오류 value는 nan값으로 만들어준다.
### 4. 365일 24시간 모든 60분의 데이터가 존재하는게 아니므로 비어있는 데이터의 값은 이전의 값으로 대체시킨다.(이전의 값을 복사하는 방법으로 대체한다.)


import pandas as pd
import numpy as np
import timeit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
'''
out_df_permin = pd.DataFrame(columns=['year', 'doy', 'hr', 'min', 'Np', 'Tp', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt'])
ls_out_df_permin = []
'''

tmp_df_per3hr = pd.DataFrame(columns=['year', 'doy', 'hr', 'min', 'Np', 'Tp', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt'])
ls_tmp_df_per3hr = []

out_df_per3hr = pd.DataFrame(columns=['date', 'year', 'doy', 'hr', 'Np', 'Tp', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt'])
out_df_per3hr2 = pd.DataFrame(columns=['date', 'year', 'doy', 'hr',
                                       'Np_mean', 'Np_std', 'Np_min', 'Np_median', 'Np_max',
                                       'Tp_mean', 'Tp_std', 'Tp_min', 'Tp_median', 'Tp_max',
                                       'Vp_mean', 'Vp_std', 'Vp_min', 'Vp_median', 'Vp_max',
                                       'B_gsm_x_mean', 'B_gsm_x_std', 'B_gsm_x_min', 'B_gsm_x_median', 'B_gsm_x_max',
                                       'B_gsm_y_mean', 'B_gsm_y_std', 'B_gsm_y_min', 'B_gsm_y_median', 'B_gsm_y_max',
                                       'B_gsm_z_mean', 'B_gsm_z_std', 'B_gsm_z_min', 'B_gsm_z_median', 'B_gsm_z_max',
                                       'Bt_mean', 'Bt_std', 'Bt_min', 'Bt_median', 'Bt_max'])
ls_out_df_per3hr = []
ls_out_df_per3hr2 = []

out_df_per3hr_nan = pd.DataFrame(columns=['date', 'year', 'doy', 'hr', 'Np', 'Tp', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt'])
out_df_per3hr2_nan = pd.DataFrame(columns=['date', 'year', 'doy', 'hr',
                                       'Np_mean', 'Np_std', 'Np_min', 'Np_median', 'Np_max',
                                       'Tp_mean', 'Tp_std', 'Tp_min', 'Tp_median', 'Tp_max',
                                       'Vp_mean', 'Vp_std', 'Vp_min', 'Vp_median', 'Vp_max',
                                       'B_gsm_x_mean', 'B_gsm_x_std', 'B_gsm_x_min', 'B_gsm_x_median', 'B_gsm_x_max',
                                       'B_gsm_y_mean', 'B_gsm_y_std', 'B_gsm_y_min', 'B_gsm_y_median', 'B_gsm_y_max',
                                       'B_gsm_z_mean', 'B_gsm_z_std', 'B_gsm_z_min', 'B_gsm_z_median', 'B_gsm_z_max',
                                       'Bt_mean', 'Bt_std', 'Bt_min', 'Bt_median', 'Bt_max'])
ls_out_df_per3hr_nan = []
ls_out_df_per3hr2_nan = []

tA = 0
tB = 0

prev_yr = 0
prev_hr = -1
prev_stat_entry = {}

is_updating_inst_mat = False


print('saving tmp results...')

### fill out the csv file name in this line
data_x = pd.read_csv('../organized_data/x_long_ts.csv',delimiter=",")
data_x = data_x.drop([0])
data_x.index = range(0, len(data_x))
#print(data_x)

data_x['doy'] = pd.to_numeric(data_x['doy'])

time_x = data_x[['doy', 'hr', 'min']]
values_x = data_x[['Np', 'Tp', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt']]

last_doy = data_x.doy[data_x.index[-1]]
doy = 1


# for doy in range(1, last_doy+1):
data_x_roi = data_x.loc[(data_x['doy'] == doy)]

for days in range(365):
    for hr in range(0, 24):
        data_x_roi_hr = data_x_roi.loc[(data_x_roi['hr'] == hr)]
        if hr != prev_hr and hr%3 == 0:

            prev_hr = hr
            anc_hr = hr

        # start = timeit.default_timer()
        for minute in range(0, 60):
            x_line = data_x_roi_hr.loc[(data_x_roi_hr['min'] == minute)]
            if not x_line.empty:
                tmp_x_line = list(x_line.to_dict('index').values())[0]

                ls_tmp_df_per3hr.append(tmp_x_line)
            # stop = timeit.default_timer()
            # print('TimeEst: ', stop - start)

        if (hr+1)%3 == 0:
            tmp_df_per3hr = pd.DataFrame(ls_tmp_df_per3hr)
            tmp_df_per3hr_nan = tmp_df_per3hr.replace(-9999.9, np.nan)

            # mean only
            summary_3hr = tmp_df_per3hr_nan.mean()[3:]
            summary_3hr['doy'] = doy
            summary_3hr['hr'] = anc_hr

            # out_df_per3hr_nan.append
            ls_out_df_per3hr_nan.append(summary_3hr.to_dict())

            #print(summary_3hr)
            if np.isnan(summary_3hr['Np']):
                summary_3hr['Np'] = prev_summary_3hr['Np']
            if np.isnan(summary_3hr['Tp']):
                summary_3hr['Tp'] = prev_summary_3hr['Tp']
            if np.isnan(summary_3hr['Vp']):
                summary_3hr['Vp'] = prev_summary_3hr['Vp']
            if np.isnan(summary_3hr['B_gsm_x']):
                    summary_3hr['B_gsm_x'] = prev_summary_3hr['B_gsm_x']
            if np.isnan(summary_3hr['B_gsm_y']):
                    summary_3hr['B_gsm_y'] = prev_summary_3hr['B_gsm_y']
            if np.isnan(summary_3hr['B_gsm_z']):
                    summary_3hr['B_gsm_z'] = prev_summary_3hr['B_gsm_z']
            if np.isnan(summary_3hr['Bt']):
                    summary_3hr['Bt'] = prev_summary_3hr['Bt']

                # out_df_per3hr = out_df_per3hr.append(summary_3hr, ignore_index=True)
                # v = summary_3hr.to_dict()
            ls_out_df_per3hr.append(summary_3hr.to_dict())

            prev_summary_3hr = summary_3hr

            # mean, std, min, median, max
            # tmp_df_per3hr_nan['year'] = pd.to_numeric(tmp_df_per3hr_nan['year'])
            # tmp_df_per3hr_nan['doy'] = pd.to_numeric(tmp_df_per3hr_nan['doy'])
            summary_3hr_mean = tmp_df_per3hr_nan.mean()[3:]
            summary_3hr_std = tmp_df_per3hr_nan.std()[3:11]
            summary_3hr_min = tmp_df_per3hr_nan.min()[3:11]
            summary_3hr_median = tmp_df_per3hr_nan.median()[3:11]
            summary_3hr_max = tmp_df_per3hr_nan.max()[3:11]

            new_stat_entry = {}
            new_stat_entry['doy'] = doy
            new_stat_entry['hr'] = anc_hr

            new_stat_entry['Np_mean'] = summary_3hr_mean['Np']
            new_stat_entry['Np_std'] = summary_3hr_std['Np']
            new_stat_entry['Np_min'] = summary_3hr_min['Np']
            new_stat_entry['Np_median'] = summary_3hr_median['Np']
            new_stat_entry['Np_max'] = summary_3hr_max['Np']

            new_stat_entry['Tp_mean'] = summary_3hr_mean['Tp']
            new_stat_entry['Tp_std'] = summary_3hr_std['Tp']
            new_stat_entry['Tp_min'] = summary_3hr_min['Tp']
            new_stat_entry['Tp_median'] = summary_3hr_median['Tp']
            new_stat_entry['Tp_max'] = summary_3hr_max['Tp']

            new_stat_entry['Vp_mean'] = summary_3hr_mean['Vp']
            new_stat_entry['Vp_std'] = summary_3hr_std['Vp']
            new_stat_entry['Vp_min'] = summary_3hr_min['Vp']
            new_stat_entry['Vp_median'] = summary_3hr_median['Vp']
            new_stat_entry['Vp_max'] = summary_3hr_max['Vp']

            # assuming the rest of the features won't be nan
            new_stat_entry['B_gsm_x_mean'] = summary_3hr_mean['B_gsm_x']
            new_stat_entry['B_gsm_x_std'] = summary_3hr_std['B_gsm_x']
            new_stat_entry['B_gsm_x_min'] = summary_3hr_min['B_gsm_x']
            new_stat_entry['B_gsm_x_median'] = summary_3hr_median['B_gsm_x']
            new_stat_entry['B_gsm_x_max'] = summary_3hr_max['B_gsm_x']

            new_stat_entry['B_gsm_y_mean'] = summary_3hr_mean['B_gsm_y']
            new_stat_entry['B_gsm_y_std'] = summary_3hr_std['B_gsm_y']
            new_stat_entry['B_gsm_y_min'] = summary_3hr_min['B_gsm_y']
            new_stat_entry['B_gsm_y_median'] = summary_3hr_median['B_gsm_y']
            new_stat_entry['B_gsm_y_max'] = summary_3hr_max['B_gsm_y']

            new_stat_entry['B_gsm_z_mean'] = summary_3hr_mean['B_gsm_z']
            new_stat_entry['B_gsm_z_std'] = summary_3hr_std['B_gsm_z']
            new_stat_entry['B_gsm_z_min'] = summary_3hr_min['B_gsm_z']
            new_stat_entry['B_gsm_z_median'] = summary_3hr_median['B_gsm_z']
            new_stat_entry['B_gsm_z_max'] = summary_3hr_max['B_gsm_z']

            new_stat_entry['Bt_mean'] = summary_3hr_mean['Bt']
            new_stat_entry['Bt_std'] = summary_3hr_std['Bt']
            new_stat_entry['Bt_min'] = summary_3hr_min['Bt']
            new_stat_entry['Bt_median'] = summary_3hr_median['Bt']
            new_stat_entry['Bt_max'] = summary_3hr_max['Bt']

            if not prev_stat_entry:
                prev_stat_entry = new_stat_entry

            # out_df_per3hr2 = out_df_per3hr2.append(new_stat_entry, ignore_index=True)
            ls_out_df_per3hr2_nan.append(new_stat_entry.copy())

            new_stat_entry['Np_mean'] = (summary_3hr_mean['Np'], prev_stat_entry['Np_mean'])[np.isnan(summary_3hr_mean['Np'])]
            new_stat_entry['Np_std'] = (summary_3hr_std['Np'], prev_stat_entry['Np_std'])[np.isnan(summary_3hr_std['Np'])]
            new_stat_entry['Np_min'] = (summary_3hr_min['Np'], prev_stat_entry['Np_min'])[np.isnan(summary_3hr_min['Np'])]
            new_stat_entry['Np_median'] = (summary_3hr_median['Np'], prev_stat_entry['Np_median'])[np.isnan(summary_3hr_median['Np'])]
            new_stat_entry['Np_max'] = (summary_3hr_max['Np'], prev_stat_entry['Np_max'])[np.isnan(summary_3hr_max['Np'])]

            new_stat_entry['Tp_mean'] = (summary_3hr_mean['Tp'], prev_stat_entry['Tp_mean'])[np.isnan(summary_3hr_mean['Tp'])]
            new_stat_entry['Tp_std'] = (summary_3hr_std['Tp'], prev_stat_entry['Tp_std'])[np.isnan(summary_3hr_std['Tp'])]
            new_stat_entry['Tp_min'] = (summary_3hr_min['Tp'], prev_stat_entry['Tp_min'])[np.isnan(summary_3hr_min['Tp'])]
            new_stat_entry['Tp_median'] = (summary_3hr_median['Tp'], prev_stat_entry['Tp_median'])[np.isnan(summary_3hr_median['Tp'])]
            new_stat_entry['Tp_max'] = (summary_3hr_max['Tp'], prev_stat_entry['Tp_max'])[np.isnan(summary_3hr_max['Tp'])]

            new_stat_entry['Vp_mean'] = (summary_3hr_mean['Vp'], prev_stat_entry['Vp_mean'])[np.isnan(summary_3hr_mean['Vp'])]
            new_stat_entry['Vp_std'] = (summary_3hr_std['Vp'], prev_stat_entry['Vp_std'])[np.isnan(summary_3hr_std['Vp'])]
            new_stat_entry['Vp_min'] = (summary_3hr_min['Vp'], prev_stat_entry['Vp_min'])[np.isnan(summary_3hr_min['Vp'])]
            new_stat_entry['Vp_median'] = (summary_3hr_median['Vp'], prev_stat_entry['Vp_median'])[np.isnan(summary_3hr_median['Vp'])]
            new_stat_entry['Vp_max'] = (summary_3hr_max['Vp'], prev_stat_entry['Vp_max'])[np.isnan(summary_3hr_max['Vp'])]

            ls_out_df_per3hr2.append(new_stat_entry)
            # stop = timeit.default_timer()
            # tB = tB + (stop - start)
            # print('$$$$ TimeEst: ', stop - start, ' - - - ', tB)

            prev_stat_entry = new_stat_entry

            # tmp_df_per3hr = pd.DataFrame(columns=['year', 'doy', 'hr', 'min', 'Np', 'Tp', 'Vp', 'B_gsm_x', 'B_gsm_y', 'B_gsm_z', 'Bt', 'Kp'])
            ls_tmp_df_per3hr = []

        doy = doy + 1

out_df_per3hr2 = pd.DataFrame(ls_out_df_per3hr2)
out_df_per3hr2.to_csv('../organized_data/new_data_long_3hour_stats.csv', index=False)

out_df_per3hr2_nan = pd.DataFrame(ls_out_df_per3hr2_nan)
out_df_per3hr2_nan.to_csv('../organized_data/new_data_long_3hour_stats_with_nan.csv', index=False)
