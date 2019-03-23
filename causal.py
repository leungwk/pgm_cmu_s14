from pathlib import Path
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from lib.HSIC import hsic_gam

SEED = 123
rand_state = np.random.RandomState(SEED)

### load data

data_dir = Path('data/pairs/')

regex_name = r'pair([0-9]{4}).txt'

meta_name = 'pairmeta.txt'
df_meta = pd.read_csv(data_dir / meta_name, names=['num', 'c1', 'cn', 'e1', 'en', 'w'], sep=' ')

acc = {}
for fname in os.listdir(data_dir):
    m = re.match(regex_name, fname)
    if not m:
        continue
    num = m.group(1)
    acc[num] = pd.read_csv(data_dir / fname, header=None, sep='\s+|\t')
data_tab = acc

## table of directions
cond_1 = (df_meta['c1'] == 1) & (df_meta['cn'] == 1) & (df_meta['e1'] == 2) & (df_meta['en'] == 2)
cond_2 = (df_meta['c1'] == 2) & (df_meta['cn'] == 2) & (df_meta['e1'] == 1) & (df_meta['en'] == 1)
two_vars = set(df_meta[cond_1 | cond_2]['num'])
ser_xy = set(df_meta[cond_1])
ser_yx = set(df_meta[cond_2])

def algorithm_one(df_in):
    df_train, df_test = train_test_split(df_in, test_size=0.4, random_state=rand_state)
    X, Y = df_train[0], df_train[1]
    f_Y = sm.OLS(Y, X).fit()
    f_X = sm.OLS(X, Y).fit()

    err_Y = df_test[1] -f_Y.predict(df_test[0])
    err_X = df_test[0] -f_X.predict(df_test[1])

    test_stat_xy, thres_xy = hsic_gam(df_test[[0]].values, err_Y.to_frame().values, 0.05)
    test_stat_yx, thres_yx = hsic_gam(df_test[[1]].values, err_X.to_frame().values, 0.05)
    return 'xy' if test_stat_xy <= test_stat_yx else 'yx'

acc = {}
for num in df_meta['num']:
    if num not in two_vars:
        continue
    base_dir = 'xy' if num in ser_xy else 'yx'
    name = f'{num:04d}'
    pred_dir = algorithm_one(data_tab[name])
    acc[name] = (base_dir, pred_dir)
    print(name, base_dir == pred_dir, base_dir, pred_dir)
df_results = pd.DataFrame.from_dict(acc, orient='index')
df_results['m'] = df_results[0] == df_results[1]

_d = {row['num']: row['w'] for _,row in df_meta.iterrows()}
df_results['w'] = df_results.index.map(lambda n: _d.get(int(n)))
p_match = len(df_results[df_results['m']]) / len(df_results['m'])
p_match_w = (df_results['m']*df_results['w']).sum()/df_results['w'].sum()
print('match rate', p_match, p_match_w)
