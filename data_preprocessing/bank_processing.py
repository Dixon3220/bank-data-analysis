import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from time import *
from scipy.stats import chi2_contingency
from scipy.stats import mode
import warnings

warnings.filterwarnings('ignore')


def drop_missing_feature(df):
    missing_rate = [((df[name] == 'unknown').sum() * 1.0 / len(df)) for name in df.columns]
    missing_df = pd.DataFrame(np.array([df.columns, missing_rate]).T, columns=['name', 'missing rate'])
    print(missing_df)
    print()

    missing_more_20 = list(missing_df[(missing_df['missing rate'] > 0.2)]['name'])
    missing_less_10 = list(missing_df[(missing_df['missing rate'] < 0.1)][(missing_df['missing rate'] > 0)]['name'])

    try:
        df.drop(missing_more_20, axis=1, inplace=True)

        missing_index = [list(df[name][(df[name] == 'unknown')].index) for name in missing_less_10]
        missing_index_unique = np.unique(sum(missing_index, []))
        df.drop(missing_index_unique, axis=0, inplace=True)

        print('missing columns dropped: ', missing_more_20)
        print('missing rows dropped (ratio): {:.2%}'.format(
            len(missing_index_unique) / (len(df) + len(missing_index_unique))))

    except KeyError:
        print('no missing features')

    return df


# calculate woe for each bin and sort bins by woe.
def woe_sort(d, column_name):
    total_yes = (d['y'] == 'yes').sum()
    total_no = (d['y'] == 'no').sum()
    names = np.unique(d[column_name])

    woe = [np.log((d['y'][(d[column_name] == name)] == 'no').sum() / total_no) - \
           np.log((d['y'][(d[column_name] == name)] == 'yes').sum() / total_yes) \
           for name in names]
    woe_sorted = sorted(dict(zip(names, woe)).items(), key=lambda x: x[1], reverse=False)

    return dict(woe_sorted)


# change the format of data.
def format_data(df):
    # deal with word or bool features.
    change_format = {'no': 0, 'yes': 1, \
                     'primary': 0, 'secondary': 1, 'tertiary': 2, \
                     'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, \
                     'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11 \
                     }
    for column in ['job', 'marital']:
        old_names = woe_sort(df[[column, 'y']].copy(), column)
        new_names = np.arange(len(old_names))
        change_format.update(dict(zip(old_names.keys(), new_names)))
    df.replace(change_format, inplace=True)
    return df

def main():
    path = 'bank-full.csv'
    df = pd.read_csv(path, header=0, low_memory=False)
    df = drop_missing_feature(df)
    df = format_data(df)
    df.to_csv('bank_processed.csv')

if __name__ == '__main__':
    main()