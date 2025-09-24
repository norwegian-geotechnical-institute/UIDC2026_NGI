
import sys, os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC as SVClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import MAD for balancing dataset
from pyod.models.iforest import IForest

from imblearn.under_sampling import RandomUnderSampler

# ensure that the correct starting location for the script is set
sys.path.append(os.path.dirname(__file__))

from X_library import dictionaries


################################
# function definitions
################################


def discretize(dataframe: pd.DataFrame, interval: float, columns: list,
               chain_col: str) -> pd.DataFrame:
    """
    XXXXX
    """
    columns = columns + ['collapse']
    # Ensure we’re working on a copy
    data = dataframe.copy()

    intervals = np.arange(data[chain_col].min(), data[chain_col].max(),
                          interval)
    df_temps = []

    for i in tqdm(range(len(intervals)-1)):
        tl_start, tl_stop = intervals[i], intervals[i+1]
        mask = (data[chain_col] >= tl_start) & (data[chain_col] < tl_stop)
        chunk = data.loc[mask, columns]

        if len(chunk) == 1:
            s = chunk.iloc[0].copy()
            n_dp = 1
        elif len(chunk) > 1:
            s = chunk.mean(axis=0)
            s['collapse'] = round(s['collapse'], 0)
            n_dp = len(chunk)
        else:
            # empty -> NaNs for the requested columns
            s = pd.Series(index=columns, dtype='float64')
            n_dp = 0

        # Add bin metadata
        s['start [m]'] = tl_start
        s['stop [m]'] = tl_stop
        s['n dp'] = n_dp
        df_temps.append(s)

    return pd.DataFrame(df_temps).reset_index(drop=True)


def plot_param(ax, dataframe, parameter):
    ax.plot(dataframe['Tunnellength [m]'], dataframe[parameter], color='black')
    ax.set_ylabel(parameter)
    ax.set_xlim(left=dataframe['Tunnellength [m]'].min(),
                right=dataframe['Tunnellength [m]'].max())
    # ax.legend()
    ax.grid(alpha=0.5)


################################
# Static variables
################################

# data from paper "Diagnosing tunnel collapse sections based on TBM tunneling
# big data and deep learning: A case study on the Yinsong Project, China" by
# Chen et al. (2021)

DATA_FOLDER = 'data'
ANALYSES_FOLDER = 'analyses'
COLLAPSE_EXCEL = 'data/collapses.xlsx'

# TODO experiment with different features
INPUT_FEATURES = [
    'T(P,V/N)-b1',
    'T(P,V/N)-b2',
    'T(P,V/N)-a',
    'T(P,V/N)-R2',
    'T(V/N)-b',
    'T(V/N)-a',
    'T(V/N)-R2',
    'P(V)-slope',
    'P(V)-a',
    'P(V)-R2',
    'Wp(WT)-slope',
    'Wp(WT)-a',
    'Wp(WT)-R2',
    'T(P)-slope',
    'T(P)-a',
    'T(P)-R2',
    'T-m',
    'T-slope',
    'T-R2',
    'P-m',
    'P-slope',
    'P-R2',
    'V-m',
    'V-slope',
    'V-R2',
    'Vs-m',
    'Vs-slope',
    'TN-m',
    'TN-slope',
    'TN-R2',
    'PV-m',
    'PV-slope',
    'PV-R2',
    'V/N-m',
    'V/N-slope',
    'V/N-R2',
    'N-m',
    'Ns-m',
    'I-m',
    'V0-m',
    'V/Vs-min',
    'T/P-min',
    'T/P-max',
    'T/P-m',
    'T/P-std',
    'Tpi-min',
    'Tpi-max',
    'Tpi-m',
    'Tpi-std',
    'Fpi-min',
    'Fpi-max',
    'Fpi-m',
    'Fpi-std',
    'Tpi-slope',
    'Tpi-R2:1-sse/sst',
    'WT/WP-slope',
    'WT/WP-R2:1-sse/sst',
    'T(P*V/N)-slope',
    'T(P*V/N)-a',
    'T(P*V/N)-R2',
    'Fpi',
    'P*V/N',
    'T/P',
    'Tpi',
    'dt',
    'dz',
    'workP',
    'workT',
    'Total T', 'Total F', 'RPM', 'Pr', 'p'
    ]


################################
# preprocessing
################################

dicts = dictionaries()

# combine datasets into one
# d02 means the distance of boring cycle is shorter than 0.2m
# d05 means the distance of boring cycle is shorter than 0.5m but longer than 0.2m
# d10 means the distance of boring cycle is shorter than 1.0m but longer than 0.5m
# d20 means the distance of boring cycle is shorter than 1.8m but longer than 1.0m

dfs = []
for ds in ['d02_eigStable.csv', 'd05_eigStable.csv', 'd10_eigStable.csv',
           'd20_eigStable.csv']:
    dfs.append(pd.read_csv(fr'{DATA_FOLDER}\{ds}'))
df = pd.concat(dfs)
df.sort_values(by='Chainage', ascending=False, inplace=True)

df.index = np.arange(len(df))


# assign collapse sections to data
df_collapses = pd.read_excel(COLLAPSE_EXCEL)
df['collapse'] = 0
for i in range(len(df_collapses)):
    ids = df[(df['Chainage'] <= df_collapses.iloc[i]['Chainage start [m]']) & 
             (df['Chainage'] >= df_collapses.iloc[i]['Chainage end [m]'])].index
    df.loc[ids, 'collapse'] = 1

# get tunnellength starting from 0 for conveniance
df['Tunnellength [m]'] = df['Chainage'] - df['Chainage'].min()
df['Tunnellength [m]'] = (df['Tunnellength [m]'] - df['Tunnellength [m]'].max()) * -1

df.to_csv("data/raw/combined_data.csv", index=False)

# TODO experiment with undersampling

# discretize dataset
# df_disc = discretize(df, 1, INPUT_FEATURES, chain_col='Tunnellength [m]')
# df_disc = df_disc[df_disc['n dp'] > 0]
# df = df_disc
# print(np.unique(df_disc['collapse'], return_counts=True))
# ################################
# # plotting
# ################################

# # rename column names to be more readable
# df.rename({'p': 'penetration\n[mm/rev]',
#            'Pr': 'advance rate\n[mm/min]',
#            'RPM': 'cutterhead rotations\n[rpm]',
#            'Total T': 'cutterhead torque\n[kNm]',
#            'Total F': 'thrust\n[kN]',
#            'Fpi': 'Field Penetration Index',
#            'Tpi': 'drilling efficiency index\nTPI'}, axis=1, inplace=True)

# # lineplot
# fig, axs = plt.subplots(ncols=1, nrows=8, figsize=(18, 10), sharex=True)

# plot_param(axs[0], df, 'penetration\n[mm/rev]')
# plot_param(axs[1], df, 'advance rate\n[mm/min]')
# plot_param(axs[2], df, 'cutterhead rotations\n[rpm]')
# plot_param(axs[3], df, 'thrust\n[kN]')
# plot_param(axs[4], df, 'cutterhead torque\n[kNm]')
# plot_param(axs[5], df, 'Field Penetration Index')
# plot_param(axs[6], df, 'drilling efficiency index\nTPI')
# plot_param(axs[7], df, 'collapse')
# axs[7].set_xlabel('Tunnellength [m]')

# plt.tight_layout()
# plt.savefig(fr'{ANALYSES_FOLDER}\00_lineplot.jpg', dpi=300)
# plt.show()


# # scatterplot
# parameters = ['penetration\n[mm/rev]', 'advance rate\n[mm/min]',
#               'cutterhead rotations\n[rpm]', 'thrust\n[kN]',
#               'cutterhead torque\n[kNm]', 'Field Penetration Index',
#               'drilling efficiency index\nTPI']



# n_params = len(parameters)
# n_figure = 1

# # from plotting import pairplot
# # c = ['grey' if x == 0 else 'red' for x in df['collapse']]
# # fig = pairplot(df, parameters, target='collapse', figsize=(18, 18), c=c)

# fig, axs = plt.subplots(nrows=n_params, ncols=n_params, figsize=(18, 18))

# df_collaps = df[df['collapse'] == 1]
# df_regular = df[df['collapse'] != 1]

# for i in range(n_params):
#     for j in range(n_params):
#         ax: plt.Axes = axs[i, j]
#         if parameters[i] == parameters[j]:
#             _, xbins, _ = ax.hist(df[parameters[i]], color='grey', bins=30,
#                     edgecolor='black', alpha=0.5, density=True)
#             ax.hist(df_collaps[parameters[i]], bins=xbins, color='red', alpha=0.5,
#                     edgecolor='black', zorder=2, density=True)
#         else:
#             ax.scatter(df_regular[parameters[j]], df_regular[parameters[i]],
#                        color='grey', edgecolor='black', alpha=0.3, s=1)
#             ax.scatter(df_collaps[parameters[j]], df_collaps[parameters[i]],
#                        color='red', alpha=1, s=2)
#             ax.set_ylabel(parameters[i].replace(' [', '\n['),
#                           fontsize=8)
#         ax.set_xlabel(parameters[j].replace(' [', '\n['),
#                         fontsize=8)
#         ax.tick_params(axis='both', labelsize=8)

# plt.tight_layout()
# plt.savefig(fr'{ANALYSES_FOLDER}\01_pairplot.jpg', dpi=300)
# plt.show()


################################
# Preprocessing for prediction
################################
random_state = 42

X = df[INPUT_FEATURES].values
y = df['collapse'].values

print(np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state, stratify=y)


# Remove outliers in the training set
iforest = IForest(n_estimators=500, contamination=0.01, random_state=random_state).fit(X_train)
probs = iforest.predict_proba(X_train)
is_outlier = probs[:, 1] > 0.8
X_train, y_train = X_train[~is_outlier], y_train[~is_outlier]
print('After removing outliers:')
print(np.unique(y_train, return_counts=True))

regular_ratio = 8  # keep some more regular data

# Undersample the majority class in the training set
rus = RandomUnderSampler(random_state=random_state,
                         sampling_strategy= {
                                0: int(np.sum(y_train == 1) * regular_ratio),
                                1: np.sum(y_train == 1)
                         })

X_train, y_train = rus.fit_resample(X_train, y_train)
print('After undersampling:')
print(np.unique(y_train, return_counts=True))

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Testing data shape:", X_test.shape)

################################
# prediction
################################

classifiers = [
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,
    SVClassifier,
    GaussianProcessClassifier,
    KNeighborsClassifier,
]

# TODO experiment with different models
# TODO hyperparameter tuning with OPTUNA

for classifier_func in classifiers:
    print(f'\nUsing classifier: {classifier_func.__name__}')
    try:
        clf = classifier_func(random_state=random_state, verbose=0)
    except TypeError:
        try:
            clf = classifier_func(random_state=random_state)
        except TypeError:
            try:
                clf = classifier_func(verbose=0)
            except TypeError:
                clf = classifier_func()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')



    cm = np.round(confusion_matrix(y_test, y_pred, normalize='true') * 100, 0)
    print(cm)



    # confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    classes = ['regular\nexcavation', 'collapse']
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=f'Confusion Matrix ({classifier_func.__name__})'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = cm[i, j]
            ax.text(j, i, f"({percent:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > 50 else "black")

    plt.tight_layout()
    plt.savefig(fr'{ANALYSES_FOLDER}\02_confusionmatrix\{classifier_func.__name__}.jpg', dpi=300)
    # plt.show()

