
import sys, os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
        elif len(ids) > 1:
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

DATA_FOLDER = r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\YS-IWHR-main\The data analyzed in section 5&6'
ANALYSES_FOLDER = r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\analyses'
COLLAPSE_EXCEL = r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\data\collapses.xlsx'
INPUT_FEATURES = [
    'T(P,V/N)-b1', 'T(P,V/N)-b2', 'T(P,V/N)-a', 'T(P,V/N)-R2',
                  'T(V/N)-b', 'T(V/N)-a', 'T(V/N)-R2', 'P(V)-slope', 'P(V)-a',
                  'P(V)-R2', 'Wp(WT)-slope', 'Wp(WT)-a', 'Wp(WT)-R2',
                   'T(P)-slope', 'T(P)-a', 'T(P)-R2', 'T-m', 'T-slope', 'T-R2',
                   'P-m', 'P-slope', 'P-R2', 'V-m', 'V-slope', 'V-R2',
                   'Vs-m', 'Vs-slope', 'TN-m', 'TN-slope', 'TN-R2', 'PV-m',
                   'PV-slope', 'PV-R2', 'V/N-m', 'V/N-slope', 'V/N-R2',
                   'N-m', 'Ns-m', 'I-m', 'V0-m', 'V/Vs-min', 'T/P-min',
                   'T/P-max', 'T/P-m', 'T/P-std', 'Tpi-min', 'Tpi-max',
                   'Tpi-m', 'Tpi-std', 'Fpi-min', 'Fpi-max', 'Fpi-m',
                   'Fpi-std', 'Tpi-slope', 'Tpi-R2:1-sse/sst', 'WT/WP-slope',
                   'WT/WP-R2:1-sse/sst', 'T(P*V/N)-slope', 'T(P*V/N)-a',
                   'T(P*V/N)-R2', 'Fpi', 'P*V/N', 'T/P', 'Tpi', 'dt', 'dz',
                   'workP', 'workT',
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

# discretize dataset
# df_disc = discretize(df, 1, INPUT_FEATURES, chain_col='Tunnellength [m]')
# df_disc = df_disc[df_disc['n dp'] > 0]
# df = df_disc
# print(np.unique(df_disc['collapse'], return_counts=True))

################################
# prediction
################################

X = df[INPUT_FEATURES].values
y = df['collapse'].values

print(np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = HistGradientBoostingClassifier()
# clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

cm = np.round(confusion_matrix(y_test, y_pred, normalize='true') * 100, 0)
print(cm)

################################
# plotting
################################

# rename column names to be more readable
df.rename({'p': 'penetration\n[mm/rev]',
           'Pr': 'advance rate\n[mm/min]',
           'RPM': 'cutterhead rotations\n[rpm]',
           'Total T': 'cutterhead torque\n[kNm]',
           'Total F': 'thrust\n[kN]',
           'Fpi': 'Field Penetration Index',
           'Tpi': 'drilling efficiency index\nTPI'}, axis=1, inplace=True)

# lineplot
fig, axs = plt.subplots(ncols=1, nrows=8, figsize=(18, 10), sharex=True)

plot_param(axs[0], df, 'penetration\n[mm/rev]')
plot_param(axs[1], df, 'advance rate\n[mm/min]')
plot_param(axs[2], df, 'cutterhead rotations\n[rpm]')
plot_param(axs[3], df, 'thrust\n[kN]')
plot_param(axs[4], df, 'cutterhead torque\n[kNm]')
plot_param(axs[5], df, 'Field Penetration Index')
plot_param(axs[6], df, 'drilling efficiency index\nTPI')
plot_param(axs[7], df, 'collapse')
axs[7].set_xlabel('Tunnellength [m]')

plt.tight_layout()
plt.savefig(fr'{ANALYSES_FOLDER}\00_lineplot.jpg', dpi=300)
plt.show()


# scatterplot
parameters = ['penetration\n[mm/rev]', 'advance rate\n[mm/min]',
              'cutterhead rotations\n[rpm]', 'thrust\n[kN]',
              'cutterhead torque\n[kNm]', 'Field Penetration Index',
              'drilling efficiency index\nTPI']

n_params = len(parameters)
n_figure = 1

fig = plt.figure(figsize=(18, 18))

for i in range(n_params):
    for j in range(n_params):
        ax = fig.add_subplot(n_params, n_params, n_figure)
        if parameters[i] == parameters[j]:
            ax.hist(df[parameters[i]], color='grey', bins=30,
                    edgecolor='black')
        else:
            df_collaps = df[df['collapse'] == 1]
            df_regular = df[df['collapse'] != 1]
            ax.scatter(df_regular[parameters[i]], df_regular[parameters[j]],
                       color='grey', edgecolor='black', alpha=0.3, s=1)
            ax.scatter(df_collaps[parameters[i]], df_collaps[parameters[j]],
                       color='red', alpha=1, s=2)
            ax.set_ylabel(parameters[j].replace(' [', '\n['),
                          fontsize=8)
        ax.set_xlabel(parameters[i].replace(' [', '\n['),
                        fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        n_figure += 1

plt.tight_layout()
plt.savefig(fr'{ANALYSES_FOLDER}\01_pairplot.jpg', dpi=300)
plt.show()


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
    title='Binary Confusion Matrix'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate cells with percentages
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        percent = cm[i, j]
        ax.text(j, i, f"({percent:.1f}%)", ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black")

plt.tight_layout()
plt.savefig(fr'{ANALYSES_FOLDER}\02_confusionmatrix.jpg', dpi=300)
plt.show()

