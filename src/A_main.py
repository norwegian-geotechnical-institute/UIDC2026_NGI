
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from X_library import dictionaries


################################
# function definitions
################################


def plot_param(ax, dataframe, parameter):
    ax.plot(dataframe['Chainage'], dataframe[parameter])
    ax.set_ylabel(parameter)
    # ax.legend()
    ax.grid(alpha=0.5)


################################
# Static variables
################################

DATA_FOLDER = r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\YS-IWHR-main\The data analyzed in section 5&6'
ANALYSES_FOLDER = r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\analyses'


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
df['collapse'] = 0

df_collapses = pd.read_excel(r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\data\collapses.xlsx')

for i in range(len(df_collapses)):
    ids = df[(df['Chainage'] <= df_collapses.iloc[i]['Chainage start [m]']) & 
             (df['Chainage'] >= df_collapses.iloc[i]['Chainage end [m]'])].index
    df.loc[ids, 'collapse'] = 1


################################
# prediction
################################

classifier = RandomForestClassifier()

print(ghjklø)


################################
# plotting
################################


# lineplot
fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(18, 9))

plot_param(axs[0], df, 'p')

plot_param(axs[1], df, 'Pr')

plot_param(axs[2], df, 'RPM')

plot_param(axs[3], df, 'Total F')

plot_param(axs[4], df, 'Total T')

plot_param(axs[5], df, 'collapse')

plt.tight_layout()
plt.savefig(fr'{ANALYSES_FOLDER}\00_lineplot.jpg', dpi=300)
plt.show()

print('\n')

# scatterplot


parameters = ['p', 'Pr', 'RPM', 'Total F', 'Total T']

n_params = len(parameters)
n_figure = 1

fig = plt.figure(figsize=(18, 18))

for i in range(n_params):
    for j in range(n_params):
        ax = fig.add_subplot(n_params, n_params, n_figure)
        if parameters[i] == parameters[j]:
            ax.hist(df_advance[parameters[i]], color='grey',
                    bins=30, edgecolor='black')
        else:
            ax.scatter(df_advance[parameters[i]],
                        df_advance[parameters[j]], color='grey',
                        edgecolor='black', alpha=0.3, s=1)
            ax.set_ylabel(parameters[j].replace(' [', '\n['),
                            fontsize=fsize)
        ax.set_xlabel(parameters[i].replace(' [', '\n['),
                        fontsize=fsize)
        ax.tick_params(axis='both', labelsize=fsize)
        n_figure += 1

fig.suptitle(f'TBM {TBM}', y=0.99)

plt.tight_layout()
plt.show()

