import pandas as pd
import matplotlib.pyplot as plt

from X_library import dictionaries

DATA_FOLDER = r'C:\Users\GEr\OneDrive - NGI\Research\UIDC2026_NGI\YS-IWHR-main\The data analyzed in section 5&6'



################################
# preprocessing
################################
dicts = dictionaries()

# combine datasets into one

dfs = []

for ds in ['d02_eigStable.csv', 'd05_eigStable.csv', 'd10_eigStable.csv',
           'd20_eigStable.csv']:
    dfs.append(pd.read_csv(fr'{DATA_FOLDER}\{ds}'))

df = pd.concat(dfs)

df.sort_values(by='Chainage', inplace=True)


# df_full = pd.read_csv(fr'{DATA_FOLDER}\CREC188_20171006.csv', encoding='gbk')
# df_full.rename(dicts.tbm_column_translation, axis=1, inplace=True)
# print(list(df_full.columns))
# print(df_full['operation_time'])

################################
# plotting
################################

fig, ax = plt.subplots()

ax.plot(df['operation_time'], df['chainage'], label='cutterhead_torque')
# ax.plot(df['chainage'], df['Total F'], label='Total F')

ax.legend()
ax.grid(alpha=0.5)

plt.tight_layout()
plt.show()

print('\n')
