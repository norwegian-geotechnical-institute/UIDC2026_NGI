import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/combined_data.csv')

# Sort by Chainage or Tunnellength to ensure proper ordering
if 'Chainage' in df.columns:
    df = df.sort_values('Chainage', ascending=False).reset_index(drop=True)
elif 'Tunnellength [m]' in df.columns:
    df = df.sort_values('Tunnellength [m]').reset_index(drop=True)

print('Total samples:', len(df))
print('Collapse samples:', df['collapse'].sum())
print('Non-collapse samples:', (df['collapse'] == 0).sum())

# Identify continuous collapse sections
# A new section starts when current row is collapse (1) and previous row is not collapse (0)
df['is_collapse'] = df['collapse'] == 1
df['collapse_section'] = (df['is_collapse'] & (~df['is_collapse'].shift(1, fill_value=False))).cumsum()

# Set section to 0 for non-collapse rows
df.loc[~df['is_collapse'], 'collapse_section'] = 0

print('\nCollapse sections found:', df[df['collapse_section'] > 0]['collapse_section'].nunique())
print('\nCollapse section sizes:')
section_sizes = df[df['collapse_section'] > 0].groupby('collapse_section').size()
print(section_sizes)
print('\nStatistics:')
print('Mean section size:', section_sizes.mean())
print('Median section size:', section_sizes.median())
print('Min section size:', section_sizes.min())
print('Max section size:', section_sizes.max())

# Show some examples
print('\n\nExample rows from first collapse section:')
first_section = df[df['collapse_section'] == 1][['Chainage', 'Tunnellength [m]', 'collapse', 'collapse_section']].head(10)
print(first_section)

print('\n\nExample rows from second collapse section:')
second_section = df[df['collapse_section'] == 2][['Chainage', 'Tunnellength [m]', 'collapse', 'collapse_section']].head(10)
print(second_section)
