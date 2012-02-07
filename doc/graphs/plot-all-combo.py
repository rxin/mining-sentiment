
from pandas import *
import matplotlib.pyplot as plt

df = read_csv('perf.csv')

# vary alpha
plt.figure()

df_multi_nostem = df_alpha = df[ (df['binary'] == False) & (df['stem'] == False) ]
series_multi_nostem = Series(df_multi_nostem['F1'], index=df_multi_nostem['alpha'])
series_multi_nostem.plot('multi', style='-')

df_multi_stem = df_alpha = df[ (df['binary'] == False) & (df['stem'] == True) ]
series_multi_stem = Series(df_multi_stem['F1'], index=df_multi_stem['alpha'])
series_multi_stem.plot('multi-stem', style='--')

df_binary_nostem = df_alpha = df[ (df['binary'] == True) & (df['stem'] == False) ]
series_binary_nostem = Series(df_binary_nostem['F1'], index=df_binary_nostem['alpha'])
series_binary_nostem.plot('binomial', style='-.')

df_binary_stem = df_alpha = df[ (df['binary'] == True) & (df['stem'] == True) ]
series_binary_stem = Series(df_binary_stem['F1'], index=df_binary_nostem['alpha'])
series_binary_stem.plot('binomial-stem', style=':')

plt.legend(loc='lower right')
#plt.margins(0)

plt.xlabel('alpha')
plt.ylabel('F1 score')

plt.savefig("all-combo.pdf")

