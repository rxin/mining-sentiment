
from pandas import *
import matplotlib.pyplot as plt

df = read_csv('perf-alpha.csv')

# vary alpha
plt.figure()

df_multi_nostem = df_alpha = df[ (df['binary'] == False) & (df['stem'] == False) ]
series_multi_nostem = Series(df_multi_nostem['F1'], index=df_multi_nostem['alpha'])
series_multi_nostem.plot('multi', style='-')

plt.xlabel('alpha')
plt.ylabel('F1 score')

plt.savefig("alpha.pdf")

