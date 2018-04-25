import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy as pt
import sklearn.linear_model as lm
import matplotlib
# from matplotlib import ggplot
import ggplot as ggp
import pylab as plt
# from ggplot import aes
from ggplot import *

def dates_interval(votedon_date):
    # print(votedon_date)
    interval = 0
    due_date = '2018-04-25'
    # for i in range(0, due_date):
    #     interval += int(due_date[i]) - int()
    interval += (int(due_date[0:4]) - int(votedon_date[0:4])) * 365 * 24 * 60 * 60
    interval += (int(due_date[5:7]) - int(votedon_date[5:7])) * 12 * 24 * 60 * 60
    interval += (int(due_date[8:10]) - int(votedon_date[8:10])) * 24 * 60 * 60
    # print(interval)
    return interval

# print(dates_interval('2018-04-24'))

# загружаем файл с данными
df = pd.read_csv('pictures-train.tsv', sep='\t',  skiprows=[0])

df.columns = ['etitle', 'region', 'takenon', 'votedon', 'author_id', 'votes', 'viewed', 'n_comments'] # 'a', 'b']

df = df.loc[df['votes'] > 0]
df = df.loc[df['viewed'] > 0]
df = df.loc[df['n_comments'] > 0]

# """
df['votedon'] = df['votedon'].astype(str)
df = df.loc[df['votedon'].map(len) == 10]
df['votedon'] = df['votedon'].apply(dates_interval)
# """

# df.sort(['c1','c2'], ascending=[False,True])
df = df.sort_values(by=['viewed'], ascending=[True])
# sort(['votedon'], ascending=[True])

x = df.iloc[:, [3, 5, 6, 7]] # 3,
# print(x)
y = df.iloc[:, -1] # -1

# print(y)

x_ = sm.add_constant(x)
smm = sm.OLS(y, x_)
# запускаем расчет модели
res = smm.fit()
# теперь выведем параметры рассчитанной модели
print(res.params)

"""
# create an empty model
skm = lm.LinearRegression() # LogisticRegression() # .LinearRegression()
# calculate parameters
skm.fit(x, y)
# show them
print(skm.intercept_, skm.coef_)
"""

plt = ggp.ggplot(aes(x="n_comments", y="y"), data=x_) + geom_point()

plt.show()

# ggp + geom_point()
# print(ggp + geom_point())

# print(x)
# print(y)