import numpy as np
import pandas as pd
from functools import reduce

# 1 Data --- -------------------------------------------------------------------
# Prep variables
df = pd.read_csv("ps3/data/ps3.csv", skiprows = [0])
df['value']=(df['EstDate Min']+df['EstDate Max'])/2
df['realisation']=df['realisation in final auAtion']
df['win'] = (df['bid']>=df['realisation'])& (df['rank']==1)
df['valwon']=df['win']*df['value']
df['id']= df['Data NuEer']
df['lotdate'] = df['lot']+[str(i) for i in df['date']]
df['kowin'] = df['rank']==1
df['recside'] = df['Net  Payment']>0
df['payside'] = df['Net  Payment']<0
# Table 1
col1234 = df.groupby("house").agg({'realisation' : [np.mean, np.std],
                        'bid' : [np.mean, np.std]})
col5 = df.groupby(['lot','date']).agg({'win' :np.sum, 'house':np.max}).\
    groupby('house').agg({'win':np.mean})
col5
col6 = df.groupby('house').agg({"valwon" : np.sum})
val = df.groupby(['lot','date']).agg({'value' : np.max, 'house' : np.max})\
    .groupby('house').agg({'value': np.sum})
col6['%valwon'] = col6['valwon']/val['value']
col6=col6.drop('valwon',axis=1)
col7 = df.groupby('house')['lotdate'].nunique()

table1 = reduce(lambda left,right: pd.merge(left,right,on='house'),
                [col1234, col5, col6, col7])
table1.columns = ['Target mean', 'Target SD',
    'Bid mean', 'Bid SD', '% lots won',
    '% value won', '# lots']
table1.index.name ="House"
table1.to_latex("ps3/tables/table1.tex",  float_format="%.2f" )

# Table 2
df['num_bidders'] = df.groupby(['lot','date']).id.transform('nunique')
ncol1234= df.groupby(['lotdate']).agg({'realisation':np.max, \
    'bid': np.median, 'num_bidders':np.max}).\
    groupby('num_bidders').agg({'realisation': [np.mean,np.std],
    'bid':[np.mean,np.std]})
ncol5 = df.groupby('lotdate').agg({'win' :np.sum, 'num_bidders': np.max}).\
    groupby('num_bidders').agg({'win':np.mean})
ncol6 = df.groupby('num_bidders')['lotdate'].nunique()

table2 = reduce(lambda left,right: pd.merge(left,right,on='num_bidders'),
                [ncol1234, ncol5, ncol6])
table2.columns = ['Target mean', 'Target SD',
    'Bid mean', 'Bid SD', '% lots won',
    '# lots']
table2.index.name ="# Bidders"
table2.to_latex("ps3/tables/table2.tex",  float_format="%.2f" )

# Remove auctions with more than 2 ring-members bidding
df2 = df.loc[df['num_bidders']<=2]

# 2 Introductory Questions -----------------------------------------------------
# Table 5
nncol12 = df.groupby('bidder').agg({'kowin':np.mean, 'id': np.ma.count})
df3 = df.loc[df['num_bidders']>=2]
nncol3456 = df3.groupby('bidder').agg({'kowin':np.mean, 'recside':np.mean,\
    'payside':np.mean, 'id': np.ma.count})

table5 = table2 = reduce(lambda left,right: pd.merge(left,right,on='bidder'),
                [nncol12, nncol3456])
table5.columns = ['% KO won', '# KOs', '% KO won', '% receive side',
'% pay side','# KOs']
table5.index.name ="Bidder #"
table5.to_latex("ps3/tables/table5.tex",  float_format="%.2f" )

# Figure 1


# 3 Structural Analysis --------------------------------------------------------


d = pd.read_csv("data/ps3.csv", skiprows = [0])
d['win'] = (d['bid']>=d['realisation in final auAtion'])& (d['rank']==1)
d['num_bidders'] = d.groupby(['lot','date'])['bidder'].transform('count')
d = d.loc[d['num_bidders'] <= 2]

d['lotdate'] = d['lot'] + [str(i) for i in d['date']]
for l in d['lotdate'].unique():
    row = d[d['lotdate'] == l].iloc[0, :]
    row['bid'] = row['realisation in final auAtion']
    row['bidder'] = 999 # 999 is code for target
    d = d.append(row)

d = d.drop(columns = ['profit', 'Net  Payment', 'rank'])
d['bidder'] = [str(i) for i in d['bidder']]
d = d.dropna()
d

import statsmodels.api as sm

vars = ['bidder', 'EstDate Min', 'EstDate Max', 'Aatalog PriAe',
                        'Grade Min ', 'Grade Max', 'ExAlusively US', 'No Value']
X = pd.get_dummies(data = d[vars])


first_stage = sm.OLS(np.log(d['bid']), X).fit()
print(first_stage.summary())
d['nb'] = np.log(d['bid']) - d.loc[:, vars[1:]] @ first_stage.params[0:7]

np.mean(d['nb'])

# target winning bids
#np.exp(d.loc[d['bidder'] == '999', 'nb'])

# max within-ring bids for each lot
b_mr = [max(np.exp(d.loc[(d['lotdate'] == i) & (d['bidder'] != '999'), 'nb']))
    for i in d['lotdate'].unique()]

# winning bids that are revealing of valuations
b_nr = np.array(np.exp(d.loc[(d['bidder'] == '999') & (d['win'] == True), 'nb']))
b_nr


grid = np.linspace(0, 3000, 10000)

from scipy import stats
import matplotlib.pyplot as plt



k = stats.gaussian_kde(b_mr, bw_method = 'silverman')
cdf = np.cumsum(k.evaluate(grid)) / np.sum(k.evaluate(grid))

plt.plot(grid, k.evaluate(grid))
plt.plot(grid, cdf)

h_bar = stats.gaussian_kde(b_nr, bw_method = 'silverman')
cdf_hbar = np.cumsum(h_bar.evaluate(grid)) / np.sum(h_bar.evaluate(grid))

plt.plot(grid, h_bar.evaluate(grid))

plt.plot(grid, cdf_hbar)


def cdf(x, dens):
    return np.sum(dens[grid < x]) / np.sum(dens)

cdf(100, k.evaluate(grid))

def h(r):
    return k.evaluate(r) / (1 - cdf(r, k.evaluate(grid)))

hr = h(grid)

plt.plot(grid, hr)
