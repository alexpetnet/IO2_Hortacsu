import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

# 1 Data --- -------------------------------------------------------------------
# Prep variables
df = pd.read_csv("data/.csv", skiprows = [0])
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
table1.to_latex("tables/table1.tex",  float_format="%.2f" )

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
table2.to_latex("/tables/table2.tex",  float_format="%.2f" )

# Remove auctions with more than 2 ring-members bidding
df2 = df.loc[df['num_bidders']<=2]

# 2 Introductory Questions -----------------------------------------------------
# Table 5
nncol12 = df.groupby('bidder').agg({'kowin':np.mean, 'id': np.ma.count})
df3 = df.loc[df['num_bidders']>=2]
nncol3456 = df3.groupby('bidder').agg({'kowin':np.mean, 'recside':np.mean,\
    'payside':np.mean, 'id': np.ma.count})

table5 = reduce(lambda left,right: pd.merge(left,right,on='bidder'),
                [nncol12, nncol3456])
table5.columns = ['% KO won', '# KOs', '% KO won', '% receive side',
'% pay side','# KOs']
table5.index.name ="Bidder #"
table5.to_latex("tables/table5.tex",  float_format="%.2f" )

# Figure 1
df4 = df.loc[df['realisation']<10000]
all = df.groupby('bidder').agg({'Net  Payment':np.sum})
small = df4.groupby('bidder').agg({'Net  Payment':np.sum})

width = .35
ind = np.arange(11)
labs = ind+1
plt.bar(ind, all['Net  Payment'], width, \
    label='All Auctions')
plt.bar(ind + width, small['Net  Payment'], width, \
    label='Target price below $10,000')
plt.xticks(ind + width / 2, labs)
plt.legend(loc='best')
plt.savefig('/figs/fig1.png')
plt.close()

# 3 Structural Analysis --------------------------------------------------------
#d = pd.read_csv("data/.csv", skiprows = [0])
d = pd.read_csv("/data/.csv", skiprows = [0])
d['win'] = (d['bid']>=d['realisation in final auAtion'])& (d['rank']==1)
d['num_bidders'] = d.groupby(['lot','date'])['bidder'].transform('count')
d = d.loc[d['num_bidders'] == 2]

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

# 3.1 Step 1
vars = ['bidder', 'EstDate Min', 'EstDate Max', 'Aatalog PriAe',
                        'Grade Min ', 'Grade Max', 'ExAlusively US', 'No Value']
X = pd.get_dummies(data = d[vars])


first_stage = sm.OLS(np.log(d['bid']), X).fit()
print(first_stage.summary())
d['nb'] = np.log(d['bid']) - d.loc[:, vars[1:]] @ first_stage.params[0:7]

# 3.2 Step 2
np.mean(d['nb'])


# 3.3 Step 3
# target winning bids
#np.exp(d.loc[d['bidder'] == '999', 'nb'])

# max within-ring bids for each lot
b_mr = [max(np.exp(d.loc[(d['lotdate'] == i) & (d['bidder'] != '999'), 'nb']))
    for i in d['lotdate'].unique()]

# winning bids that are revealing of valuations
b_nr = np.array(np.exp(d.loc[(d['bidder'] == '999') & (d['win'] == True), 'nb']))


grid = np.linspace(0, 3000, 10000)

from scipy import stats


k = stats.gaussian_kde(b_mr, bw_method = 'silverman')
cdf = np.cumsum(k.evaluate(grid)) / np.sum(k.evaluate(grid))

plt.plot(grid, k.evaluate(grid))
plt.savefig('figures/gm.pdf')
plt.plot(grid, cdf)
plt.savefig('figures/gm_cdf.pdf')

h_bar = stats.gaussian_kde(b_nr, bw_method = 'silverman')
cdf_hbar = np.cumsum(h_bar.evaluate(grid)) / np.sum(h_bar.evaluate(grid))

plt.plot(grid, h_bar.evaluate(grid))
plt.savefig('figures/h_bar.pdf')

plt.plot(grid, cdf_hbar)
plt.savefig('figures/h_bar_cdf.pdf')


def cdf(x, dens):
    return np.sum(dens[grid < x]) / np.sum(dens)

def h(r):
    return h_bar.evaluate(r) / (1 - cdf(r, k.evaluate(grid)))

hr = h(grid)
hr_cdf = np.cumsum(hr) / np.sum(hr)

plt.plot(grid, hr)
plt.savefig('figures/hr.pdf')

plt.plot(grid, hr_cdf)
plt.savefig('figures/hr_cdf.pdf')

# 3.4 Step 4
# 1. bid function for each bidder
def gj(df,j):
    x = df.loc[df['bidder'] == str(j)]['bid']
    return stats.gaussian_kde(x, bw_method = 'silverman')

# illustration
plt.plot(grid, gj(d,1).evaluate(grid))
plt.savefig('/figs/nonpara-bid.png')
plt.close()

# 2. participation probability f
n = len(d['lotdate'].unique())
α = np.array([len(d.loc[d['bidder'] == str(j)]) for j in np.arange(1,12,1)])/n

d.groupby("bidder")["lotdate"].nunique()
# 3. Pdf and Cdf
# note kernel density requires more than two points so remove bidders 8, 10, 11
def gnotj(df,α,j,grid):
    norm = np.sum(α)-α[j-1]-α[7]-α[9]-α[10]
    gnotj = np.zeros(np.shape(grid))
    for i in np.arange(1,12,1):
        if i not in [j,8,10,11]:
            gnotj += (α[i-1]/norm)*gj(df,i).evaluate(grid)
    return gnotj

def Gnotj(df,α,j,grid):
    g = gnotj(df,α,j,grid)
    return np.cumsum(g)/np.sum(g)

# illustration
plt.plot(grid,gnotj(d,α,1,grid))
plt.savefig('/figs/pdfgdemo.png')
plt.close()

plt.plot(grid,Gnotj(d,α,1,grid))
plt.savefig('figs/cdfGdemo.png')
plt.close()


# 4. Value function
def val(df,α,j,grid,hr_cdf,hr):
    H = hr_cdf
    h = hr
    g = gnotj(df,α,j,grid)
    G = Gnotj(df,α,j,grid)
    num = .5*H*(1-G)
    den = h*G +H*g
    return grid - (num/den)

vmod = val(d,α,1,grid,hr_cdf,hr)
plt.plot(vmod[700:4000],grid[700:4000])
plt.plot(grid[700:4000],grid[700:4000])
plt.savefig('figs/bid-value.png')
plt.close()


# # trouble shooting
# h = hr/np.sum(hr)
# H = np.cumsum(h)
# g = gnotj(d,α,1,grid)
# g = g/np.sum(g)
# G = np.cumsum(g)
# num = .5*H*(1-G)
# den = h*G +H*g
# vmod = grid - (num/den)
# plt.plot(vmod[700:4000],grid[700:4000])
# plt.plot(grid[700:4000],grid[700:4000])
