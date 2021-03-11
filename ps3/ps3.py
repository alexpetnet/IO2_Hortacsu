import numpy as np
import pandas as pd
from functools import reduce

# 1 Data --- -------------------------------------------------------------------
df = pd.read_csv("ps3/data/ps3.csv", skiprows = [0])

# Table 1:
df.columns
# Prep variables
df['value']=(df['EstDate Min']+df['EstDate Max'])/2
df['realisation']=df['realisation in final auAtion']
df['win'] = (df['realisation']>=df['bid'])& (df['rank']==1)
df['valwon']=df['win']*df['value']

# Make the table
col1234 = df.groupby("house").agg({'realisation' : [np.mean, np.std],
                        'bid' : [np.mean, np.std]})
col5 = df.groupby(['lot','date']).agg({'win' :np.sum, 'house':np.max}).\
    groupby('house').agg({'win':np.mean})
col6 = df.groupby('house').agg({"valwon" : np.sum})
val = df.groupby(['lot','date']).agg({'value' : np.max, 'house' : np.max})\
    .groupby('house').agg({'value': np.sum})
col6['%valwon'] = col6['valwon']/val['value']
col6=col6.drop('valwon',axis=1)
col7 = df.groupby('house')['lot'].nunique()

table1 = reduce(lambda left,right: pd.merge(left,right,on='house'),
                [col1234, col5, col6, col7])
table1.columns = ['Target auction mean', 'Target auction SD',
    'Knockout auction mean', 'Knockout auction SD', '% lots won',
    '% total est. value won', 'number of lots']
table1.to_latex("ps3/tables/table1.tex",index=False)

# 2 Introductory Questions -----------------------------------------------------

# 3 Structural Analysis --------------------------------------------------------
