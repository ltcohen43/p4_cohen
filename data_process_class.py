# %%
import re

import pandas as pd
import numpy as np

from plotnine import *
# %%
# select variables we will use in class.
dat = (pd.read_csv('SalesBook_2013.csv', low_memory=False)
    .filter(['NBHD', 'PARCEL', 'LIVEAREA', 'FINBSMNT',  
        'BASEMENT', 'YRBUILT', 'CONDITION', 'QUALITY',
        'TOTUNITS', 'STORIES', 'GARTYPE', 'NOCARS',
        'NUMBDRM', 'NUMBATHS', 'ARCSTYLE', 'SPRICE',
        'DEDUCT', 'NETPRICE', 'TASP', 'SMONTH',
        'SYEAR', 'QUALIFIED', 'STATUS'])
    .rename(columns=str.lower)
    # drop homes that are not single family or duplexes
    .query('totunits <= 2 & yrbuilt != 0 & condition != "None"')
    .assign(
        before1980 = lambda x: np.where(x.yrbuilt < 1980, 1, 0),
        gartype = lambda x: np.where(x.gartype.isnull(), "Missing", x.gartype),
        )
    .sort_values(['parcel','syear', 'smonth'], ascending = False)
    .groupby(['parcel'])
    .first() # removes older selling moments of duplicate homes
    .reset_index()
    .drop(['nbhd', # don't want to deal with nbhd 
        'parcel', # don't want to have unique identifier in model
        'status'], # almost all 'I'.
        axis=1)) # don't want to deal with nbhd and dropcing parcel

# %%
arc_dummies = pd.get_dummies(dat.filter(['arcstyle']),
    drop_first=True)

# %%

replace_quality = {
    "E-":-0.3 , "E":0, "E+":0.3,
    "D-":0.7, "D":1, "D+":1.3,
    "C-":1.7, "C":2, "C+":2.3,
    "B-":2.7, "B":3, "B+":3.3,
    "A-":3.7, "A":4, "A+":4.3,
    "X-":4.7, "X":5, "X+":5.3 
}

replace_condition = {
    "Excel":3,
    "VGood":2,
    "Good":1,
    "AVG":0,
    "Avg":0,
    "Fair":-1,
    "Poor":-2
}

values_missing = {
    "basement":0, 
    "nocars": dat.nocars.median(),
    "numbdrm": dat.numbdrm.median(),
    'numbaths': dat.numbaths.median()}


# dat_ml.qualified.value_counts()
# dat_ml.gartype.str.contains("att", flags=re.IGNORECASE, regex=True).astype(int)

dat_ml = (dat.assign(
    quality = lambda x: x.quality.replace(replace_quality),
    condition = lambda x: x.condition.replace(replace_condition),
    attachedGarage = lambda x: x.gartype.str.contains("att",
        flags=re.IGNORECASE, regex=True).astype(int),
    detachedGarage = lambda x: x.gartype.str.contains("det",
        flags=re.IGNORECASE, regex=True).astype(int),
    carportGaragae = lambda x: x.gartype.str.contains("cp",
        flags=re.IGNORECASE, regex=True).astype(int),
    noGarage = lambda x: x.gartype.str.contains("none",
        flags=re.IGNORECASE, regex=True).astype(int),
    qualified = lambda x: np.where(x.qualified == "Q", 1, 0))
.drop(columns = ['gartype', 'qualified', 'arcstyle'])
.fillna(values_missing))

dat_ml = pd.concat([dat_ml, arc_dummies], axis=1)

# %%
# now fix missing
dat_ml.isnull().sum()/len(dat_ml)*100


# %%
dat_ml.to_pickle('dat_ml.pkl')



# %%
# dat_ml.gartype.value_counts()