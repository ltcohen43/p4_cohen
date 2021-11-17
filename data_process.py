# %%
import pandas as pd
import numpy as np

from plotnine import *

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# %%
#importing CSV
 dat = pd.read_csv('SalesBook_2013.csv')
 # select variables we will use in class.
 dat_ml = (dat
     .filter(['NBHD', 'PARCEL', 'LIVEAREA', 'FINBSMNT',  
         'BASEMENT', 'YRBUILT', 'CONDITION', 'QUALITY',
         'TOTUNITS', 'STORIES', 'GARTYPE', 'NOCARS',
         'NUMBDRM', 'NUMBATHS', 'ARCSTYLE', 'SPRICE',
         'DEDUCT', 'NETPRICE', 'TASP', 'SMONTH',
         'SYEAR', 'QUALIFIED', 'STATUS'])
     .rename(columns=str.lower) # changing the variable names to lowercase
     .query('totunits <= 2')) # drop homes that are not single family or duplexes
# %%

dat_ml =(
            dat_ml
             .query('yrbuilt!=0 & condition != "None"') # removing the houses with year built as 0 and condtion that are listed as None
             .assign(before1980 = lambda x: np.where(x.yrbuilt < 1980,1,0)) #making new variable to say if house is > 1980=0
)

# %%
#checking how many house have 0 as year built
dat_ml.yrbuilt.value_counts()
# %%
# look at our variable/features and imagine which ones might be good predictors of thea ge of a house.
# bathroom and quality
#do .astype to show the different categories instead of one boxplot
#bathroom seems to make show if you have more than 4 bathrooms the house it older than 1980
(ggplot(dat_ml, aes(x='numbaths.astype(str)',y='yrbuilt'))+geom_boxplot(color='orange'))
# %%
#quality : ordinal variable as E is the lowest and X is the highest
(ggplot(dat_ml, aes(x='quality.astype(str)',y='yrbuilt'))+geom_boxplot(color='orange'))
# %%

#https://hcad.org/hcad-resources/hcad-appraisal-codes/hcad-building-grade-adjustment/
# E- (lowest), E, E+, D-, D, D+, C-, C, C+, B-, B, B+, A-, A, A+, X-, X, X+ (highest)

dat_ml.quality.value_counts() 
replace_dictionary = {
    "E-":-0.3,
    "E":0.0,
    "E+":0.3,
    "D-":0.7,
    "D":1.0,
    "D+":1.3,
    "C-":1.7,
    "C":2.0,
    "C+":2.3,
    "B-":2.7,
    "B":3.0,
    "B+":3.3,
    "A-":3.7,
    "A":4.0,
    "A+":4.3,
    "X-":4.7,
    "X":5.0,
    "X+":5.3,
}
qual_ord = dat_ml.quality.replace(replace_dictionary)

dat_ml.condition.value_counts()
replace_dictionary = {
    "Excel":3,
    "VGood":2,
    "Good":1,
    "AVG":0,
    "Avg":0,
    "Fair":-1,
    "Poor":-2,
}

cond_ord = dat_ml.condition.replace(replace_dictionary)
# %%
#one-hot-encode or dummy variables
# arcstyle, ndhb, gartype

dat_ml.arcstyle.value_counts() #looking at each of the categories
dat_ml.gartype.value_counts()

pd.get_dummies(dat_ml.filter(['arcstyle']), drop_first=True)


# %%
