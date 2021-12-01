# %%

#Installation


# import sys
# !{sys.executable} -m pip install dalex 

# import sys
# !{sys.executable} -m pip install scikit-learn

# shap install did not work
# shap
# %%

# Building our model
import pandas as pd
import numpy as np
import joblib # to save ml models
from plotnine import *

from sklearn import metrics
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# %%
#Build training and testing data
dat_ml = pd.read_pickle('dat_ml.pkl')

# Split our data into training and testing sets.
X_pred = dat_ml.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = dat_ml.before1980
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, y_pred, test_size = .34, random_state = 76) 
# %%
# Building our models
clfNB = GaussianNB()
clfGB = GradientBoostingClassifier()
clfNB.fit(X_train, y_train)
clfGB.fit(X_train, y_train)
ypred_clfNB = clfNB.predict(X_test)
ypred_clfGB = clfGB.predict(X_test) # set at 0.5 to decide 0's and 1's
# %%
 ypred_clfGB_prop = clfGB.predict_proba(X_test)
# Diagnosing our model with sklearn

# %%
metrics.plot_roc_curve(clfGB, X_test, y_test) # the curve shows its under 1 and good model
metrics.plot_roc_curve(clfNB, X_test, y_test) # straight like is like 50/50 chance, like flipping a coin
# %%
metrics.confusion_matrix(y_test, ypred_clfNB)
# %%
metrics.confusion_matrix(y_test, ypred_clfGB)

# Now we can build our own feature importance chart

# %%
df_features = pd.DataFrame(
    {'f_names': X_train.columns, 
    'f_values': clfGB.feature_importances_}).sort_values('f_values', ascending = False).head(12)

# Python sequence slice addresses 
# can be written as a[start:end:step]
# and any of start, stop or end can be dropped.
# a[::-1] is reverse sequence.
f_names_cat = pd.Categorical(
    df_features.f_names,
    categories=df_features.f_names[::-1])

df_features = df_features.assign(f_cat = f_names_cat)

(ggplot(df_features,
    aes(x = 'f_cat', y = 'f_values')) +
    geom_col() +
    coord_flip() +
    theme_bw()
)
# %%
# Variable reduction

# build reduced model
compVars = df_features.f_names[::-1].tolist()# flip order of the features

X_pred_reduced = dat_ml.filter(compVars, axis = 1)
y_pred = dat_ml.before1980

X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    X_pred_reduced, y_pred, test_size = .34, random_state = 76)  

clfGB_reduced = GradientBoostingClassifier()
clfGB_reduced.fit(X_train_reduced, y_train)
ypred_clfGB_red = clfGB_reduced.predict(X_test_reduced)

# %%
print(metrics.classification_report(ypred_clfGB_red, y_test)) #use the weighted average value
metrics.confusion_matrix(y_test, ypred_clfGB_red)

# %%
#Saving our models
joblib.dump(clfNB, 'models/clfNB.pkl')
joblib.dump(clfGB, 'models/clfGB.pkl')
joblib.dump(clfGB_reduced, 'models/clfGB_final.pkl')
df_features.f_names[::-1].to_pickle('models/compVars.pkl')